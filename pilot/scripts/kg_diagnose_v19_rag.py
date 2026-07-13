#!/usr/bin/env python3
"""진단 v19: KG-RAG 방식 re-ranking.

v17(56.2%): 환자 프로필 + 질환 이름 → LLM 선택
v19: 환자 프로필 + 질환별 "환자 증상과 매칭되는 KG 증상" → LLM 판단

핵심: KG를 단순 후보 선정이 아닌 re-ranking의 참고 자료로 활용.
각 후보 질환에 대해 "환자가 가진 증상 중 이 질환에 해당하는 것"을 명시.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")
KG_CACHE = RESULTS_DIR / "kg_v3_cache.json"

NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'does','have','your','you','the','and','for','are','with','that','this','from','been','were','being','which','their','than','other','about','into','over','some','only','very','also','just','more','most','such','much','will','would','could','should','make','like','time','when','what','where','how','who','all','each','every','both','few','any','not','can','may','her','his','its','our','they','them','then','had','has','him','but','one','two','way','day','did','get','got','let','say','she','too','use','yes','yet','now','new','old','see','own','why','try','ask','set','related','reason','consulting','significant','measured','thermometer','either','believe','racing','missing','beat','fast','irregularly','problems','situation','associated','inability','speak','trouble','keeping','opening','raising','annoying','else','body','somewhere','anywhere','nowhere','recently','currently','usually','often','sometimes','worse','better'}

RERANK_PROMPT = """Patient: {age}yo {sex}
Chief complaint: {chief_complaint}

{clinical_info}

For each candidate, matching symptoms from medical literature (PubMed KG):
{candidates_with_matches}

Which diagnosis BEST explains this patient's presentation?
Answer with ONLY the number (1-{n})."""


def load_umls_names():
    cp = {}; can = defaultdict(set)
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cp: cp[p[0]] = p[14].strip()
    return dict(can), cp


def load_ddxplus():
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]
        diseases[dn] = {"cui": dc, "umls_name": icd_map[dn]["umls_name"], "fr": info.get("cond-name-fr", "")}
        fr2cui[info.get("cond-name-fr", "")] = dc; cui2name[dc] = dn
    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]
    return diseases, fr2cui, ev_info, cui2name


def load_kg():
    with open(KG_CACHE) as f: cache = json.load(f)
    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v
    return pc


def build_ds(pc, dcs):
    ds = defaultdict(dict); scuis = set()
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt; scuis.add(b)
        if b in dcs: ds[b][a] = cnt; scuis.add(a)
    return dict(ds), scuis


def build_aho(scuis, can):
    aho = ahocorasick.Automaton()
    for cui in scuis:
        for name in can.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) < 4 or lo in STOPWORDS: continue
            try: aho.add_word(lo, (lo, cui))
            except: pass
    aho.make_automaton()
    return aho


def text_match(evidences, ev_info, aho):
    cuis = set()
    for ev in evidences:
        parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
        info = ev_info.get(base, {})
        if info.get("is_antecedent"): continue
        terms = []
        bc = re.sub(r"_.*", "", base); bc = re.sub(r"xx$", "", bc)
        if len(bc) >= 3 and bc not in STOPWORDS: terms.append(bc)
        q = info.get("question_en", "")
        if q:
            text = re.sub(r"\(.*?\)", "", q); text = re.sub(r"[?.,;:!]", "", text)
            terms.extend(w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) >= 3)
            ql = q.lower()
            for ph in ["chest pain","sore throat","shortness of breath","difficulty breathing","weight loss","weight gain","loss of consciousness","muscle pain","muscle spasm","nasal congestion","runny nose","skin lesion","skin rash","black stool","bloody stool","heart palpitation","double vision","swollen"]:
                if ph in ql: terms.append(ph)
        if value:
            val_en = info.get("value_en", {}).get(value, "")
            if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                vc = re.sub(r"\([rl]\)", "", val_en.lower()).strip()
                if vc and vc not in STOPWORDS:
                    terms.append(vc)
                    if "pain" in q.lower():
                        terms.append(f"{vc} pain")
                        for part in vc.split():
                            if part not in STOPWORDS and len(part) >= 4: terms.append(f"{part} pain")
        pt = " . ".join(terms)
        for ei, (n, cui) in aho.iter(pt):
            si = ei - len(n) + 1
            if si > 0 and pt[si - 1].isalpha(): continue
            if ei + 1 < len(pt) and pt[ei + 1].isalpha(): continue
            cuis.add(cui)
    return cuis


def structured_patient_profile(evidences, ev_info, age, sex):
    pain_info = []; skin_info = []; respiratory = []; cardiac = []
    gi = []; neuro = []; general = []; history = []
    for ev in evidences:
        parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
        info = ev_info.get(base, {}); q = info.get("question_en", "")
        if info.get("is_antecedent"):
            if value:
                val_en = info.get("value_en", {}).get(value, value)
                history.append(f"{q}: {val_en}" if q else value)
            elif q:
                history.append(re.sub(r"Do you |Have you |Are you |Did you ", "", q).rstrip("?"))
            continue
        val_en = ""
        if value:
            val_en = info.get("value_en", {}).get(value, value)
            if val_en and val_en.lower() in ("na", "nowhere", "n"): val_en = ""
        q_lower = q.lower() if q else ""
        if "douleur" in base or "pain" in q_lower or "_dlr" in base:
            if "endroitducorps" in base and val_en: pain_info.append(f"Location: {val_en}")
            elif "carac" in base and val_en: pain_info.append(f"Character: {val_en}")
            elif "intens" in base and value: pain_info.append(f"Intensity: {value}/10")
            elif "irrad" in base and val_en: pain_info.append(f"Radiation: {val_en}")
            elif "soudain" in base and value: pain_info.append(f"Onset speed: {value}/10")
            elif "precis" in base and value: pain_info.append(f"Precision: {value}/10")
            elif base == "douleurxx": pain_info.insert(0, "Pain: present")
            continue
        if "lesions_peau" in base or "skin" in q_lower or "rash" in q_lower:
            if "endroitducorps" in base and val_en: skin_info.append(f"Location: {val_en}")
            elif "couleur" in base and val_en: skin_info.append(f"Color: {val_en}")
            elif "desquame" in base: skin_info.append("Peeling: yes")
            elif "elevee" in base: skin_info.append("Elevated: yes")
            elif "plusqu1cm" in base: skin_info.append("Size >1cm: yes")
            elif "prurit" in base: skin_info.append("Pruritic: yes")
            elif base == "lesions_peau": skin_info.insert(0, "Skin lesion: present")
            continue
        if "oedeme" in base:
            if "endroitducorps" in base and val_en: general.append(f"Edema location: {val_en}")
            elif base == "oedeme": general.append("Edema: present")
            continue
        q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your |Does the ", "", q).rstrip("?").strip() if q else base
        if any(k in q_lower for k in ["breath", "cough", "wheez", "dyspn", "toux", "expecto"]):
            respiratory.append(q_clean + (f": {val_en}" if val_en else ""))
        elif any(k in q_lower for k in ["heart", "palpit", "chest"]):
            cardiac.append(q_clean + (f": {val_en}" if val_en else ""))
        elif any(k in q_lower for k in ["nausea", "vomit", "diarr", "stool", "abdom", "swallow"]):
            gi.append(q_clean + (f": {val_en}" if val_en else ""))
        elif any(k in q_lower for k in ["consciousness", "dizz", "numb", "weak", "paralys", "vision", "speech"]):
            neuro.append(q_clean + (f": {val_en}" if val_en else ""))
        else:
            general.append(q_clean + (f": {val_en}" if val_en else ""))
    sections = []
    if pain_info: sections.append("PAIN: " + "; ".join(pain_info))
    if skin_info: sections.append("SKIN: " + "; ".join(skin_info))
    if respiratory: sections.append("RESPIRATORY: " + "; ".join(respiratory))
    if cardiac: sections.append("CARDIAC: " + "; ".join(cardiac))
    if gi: sections.append("GI: " + "; ".join(gi))
    if neuro: sections.append("NEURO: " + "; ".join(neuro))
    if general: sections.append("OTHER: " + "; ".join(general))
    if history: sections.append("HISTORY: " + "; ".join(history))
    return "\n".join(sections)


def main():
    print("=" * 80, flush=True)
    print("진단 v19: KG-RAG re-ranking", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] 로드...", flush=True)
    can, cp = load_umls_names()
    diseases, fr2cui, ev_info, cui2name = load_ddxplus()
    dcs = {v["cui"] for v in diseases.values()}
    pc = load_kg()
    ds, scuis = build_ds(pc, dcs)
    aho = build_aho(scuis, can)
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    # Prior
    age_sex_disease = defaultdict(Counter)
    with open("data/ddxplus/release_train_patients.csv") as f:
        for row in csv.DictReader(f):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            age_bin = min(int(row.get("AGE", 0)) // 10 * 10, 80)
            age_sex_disease[(age_bin, row.get("SEX", "M"))][tdc] += 1
    print(f"  KG: {len(pc):,} 쌍", flush=True)

    print("\n[2] 테스트 데이터...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "0"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})
    print(f"  {len(patients):,}명", flush=True)

    # [3] Bayesian+prior top-10
    print("\n[3] Bayesian+prior top-10...", flush=True)
    t0 = time.time()
    candidates = []; baseline = 0; n = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        ps = text_match(p["evidences"], ev_info, aho)
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            ll = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in ps)
            age_bin = min(int(p["age"]) // 10 * 10, 80)
            pr = age_sex_disease[(age_bin, p["sex"])]
            pt = sum(pr.values())
            prior = (pr.get(dc, 0) + 1) / (pt + len(dcs)) if pt > 0 else 1.0/len(dcs)
            sc[dc] = ll + math.log(prior)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        topk = [dc for dc, _ in ranked[:10]]
        candidates.append({"patient": p, "true_dc": tdc, "topk": topk, "matched_cuis": ps})
        if topk and topk[0] == tdc: baseline += 1
    in_topk = sum(1 for c in candidates if c["true_dc"] in c["topk"])
    print(f"  Baseline @1={100*baseline/n:.1f}%, in_top10={100*in_topk/n:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    # [4] RAG 프롬프트 생성
    print("\n[4] KG-RAG 프롬프트...", flush=True)
    prompts = []
    for c in candidates:
        p = c["patient"]
        profile = structured_patient_profile(p["evidences"], ev_info, p["age"], p["sex"])
        ie = p.get("initial", "")
        ie_info = ev_info.get(ie, {})
        chief = ie_info.get("question_en", ie) if ie else "not specified"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()

        # 각 후보 질환에 대해 환자 CUI와 매칭되는 KG 증상 표시
        patient_cuis = c["matched_cuis"]
        cand_lines = []
        for i, dc in enumerate(c["topk"]):
            dname = cui2name.get(dc, cp.get(dc, dc))
            disease_syms = ds.get(dc, {})
            # 환자 CUI 중 이 질환의 KG 증상과 겹치는 것
            matched = [(cui, disease_syms[cui]) for cui in patient_cuis if cui in disease_syms]
            matched.sort(key=lambda x: -x[1])
            if matched:
                match_str = ", ".join(cp.get(cui, cui) for cui, _ in matched[:6])
                cand_lines.append(f"{i+1}. {dname} (matching: {match_str})")
            else:
                cand_lines.append(f"{i+1}. {dname}")
        cands = "\n".join(cand_lines)

        prompt = RERANK_PROMPT.format(
            age=p["age"], sex="Male" if p["sex"] == "M" else "Female",
            chief_complaint=chief, clinical_info=profile,
            candidates_with_matches=cands, n=len(c["topk"]),
        )
        prompts.append(prompt)

    print(f"  프롬프트: {len(prompts):,}개", flush=True)

    # [5] vLLM
    print("\n[5] vLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=16)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)", flush=True)

    # [6] 평가
    print("\n[6] 평가...", flush=True)
    t1 = t3 = t5 = pf = 0
    for c, out in zip(candidates, outputs):
        answer = out.outputs[0].text.strip()
        m = re.search(r"(\d+)", answer)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(c["topk"]):
                reranked = list(c["topk"]); chosen = reranked.pop(idx); reranked.insert(0, chosen)
            else: reranked = c["topk"]; pf += 1
        else: reranked = c["topk"]; pf += 1
        tdc = c["true_dc"]
        if reranked[0] == tdc: t1 += 1
        if tdc in reranked[:3]: t3 += 1
        if tdc in reranked[:5]: t5 += 1
    nt = len(candidates)
    print(f"\n  Baseline: @1={100*baseline/nt:.1f}%", flush=True)
    print(f"  Re-ranked: @1={100*t1/nt:.1f}% @3={100*t3/nt:.1f}% @5={100*t5/nt:.1f}%", flush=True)
    print(f"  parse_fail={pf:,}", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"Baseline @1={100*baseline/nt:.1f}% → Re-ranked @1={100*t1/nt:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
