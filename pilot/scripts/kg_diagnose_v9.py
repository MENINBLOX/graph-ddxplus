#!/usr/bin/env python3
"""진단 v9: 다단계 re-ranking (10→3→1).

v7(48.1%): Bayesian top-10 → LLM 1회 → top-1
v9: Bayesian top-10 → LLM round1 top-3 → LLM round2 top-1

LLM이 3개 중 선택하는 것이 10개 중 선택보다 정확할 것으로 기대.
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

NOISE_CUIS = {
    "C0150312", "C0442743", "C0039082", "C0221423", "C1457887",
    "C0205390", "C0442804", "C3839861", "C0332157", "C1457868",
    "C0445223", "C1272751", "C0015663", "C0277814", "C5202885",
    "C0153933", "C0585362",
}
STOPWORDS = {
    "does", "have", "your", "you", "the", "and", "for", "are", "with",
    "that", "this", "from", "been", "were", "being", "which", "their",
    "than", "other", "about", "into", "over", "some", "only", "very",
    "also", "just", "more", "most", "such", "much", "will", "would",
    "could", "should", "make", "like", "time", "when", "what", "where",
    "how", "who", "all", "each", "every", "both", "few", "any", "not",
    "can", "may", "her", "his", "its", "our", "they", "them", "then",
    "had", "has", "him", "but", "one", "two", "way", "day", "did",
    "get", "got", "let", "say", "she", "too", "use", "yes", "yet",
    "now", "new", "old", "see", "own", "why", "try", "ask", "set",
    "related", "reason", "consulting", "significant", "measured",
    "thermometer", "either", "believe", "racing", "missing", "beat",
    "fast", "irregularly", "problems", "situation", "associated",
    "inability", "speak", "trouble", "keeping", "opening", "raising",
    "annoying", "else", "body", "somewhere", "anywhere", "nowhere",
    "recently", "currently", "usually", "often", "sometimes",
    "worse", "better",
}

PROMPT_R1 = """A patient presents with:
{symptoms}

Which 3 of these diseases are MOST LIKELY? List the 3 numbers separated by commas.
{candidates}

Answer format: N,N,N"""

PROMPT_R2 = """A patient presents with:
{symptoms}

Which disease is MOST LIKELY?
{candidates}

Answer with ONLY the number (1-3)."""


def load_umls_names():
    print("  MRCONSO...", end="", flush=True)
    can = defaultdict(set)
    cp = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cp:
                    cp[p[0]] = p[14].strip()
    print(f" {len(can):,}", flush=True)
    return dict(can), cp


def load_ddxplus():
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f:
        cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)
    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]
        diseases[dn] = {"cui": dc, "umls_name": icd_map[dn]["umls_name"],
                        "fr": info.get("cond-name-fr", "")}
        fr2cui[info.get("cond-name-fr", "")] = dc
        cui2name[dc] = dn
    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""),
                        "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"):
                    ev_info[eid]["value_en"][k] = v["en"]
    return diseases, fr2cui, ev_info, cui2name


def load_kg():
    with open(KG_CACHE) as f:
        cache = json.load(f)
    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v
    return pc


def build_ds(pc, dcs):
    ds = defaultdict(dict); scuis = set()
    for (a, b), cnt in pc.items():
        if a in NOISE_CUIS or b in NOISE_CUIS: continue
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
        parts = ev.split("_@_"); base = parts[0]
        value = parts[1] if len(parts) > 1 else None
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
            for ph in ["chest pain","sore throat","shortness of breath","difficulty breathing",
                        "weight loss","weight gain","loss of consciousness","muscle pain",
                        "muscle spasm","nasal congestion","runny nose","skin lesion","skin rash",
                        "black stool","bloody stool","heart palpitation","double vision","swollen"]:
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
                            if part not in STOPWORDS and len(part) >= 4:
                                terms.append(f"{part} pain")
        pt = " . ".join(terms)
        for ei, (n, cui) in aho.iter(pt):
            si = ei - len(n) + 1
            if si > 0 and pt[si-1].isalpha(): continue
            if ei+1 < len(pt) and pt[ei+1].isalpha(): continue
            cuis.add(cui)
    return cuis


def d_bayesian(ps, ds, dcs, all_s):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = -1e6; continue
        tw = sum(s.values()) + len(all_s) * 0.1
        sc[dc] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in ps)
    return sorted(sc.items(), key=lambda x: -x[1])


def patient_symptoms_text(evidences, ev_info):
    symptoms = []
    for ev in evidences:
        parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
        info = ev_info.get(base, {})
        if info.get("is_antecedent"): continue
        q = info.get("question_en", "")
        if not q: continue
        q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your |Does the ", "", q)
        q_clean = re.sub(r"\?$", "", q_clean).strip()
        if value:
            val_en = info.get("value_en", {}).get(value, "")
            if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                symptoms.append(f"{q_clean}: {val_en}")
            else: symptoms.append(q_clean)
        else: symptoms.append(q_clean)
    return "\n".join(f"- {s}" for s in symptoms[:20])


def main():
    print("=" * 80, flush=True)
    print("진단 v9: 다단계 re-ranking (10→3→1)", flush=True)
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
    print(f"  KG: {len(pc):,} 쌍", flush=True)

    print("\n[2] 테스트 데이터...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"]})
    print(f"  {len(patients):,}명", flush=True)

    # [3] Bayesian top-10
    print("\n[3] Bayesian top-10...", flush=True)
    t0 = time.time()
    candidates = []
    baseline = 0; n = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        ps = text_match(p["evidences"], ev_info, aho)
        ranked = d_bayesian(ps, ds, dcs, all_s)
        topk = [dc for dc, _ in ranked[:10]]
        candidates.append({"patient": p, "true_dc": tdc, "topk": topk})
        if topk and topk[0] == tdc: baseline += 1

    in_topk = sum(1 for c in candidates if c["true_dc"] in c["topk"])
    print(f"  Baseline @1={100*baseline/n:.1f}%, in_top10={100*in_topk/n:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    # Pre-generate symptom texts
    sym_texts = []
    for c in candidates:
        sym_texts.append(patient_symptoms_text(c["patient"]["evidences"], ev_info))

    # [4] Round 1: top-10 → top-3
    print("\n[4] Round 1: top-10 → top-3...", flush=True)
    r1_prompts = []
    for c, st in zip(candidates, sym_texts):
        cands = "\n".join(f"{i+1}. {cui2name.get(dc, cp.get(dc,dc))}" for i, dc in enumerate(c["topk"]))
        r1_prompts.append(PROMPT_R1.format(symptoms=st, candidates=cands))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=32)

    convs = [[{"role": "user", "content": p}] for p in r1_prompts]
    t0 = time.time()
    r1_outputs = llm.chat(convs, sampling)
    elapsed1 = time.time() - t0
    print(f"  Round 1 완료: {elapsed1:.0f}초 ({len(r1_outputs)/elapsed1:.1f}/s)", flush=True)

    # Parse round 1: extract 3 indices
    top3_candidates = []
    r1_pf = 0
    for c, out in zip(candidates, r1_outputs):
        answer = out.outputs[0].text.strip()
        nums = re.findall(r"\d+", answer)
        selected = []
        for num in nums[:3]:
            idx = int(num) - 1
            if 0 <= idx < len(c["topk"]):
                selected.append(c["topk"][idx])
        if len(selected) < 3:
            # fallback: use bayesian top 3
            selected = c["topk"][:3]
            r1_pf += 1
        top3_candidates.append(selected[:3])

    # Check round 1 quality
    r1_in_top3 = sum(1 for c, t3 in zip(candidates, top3_candidates) if c["true_dc"] in t3)
    print(f"  Round 1: 정답 in top3={100*r1_in_top3/n:.1f}%, parse_fail={r1_pf:,}", flush=True)

    # Also evaluate round 1 as single-stage (pick first from LLM's 3)
    r1_top1 = sum(1 for c, t3 in zip(candidates, top3_candidates) if t3 and t3[0] == c["true_dc"])
    print(f"  Round 1 @1={100*r1_top1/n:.1f}%", flush=True)

    # [5] Round 2: top-3 → top-1
    print("\n[5] Round 2: top-3 → top-1...", flush=True)
    r2_prompts = []
    for c, t3, st in zip(candidates, top3_candidates, sym_texts):
        cands = "\n".join(f"{i+1}. {cui2name.get(dc, cp.get(dc,dc))}" for i, dc in enumerate(t3))
        r2_prompts.append(PROMPT_R2.format(symptoms=st, candidates=cands))

    convs2 = [[{"role": "user", "content": p}] for p in r2_prompts]
    t0 = time.time()
    r2_outputs = llm.chat(convs2, sampling)
    elapsed2 = time.time() - t0
    print(f"  Round 2 완료: {elapsed2:.0f}초 ({len(r2_outputs)/elapsed2:.1f}/s)", flush=True)

    # Evaluate
    t1 = t3_acc = 0
    r2_pf = 0
    for c, t3, out in zip(candidates, top3_candidates, r2_outputs):
        answer = out.outputs[0].text.strip()
        m = re.search(r"(\d+)", answer)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(t3):
                reranked = list(t3)
                chosen = reranked.pop(idx)
                reranked.insert(0, chosen)
            else:
                reranked = t3; r2_pf += 1
        else:
            reranked = t3; r2_pf += 1

        tdc = c["true_dc"]
        if reranked and reranked[0] == tdc: t1 += 1
        if tdc in reranked[:3]: t3_acc += 1

    print(f"\n  === 결과 ===", flush=True)
    print(f"  Baseline (Bayesian):  @1={100*baseline/n:.1f}%", flush=True)
    print(f"  Round 1 (10→3):      @1={100*r1_top1/n:.1f}%, in_top3={100*r1_in_top3/n:.1f}%", flush=True)
    print(f"  Round 2 (3→1):       @1={100*t1/n:.1f}% (parse_fail={r2_pf:,})", flush=True)
    print(f"  총 시간: {elapsed1+elapsed2:.0f}초", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"Baseline @1={100*baseline/n:.1f}% → Final @1={100*t1/n:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
