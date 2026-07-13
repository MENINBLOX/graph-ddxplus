#!/usr/bin/env python3
"""KG 구축 V7: DDXPlus 패턴 기반 프롬프트 + vLLM batch.

DDXPlus 분석 결과:
- 증상은 "환자가 호소하는 임상 증상" (Pain, Dyspnea, Nausea, Fever 등)
- 87개 증상 CUI 중 59개가 V4 KG에 없음
- Aho-Corasick은 매칭 가능 → LLM이 관계로 판정하지 않는 게 문제

개선:
1. 프롬프트를 "임상 증상" 초점으로 변경
2. DDXPlus 87개 증상 CUI를 "타겟 증상 사전"으로 활용하되,
   KG 구축에는 PubMed만 사용 (DDXPlus 답을 직접 넣지 않음)
3. 500편/질환, vLLM batch
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049",
                "T033", "T031", "T040"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

# 프롬프트: "임상 증상" 초점
PROMPT_CLINICAL = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Clinical findings and symptoms found in this abstract:
{keywords}

From the abstract, identify which of the above findings are CLINICAL SYMPTOMS or SIGNS that a patient with {disease_name} would present with.

Include:
- Symptoms the patient reports (pain, nausea, fever, cough, dyspnea, fatigue, etc.)
- Physical examination findings (edema, skin lesions, pallor, etc.)
- Vital sign abnormalities (tachycardia, hypotension, etc.)

Exclude:
- Laboratory test results
- Imaging findings
- Other diseases or conditions (these are NOT symptoms)
- Synonyms or subtypes of {disease_name} itself

JSON only: [{{"cui":"...","relation":"symptom-of|sign-of|risk-factor-for"}}]
If none: []"""

# S2-J 스타일 (쌍별 분류) + 임상 초점
PROMPT_S2J_CLINICAL = """For each concept pair, determine if the text describes the second concept as a clinical symptom or sign of the first concept (the disease).

Classify as:
- "present": The text states this is a symptom, sign, or clinical finding of the disease
- "not_related": No symptom/sign relationship described

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""

MAX_ABSTRACTS = 500
RESULTS_FILE = RESULTS_DIR / "kg_v7_results.json"


def load_umls():
    cui_stys = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set)
    synonym_map = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"): parent_map[p[0]].add(p[4])
            if p[3] == "SY": synonym_map[p[0]].add(p[4]); synonym_map[p[4]].add(p[0])
    mesh_to_cui = {}
    cui_all_names = defaultdict(set)
    cui_preferred = {}
    target = {c for c, s in cui_stys.items() if s & ALLOWED_STYS} - BLACKLIST
    A = ahocorasick.Automaton()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui:
                mesh_to_cui[p[13]] = p[0]
            if p[1] == "ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cui_preferred:
                    cui_preferred[p[0]] = p[14].strip()
            if p[0] in target and p[1] == "ENG":
                lower = p[14].strip().lower()
                if len(lower) >= 4:
                    try: A.add_word(lower, (lower, p[0]))
                    except: pass
    A.make_automaton()
    return dict(cui_stys), dict(parent_map), dict(synonym_map), mesh_to_cui, dict(cui_all_names), cui_preferred, A


def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences_en.json") as f: ev_en = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
    with open("data/ddxplus/umls_mapping.json") as f: umap = json.load(f)["mapping"]
    with open("data/ddxplus/disease_umls_mapping.json") as f: dm = json.load(f)["mapping"]
    eid_to_fr = {}
    for eid, en in ev_en.items():
        for fn, fr in ev_fr.items():
            if en.get("question_en") == fr.get("question_en") and en.get("question_en"):
                eid_to_fr[eid] = fn; break
    gold, dcuis = set(), {}
    for dn, info in cond.items():
        dc = dm.get(dn, {}).get("umls_cui"); un = dm.get(dn, {}).get("umls_name", dn)
        if not dc: continue
        dcuis[dn] = {"cui": dc, "umls_name": un}
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False): continue
            fn = eid_to_fr.get(eid)
            if fn and fn in umap:
                cui = umap[fn].get("cui")
                if cui: gold.add(tuple(sorted([dc, cui])))
    return gold, dcuis


def text_match(text_lower, A, exclude=None):
    matched = set()
    for ei, (n, c) in A.iter(text_lower):
        if c == exclude: continue
        si = ei - len(n) + 1
        if si > 0 and text_lower[si-1].isalpha(): continue
        if ei+1 < len(text_lower) and text_lower[ei+1].isalpha(): continue
        matched.add(c)
    return matched


def search_abs(conn, dc, dn, un, c2m, can, limit):
    c = conn.cursor(); rows, seen = [], set()
    muids = c2m.get(dc, set())
    if muids:
        mc = " OR ".join("mesh_terms LIKE '%%%s%%'" % m for m in muids)
        c.execute(f"SELECT pmid, abstract FROM abstracts WHERE ({mc}) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?", (limit,))
        for p, a in c.fetchall():
            if p not in seen: seen.add(p); rows.append((p, a))
    if len(rows) < limit:
        kws = []
        for raw in [dn, un]:
            kw = re.sub(r'\(.*?\)', '', raw).strip()
            kw = re.sub(r'\b(NOS|unspecified)\b', '', kw, flags=re.IGNORECASE).strip()
            if len(kw) >= 4: kws.append(kw)
            for part in raw.split('/'):
                part = part.strip()
                if len(part) >= 4 and part not in kws: kws.append(part)
        syns = set()
        for name in can.get(dc, set()):
            kw = re.sub(r'\(.*?\)', '', name).strip()
            kw = re.sub(r'\b(NOS|unspecified|disease)\b', '', kw, flags=re.IGNORECASE).strip().strip(',./').strip()
            if len(kw) >= 4 and kw not in kws: syns.add(kw)
        kws.extend(sorted(syns, key=len)[:10])
        for kw in kws:
            if len(rows) >= limit: break
            c.execute("SELECT pmid, abstract FROM abstracts WHERE (title LIKE ? OR abstract LIKE ?) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?",
                      (f"%{kw}%", f"%{kw}%", limit-len(rows)))
            for p, a in c.fetchall():
                if p not in seen: seen.add(p); rows.append((p, a))
    return rows


def parse_json_r(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text); text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return []


def evaluate(our, gold, pm):
    def cm(a, b):
        if a == b: return True
        return b in pm.get(a, set()) or a in pm.get(b, set())
    mg, mo = set(), set()
    for op in our:
        for gp in gold:
            if ((cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0]))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our) if our else 0; r = len(mg)/len(gold) if gold else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return round(p,4), round(r,4), round(f1,4), len(mg)


def main():
    print("=" * 80)
    print("KG V7: DDXPlus 패턴 기반 (임상 증상 초점)")
    print("=" * 80)

    print("\n[1] UMLS 로드...")
    cui_stys, parent_map, synonym_map, mesh_to_cui, cui_all_names, cui_preferred, automaton = load_umls()
    c2m = defaultdict(set)
    for m, c in mesh_to_cui.items(): c2m[c].add(m)
    gold, dcuis = prepare_gold()

    # 초록 수집
    print(f"\n[2] 초록 수집 ({MAX_ABSTRACTS}편/질환)...")
    conn = sqlite3.connect(str(DB_PATH))
    all_docs = {}
    for idx, (dn, dinfo) in enumerate(sorted(dcuis.items())):
        dc, un = dinfo["cui"], dinfo["umls_name"]
        rows = search_abs(conn, dc, dn, un, c2m, cui_all_names, MAX_ABSTRACTS)
        docs = []
        for pmid, ab in rows:
            cuis = text_match(ab.lower(), automaton, exclude=dc)
            if cuis:
                docs.append({"pmid": pmid, "abstract": ab, "cuis": sorted(cuis)})
        all_docs[dc] = docs
        print(f"  [{idx+1}/49] {dn}: {len(docs)}편")
    conn.close()
    total = sum(len(d) for d in all_docs.values())
    print(f"\n  총: {total}편")

    # vLLM 로드
    print("\n[3] vLLM 로드...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=4096)

    # === 실험 1: V2 프롬프트 (기존, baseline) ===
    PROMPT_V2 = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Other medical concepts found in this abstract:
{keywords}

Which of the above concepts does the abstract describe as being related to {disease_name}?
For each related concept, classify the relationship type:
symptom-of, causes, complication-of, risk-factor-for, diagnostic-finding-of, manifestation-of, co-occurs-with.

Rules:
- ONLY include concepts that the abstract EXPLICITLY links to {disease_name}
- Do NOT infer relationships not stated in the text

JSON array only: [{{"cui":"...","relation":"..."}}]
If none related: []"""

    prompts_v2 = []
    prompts_clinical = []
    prompts_s2j = []
    task_meta = []  # 공유 메타데이터

    for dc, docs in all_docs.items():
        dn = cui_preferred.get(dc, dc)
        for doc in docs:
            kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in doc["cuis"])
            # V2
            prompts_v2.append(PROMPT_V2.format(
                abstract=doc["abstract"][:3000], disease_name=dn, disease_cui=dc, keywords=kw))
            # Clinical
            prompts_clinical.append(PROMPT_CLINICAL.format(
                abstract=doc["abstract"][:3000], disease_name=dn, disease_cui=dc, keywords=kw))
            # S2-J clinical (질환-CUI 쌍)
            pairs = "\n".join(
                f"- ({dn[:40]}, {cui_preferred.get(c,c)[:40]}) [CUI: {dc}, {c}]"
                for c in doc["cuis"][:15]
            )
            prompts_s2j.append(PROMPT_S2J_CLINICAL.format(
                text=doc["abstract"][:2500], pairs=pairs))
            task_meta.append({"dc": dc, "cuis": doc["cuis"][:15]})

    print(f"\n  프롬프트: {len(prompts_v2)}편 × 3 실험")

    # === 실험 실행 ===
    results = {}

    for exp_name, prompts, is_s2j in [
        ("V2_baseline", prompts_v2, False),
        ("Clinical_V2", prompts_clinical, False),
        ("Clinical_S2J", prompts_s2j, True),
    ]:
        print(f"\n{'='*60}")
        print(f"실험: {exp_name}")
        print(f"{'='*60}")

        convs = [[{"role": "user", "content": p}] for p in prompts]
        t0 = time.time()
        outputs = llm.chat(convs, sampling)
        elapsed = time.time() - t0
        print(f"  LLM: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

        all_cls = []
        for i, out in enumerate(outputs):
            parsed = parse_json_r(out.outputs[0].text)
            meta = task_meta[i]
            if is_s2j:
                for item in parsed:
                    cls = item.get("classification", "").lower().replace(" ", "_")
                    if cls == "present":
                        a, b = item.get("cui_a", ""), item.get("cui_b", "")
                        if a and b:
                            other = b if a == meta["dc"] else a
                            all_cls.append({"dc": meta["dc"], "cui": other})
            else:
                for item in parsed:
                    cui = item.get("cui", "")
                    rel = item.get("relation", "")
                    if cui and rel:
                        all_cls.append({"dc": meta["dc"], "cui": cui, "rel": rel})

        # 후처리
        filtered = []
        for c in all_cls:
            if c.get("rel") == "manifestation-of": continue
            dc, cui = c["dc"], c["cui"]
            if cui in synonym_map.get(dc, set()): continue
            if cui in parent_map.get(dc, set()) or dc in parent_map.get(cui, set()): continue
            filtered.append(c)

        pc = Counter(tuple(sorted([c["dc"], c["cui"]])) for c in filtered)
        print(f"  Raw: {len(all_cls):,}, Filtered: {len(filtered):,}, Pairs: {len(pc):,}")

        best_f1, best_mc = 0, 1
        for mc in [1, 2, 3, 5, 7, 10]:
            kg = {p for p, cnt in pc.items() if cnt >= mc}
            exp = set(kg)
            for (a, b) in list(kg):
                for pa in parent_map.get(a, set()):
                    if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                        exp.add(tuple(sorted([pa, b])))
                for pb in parent_map.get(b, set()):
                    if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                        exp.add(tuple(sorted([a, pb])))
            p, r, f1, m = evaluate(exp, gold, parent_map)
            marker = " ★" if f1 > best_f1 else ""
            if f1 > best_f1: best_f1, best_mc = f1, mc
            print(f"    MC={mc:>2} edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324{marker}")

        results[exp_name] = (best_f1, best_mc)

    # 요약
    print(f"\n{'='*80}")
    print(f"V7 요약:")
    for name, (f1, mc) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"  {name:<20} F1={f1:.3f} (MC={mc})")
    print(f"{'='*80}")

    # 저장
    with open(RESULTS_FILE, "w") as f:
        json.dump({"results": {k: {"f1": v[0], "mc": v[1]} for k, v in results.items()}}, f, indent=2)


if __name__ == "__main__":
    main()
