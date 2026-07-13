#!/usr/bin/env python3
"""KG V10: DDXPlus 어휘 폐쇄형 KG 구축.

DDXPlus의 49개 질환 × 87개 증상 CUI = 4,263쌍에 대해
PubMed 초록 기반 이진 분류.

각 (질환, 증상) 쌍마다:
  1. 질환 초록에서 증상 CUI가 텍스트 매칭되는 초록 검색
  2. 해당 초록에서 LLM에게 "이 증상이 이 질환의 임상 증상인가?" 질문
  3. 여러 초록의 판정 집계
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

ALLOWED_STYS = {"T047","T184","T191","T046","T048","T037","T019","T020","T190","T049",
                "T033","T031","T040"}
BLACKLIST = {"C1457887","C3257980","C0012634","C0699748","C3839861"}

PROMPT_BINARY = """Abstract: {abstract}

Question: Does this abstract describe "{symptom_name}" as a clinical symptom, sign, or presenting complaint of "{disease_name}"?

Answer ONLY "yes" or "no"."""

MAX_ABSTRACTS = 500
MAX_ABS_PER_PAIR = 10


def load_umls_minimal():
    """최소 UMLS 로드 (증상 CUI 텍스트 매칭용)."""
    cui_stys = defaultdict(set)
    with open(UMLS_DIR/"MRSTY.RRF") as f:
        for l in f: p=l.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set)
    with open(UMLS_DIR/"MRREL.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[3] in("PAR","RB"): parent_map[p[0]].add(p[4])
    cui_preferred = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[1]=="ENG" and p[2]=="P" and p[0] not in cui_preferred:
                cui_preferred[p[0]]=p[14].strip()
    mesh_to_cui = {}
    cui_all_names = defaultdict(set)
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[11]=="MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui:
                mesh_to_cui[p[13]]=p[0]
            if p[1]=="ENG":
                cui_all_names[p[0]].add(p[14].strip())
    return dict(cui_stys), dict(parent_map), cui_preferred, mesh_to_cui, dict(cui_all_names)


def build_symptom_automaton(symptom_cuis, cui_all_names):
    """DDXPlus 증상 CUI에 대해서만 Aho-Corasick 구축."""
    A = ahocorasick.Automaton()
    for cui in symptom_cuis:
        for name in cui_all_names.get(cui, set()):
            lower = name.lower()
            if len(lower) >= 4:
                try: A.add_word(lower, (lower, cui))
                except: pass
    A.make_automaton()
    return A


def prepare_ddxplus():
    """DDXPlus 질환-증상 관계 및 어휘."""
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

    # 질환 정보
    diseases = {}
    for dn, info in cond.items():
        dc = dm.get(dn, {}).get("umls_cui")
        un = dm.get(dn, {}).get("umls_name", dn)
        if dc:
            diseases[dn] = {"cui": dc, "umls_name": un}

    # 증상 CUI 집합 (antecedent 제외)
    symptom_cuis = set()
    symptom_names = {}  # cui -> preferred name from DDXPlus
    gold_pairs = set()

    for dn, info in cond.items():
        dc = dm.get(dn, {}).get("umls_cui")
        if not dc: continue
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False): continue
            fn = eid_to_fr.get(eid)
            if fn and fn in umap:
                cui = umap[fn].get("cui")
                name = umap[fn].get("name", fn)
                if cui:
                    symptom_cuis.add(cui)
                    symptom_names[cui] = name
                    gold_pairs.add(tuple(sorted([dc, cui])))

    return diseases, symptom_cuis, symptom_names, gold_pairs


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


def evaluate(our, gold, pm):
    def cm(a, b):
        if a == b: return True
        return b in pm.get(a, set()) or a in pm.get(b, set())
    mg, mo = set(), set()
    for op in our:
        for gp in gold:
            if (cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0])):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our) if our else 0
    r = len(mg)/len(gold) if gold else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return round(p,4), round(r,4), round(f1,4), len(mg)


def main():
    print("="*80)
    print("KG V10: DDXPlus 어휘 폐쇄형 KG 구축")
    print("="*80)

    print("\n[1] 데이터 로드...")
    cui_stys, parent_map, cui_preferred, mesh_to_cui, cui_all_names = load_umls_minimal()
    c2m = defaultdict(set)
    for m, c in mesh_to_cui.items(): c2m[c].add(m)
    diseases, symptom_cuis, symptom_names, gold_pairs = prepare_ddxplus()

    print(f"  질환: {len(diseases)}개")
    print(f"  증상 CUI: {len(symptom_cuis)}개")
    print(f"  Gold: {len(gold_pairs)}쌍")
    print(f"  전체 쌍: {len(diseases)*len(symptom_cuis)}개")

    # 증상 CUI용 Aho-Corasick (DDXPlus 87개 CUI만)
    symptom_aho = build_symptom_automaton(symptom_cuis, cui_all_names)

    # 질환별 초록 수집 + 증상 매칭
    print(f"\n[2] 초록 수집 + 증상 매칭...")
    conn = sqlite3.connect(str(DB_PATH))

    # (disease_cui, symptom_cui) → [abstracts where symptom appears]
    pair_abstracts = defaultdict(list)

    for idx, (dn, dinfo) in enumerate(sorted(diseases.items())):
        dc, un = dinfo["cui"], dinfo["umls_name"]
        rows = search_abs(conn, dc, dn, un, c2m, cui_all_names, MAX_ABSTRACTS)

        for pmid, ab in rows:
            ab_lower = ab.lower()
            # 이 초록에서 어떤 DDXPlus 증상이 발견되는지
            found_symptoms = set()
            for ei, (name, cui) in symptom_aho.iter(ab_lower):
                si = ei - len(name) + 1
                if si > 0 and ab_lower[si-1].isalpha(): continue
                if ei+1 < len(ab_lower) and ab_lower[ei+1].isalpha(): continue
                found_symptoms.add(cui)

            for scui in found_symptoms:
                if len(pair_abstracts[(dc, scui)]) < MAX_ABS_PER_PAIR:
                    pair_abstracts[(dc, scui)].append(ab)

        n_pairs = sum(1 for (d, s) in pair_abstracts if d == dc)
        print(f"  [{idx+1}/49] {dn}: {len(rows)}편, {n_pairs} 증상쌍")

    conn.close()

    # 후보 쌍 통계
    total_pairs = len(pair_abstracts)
    total_prompts = sum(len(v) for v in pair_abstracts.values())
    print(f"\n  후보 (질환,증상) 쌍: {total_pairs}")
    print(f"  총 프롬프트: {total_prompts}")

    # LLM 이진 분류
    print(f"\n[3] vLLM 이진 분류...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=10)

    prompts = []
    meta = []
    for (dc, scui), abstracts in pair_abstracts.items():
        dn = cui_preferred.get(dc, dc)
        sn = symptom_names.get(scui, cui_preferred.get(scui, scui))
        for ab in abstracts:
            prompts.append(PROMPT_BINARY.format(
                abstract=ab[:2000], symptom_name=sn, disease_name=dn))
            meta.append((dc, scui))

    print(f"  프롬프트: {len(prompts):,}건")
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

    # 집계
    print(f"\n[4] 집계...")
    pair_yes = Counter()
    pair_total = Counter()

    for (dc, scui), out in zip(meta, outputs):
        pair_key = tuple(sorted([dc, scui]))
        pair_total[pair_key] += 1
        answer = out.outputs[0].text.strip().lower()
        if "yes" in answer:
            pair_yes[pair_key] += 1

    print(f"  총 쌍: {len(pair_total):,}")
    print(f"  yes 있는 쌍: {len(pair_yes):,}")

    # 벤치마크
    print(f"\n[5] 벤치마크...")

    print(f"\n  yes 횟수 threshold (1-level 매칭):")
    for mc in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        kg = {p for p, cnt in pair_yes.items() if cnt >= mc}
        # 전파
        exp = set(kg)
        for (a, b) in list(kg):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a, pb])))
        p, r, f1, m = evaluate(exp, gold_pairs, parent_map)
        print(f"    yes>={mc:>2}: edges={len(exp):>5,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    print(f"\n  yes 횟수 (전파 없이):")
    for mc in [1, 2, 3, 4, 5]:
        kg = {p for p, cnt in pair_yes.items() if cnt >= mc}
        p, r, f1, m = evaluate(kg, gold_pairs, parent_map)
        print(f"    yes>={mc}: edges={len(kg):>5,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    print(f"\n  yes 비율 threshold:")
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        kg = set()
        for pair in pair_yes:
            if pair_yes[pair] / pair_total[pair] >= thr:
                kg.add(pair)
        exp = set(kg)
        for (a, b) in list(kg):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a, pb])))
        p, r, f1, m = evaluate(exp, gold_pairs, parent_map)
        print(f"    ratio>={thr:.1f}: edges={len(exp):>5,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/324")

    # 저장
    with open(RESULTS_DIR/"kg_v10_results.json", "w") as f:
        json.dump({
            "pair_yes": [[list(k), v] for k, v in pair_yes.most_common()],
            "pair_total": [[list(k), v] for k, v in pair_total.most_common()],
            "stats": {"diseases": len(diseases), "symptoms": len(symptom_cuis),
                      "candidate_pairs": total_pairs, "prompts": len(prompts),
                      "time_seconds": round(elapsed, 1)},
        }, f)

    print(f"\n{'='*80}")
    print("V10 완료")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
