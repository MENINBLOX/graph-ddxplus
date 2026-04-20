#!/usr/bin/env python3
"""MeSH 후보 선별 + LLM 검증 KG 구축.

1단계: MeSH 공출현에서 통계적으로 유의한 후보 쌍 선별
2단계: 해당 초록을 LLM으로 관계 검증
3단계: KG 구축 + 벤치마크
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
import scipy.stats as stats

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT = """Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""

# 설정
TOP_CANDIDATES_PER_DISEASE = 30  # 질환당 상위 후보 수
MAX_ABSTRACTS_PER_PAIR = 5       # 쌍당 검증 초록 수
MAX_PAIRS_PER_LLM_CALL = 10     # LLM 호출당 최대 쌍 수


def load_mesh_to_cui():
    m = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D"):
                if p[13] not in m:
                    m[p[13]] = p[0]
    return m

def load_cui_to_mesh(mesh_to_cui):
    r = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        r[cui].add(mesh)
    return dict(r)

def load_cui_stys():
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)

def load_cui_names():
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and (p[0] not in names or p[2] == "P"):
                names[p[0]] = p[14]
    return names

def load_parent_map():
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parents[p[0]].add(p[4])
    return dict(parents)

def call_ollama(prompt):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }, timeout=300)
    return resp.json().get("response", "")

def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return []

def cui_match(a, b, parent_map):
    if a == b: return True
    return b in parent_map.get(a, set()) or a in parent_map.get(b, set())

def evaluate(our_pairs, gold_pairs, parent_map):
    mg = set(); mo = set()
    for op in our_pairs:
        for gp in gold_pairs:
            if ((cui_match(op[0],gp[0],parent_map) and cui_match(op[1],gp[1],parent_map)) or
                (cui_match(op[0],gp[1],parent_map) and cui_match(op[1],gp[0],parent_map))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our_pairs) if our_pairs else 0
    r = len(mg)/len(gold_pairs) if gold_pairs else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return {"P":round(p,4),"R":round(r,4),"F1":round(f1,4),
            "matched":len(mg),"our":len(our_pairs),"gold":len(gold_pairs)}

def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f:
        conditions = json.load(f)
    with open("data/ddxplus/release_evidences_en.json") as f:
        ev_en = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)
    with open("data/ddxplus/umls_mapping.json") as f:
        umap = json.load(f)["mapping"]
    with open("data/ddxplus/disease_umls_mapping.json") as f:
        disease_map = json.load(f)["mapping"]

    eid_to_fr = {}
    for eid, en in ev_en.items():
        for fr_name, fr in ev_fr.items():
            if en.get("question_en") == fr.get("question_en") and en.get("question_en"):
                eid_to_fr[eid] = fr_name; break

    gold_sym = set(); gold_full = set()
    disease_cuis = {}
    for dn, info in conditions.items():
        dc = disease_map.get(dn, {}).get("umls_cui")
        if not dc: continue
        disease_cuis[dn] = dc
        for eid in info.get("symptoms", {}):
            fr = eid_to_fr.get(eid)
            if fr and fr in umap:
                cui = umap[fr].get("cui")
                if cui:
                    pair = tuple(sorted([dc, cui]))
                    gold_full.add(pair)
                    if not ev_en.get(eid, {}).get("is_antecedent", False):
                        gold_sym.add(pair)
        for eid in info.get("antecedents", {}):
            fr = eid_to_fr.get(eid)
            if fr and fr in umap:
                cui = umap[fr].get("cui")
                if cui: gold_full.add(tuple(sorted([dc, cui])))
    return gold_sym, gold_full, disease_cuis


def main():
    print("=" * 80)
    print("MeSH 후보 선별 + LLM 검증 KG 구축")
    print("=" * 80)

    # [1] 로드
    print("\n[1/6] 데이터 로드...")
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = load_cui_to_mesh(mesh_to_cui)
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    parent_map = load_parent_map()
    gold_sym, gold_full, disease_cuis = prepare_gold()
    print(f"  MeSH→CUI: {len(mesh_to_cui):,}, Gold(증상): {len(gold_sym)}, 질환: {len(disease_cuis)}")

    # [2] MeSH 공출현에서 후보 쌍 선별
    print(f"\n[2/6] 질환별 MeSH 공출현 후보 선별...")
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # 각 DDXPlus 질환의 MeSH UID 확인
    disease_mesh = {}
    for dn, dc in disease_cuis.items():
        meshes = cui_to_mesh.get(dc, set())
        if meshes:
            disease_mesh[dn] = {"cui": dc, "mesh_uids": meshes}

    print(f"  MeSH 매핑된 질환: {len(disease_mesh)}/49")

    # 질환별 공출현 후보 수집
    candidate_pairs = {}  # (disease_cui, candidate_cui) -> count
    candidate_abstracts = defaultdict(list)  # (disease_cui, candidate_cui) -> [pmid]

    for dn, info in disease_mesh.items():
        dc = info["cui"]
        mesh_uids = info["mesh_uids"]

        # 이 질환의 MeSH가 포함된 초록 검색
        mesh_conditions = " OR ".join(f"mesh_terms LIKE '%{m}%'" for m in mesh_uids)
        c.execute(f"""
            SELECT pmid, mesh_terms, abstract
            FROM abstracts
            WHERE ({mesh_conditions})
            AND abstract IS NOT NULL AND length(abstract) > 100
            LIMIT 500
        """)

        rows = c.fetchall()
        cooc_count = Counter()

        for pmid, mesh_json, abstract in rows:
            mesh_list = json.loads(mesh_json)
            diso_cuis = []
            for m in mesh_list:
                cui = mesh_to_cui.get(m)
                if cui and cui != dc and (cui_stys.get(cui, set()) & ALLOWED_STYS) and cui not in BLACKLIST:
                    diso_cuis.append(cui)

            for ccui in set(diso_cuis):
                pair = tuple(sorted([dc, ccui]))
                cooc_count[pair] += 1
                if len(candidate_abstracts[pair]) < MAX_ABSTRACTS_PER_PAIR:
                    candidate_abstracts[pair].append({"pmid": pmid, "abstract": abstract})

        # 상위 후보 선택
        top = cooc_count.most_common(TOP_CANDIDATES_PER_DISEASE)
        for pair, cnt in top:
            candidate_pairs[pair] = cnt

    print(f"  후보 쌍: {len(candidate_pairs):,}")
    print(f"  검증 필요 초록: {sum(len(v) for v in candidate_abstracts.values() if tuple(sorted(v[0]['pmid'] if v else '')) in candidate_pairs or True):,}")

    # 실제 검증할 초록 수
    abstracts_to_process = {}
    for pair in candidate_pairs:
        for ab in candidate_abstracts.get(pair, []):
            if ab["pmid"] not in abstracts_to_process:
                abstracts_to_process[ab["pmid"]] = {
                    "abstract": ab["abstract"],
                    "pairs": [],
                }
            abstracts_to_process[ab["pmid"]]["pairs"].append(pair)

    # 쌍 수 제한
    for pmid in abstracts_to_process:
        abstracts_to_process[pmid]["pairs"] = abstracts_to_process[pmid]["pairs"][:MAX_PAIRS_PER_LLM_CALL]

    print(f"  고유 초록: {len(abstracts_to_process):,}")
    conn.close()

    # [3] LLM 분류
    print(f"\n[3/6] LLM 분류 ({len(abstracts_to_process):,}편)...")

    ckpt_file = DATA_DIR / "mesh_llm_checkpoint.json"
    all_cls = []
    processed = set()
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = json.load(f)
            all_cls = ckpt.get("cls", [])
            processed = set(c_["pmid"] for c_ in all_cls)
        print(f"  체크포인트: {len(processed)}편 완료")

    start = time.time()
    actual = 0

    for idx, (pmid, info) in enumerate(abstracts_to_process.items()):
        if pmid in processed:
            continue

        text = info["abstract"]
        pairs = info["pairs"]
        if not text or not pairs:
            continue

        pairs_text = "\n".join(
            f"- ({cui_names.get(p[0],p[0])[:40]}, {cui_names.get(p[1],p[1])[:40]}) [CUI: {p[0]}, {p[1]}]"
            for p in pairs
        )
        prompt = PROMPT.format(text=text[:2500], pairs=pairs_text)

        try:
            response = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                cls = item.get("classification", "").lower().strip().replace(" ", "_")
                if cls in ("present", "not_related"):
                    all_cls.append({
                        "pmid": pmid, "cui_a": item.get("cui_a", ""),
                        "cui_b": item.get("cui_b", ""), "classification": cls,
                    })
            actual += 1
        except:
            pass

        if actual % 20 == 0 and actual > 0:
            elapsed = time.time() - start
            rate = actual / elapsed
            remaining = len(abstracts_to_process) - len(processed) - actual
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{len(processed)+actual:5d}/{len(abstracts_to_process)}] "
                  f"cls={len(all_cls):,} {rate:.2f}/s ETA={eta/60:.0f}분")
            with open(ckpt_file, "w") as f:
                json.dump({"cls": all_cls}, f)

    with open(ckpt_file, "w") as f:
        json.dump({"cls": all_cls}, f)

    elapsed = time.time() - start
    dist = Counter(c_["classification"] for c_ in all_cls)
    print(f"\n  완료: {len(all_cls):,}건 ({elapsed/60:.1f}분)")
    print(f"  present={dist.get('present',0)}, not_related={dist.get('not_related',0)}")

    # [4] KG 구축
    print(f"\n[4/6] KG 구축 (MC=3 + CUI 전파)...")

    pair_counts = Counter()
    for c_ in all_cls:
        if c_["classification"] == "present":
            pair = tuple(sorted([c_["cui_a"], c_["cui_b"]]))
            pair_counts[pair] += 1

    # MC=3
    kg_pairs = {p for p, cnt in pair_counts.items() if cnt >= 3}

    # CUI 1-level 전파
    expanded = set(kg_pairs)
    for (a, b) in list(kg_pairs):
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                expanded.add(tuple(sorted([pa, b])))
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                expanded.add(tuple(sorted([a, pb])))

    print(f"  present 쌍: {len(pair_counts):,}")
    print(f"  MC>=3: {len(kg_pairs):,}")
    print(f"  전파 후: {len(expanded):,}")

    # [5] 벤치마크
    print(f"\n[5/6] 벤치마크...")

    eval_sym = evaluate(expanded, gold_sym, parent_map)
    eval_full = evaluate(expanded, gold_full, parent_map)

    print(f"  DDXPlus (증상 {len(gold_sym)}쌍): P={eval_sym['P']:.3f} R={eval_sym['R']:.3f} F1={eval_sym['F1']:.3f} ({eval_sym['matched']}/{len(gold_sym)})")
    print(f"  DDXPlus (전체 {len(gold_full)}쌍): P={eval_full['P']:.3f} R={eval_full['R']:.3f} F1={eval_full['F1']:.3f} ({eval_full['matched']}/{len(gold_full)})")

    # MC sweep
    print(f"\n  MC 파라미터 sweep:")
    for mc in [1, 2, 3, 5, 10]:
        mc_pairs = {p for p, cnt in pair_counts.items() if cnt >= mc}
        mc_exp = set(mc_pairs)
        for (a, b) in list(mc_pairs):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                    mc_exp.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                    mc_exp.add(tuple(sorted([a, pb])))
        ev = evaluate(mc_exp, gold_sym, parent_map)
        print(f"    MC={mc}: edges={len(mc_exp):>6,} P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} match={ev['matched']}")

    # [6] 저장
    print(f"\n[6/6] 저장...")
    output = {
        "config": {"model": MODEL, "top_candidates": TOP_CANDIDATES_PER_DISEASE,
                   "max_abstracts_per_pair": MAX_ABSTRACTS_PER_PAIR},
        "stats": {"candidate_pairs": len(candidate_pairs),
                  "abstracts_processed": len(abstracts_to_process),
                  "classifications": len(all_cls), "present": dist.get("present", 0)},
        "benchmarks": {"ddxplus_sym": eval_sym, "ddxplus_full": eval_full},
        "kg_edges": [{"cui_a": p[0], "cui_b": p[1], "n": pair_counts[p]}
                     for p in sorted(kg_pairs, key=lambda x: -pair_counts[x])],
    }
    with open(RESULTS_DIR / "mesh_llm_kg_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  저장: {RESULTS_DIR / 'mesh_llm_kg_results.json'}")
    print("완료!")


if __name__ == "__main__":
    main()
