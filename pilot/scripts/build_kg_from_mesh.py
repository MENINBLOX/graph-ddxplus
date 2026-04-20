#!/usr/bin/env python3
"""PubMed MeSH 기반 KG 구축 + DDXPlus 벤치마크.

28M PubMed 초록의 MeSH 태그에서:
1. MeSH Descriptor UI → UMLS CUI 매핑
2. DISO CUI만 필터
3. 문서별 CUI 쌍 공출현 집계
4. Jensen Lab score + G² + FDR
5. CUI 계층 전파
6. DDXPlus/HPO 벤치마크
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import scipy.stats as stats

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}


def load_mesh_to_cui() -> dict[str, str]:
    """MRCONSO에서 MeSH Descriptor UI → UMLS CUI 매핑."""
    mapping = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, sab, code = p[0], p[11], p[13]
            if sab == "MSH" and code.startswith("D"):
                if code not in mapping:
                    mapping[code] = cui
    return mapping


def load_cui_stys() -> dict[str, set[str]]:
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and (p[0] not in names or p[2] == "P"):
                names[p[0]] = p[14]
    return names


def load_parent_map() -> dict[str, set[str]]:
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parents[p[0]].add(p[4])
    return dict(parents)


def build_ancestor_fn(parent_map, max_depth=1):
    cache = {}
    def get(cui, d=0):
        if cui in cache: return cache[cui]
        if d >= max_depth or cui not in parent_map:
            cache[cui] = set(); return set()
        anc = set()
        for p in parent_map[cui]:
            anc.add(p)
        cache[cui] = anc
        return anc
    return get


def dunning_g2(a, b, c, d):
    n = a+b+c+d
    if n == 0: return 0
    def g(o, e): return o*math.log(o/e) if o>0 and e>0 else 0
    ea=(a+b)*(a+c)/n; eb=(a+b)*(b+d)/n; ec=(c+d)*(a+c)/n; ed=(c+d)*(b+d)/n
    return 2*(g(a,ea)+g(b,eb)+g(c,ec)+g(d,ed))


def bh_fdr(pvals):
    n=len(pvals)
    if not n: return []
    idx=sorted(range(n),key=lambda i:pvals[i])
    q=[0.0]*n; prev=1.0
    for i in range(n-1,-1,-1):
        j=idx[i]; q[j]=min(prev,pvals[j]*n/(i+1)); prev=q[j]
    return q


def cui_match(a, b, get_ancestors):
    if a == b: return True
    return b in get_ancestors(a) or a in get_ancestors(b)


def evaluate(our_pairs, gold_pairs, get_ancestors):
    mg = set(); mo = set()
    for op in our_pairs:
        for gp in gold_pairs:
            if ((cui_match(op[0],gp[0],get_ancestors) and cui_match(op[1],gp[1],get_ancestors)) or
                (cui_match(op[0],gp[1],get_ancestors) and cui_match(op[1],gp[0],get_ancestors))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our_pairs) if our_pairs else 0
    r = len(mg)/len(gold_pairs) if gold_pairs else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return {"precision":round(p,4),"recall":round(r,4),"f1":round(f1,4),
            "n_our":len(our_pairs),"n_gold":len(gold_pairs),"matched_gold":len(mg)}


def prepare_gold():
    """DDXPlus symptom-only gold standard."""
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
    for eid, en_info in ev_en.items():
        for fr_name, fr_info in ev_fr.items():
            if en_info.get("question_en") == fr_info.get("question_en") and en_info.get("question_en"):
                eid_to_fr[eid] = fr_name
                break

    gold_sym = set()
    gold_full = set()
    for disease_name, info in conditions.items():
        d_cui = disease_map.get(disease_name, {}).get("umls_cui")
        if not d_cui: continue
        for eid in info.get("symptoms", {}):
            fr = eid_to_fr.get(eid)
            if fr and fr in umap:
                cui = umap[fr].get("cui")
                if cui:
                    pair = tuple(sorted([d_cui, cui]))
                    gold_full.add(pair)
                    if not ev_en.get(eid, {}).get("is_antecedent", False):
                        gold_sym.add(pair)
        for eid in info.get("antecedents", {}):
            fr = eid_to_fr.get(eid)
            if fr and fr in umap:
                cui = umap[fr].get("cui")
                if cui:
                    gold_full.add(tuple(sorted([d_cui, cui])))

    return gold_sym, gold_full


def main():
    print("=" * 80)
    print("PubMed MeSH 기반 KG 구축 (28M 초록)")
    print("=" * 80)

    # [1/7] 매핑 로드
    print("\n[1/7] UMLS 매핑 로드...")
    t0 = time.time()
    mesh_to_cui = load_mesh_to_cui()
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    parent_map = load_parent_map()
    get_ancestors = build_ancestor_fn(parent_map, max_depth=1)
    print(f"  MeSH→CUI 매핑: {len(mesh_to_cui):,}")

    # DISO CUI만 필터하는 함수
    def mesh_to_diso_cuis(mesh_list: list[str]) -> list[str]:
        cuis = []
        for m in mesh_list:
            cui = mesh_to_cui.get(m)
            if cui and (cui_stys.get(cui, set()) & ALLOWED_STYS) and cui not in BLACKLIST:
                cuis.append(cui)
        return cuis

    print(f"  로드 완료 ({time.time()-t0:.0f}초)")

    # [2/7] 공출현 집계
    print("\n[2/7] MeSH 공출현 집계 (28M 초록)...")
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    pair_counts = Counter()
    cui_doc_freq = Counter()
    total_docs = 0
    docs_with_diso = 0

    c.execute("SELECT mesh_terms FROM abstracts WHERE mesh_terms IS NOT NULL AND mesh_terms != '[]'")

    batch_size = 100000
    start = time.time()

    while True:
        rows = c.fetchmany(batch_size)
        if not rows:
            break

        for (mesh_json,) in rows:
            total_docs += 1
            mesh_list = json.loads(mesh_json)
            diso_cuis = mesh_to_diso_cuis(mesh_list)

            if len(diso_cuis) < 2:
                continue
            docs_with_diso += 1

            # CUI 빈도
            for cui in set(diso_cuis):
                cui_doc_freq[cui] += 1

            # 쌍 공출현 (문서 내 고유 CUI 쌍)
            unique_cuis = sorted(set(diso_cuis))
            for i in range(len(unique_cuis)):
                for j in range(i+1, len(unique_cuis)):
                    pair_counts[(unique_cuis[i], unique_cuis[j])] += 1

        elapsed = time.time() - start
        rate = total_docs / elapsed if elapsed > 0 else 0
        print(f"  {total_docs:>12,}건 처리 | DISO쌍 있는 문서: {docs_with_diso:,} | "
              f"고유 쌍: {len(pair_counts):,} | {rate:.0f}건/초")

    conn.close()
    elapsed = time.time() - start
    print(f"\n  완료: {total_docs:,}건, {elapsed/60:.1f}분")
    print(f"  DISO 쌍 있는 문서: {docs_with_diso:,} ({docs_with_diso/total_docs*100:.1f}%)")
    print(f"  고유 CUI: {len(cui_doc_freq):,}")
    print(f"  고유 CUI 쌍: {len(pair_counts):,}")

    # [3/7] 통계 필터링
    print("\n[3/7] 통계 필터링...")

    # 최소 공출현 sweep
    for mc in [3, 5, 10, 20, 50, 100]:
        n = sum(1 for v in pair_counts.values() if v >= mc)
        print(f"  MC>={mc}: {n:,}쌍")

    # 최적 설정: MC=3 (파일럿 결과)
    MC = 3
    candidates = {k: v for k, v in pair_counts.items() if v >= MC}
    print(f"\n  MC={MC} 적용: {len(candidates):,}쌍")

    # Jensen Lab + G²
    print("\n[4/7] Jensen Lab score + G² 계산...")
    edges = []
    for (a, b), cnt in candidates.items():
        c_a = max(cui_doc_freq.get(a, 1), 1)
        c_b = max(cui_doc_freq.get(b, 1), 1)
        alpha = 0.6
        oe = (cnt * total_docs) / (c_a * c_b) if c_a*c_b > 0 else 0
        jensen = (cnt**alpha) * (oe**(1-alpha)) if cnt > 0 and oe > 0 else 0

        ga=cnt; gb=max(c_a-cnt,0); gc=max(c_b-cnt,0); gd=max(total_docs-ga-gb-gc,0)
        g2 = dunning_g2(ga, gb, gc, gd)
        pv = 1 - stats.chi2.cdf(g2, df=1) if g2 > 0 else 1.0

        edges.append({
            "cui_a": a, "cui_b": b,
            "n": cnt, "jensen": round(jensen, 3),
            "g2": round(g2, 2), "p_value": pv,
        })

    # FDR
    pvals = [e["p_value"] for e in edges]
    qvals = bh_fdr(pvals)
    for e, q in zip(edges, qvals):
        e["q_value"] = q

    # FDR sweep
    print("\n  FDR 임계값별 엣지 수:")
    for fdr in [0.001, 0.01, 0.05, 0.10, 1.0]:
        n = sum(1 for e in edges if e["q_value"] < fdr)
        print(f"    FDR<{fdr}: {n:,}")

    # FDR<0.05 기본
    sig_edges = [e for e in edges if e["q_value"] < 0.05]
    sig_edges.sort(key=lambda e: -e["jensen"])

    unique_cuis = set()
    for e in sig_edges:
        unique_cuis.add(e["cui_a"])
        unique_cuis.add(e["cui_b"])

    print(f"\n  KG (FDR<0.05): {len(unique_cuis):,} 노드, {len(sig_edges):,} 엣지")

    # CUI 계층 전파
    print("\n[5/7] CUI 계층 전파...")
    expanded_edges = {}
    for e in sig_edges:
        a, b = e["cui_a"], e["cui_b"]
        key = tuple(sorted([a, b]))
        if key not in expanded_edges or e["n"] > expanded_edges[key]["n"]:
            expanded_edges[key] = e

        # 1-level 전파
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                pk = tuple(sorted([pa, b]))
                if pk not in expanded_edges:
                    expanded_edges[pk] = {"cui_a": pk[0], "cui_b": pk[1], "n": e["n"],
                                         "jensen": e["jensen"], "g2": e["g2"],
                                         "p_value": e["p_value"], "q_value": e["q_value"]}
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                pk = tuple(sorted([a, pb]))
                if pk not in expanded_edges:
                    expanded_edges[pk] = {"cui_a": pk[0], "cui_b": pk[1], "n": e["n"],
                                         "jensen": e["jensen"], "g2": e["g2"],
                                         "p_value": e["p_value"], "q_value": e["q_value"]}

    print(f"  전파 후: {len(expanded_edges):,} 엣지 (전파 전: {len(sig_edges):,})")

    # 상위 엣지
    top_edges = sorted(expanded_edges.values(), key=lambda e: -e["n"])
    print(f"\n  상위 20 엣지 (공출현 횟수 기준):")
    for e in top_edges[:20]:
        a = cui_names.get(e["cui_a"], e["cui_a"])[:30]
        b = cui_names.get(e["cui_b"], e["cui_b"])[:30]
        print(f"    {a:30s} - {b:30s} n={e['n']:>6,} J={e['jensen']:.1f}")

    # [6/7] 벤치마크
    print(f"\n[6/7] 벤치마크 평가...")
    gold_sym, gold_full = prepare_gold()
    our_pairs = set((e["cui_a"], e["cui_b"]) for e in expanded_edges.values())

    eval_sym = evaluate(our_pairs, gold_sym, get_ancestors)
    eval_full = evaluate(our_pairs, gold_full, get_ancestors)

    print(f"  DDXPlus (증상만 {len(gold_sym)}쌍):")
    print(f"    P={eval_sym['precision']:.3f} R={eval_sym['recall']:.3f} F1={eval_sym['f1']:.3f} ({eval_sym['matched_gold']}/{len(gold_sym)})")
    print(f"  DDXPlus (전체 {len(gold_full)}쌍):")
    print(f"    P={eval_full['precision']:.3f} R={eval_full['recall']:.3f} F1={eval_full['f1']:.3f} ({eval_full['matched_gold']}/{len(gold_full)})")

    # HPO
    with open(DATA_DIR / "gold_standard.json") as f:
        hpo_gold = set(tuple(p) for p in json.load(f)["hpo"]["pairs"])
    if hpo_gold:
        eval_hpo = evaluate(our_pairs, hpo_gold, get_ancestors)
        print(f"  HPO ({len(hpo_gold)}쌍):")
        print(f"    P={eval_hpo['precision']:.3f} R={eval_hpo['recall']:.3f} F1={eval_hpo['f1']:.3f} ({eval_hpo['matched_gold']}/{len(hpo_gold)})")

    # SemMedDB
    semmed_file = DATA_DIR / "semmed_baseline_pairs.json"
    if semmed_file.exists():
        with open(semmed_file) as f:
            semmed_pairs = set(tuple(p) for p in json.load(f))
        overlap = our_pairs & semmed_pairs
        print(f"  SemMedDB ({len(semmed_pairs):,}쌍):")
        print(f"    우리→SemMedDB: {len(overlap):,}/{len(our_pairs):,} ({len(overlap)/len(our_pairs)*100:.1f}%)")

    # [7/7] MC sweep로 최적화
    print(f"\n[7/7] MC 파라미터 최적화 (FDR<0.05 고정)...")
    for mc in [3, 5, 10, 20, 50, 100, 200]:
        mc_edges = {k: v for k, v in expanded_edges.items()
                    if v["n"] >= mc and v["q_value"] < 0.05}
        mc_pairs = set(mc_edges.keys())
        if not mc_pairs: continue
        ev = evaluate(mc_pairs, gold_sym, get_ancestors)
        print(f"  MC={mc:>4}: edges={len(mc_pairs):>8,} P={ev['precision']:.3f} R={ev['recall']:.3f} F1={ev['f1']:.3f} match={ev['matched_gold']}")

    # 저장
    output = {
        "stats": {
            "total_docs": total_docs, "docs_with_diso": docs_with_diso,
            "unique_cuis": len(cui_doc_freq), "unique_pairs": len(pair_counts),
            "sig_edges_fdr05": len(sig_edges), "expanded_edges": len(expanded_edges),
        },
        "benchmarks": {
            "ddxplus_sym": eval_sym, "ddxplus_full": eval_full,
        },
        "top_edges": [{"cui_a": e["cui_a"], "cui_b": e["cui_b"],
                       "name_a": cui_names.get(e["cui_a"], "?")[:50],
                       "name_b": cui_names.get(e["cui_b"], "?")[:50],
                       "n": e["n"], "jensen": e["jensen"]}
                      for e in top_edges[:500]],
    }
    with open(RESULTS_DIR / "mesh_kg_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {RESULTS_DIR / 'mesh_kg_results.json'}")
    print("완료!")


if __name__ == "__main__":
    main()
