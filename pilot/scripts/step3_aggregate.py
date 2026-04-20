#!/usr/bin/env python3
"""Step 3-4: Jensen Lab score + Dunning's G² + BH-FDR.

Step 2의 LLM 분류 결과를 통계적으로 집계하여 KG 엣지를 생성한다.

분기점:
- α 값: 0.5, 0.6, 0.7
- FDR 임계값: 0.01, 0.05, 0.10
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import scipy.stats as stats

DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
UMLS_DIR = Path("data/umls_extracted")

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, lang, ts, name = p[0], p[1], p[2], p[14]
            if lang == "ENG" and (cui not in names or ts == "P"):
                names[cui] = name
    return names


def load_cui_stys() -> dict[str, set[str]]:
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui_stys[p[0]].add(p[1])
    return dict(cui_stys)


def dunning_g2(a: int, b: int, c: int, d: int) -> float:
    """Dunning's Log-Likelihood Ratio (G²).

    Contingency table:
        | co-occur | not co-occur |
    a-present |    a    |      b       |
    a-absent  |    c    |      d       |

    G² = 2 * Σ O * ln(O/E)
    """
    n = a + b + c + d
    if n == 0:
        return 0.0

    def g_term(obs: int, exp: float) -> float:
        if obs == 0 or exp <= 0:
            return 0.0
        return obs * math.log(obs / exp)

    e_a = (a + b) * (a + c) / n
    e_b = (a + b) * (b + d) / n
    e_c = (c + d) * (a + c) / n
    e_d = (c + d) * (b + d) / n

    g2 = 2 * (g_term(a, e_a) + g_term(b, e_b) + g_term(c, e_c) + g_term(d, e_d))
    return g2


def jensen_lab_score(c_ab: float, c_a: float, c_b: float, c_total: float, alpha: float) -> float:
    """Jensen Lab weighted co-occurrence score.

    S(a,b) = C(a,b)^α * [C(a,b) * C_total / (C(a) * C(b))]^(1-α)
    """
    if c_ab <= 0 or c_a <= 0 or c_b <= 0 or c_total <= 0:
        return 0.0
    oe_ratio = (c_ab * c_total) / (c_a * c_b)
    return (c_ab ** alpha) * (oe_ratio ** (1 - alpha))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_values = [0.0] * n

    prev_q = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1
        q = min(prev_q, p * n / rank)
        q_values[orig_idx] = q
        prev_q = q

    return q_values


def main():
    print("=" * 80)
    print("Step 3-4: 통계 집계 + 유의성 검정")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    with open(DATA_DIR / "step2_classifications.json") as f:
        classifications = json.load(f)

    cui_names = load_cui_names()
    cui_stys = load_cui_stys()

    print(f"  분류 데이터: {len(classifications):,}건")

    # 분류 분포
    dist = defaultdict(int)
    for c in classifications:
        dist[c["classification"]] += 1
    for cls, cnt in sorted(dist.items()):
        print(f"    {cls}: {cnt:,}")

    # 쌍별 집계
    print("\n[2/4] 쌍별 집계...")
    pair_stats: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "n_present": 0, "n_absent": 0, "n_not_related": 0, "pmids": set(),
    })
    # 각 CUI의 전체 등장 횟수
    cui_doc_count: dict[str, int] = defaultdict(int)

    # 문서별 CUI 목록 (전체 등장 횟수 계산용)
    with open(DATA_DIR / "step1_documents.json") as f:
        docs_data = json.load(f)
    for doc in docs_data["documents"]:
        for cui in set(doc["diso_cuis"]):
            cui_doc_count[cui] += 1
    total_docs = len(docs_data["documents"])

    for c in classifications:
        a, b = c["cui_a"], c["cui_b"]
        key = tuple(sorted([a, b]))
        pair_stats[key][f"n_{c['classification']}"] += 1
        pair_stats[key]["pmids"].add(c["pmid"])

    print(f"  고유 쌍 수: {len(pair_stats):,}")

    # present가 1건 이상인 쌍만 대상
    candidate_pairs = {k: v for k, v in pair_stats.items() if v["n_present"] >= 1}
    print(f"  present >= 1: {len(candidate_pairs):,}쌍")

    # Step 3: Jensen Lab score + G² 계산
    print("\n[3/4] Jensen Lab score + G² 계산...")
    alpha_values = [0.5, 0.6, 0.7]
    fdr_thresholds = [0.01, 0.05, 0.10]

    edges_data = []
    for (cui_a, cui_b), ps in candidate_pairs.items():
        n_present = ps["n_present"]
        n_absent = ps["n_absent"]
        n_not_related = ps["n_not_related"]
        n_total = n_present + n_absent + n_not_related

        # G² contingency table
        # co-occur as present | not co-occur as present
        # co-occur as other  | not co-occur as other
        a = n_present
        b = cui_doc_count.get(cui_a, 1) - n_present  # a가 등장했으나 b와 present 아닌 경우
        c = cui_doc_count.get(cui_b, 1) - n_present  # b가 등장했으나 a와 present 아닌 경우
        d = total_docs - a - b - c

        # 음수 방지
        b = max(b, 0)
        c = max(c, 0)
        d = max(d, 0)

        g2 = dunning_g2(a, b, c, d)
        p_value = 1 - stats.chi2.cdf(g2, df=1) if g2 > 0 else 1.0

        # Jensen Lab score (각 alpha)
        c_ab = float(n_present)
        c_a = float(cui_doc_count.get(cui_a, 1))
        c_b = float(cui_doc_count.get(cui_b, 1))

        scores = {}
        for alpha in alpha_values:
            scores[f"jensen_{alpha}"] = jensen_lab_score(c_ab, c_a, c_b, total_docs, alpha)

        # polarity 결정
        if n_absent > n_present:
            polarity = "absent"
        elif n_present > 0:
            polarity = "present"
        else:
            polarity = "ambiguous"

        edges_data.append({
            "cui_a": cui_a,
            "cui_b": cui_b,
            "polarity": polarity,
            "n_present": n_present,
            "n_absent": n_absent,
            "n_not_related": n_not_related,
            "g2": g2,
            "p_value": p_value,
            **scores,
            "pmids": sorted(ps["pmids"]),
        })

    # BH-FDR 보정
    p_values = [e["p_value"] for e in edges_data]
    q_values = benjamini_hochberg(p_values)
    for e, q in zip(edges_data, q_values):
        e["q_value"] = q

    # Step 4: FDR 임계값별 결과
    print("\n[4/4] FDR 임계값별 결과")
    print("=" * 80)

    for fdr in fdr_thresholds:
        significant = [e for e in edges_data if e["q_value"] < fdr]
        present_sig = [e for e in significant if e["polarity"] == "present"]
        absent_sig = [e for e in significant if e["polarity"] == "absent"]

        unique_cuis = set()
        for e in significant:
            unique_cuis.add(e["cui_a"])
            unique_cuis.add(e["cui_b"])

        print(f"\n  FDR < {fdr}:")
        print(f"    유의한 엣지: {len(significant):,}")
        print(f"    present: {len(present_sig):,}")
        print(f"    absent: {len(absent_sig):,}")
        print(f"    고유 CUI: {len(unique_cuis)}")

    # α 값별 score 분포 (FDR < 0.05 기준)
    sig_005 = [e for e in edges_data if e["q_value"] < 0.05]
    for alpha in alpha_values:
        key = f"jensen_{alpha}"
        scores = sorted([e[key] for e in sig_005], reverse=True)
        if scores:
            print(f"\n  Jensen score (α={alpha}, FDR<0.05):")
            print(f"    평균: {sum(scores)/len(scores):.2f}")
            print(f"    중앙값: {scores[len(scores)//2]:.2f}")
            print(f"    최대: {scores[0]:.2f}")

    # 상위 엣지 (FDR < 0.05, Jensen 0.6 기준)
    sig_sorted = sorted(sig_005, key=lambda e: -e["jensen_0.6"])
    print(f"\n  상위 present 관계 (FDR<0.05, Jensen α=0.6):")
    for e in sig_sorted[:20]:
        if e["polarity"] == "present":
            a_name = cui_names.get(e["cui_a"], e["cui_a"])[:35]
            b_name = cui_names.get(e["cui_b"], e["cui_b"])[:35]
            print(f"    {a_name:35s} - {b_name:35s} n={e['n_present']:3d} G²={e['g2']:6.1f} q={e['q_value']:.4f}")

    print(f"\n  상위 absent 관계 (FDR<0.05):")
    absent_sorted = [e for e in sig_sorted if e["polarity"] == "absent"]
    for e in absent_sorted[:10]:
        a_name = cui_names.get(e["cui_a"], e["cui_a"])[:35]
        b_name = cui_names.get(e["cui_b"], e["cui_b"])[:35]
        print(f"    {a_name:35s} - {b_name:35s} n_abs={e['n_absent']:3d} n_pres={e['n_present']:3d}")

    # DDXPlus 5개 질환 관련 엣지
    ddx_cuis = {"C0032285", "C0034065", "C0017168", "C0086769", "C0006277"}
    print(f"\n  DDXPlus 5개 질환 관련 엣지 (FDR<0.05):")
    for e in sig_sorted:
        if e["cui_a"] in ddx_cuis or e["cui_b"] in ddx_cuis:
            disease_cui = e["cui_a"] if e["cui_a"] in ddx_cuis else e["cui_b"]
            other_cui = e["cui_b"] if disease_cui == e["cui_a"] else e["cui_a"]
            d_name = cui_names.get(disease_cui, "?")[:25]
            o_name = cui_names.get(other_cui, "?")[:35]
            o_stys = cui_stys.get(other_cui, set()) & DISO_TYPES
            print(f"    {d_name:25s} → {o_name:35s} ({e['polarity']}) n={e['n_present']} {','.join(sorted(o_stys))}")

    # 저장 (FDR < 0.05 필터된 엣지)
    output = {
        "config": {
            "model": "qwen3:4b-instruct-2507-fp16",
            "prompt": "v2",
            "alpha": 0.6,
            "fdr_threshold": 0.05,
            "total_docs": total_docs,
        },
        "all_edges": edges_data,
        "significant_edges": sig_005,
    }
    with open(DATA_DIR / "step3_kg_edges.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {DATA_DIR / 'step3_kg_edges.json'}")


if __name__ == "__main__":
    main()
