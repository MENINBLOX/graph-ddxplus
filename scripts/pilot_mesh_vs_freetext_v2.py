#!/usr/bin/env python3
"""MeSH 매핑된 DISO CUI 100개로 MeSH vs 자유 텍스트 검색 비교.

v1에서 random DISO CUI는 MeSH에 매핑되지 않아 검색 결과가 0건이었음.
v2는 SAB=MSH인 CUI만 대상으로 비교한다.
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from Bio import Entrez

Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")
OUTPUT_DIR = Path("results/pilot_mesh_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}


def load_mesh_diso_terms() -> list[tuple[str, str, set[str]]]:
    """MRCONSO에서 SAB=MSH인 DISO CUI와 MeSH term을 로드한다."""
    print("[1/4] MRSTY에서 DISO CUI 로드...")
    diso_cuis: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            if parts[1] in DISO_TYPES:
                diso_cuis[parts[0]].add(parts[1])
    print(f"  DISO CUI 수: {len(diso_cuis):,}")

    print("[2/4] MRCONSO에서 MeSH 매핑 CUI 로드...")
    mesh_terms: dict[str, str] = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, ts, sab, name = parts[0], parts[1], parts[2], parts[11], parts[14]
            if cui not in diso_cuis:
                continue
            if lang != "ENG" or sab != "MSH":
                continue
            if cui not in mesh_terms or ts == "P":
                mesh_terms[cui] = name
    print(f"  MeSH 매핑 DISO CUI: {len(mesh_terms):,}")

    result = [
        (cui, name, diso_cuis[cui])
        for cui, name in mesh_terms.items()
    ]
    return result


def search_pubmed(query: str, retmax: int = 200) -> list[str]:
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])
    except Exception as e:
        print(f"    검색 오류: {e}")
        return []


def compare_search_methods(
    sample: list[tuple[str, str, set[str]]],
) -> list[dict]:
    print(f"[3/4] {len(sample)}개 MeSH 키워드 검색 비교...")

    processed_pmids: set[str] = set()
    results = []

    for i, (cui, name, stys) in enumerate(sample):
        # 자유 텍스트 검색
        freetext_pmids = search_pubmed(f'"{name}"', retmax=200)
        time.sleep(0.12)

        # MeSH 검색
        mesh_pmids = search_pubmed(f'"{name}"[MeSH Terms]', retmax=200)
        time.sleep(0.12)

        freetext_set = set(freetext_pmids)
        mesh_set = set(mesh_pmids)
        overlap = freetext_set & mesh_set

        all_pmids = freetext_set | mesh_set
        new_pmids = all_pmids - processed_pmids
        processed_pmids.update(all_pmids)

        result = {
            "cui": cui,
            "name": name,
            "semantic_types": sorted(stys),
            "freetext_count": len(freetext_set),
            "mesh_count": len(mesh_set),
            "overlap_count": len(overlap),
            "freetext_only": len(freetext_set - mesh_set),
            "mesh_only": len(mesh_set - freetext_set),
            "total_unique": len(all_pmids),
            "new_pmids": len(new_pmids),
            "cumulative_processed": len(processed_pmids),
        }
        results.append(result)

        print(
            f"  [{i+1:3d}/100] {name[:40]:<40} "
            f"free={result['freetext_count']:>4d}  "
            f"mesh={result['mesh_count']:>4d}  "
            f"overlap={result['overlap_count']:>4d}  "
            f"mesh_only={result['mesh_only']:>4d}  "
            f"new={result['new_pmids']:>4d}  "
            f"cumul={result['cumulative_processed']:>6d}"
        )

    return results


def analyze_results(results: list[dict]) -> None:
    print("\n[4/4] 분석 결과")
    print("=" * 80)

    n = len(results)
    ft = [r["freetext_count"] for r in results]
    ms = [r["mesh_count"] for r in results]
    ov = [r["overlap_count"] for r in results]
    mo = [r["mesh_only"] for r in results]

    print(f"\n{'':>30} {'자유 텍스트':>12} {'MeSH':>12} {'Overlap':>12} {'MeSH only':>12}")
    print(f"{'평균':>30} {sum(ft)/n:>12.1f} {sum(ms)/n:>12.1f} {sum(ov)/n:>12.1f} {sum(mo)/n:>12.1f}")
    print(f"{'중앙값':>30} {sorted(ft)[n//2]:>12d} {sorted(ms)[n//2]:>12d} {sorted(ov)[n//2]:>12d} {sorted(mo)[n//2]:>12d}")
    print(f"{'합계':>30} {sum(ft):>12d} {sum(ms):>12d} {sum(ov):>12d} {sum(mo):>12d}")

    # MeSH 결과가 0인 케이스
    mesh_zero = [r for r in results if r["mesh_count"] == 0]
    ft_zero = [r for r in results if r["freetext_count"] == 0]
    print(f"\n  MeSH 결과 0건: {len(mesh_zero)}개")
    if mesh_zero:
        for r in mesh_zero[:10]:
            print(f"    - {r['name']} (free={r['freetext_count']})")
    print(f"  자유 텍스트 결과 0건: {len(ft_zero)}개")

    # 비교
    mesh_more = sum(1 for r in results if r["mesh_count"] > r["freetext_count"])
    ft_more = sum(1 for r in results if r["freetext_count"] > r["mesh_count"])
    equal = sum(1 for r in results if r["freetext_count"] == r["mesh_count"])
    print(f"\n  자유 텍스트 > MeSH: {ft_more}건")
    print(f"  MeSH > 자유 텍스트: {mesh_more}건")
    print(f"  동일: {equal}건")

    # MeSH 커버리지 (MeSH가 자유 텍스트 결과의 몇 %를 커버하는가)
    coverages = []
    for r in results:
        if r["freetext_count"] > 0:
            coverages.append(r["overlap_count"] / r["freetext_count"])
    if coverages:
        print(f"\n  MeSH → 자유 텍스트 커버리지 (평균): {sum(coverages)/len(coverages):.1%}")
        print(f"  MeSH → 자유 텍스트 커버리지 (중앙값): {sorted(coverages)[len(coverages)//2]:.1%}")

    # MeSH only 비율 (MeSH에만 있고 자유 텍스트에 없는 것)
    mesh_only_rates = []
    for r in results:
        if r["mesh_count"] > 0:
            mesh_only_rates.append(r["mesh_only"] / r["mesh_count"])
    if mesh_only_rates:
        print(f"  MeSH only 비율 (평균): {sum(mesh_only_rates)/len(mesh_only_rates):.1%}")

    # 중복 처리
    total_fetched = sum(r["total_unique"] for r in results)
    cumulative = results[-1]["cumulative_processed"]
    dedup_rate = 1 - (cumulative / max(total_fetched, 1))
    print(f"\n  총 PMID 수 (키워드별 합산): {total_fetched:,}")
    print(f"  고유 PMID 수 (중복 제거 후): {cumulative:,}")
    print(f"  중복 제거율: {dedup_rate:.1%}")

    # Semantic type 분포
    sty_dist = defaultdict(int)
    for r in results:
        for st in r["semantic_types"]:
            sty_dist[st] += 1
    print(f"\n  Semantic type 분포:")
    for st, cnt in sorted(sty_dist.items(), key=lambda x: -x[1]):
        print(f"    {st}: {cnt}건")


def main():
    print("=" * 80)
    print("MeSH 매핑 DISO CUI 100개: MeSH vs 자유 텍스트 비교 (v2)")
    print("=" * 80)

    all_mesh_terms = load_mesh_diso_terms()

    # Semantic type별 균등 샘플링
    random.seed(42)
    by_sty: dict[str, list] = defaultdict(list)
    for item in all_mesh_terms:
        for sty in item[2]:
            by_sty[sty].append(item)

    sample_set: set[str] = set()
    sample_list: list[tuple[str, str, set[str]]] = []
    per_type = max(1, 100 // len(DISO_TYPES))

    for sty in sorted(DISO_TYPES):
        available = [x for x in by_sty.get(sty, []) if x[0] not in sample_set]
        n = min(per_type, len(available))
        chosen = random.sample(available, n)
        for item in chosen:
            if item[0] not in sample_set:
                sample_set.add(item[0])
                sample_list.append(item)

    # 100개 맞추기
    remaining = [x for x in all_mesh_terms if x[0] not in sample_set]
    while len(sample_list) < 100 and remaining:
        item = random.choice(remaining)
        remaining.remove(item)
        sample_set.add(item[0])
        sample_list.append(item)

    sample_list = sample_list[:100]
    print(f"  최종 샘플: {len(sample_list)}개")

    results = compare_search_methods(sample_list)
    analyze_results(results)

    output_file = OUTPUT_DIR / "mesh_vs_freetext_v2_100.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
