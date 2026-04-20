#!/usr/bin/env python3
"""MeSH 검색 vs 자유 텍스트 검색 비교 파일럿.

UMLS DISO 키워드 100개로 PubMed 검색 결과를 비교한다.
- 자유 텍스트 검색: keyword 그대로 검색
- MeSH 검색: keyword[MeSH Terms]로 검색

또한 중복 PMID 추적 패턴을 적용하여
동일 초록이 여러 키워드에서 반복 처리되지 않도록 한다.
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from Bio import Entrez

# API 설정
Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")
OUTPUT_DIR = Path("results/pilot_mesh_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DISO semantic types
DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}


def load_diso_cuis() -> dict[str, set[str]]:
    """MRSTY에서 DISO CUI와 semantic type을 로드한다."""
    print("[1/4] MRSTY에서 DISO CUI 로드...")
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, sty = parts[0], parts[1]
            if sty in DISO_TYPES:
                cui_stys[cui].add(sty)
    print(f"  DISO CUI 수: {len(cui_stys):,}")
    return dict(cui_stys)


def load_preferred_names(target_cuis: set[str]) -> dict[str, str]:
    """MRCONSO에서 영문 preferred term을 로드한다."""
    print("[2/4] MRCONSO에서 preferred term 로드...")
    cui_names: dict[str, str] = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, ts, name = parts[0], parts[1], parts[2], parts[14]
            if cui not in target_cuis:
                continue
            if lang != "ENG":
                continue
            # Preferred term 우선
            if cui not in cui_names or ts == "P":
                cui_names[cui] = name
    print(f"  영문 이름 확보: {len(cui_names):,}")
    return cui_names


def search_pubmed(query: str, retmax: int = 200) -> list[str]:
    """PubMed 검색하여 PMID 목록을 반환한다."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])
    except Exception as e:
        print(f"    검색 오류: {e}")
        return []


def compare_search_methods(
    sample_cuis: list[tuple[str, str, set[str]]],
) -> list[dict]:
    """자유 텍스트 vs MeSH 검색 결과를 비교한다."""
    print(f"[3/4] {len(sample_cuis)}개 키워드 검색 비교...")

    # 중복 PMID 추적 (프로세스 개선: 이미 처리된 논문 기록)
    processed_pmids: set[str] = set()
    results = []

    for i, (cui, name, stys) in enumerate(sample_cuis):
        # 자유 텍스트 검색
        freetext_pmids = search_pubmed(f'"{name}"', retmax=200)
        time.sleep(0.15)  # rate limit

        # MeSH 검색
        mesh_pmids = search_pubmed(f'"{name}"[MeSH Terms]', retmax=200)
        time.sleep(0.15)

        freetext_set = set(freetext_pmids)
        mesh_set = set(mesh_pmids)
        overlap = freetext_set & mesh_set

        # 새로 발견된 PMID (중복 제거)
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

        status = (
            f"  [{i+1:3d}/100] {name[:40]:<40} "
            f"free={result['freetext_count']:>4d}  "
            f"mesh={result['mesh_count']:>4d}  "
            f"overlap={result['overlap_count']:>4d}  "
            f"new={result['new_pmids']:>4d}  "
            f"cumul={result['cumulative_processed']:>6d}"
        )
        print(status)

    return results


def analyze_results(results: list[dict]) -> None:
    """결과를 분석하고 보고한다."""
    print("\n[4/4] 분석 결과")
    print("=" * 80)

    n = len(results)

    # 기본 통계
    ft_counts = [r["freetext_count"] for r in results]
    mesh_counts = [r["mesh_count"] for r in results]
    overlap_counts = [r["overlap_count"] for r in results]

    print(f"\n{'':>40} {'자유 텍스트':>12} {'MeSH':>12} {'Overlap':>12}")
    print(f"{'평균':>40} {sum(ft_counts)/n:>12.1f} {sum(mesh_counts)/n:>12.1f} {sum(overlap_counts)/n:>12.1f}")
    print(f"{'중앙값':>40} {sorted(ft_counts)[n//2]:>12d} {sorted(mesh_counts)[n//2]:>12d} {sorted(overlap_counts)[n//2]:>12d}")
    print(f"{'합계':>40} {sum(ft_counts):>12d} {sum(mesh_counts):>12d} {sum(overlap_counts):>12d}")

    # MeSH 결과가 0인 케이스
    mesh_zero = [r for r in results if r["mesh_count"] == 0]
    ft_zero = [r for r in results if r["freetext_count"] == 0]
    print(f"\n  MeSH 결과 0건: {len(mesh_zero)}개")
    if mesh_zero:
        for r in mesh_zero[:10]:
            print(f"    - {r['name']} (CUI: {r['cui']}, free={r['freetext_count']})")

    print(f"  자유 텍스트 결과 0건: {len(ft_zero)}개")

    # MeSH가 자유 텍스트보다 많은 케이스
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
            coverage = r["overlap_count"] / r["freetext_count"]
            coverages.append(coverage)
    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        print(f"\n  MeSH → 자유 텍스트 커버리지 (평균): {avg_coverage:.1%}")
        print(f"  MeSH → 자유 텍스트 커버리지 (중앙값): {sorted(coverages)[len(coverages)//2]:.1%}")

    # 중복 처리 효과
    total_fetched = sum(r["total_unique"] for r in results)
    cumulative = results[-1]["cumulative_processed"]
    dedup_rate = 1 - (cumulative / max(total_fetched, 1))
    print(f"\n  총 PMID 수 (키워드별 합산): {total_fetched:,}")
    print(f"  고유 PMID 수 (중복 제거 후): {cumulative:,}")
    print(f"  중복 제거율: {dedup_rate:.1%}")

    # Semantic type별 분포
    sty_dist = defaultdict(int)
    for r in results:
        for st in r["semantic_types"]:
            sty_dist[st] += 1
    print(f"\n  Semantic type 분포:")
    for st, cnt in sorted(sty_dist.items(), key=lambda x: -x[1]):
        print(f"    {st}: {cnt}건")


def main():
    print("=" * 80)
    print("MeSH vs 자유 텍스트 PubMed 검색 비교 (100 DISO 키워드)")
    print("=" * 80)

    # DISO CUI 로드
    diso_cuis = load_diso_cuis()

    # 다양한 semantic type에서 균등 샘플링
    cuis_by_sty: dict[str, list[str]] = defaultdict(list)
    for cui, stys in diso_cuis.items():
        for sty in stys:
            cuis_by_sty[sty].append(cui)

    # 각 semantic type에서 비례적으로 샘플링
    random.seed(42)
    sample_cuis_set: set[str] = set()
    per_type = max(1, 100 // len(DISO_TYPES))
    for sty in sorted(DISO_TYPES):
        available = cuis_by_sty.get(sty, [])
        n_sample = min(per_type, len(available))
        sample_cuis_set.update(random.sample(available, n_sample))

    # 100개로 맞추기
    all_diso_list = list(diso_cuis.keys())
    while len(sample_cuis_set) < 100:
        sample_cuis_set.add(random.choice(all_diso_list))
    sample_cuis_set = set(list(sample_cuis_set)[:100])

    # Preferred name 로드
    cui_names = load_preferred_names(sample_cuis_set)

    # 이름이 있는 CUI만 사용
    sample = [
        (cui, cui_names[cui], diso_cuis[cui])
        for cui in sample_cuis_set
        if cui in cui_names
    ][:100]

    print(f"  최종 샘플: {len(sample)}개")

    # 검색 비교 실행
    results = compare_search_methods(sample)

    # 분석
    analyze_results(results)

    # 결과 저장
    output_file = OUTPUT_DIR / "mesh_vs_freetext_100.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
