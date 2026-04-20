#!/usr/bin/env python3
"""Abstract vs Full-text 관계 추출 비교 파일럿.

PMC Open Access 논문 5편에서:
1. 초록만으로 DISO CUI 쌍 추출
2. 전문(full text)으로 DISO CUI 쌍 추출
3. 차이 비교

MetaMap 미설치 상태이므로 MRCONSO 기반 사전 매칭으로 대리 테스트.
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from Bio import Entrez

Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}


def build_diso_dictionary() -> dict[str, str]:
    """MRCONSO에서 DISO 용어 사전을 구축한다.

    key: 소문자 영문 term
    value: CUI

    너무 짧거나(2글자 이하) 너무 긴(6단어 이상) 용어는 제외.
    """
    print("[1/5] DISO 용어 사전 구축...")

    # DISO CUI 로드
    diso_cuis = set()
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            if parts[1] in DISO_TYPES:
                diso_cuis.add(parts[0])

    # 영문 preferred term 로드
    term_to_cui: dict[str, str] = {}
    cui_to_name: dict[str, str] = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, ts, sab, name = parts[0], parts[1], parts[2], parts[11], parts[14]
            if cui not in diso_cuis or lang != "ENG":
                continue

            # preferred term 우선 저장
            if cui not in cui_to_name or ts == "P":
                cui_to_name[cui] = name

            # 사전에 추가 (필터링 적용)
            name_lower = name.lower()
            word_count = len(name_lower.split())
            if len(name_lower) <= 2 or word_count >= 7:
                continue
            # 코드 같은 것 제외
            if re.match(r'^[A-Z0-9\.\-]+$', name):
                continue
            term_to_cui[name_lower] = cui

    print(f"  DISO CUI: {len(diso_cuis):,}")
    print(f"  사전 항목: {len(term_to_cui):,}")
    print(f"  CUI 이름: {len(cui_to_name):,}")
    return term_to_cui, cui_to_name


def find_diso_concepts(text: str, term_dict: dict[str, str]) -> dict[str, list[str]]:
    """텍스트에서 DISO 개념을 사전 매칭으로 찾는다.

    Returns: {cui: [matched_terms]}
    """
    text_lower = text.lower()
    found: dict[str, list[str]] = defaultdict(list)

    # 길이순 정렬 (긴 것부터 매칭하여 부분 매칭 최소화)
    sorted_terms = sorted(term_dict.keys(), key=len, reverse=True)

    # 성능을 위해 3단어 이상 용어만 정확 매칭
    for term in sorted_terms:
        if term in text_lower:
            # 단어 경계 확인 (부분 매칭 방지)
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                cui = term_dict[term]
                if term not in found[cui]:
                    found[cui].append(term)

    return dict(found)


def get_pmc_fulltext(pmc_ids: list[str]) -> list[dict]:
    """PMC에서 full text XML을 가져와 abstract와 body를 분리한다."""
    print(f"[3/5] PMC에서 {len(pmc_ids)}편 full text 다운로드...")
    papers = []

    for pmc_id in pmc_ids:
        try:
            handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml")
            xml_text = handle.read()
            handle.close()

            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")

            root = ET.fromstring(xml_text)

            # 제목
            title = ""
            title_elem = root.find(".//article-title")
            if title_elem is not None:
                title = ET.tostring(title_elem, encoding="unicode", method="text").strip()

            # 초록
            abstract_parts = []
            for abs_elem in root.findall(".//abstract"):
                abstract_parts.append(
                    ET.tostring(abs_elem, encoding="unicode", method="text").strip()
                )
            abstract_text = " ".join(abstract_parts)

            # 본문
            body_parts = []
            for body_elem in root.findall(".//body"):
                body_parts.append(
                    ET.tostring(body_elem, encoding="unicode", method="text").strip()
                )
            body_text = " ".join(body_parts)

            full_text = abstract_text + " " + body_text

            papers.append({
                "pmc_id": pmc_id,
                "title": title[:100],
                "abstract": abstract_text,
                "body": body_text,
                "full_text": full_text,
                "abstract_words": len(abstract_text.split()),
                "body_words": len(body_text.split()),
                "full_words": len(full_text.split()),
            })
            print(f"  {pmc_id}: {title[:60]}... (abstract={len(abstract_text.split())}w, body={len(body_text.split())}w)")
            time.sleep(0.3)

        except Exception as e:
            print(f"  {pmc_id}: 오류 - {e}")

    return papers


def search_pmc_open_access(query: str = "pneumonia OR diabetes OR hypertension", n: int = 5) -> list[str]:
    """PMC Open Access에서 논문 검색."""
    print(f"[2/5] PMC Open Access 검색: {query}")
    handle = Entrez.esearch(
        db="pmc",
        term=f'({query}) AND "open access"[filter]',
        retmax=n,
        sort="relevance",
    )
    record = Entrez.read(handle)
    handle.close()
    pmc_ids = record.get("IdList", [])
    print(f"  검색 결과: {len(pmc_ids)}건")
    return pmc_ids


def main():
    print("=" * 80)
    print("Abstract vs Full-text DISO 관계 추출 비교 (5편)")
    print("=" * 80)

    # 용어 사전 구축
    term_dict, cui_names = build_diso_dictionary()

    # PMC에서 다양한 질환의 논문 5편 검색
    pmc_ids = search_pmc_open_access(
        query="(pneumonia symptoms) OR (diabetes complications) OR (heart failure diagnosis)",
        n=5,
    )

    if not pmc_ids:
        print("PMC 검색 결과가 없습니다.")
        return

    # Full text 다운로드
    papers = get_pmc_fulltext(pmc_ids)

    # 비교 분석
    print(f"\n[4/5] Abstract vs Full-text 비교 분석...")
    print("=" * 80)

    all_results = []

    for paper in papers:
        # Abstract에서 CUI 추출
        abs_concepts = find_diso_concepts(paper["abstract"], term_dict)
        # Full text에서 CUI 추출
        full_concepts = find_diso_concepts(paper["full_text"], term_dict)

        abs_cuis = set(abs_concepts.keys())
        full_cuis = set(full_concepts.keys())

        # CUI 쌍 생성
        abs_pairs = set()
        for c1 in abs_cuis:
            for c2 in abs_cuis:
                if c1 < c2:
                    abs_pairs.add((c1, c2))

        full_pairs = set()
        for c1 in full_cuis:
            for c2 in full_cuis:
                if c1 < c2:
                    full_pairs.add((c1, c2))

        new_cuis = full_cuis - abs_cuis
        new_pairs = full_pairs - abs_pairs

        result = {
            "pmc_id": paper["pmc_id"],
            "title": paper["title"],
            "abstract_words": paper["abstract_words"],
            "body_words": paper["body_words"],
            "abs_cui_count": len(abs_cuis),
            "full_cui_count": len(full_cuis),
            "new_cui_count": len(new_cuis),
            "abs_pair_count": len(abs_pairs),
            "full_pair_count": len(full_pairs),
            "new_pair_count": len(new_pairs),
        }
        all_results.append(result)

        print(f"\n--- {paper['pmc_id']}: {paper['title'][:70]} ---")
        print(f"  텍스트 길이: abstract={paper['abstract_words']}w, body={paper['body_words']}w")
        print(f"  DISO 개념: abstract={len(abs_cuis)}, full={len(full_cuis)}, 추가={len(new_cuis)}")
        print(f"  CUI 쌍:    abstract={len(abs_pairs)}, full={len(full_pairs)}, 추가={len(new_pairs)}")

        if new_cuis:
            print(f"  Full text에서만 발견된 개념 (상위 10):")
            for cui in sorted(new_cuis)[:10]:
                name = cui_names.get(cui, cui)
                terms = full_concepts[cui]
                print(f"    {cui}: {name} (매칭: {terms[0]})")

    # 종합
    print(f"\n[5/5] 종합 결과")
    print("=" * 80)
    print(f"\n{'PMC ID':<12} {'Abstract CUI':>14} {'Full CUI':>10} {'추가 CUI':>10} {'증가율':>8} {'Abstract 쌍':>13} {'Full 쌍':>9} {'추가 쌍':>9} {'증가율':>8}")
    print("-" * 105)

    total_abs_cui = 0
    total_full_cui = 0
    total_abs_pair = 0
    total_full_pair = 0

    for r in all_results:
        cui_rate = (r["full_cui_count"] / max(r["abs_cui_count"], 1) - 1) * 100
        pair_rate = (r["full_pair_count"] / max(r["abs_pair_count"], 1) - 1) * 100
        print(
            f"{r['pmc_id']:<12} {r['abs_cui_count']:>14} {r['full_cui_count']:>10} "
            f"{r['new_cui_count']:>10} {cui_rate:>7.0f}% {r['abs_pair_count']:>13} "
            f"{r['full_pair_count']:>9} {r['new_pair_count']:>9} {pair_rate:>7.0f}%"
        )
        total_abs_cui += r["abs_cui_count"]
        total_full_cui += r["full_cui_count"]
        total_abs_pair += r["abs_pair_count"]
        total_full_pair += r["full_pair_count"]

    print("-" * 105)
    avg_cui_rate = (total_full_cui / max(total_abs_cui, 1) - 1) * 100
    avg_pair_rate = (total_full_pair / max(total_abs_pair, 1) - 1) * 100
    print(f"{'합계':<12} {total_abs_cui:>14} {total_full_cui:>10} "
          f"{total_full_cui - total_abs_cui:>10} {avg_cui_rate:>7.0f}% {total_abs_pair:>13} "
          f"{total_full_pair:>9} {total_full_pair - total_abs_pair:>9} {avg_pair_rate:>7.0f}%")


if __name__ == "__main__":
    main()
