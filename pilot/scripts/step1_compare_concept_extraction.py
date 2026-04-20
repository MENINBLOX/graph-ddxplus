#!/usr/bin/env python3
"""Step 1 분기: 개념 식별 방법 3가지 비교.

동일한 PubMed 초록 50건에 대해:
(A) MRCONSO 사전 매칭
(B) scispaCy + UMLS linker
(C) QuickUMLS

정밀도, 재현율, 속도, DISO CUI 수를 비교한다.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from Bio import Entrez

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}


# ============================================================
# 공통: PubMed 초록 수집
# ============================================================

def fetch_abstracts(query: str, n: int = 50) -> list[dict]:
    """PubMed에서 초록을 수집한다."""
    print(f"  PubMed 검색: {query} (n={n})")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=n, sort="relevance")
    pmids = Entrez.read(handle)["IdList"]
    handle.close()
    time.sleep(0.2)

    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml")
    records = Entrez.read(handle)
    handle.close()

    abstracts = []
    for article in records["PubmedArticle"]:
        pmid = str(article["MedlineCitation"]["PMID"])
        abs_parts = article["MedlineCitation"]["Article"].get("Abstract", {})
        text = " ".join(str(t) for t in abs_parts.get("AbstractText", []))
        if text and len(text.split()) > 30:
            abstracts.append({"pmid": pmid, "text": text})
    print(f"  수집: {len(abstracts)}건")
    return abstracts


# ============================================================
# 공통: DISO CUI 로드
# ============================================================

def load_diso_cuis() -> set[str]:
    """MRSTY에서 DISO CUI 목록을 로드한다."""
    diso = set()
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            if parts[1] in DISO_TYPES:
                diso.add(parts[0])
    return diso


def load_cui_stys() -> dict[str, set[str]]:
    """MRSTY에서 CUI -> semantic types 매핑을 로드한다."""
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui_stys[parts[0]].add(parts[1])
    return dict(cui_stys)


# ============================================================
# 방법 A: MRCONSO 사전 매칭
# ============================================================

def build_mrconso_dict(diso_cuis: set[str]) -> dict[str, str]:
    """MRCONSO에서 DISO 용어 사전을 구축한다."""
    term_to_cui: dict[str, str] = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, sab, name = parts[0], parts[1], parts[11], parts[14]
            if cui not in diso_cuis or lang != "ENG":
                continue
            name_lower = name.lower().strip()
            word_count = len(name_lower.split())
            # 2~5 단어 용어만 (너무 짧으면 노이즈, 너무 길면 매칭 안됨)
            if len(name_lower) <= 3 or word_count > 5:
                continue
            if re.match(r'^[A-Z0-9\.\-\:]+$', name):
                continue
            term_to_cui[name_lower] = cui
    return term_to_cui


def extract_method_a(text: str, term_dict: dict[str, str]) -> set[str]:
    """MRCONSO 사전 매칭으로 DISO CUI를 추출한다."""
    text_lower = text.lower()
    found = set()
    for term, cui in term_dict.items():
        if term in text_lower:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                found.add(cui)
    return found


# ============================================================
# 방법 B: scispaCy
# ============================================================

def setup_scispacy():
    """scispaCy 모델을 로드한다."""
    import spacy
    nlp = spacy.load("en_core_sci_sm")
    # UMLS linker 추가
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
    })
    return nlp


def extract_method_b(text: str, nlp, diso_cuis: set[str]) -> set[str]:
    """scispaCy + UMLS linker로 DISO CUI를 추출한다."""
    doc = nlp(text)
    found = set()
    linker = nlp.get_pipe("scispacy_linker")
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if cui in diso_cuis and score > 0.7:
                found.add(cui)
    return found


# ============================================================
# 방법 C: QuickUMLS
# ============================================================

def setup_quickumls():
    """QuickUMLS를 설정한다."""
    try:
        from quickumls import QuickUMLS
        # QuickUMLS는 사전 설치된 데이터가 필요
        # 없으면 None 반환
        quickumls_path = Path("data/quickumls_data")
        if quickumls_path.exists():
            matcher = QuickUMLS(str(quickumls_path))
            return matcher
        else:
            print("  QuickUMLS 데이터 미설치 (data/quickumls_data 없음)")
            return None
    except Exception as e:
        print(f"  QuickUMLS 설정 실패: {e}")
        return None


def extract_method_c(text: str, matcher, diso_cuis: set[str]) -> set[str]:
    """QuickUMLS로 DISO CUI를 추출한다."""
    if matcher is None:
        return set()
    found = set()
    matches = matcher.match(text)
    for match_group in matches:
        for match in match_group:
            cui = match["cui"]
            if cui in diso_cuis:
                found.add(cui)
    return found


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 80)
    print("Step 1 분기: 개념 식별 방법 3가지 비교")
    print("=" * 80)

    # 1. 데이터 로드
    print("\n[1/5] 데이터 로드...")
    diso_cuis = load_diso_cuis()
    cui_stys = load_cui_stys()
    print(f"  DISO CUI: {len(diso_cuis):,}")

    # 2. 초록 수집 (pneumonia 관련 50건)
    print("\n[2/5] 초록 수집...")
    abstracts = fetch_abstracts("pneumonia diagnosis symptoms", n=50)

    # 3. 방법 A 준비
    print("\n[3/5] 방법별 준비...")
    print("  (A) MRCONSO 사전 구축...")
    t0 = time.time()
    term_dict = build_mrconso_dict(diso_cuis)
    print(f"  사전 크기: {len(term_dict):,} 항목, 구축 시간: {time.time()-t0:.1f}s")

    # 방법 B 준비
    print("  (B) scispaCy 로드...")
    t0 = time.time()
    try:
        nlp = setup_scispacy()
        scispacy_ok = True
        print(f"  scispaCy 로드 시간: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  scispaCy 실패: {e}")
        nlp = None
        scispacy_ok = False

    # 방법 C 준비
    print("  (C) QuickUMLS 설정...")
    matcher = setup_quickumls()
    quickumls_ok = matcher is not None

    # 4. 추출 실행
    print(f"\n[4/5] {len(abstracts)}건 초록에서 추출 실행...")
    results = []

    for i, abstract in enumerate(abstracts):
        row = {"pmid": abstract["pmid"], "text_len": len(abstract["text"].split())}

        # 방법 A
        t0 = time.time()
        cuis_a = extract_method_a(abstract["text"], term_dict)
        row["a_time"] = time.time() - t0
        row["a_count"] = len(cuis_a)
        row["a_cuis"] = sorted(cuis_a)

        # 방법 B
        if scispacy_ok:
            t0 = time.time()
            cuis_b = extract_method_b(abstract["text"], nlp, diso_cuis)
            row["b_time"] = time.time() - t0
            row["b_count"] = len(cuis_b)
            row["b_cuis"] = sorted(cuis_b)
        else:
            cuis_b = set()
            row["b_time"] = 0
            row["b_count"] = 0
            row["b_cuis"] = []

        # 방법 C
        if quickumls_ok:
            t0 = time.time()
            cuis_c = extract_method_c(abstract["text"], matcher, diso_cuis)
            row["c_time"] = time.time() - t0
            row["c_count"] = len(cuis_c)
            row["c_cuis"] = sorted(cuis_c)
        else:
            cuis_c = set()
            row["c_time"] = 0
            row["c_count"] = 0
            row["c_cuis"] = []

        # 교집합/합집합
        all_methods = [("A", cuis_a)]
        if scispacy_ok:
            all_methods.append(("B", cuis_b))
        if quickumls_ok:
            all_methods.append(("C", cuis_c))

        union = set()
        for _, s in all_methods:
            union |= s
        row["union_count"] = len(union)

        results.append(row)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(abstracts)}] A={row['a_count']:3d}  B={row['b_count']:3d}  C={row['c_count']:3d}")

    # 5. 분석
    print(f"\n[5/5] 분석 결과")
    print("=" * 80)

    methods = ["A (MRCONSO)"]
    if scispacy_ok:
        methods.append("B (scispaCy)")
    if quickumls_ok:
        methods.append("C (QuickUMLS)")

    for method_label, key in [("A (MRCONSO)", "a"), ("B (scispaCy)", "b"), ("C (QuickUMLS)", "c")]:
        if key == "b" and not scispacy_ok:
            continue
        if key == "c" and not quickumls_ok:
            continue

        counts = [r[f"{key}_count"] for r in results]
        times = [r[f"{key}_time"] for r in results]
        print(f"\n  {method_label}:")
        print(f"    DISO CUI 수 (평균): {sum(counts)/len(counts):.1f}")
        print(f"    DISO CUI 수 (중앙값): {sorted(counts)[len(counts)//2]}")
        print(f"    DISO CUI 수 (합계): {sum(counts)}")
        print(f"    처리 시간 (평균): {sum(times)/len(times)*1000:.1f}ms")
        print(f"    처리 시간 (합계): {sum(times):.1f}s")

    # 방법 간 겹침 분석 (A vs B)
    if scispacy_ok:
        print(f"\n  A ∩ B 분석:")
        total_a_only = 0
        total_b_only = 0
        total_both = 0
        for r in results:
            a_set = set(r["a_cuis"])
            b_set = set(r["b_cuis"])
            total_a_only += len(a_set - b_set)
            total_b_only += len(b_set - a_set)
            total_both += len(a_set & b_set)
        print(f"    A에만: {total_a_only}")
        print(f"    B에만: {total_b_only}")
        print(f"    양쪽 모두: {total_both}")

    # CUI 쌍 수 (진단에 실제 사용될 규모)
    print(f"\n  CUI 쌍 수 (n*(n-1)/2):")
    for method_label, key in [("A", "a"), ("B", "b"), ("C", "c")]:
        if key == "b" and not scispacy_ok:
            continue
        if key == "c" and not quickumls_ok:
            continue
        total_pairs = sum(r[f"{key}_count"] * (r[f"{key}_count"] - 1) // 2 for r in results)
        print(f"    {method_label}: {total_pairs:,} 쌍 (LLM 호출 수)")

    # 결과 저장
    output_file = RESULTS_DIR / "step1_concept_extraction_comparison.json"
    with open(output_file, "w") as f:
        json.dump({
            "methods_tested": methods,
            "n_abstracts": len(abstracts),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
