#!/usr/bin/env python3
"""Step 1 분기: MRCONSO 사전 vs scispaCy 비교 (v2).

동일 초록 50건에 대해 두 방법을 비교한다.
QuickUMLS는 데이터 미설치로 제외.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import scispacy
from scispacy.linking import EntityLinker
import spacy
from Bio import Entrez

Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}


def fetch_abstracts(query: str, n: int = 50) -> list[dict]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=n, sort="relevance")
    pmids = Entrez.read(handle)["IdList"]
    handle.close()
    time.sleep(0.2)
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml")
    records = Entrez.read(handle)
    handle.close()
    results = []
    for article in records["PubmedArticle"]:
        pmid = str(article["MedlineCitation"]["PMID"])
        abs_parts = article["MedlineCitation"]["Article"].get("Abstract", {})
        text = " ".join(str(t) for t in abs_parts.get("AbstractText", []))
        if text and len(text.split()) > 30:
            results.append({"pmid": pmid, "text": text})
    return results


def load_diso_cuis() -> set[str]:
    diso = set()
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            if parts[1] in DISO_TYPES:
                diso.add(parts[0])
    return diso


def load_cui_names(target_cuis: set[str] = None) -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, ts, name = parts[0], parts[1], parts[2], parts[14]
            if lang != "ENG":
                continue
            if target_cuis and cui not in target_cuis:
                continue
            if cui not in names or ts == "P":
                names[cui] = name
    return names


def load_cui_stys() -> dict[str, set[str]]:
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui_stys[parts[0]].add(parts[1])
    return dict(cui_stys)


# 방법 A: MRCONSO 사전 매칭
def build_mrconso_dict(diso_cuis: set[str]) -> dict[str, str]:
    term_to_cui: dict[str, str] = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.strip().split("|")
            cui, lang, name = parts[0], parts[1], parts[14]
            if cui not in diso_cuis or lang != "ENG":
                continue
            name_lower = name.lower().strip()
            if len(name_lower) <= 3 or len(name_lower.split()) > 5:
                continue
            if re.match(r'^[A-Z0-9\.\-\:]+$', name):
                continue
            term_to_cui[name_lower] = cui
    return term_to_cui


def extract_mrconso(text: str, term_dict: dict[str, str]) -> set[str]:
    text_lower = text.lower()
    found = set()
    for term, cui in term_dict.items():
        if term in text_lower:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                found.add(cui)
    return found


# 방법 B: scispaCy
def extract_scispacy(text: str, nlp, diso_cuis: set[str], threshold: float = 0.85) -> set[str]:
    doc = nlp(text)
    found = set()
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if cui in diso_cuis and score >= threshold:
                found.add(cui)
    return found


def main():
    print("=" * 80)
    print("Step 1 분기: MRCONSO 사전 vs scispaCy 비교")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/5] 데이터 로드...")
    diso_cuis = load_diso_cuis()
    cui_stys = load_cui_stys()
    print(f"  DISO CUI: {len(diso_cuis):,}")

    # 초록 수집
    print("\n[2/5] 초록 수집 (pneumonia 50건)...")
    abstracts = fetch_abstracts("pneumonia diagnosis symptoms", n=50)
    print(f"  수집: {len(abstracts)}건")

    # 방법 A 준비
    print("\n[3/5] 방법 A: MRCONSO 사전 구축...")
    t0 = time.time()
    term_dict = build_mrconso_dict(diso_cuis)
    print(f"  사전: {len(term_dict):,} 항목 ({time.time()-t0:.1f}s)")

    # 방법 B 준비
    print("  방법 B: scispaCy + UMLS linker 로드...")
    t0 = time.time()
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
    })
    print(f"  scispaCy 로드: {time.time()-t0:.1f}s")

    # 추출 실행
    print(f"\n[4/5] {len(abstracts)}건 추출...")
    results_per_abstract = []

    for i, ab in enumerate(abstracts):
        # 방법 A
        t0 = time.time()
        cuis_a = extract_mrconso(ab["text"], term_dict)
        time_a = time.time() - t0

        # 방법 B (threshold 0.85)
        t0 = time.time()
        cuis_b = extract_scispacy(ab["text"], nlp, diso_cuis, threshold=0.85)
        time_b = time.time() - t0

        # 방법 B (threshold 0.70 - 더 관대)
        t0 = time.time()
        cuis_b_loose = extract_scispacy(ab["text"], nlp, diso_cuis, threshold=0.70)
        time_b_loose = time.time() - t0

        overlap = cuis_a & cuis_b
        a_only = cuis_a - cuis_b
        b_only = cuis_b - cuis_a

        results_per_abstract.append({
            "pmid": ab["pmid"],
            "words": len(ab["text"].split()),
            "a_count": len(cuis_a),
            "b_count": len(cuis_b),
            "b_loose_count": len(cuis_b_loose),
            "overlap": len(overlap),
            "a_only": len(a_only),
            "b_only": len(b_only),
            "a_time_ms": time_a * 1000,
            "b_time_ms": time_b * 1000,
            "a_cuis": sorted(cuis_a),
            "b_cuis": sorted(cuis_b),
            "a_only_cuis": sorted(a_only),
            "b_only_cuis": sorted(b_only),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(abstracts)}] "
                  f"A={len(cuis_a):3d}  B={len(cuis_b):3d}  B_loose={len(cuis_b_loose):3d}  "
                  f"overlap={len(overlap):3d}  A_only={len(a_only):3d}  B_only={len(b_only):3d}")

    # 분석
    print(f"\n[5/5] 분석 결과")
    print("=" * 80)

    n = len(results_per_abstract)

    for label, key in [("A (MRCONSO 사전)", "a"), ("B (scispaCy 0.85)", "b"), ("B (scispaCy 0.70)", "b_loose")]:
        counts = [r[f"{key}_count"] for r in results_per_abstract]
        print(f"\n  {label}:")
        print(f"    CUI 수 (평균): {sum(counts)/n:.1f}")
        print(f"    CUI 수 (중앙값): {sorted(counts)[n//2]}")
        print(f"    CUI 수 (최소~최대): {min(counts)}~{max(counts)}")
        if key in ("a", "b"):
            times = [r[f"{key}_time_ms"] for r in results_per_abstract]
            print(f"    처리 시간 (평균): {sum(times)/n:.1f}ms/건")

    # 겹침 분석
    total_a_only = sum(r["a_only"] for r in results_per_abstract)
    total_b_only = sum(r["b_only"] for r in results_per_abstract)
    total_overlap = sum(r["overlap"] for r in results_per_abstract)
    total_a = sum(r["a_count"] for r in results_per_abstract)
    total_b = sum(r["b_count"] for r in results_per_abstract)

    print(f"\n  A vs B (threshold=0.85) 겹침 분석:")
    print(f"    A 전체: {total_a}")
    print(f"    B 전체: {total_b}")
    print(f"    양쪽 모두: {total_overlap}")
    print(f"    A에만: {total_a_only}")
    print(f"    B에만: {total_b_only}")
    if total_a > 0:
        print(f"    A의 B 커버율: {total_overlap/total_a:.1%}")
    if total_b > 0:
        print(f"    B의 A 커버율: {total_overlap/total_b:.1%}")

    # CUI 쌍 수 비교
    pairs_a = sum(r["a_count"] * (r["a_count"] - 1) // 2 for r in results_per_abstract)
    pairs_b = sum(r["b_count"] * (r["b_count"] - 1) // 2 for r in results_per_abstract)
    print(f"\n  CUI 쌍 수 (= LLM 호출 수):")
    print(f"    A: {pairs_a:,}")
    print(f"    B: {pairs_b:,}")

    # A에만 있는 CUI 샘플 (어떤 것들이 사전 매칭에만 걸리는지)
    cui_names = load_cui_names(diso_cuis)
    a_only_counter = defaultdict(int)
    b_only_counter = defaultdict(int)
    for r in results_per_abstract:
        for cui in r["a_only_cuis"]:
            a_only_counter[cui] += 1
        for cui in r["b_only_cuis"]:
            b_only_counter[cui] += 1

    print(f"\n  A에만 있는 상위 CUI (사전 매칭의 고유 발견):")
    for cui, cnt in sorted(a_only_counter.items(), key=lambda x: -x[1])[:10]:
        stys = cui_stys.get(cui, set())
        print(f"    {cui}: {cui_names.get(cui, '?')[:50]} ({','.join(sorted(stys))}) x{cnt}")

    print(f"\n  B에만 있는 상위 CUI (scispaCy의 고유 발견):")
    for cui, cnt in sorted(b_only_counter.items(), key=lambda x: -x[1])[:10]:
        stys = cui_stys.get(cui, set())
        print(f"    {cui}: {cui_names.get(cui, '?')[:50]} ({','.join(sorted(stys))}) x{cnt}")

    # 결과 저장
    output = {
        "summary": {
            "n_abstracts": n,
            "a_avg_cui": sum(r["a_count"] for r in results_per_abstract) / n,
            "b_avg_cui": sum(r["b_count"] for r in results_per_abstract) / n,
            "b_loose_avg_cui": sum(r["b_loose_count"] for r in results_per_abstract) / n,
            "overlap_total": total_overlap,
            "a_only_total": total_a_only,
            "b_only_total": total_b_only,
            "a_pairs": pairs_a,
            "b_pairs": pairs_b,
        },
        "per_abstract": results_per_abstract,
    }
    output_file = RESULTS_DIR / "step1_mrconso_vs_scispacy.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
