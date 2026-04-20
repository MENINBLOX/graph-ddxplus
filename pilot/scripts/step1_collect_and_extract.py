#!/usr/bin/env python3
"""Step 1 실행: 5개 질환 PubMed/PMC 수집 + scispaCy CUI 식별.

5개 질환 × 100건 = 500건 초록 수집
PMC full text 가능한 건은 full text 처리
scispaCy (threshold=0.85)로 DISO CUI 식별
처리 완료 PMID 추적으로 중복 방지
"""
from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import scispacy
from scispacy.linking import EntityLinker
import spacy
from Bio import Entrez

Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}

SEED_DISEASES = {
    "Pneumonia": "pneumonia",
    "Pulmonary embolism": "pulmonary embolism",
    "GERD": "gastroesophageal reflux disease",
    "Panic attack": "panic attack",
    "Bronchitis": "acute bronchitis",
}

SCISPACY_THRESHOLD = 0.85


def load_diso_cuis() -> set[str]:
    diso = set()
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] in DISO_TYPES:
                diso.add(p[0])
    return diso


def load_cui_stys() -> dict[str, set[str]]:
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui_stys[p[0]].add(p[1])
    return dict(cui_stys)


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, lang, ts, name = p[0], p[1], p[2], p[14]
            if lang != "ENG":
                continue
            if cui not in names or ts == "P":
                names[cui] = name
    return names


def fetch_pubmed_abstracts(query: str, n: int = 100) -> list[dict]:
    """PubMed 초록 수집. Publication type 필터링 포함."""
    handle = Entrez.esearch(
        db="pubmed",
        term=f'({query}) AND (Journal Article[pt] OR Clinical Trial[pt] OR Case Reports[pt] OR Review[pt])',
        retmax=n,
        sort="relevance",
    )
    pmids = Entrez.read(handle)["IdList"]
    handle.close()
    time.sleep(0.2)

    if not pmids:
        return []

    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml")
    records = Entrez.read(handle)
    handle.close()

    results = []
    for article in records["PubmedArticle"]:
        pmid = str(article["MedlineCitation"]["PMID"])
        abs_parts = article["MedlineCitation"]["Article"].get("Abstract", {})
        text = " ".join(str(t) for t in abs_parts.get("AbstractText", []))

        # PMC ID 확인
        pmc_id = None
        for id_item in article.get("PubmedData", {}).get("ArticleIdList", []):
            if str(id_item.attributes.get("IdType", "")) == "pmc":
                pmc_id = str(id_item)

        if text and len(text.split()) > 30:
            results.append({
                "pmid": pmid,
                "pmc_id": pmc_id,
                "abstract": text,
                "has_pmc": pmc_id is not None,
            })

    return results


def fetch_pmc_fulltext(pmc_id: str) -> str | None:
    """PMC에서 full text를 가져온다."""
    try:
        handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml")
        xml_text = handle.read()
        handle.close()
        if isinstance(xml_text, bytes):
            xml_text = xml_text.decode("utf-8")
        root = ET.fromstring(xml_text)

        parts = []
        for elem in root.findall(".//abstract"):
            parts.append(ET.tostring(elem, encoding="unicode", method="text").strip())
        for elem in root.findall(".//body"):
            parts.append(ET.tostring(elem, encoding="unicode", method="text").strip())
        full_text = " ".join(parts)
        return full_text if len(full_text.split()) > 50 else None
    except Exception:
        return None


def extract_diso_cuis(text: str, nlp, diso_cuis: set[str]) -> list[dict]:
    """scispaCy로 텍스트에서 DISO CUI를 추출한다."""
    doc = nlp(text)
    found = {}
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if cui in diso_cuis and score >= SCISPACY_THRESHOLD:
                if cui not in found or score > found[cui]["score"]:
                    found[cui] = {
                        "cui": cui,
                        "score": score,
                        "matched_text": ent.text,
                    }
    return list(found.values())


def main():
    print("=" * 80)
    print("Step 1 실행: 5개 질환 PubMed/PMC 수집 + scispaCy CUI 식별")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    diso_cuis = load_diso_cuis()
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    print(f"  DISO CUI: {len(diso_cuis):,}")

    # scispaCy 로드
    print("\n[2/4] scispaCy 로드...")
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
    })
    print("  OK")

    # 질환별 수집 + CUI 추출
    print("\n[3/4] 질환별 수집 + CUI 추출...")
    processed_pmids: set[str] = set()
    all_documents: list[dict] = []
    all_cui_pairs: list[dict] = []  # (cui_a, cui_b, pmid, source_type)

    for disease_name, search_term in SEED_DISEASES.items():
        print(f"\n  --- {disease_name} ---")

        # PubMed 초록 수집
        abstracts = fetch_pubmed_abstracts(search_term, n=100)
        print(f"  PubMed 초록: {len(abstracts)}건")
        time.sleep(0.5)

        new_count = 0
        fulltext_count = 0
        cui_count = 0

        for ab in abstracts:
            if ab["pmid"] in processed_pmids:
                continue
            processed_pmids.add(ab["pmid"])
            new_count += 1

            # full text 시도
            text = ab["abstract"]
            source_type = "abstract"
            if ab["has_pmc"]:
                ft = fetch_pmc_fulltext(ab["pmc_id"])
                if ft:
                    text = ft
                    source_type = "fulltext"
                    fulltext_count += 1
                time.sleep(0.3)

            # CUI 추출
            concepts = extract_diso_cuis(text, nlp, diso_cuis)
            cui_list = [c["cui"] for c in concepts]
            cui_count += len(cui_list)

            # CUI 쌍 생성
            pairs = []
            for i in range(len(cui_list)):
                for j in range(i + 1, len(cui_list)):
                    a, b = sorted([cui_list[i], cui_list[j]])
                    pairs.append({
                        "cui_a": a,
                        "cui_b": b,
                        "pmid": ab["pmid"],
                        "source_type": source_type,
                        "seed_disease": disease_name,
                    })
            all_cui_pairs.extend(pairs)

            doc = {
                "pmid": ab["pmid"],
                "pmc_id": ab.get("pmc_id"),
                "source_type": source_type,
                "seed_disease": disease_name,
                "text_words": len(text.split()),
                "n_diso_cuis": len(cui_list),
                "diso_cuis": cui_list,
                "concepts": concepts,
            }
            all_documents.append(doc)

        print(f"  신규 처리: {new_count}건 (중복 제외)")
        print(f"  Full text: {fulltext_count}건")
        print(f"  DISO CUI 추출: {cui_count}개")
        print(f"  누적 PMID: {len(processed_pmids)}건")

    # 통계
    print(f"\n[4/4] 종합 통계")
    print("=" * 80)
    print(f"  총 문서 수: {len(all_documents)}")
    print(f"  총 CUI 쌍 수: {len(all_cui_pairs):,}")

    # 고유 CUI 수
    unique_cuis = set()
    for doc in all_documents:
        unique_cuis.update(doc["diso_cuis"])
    print(f"  고유 DISO CUI 수: {len(unique_cuis)}")

    # 고유 CUI 쌍 수
    unique_pairs = set()
    for pair in all_cui_pairs:
        unique_pairs.add((pair["cui_a"], pair["cui_b"]))
    print(f"  고유 CUI 쌍 수: {len(unique_pairs):,}")

    # source type 분포
    abs_count = sum(1 for d in all_documents if d["source_type"] == "abstract")
    ft_count = sum(1 for d in all_documents if d["source_type"] == "fulltext")
    print(f"  초록만: {abs_count}건, Full text: {ft_count}건")

    # 질환별 분포
    for disease in SEED_DISEASES:
        docs = [d for d in all_documents if d["seed_disease"] == disease]
        cuis = set()
        for d in docs:
            cuis.update(d["diso_cuis"])
        print(f"  {disease}: {len(docs)}건, {len(cuis)} CUI")

    # 상위 CUI
    cui_freq = defaultdict(int)
    for doc in all_documents:
        for cui in doc["diso_cuis"]:
            cui_freq[cui] += 1
    print(f"\n  가장 빈번한 DISO CUI (상위 20):")
    for cui, cnt in sorted(cui_freq.items(), key=lambda x: -x[1])[:20]:
        name = cui_names.get(cui, "?")[:50]
        stys = cui_stys.get(cui, set())
        print(f"    {cui}: {name} ({','.join(sorted(stys))}) x{cnt}")

    # 저장
    output = {
        "config": {
            "seed_diseases": list(SEED_DISEASES.keys()),
            "scispacy_threshold": SCISPACY_THRESHOLD,
            "n_documents": len(all_documents),
            "n_unique_cuis": len(unique_cuis),
            "n_unique_pairs": len(unique_pairs),
            "n_total_pairs": len(all_cui_pairs),
        },
        "documents": all_documents,
    }
    with open(DATA_DIR / "step1_documents.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # CUI 쌍도 별도 저장 (Step 2 입력용)
    with open(DATA_DIR / "step1_cui_pairs.json", "w") as f:
        json.dump(all_cui_pairs, f, indent=2, ensure_ascii=False)

    print(f"\n문서 저장: {DATA_DIR / 'step1_documents.json'}")
    print(f"CUI 쌍 저장: {DATA_DIR / 'step1_cui_pairs.json'}")
    print("완료!")


if __name__ == "__main__":
    main()
