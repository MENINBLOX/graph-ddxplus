#!/usr/bin/env python3
"""49개 질환 PubMed 데이터 수집 + scispaCy CUI 추출.

DDXPlus 49개 질환의 영문명으로 PubMed 검색,
질환당 50편 초록 수집, scispaCy로 DISO CUI 추출.
"""
from __future__ import annotations

import json
import os
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
DATA_DIR = Path("pilot/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST_CUIS = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}
SCISPACY_THRESHOLD = 0.85
ABSTRACTS_PER_DISEASE = 50


def load_allowed_cuis() -> set[str]:
    """MRSTY에서 허용 CUI 목록을 로드."""
    allowed = set()
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] in ALLOWED_STYS:
                allowed.add(p[0])
    return allowed - BLACKLIST_CUIS


def load_ddxplus_diseases() -> dict[str, str]:
    """DDXPlus 49개 질환명과 검색어를 로드."""
    with open("data/ddxplus/disease_umls_mapping.json") as f:
        mapping = json.load(f)["mapping"]

    diseases = {}
    for name, info in mapping.items():
        cui = info.get("umls_cui")
        if cui:
            diseases[name] = cui
    return diseases


def fetch_abstracts(query: str, n: int = 50) -> list[dict]:
    """PubMed에서 초록 수집."""
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=f'({query}) AND (Journal Article[pt] OR Clinical Trial[pt] OR Case Reports[pt] OR Review[pt])',
            retmax=n,
            sort="relevance",
        )
        pmids = Entrez.read(handle)["IdList"]
        handle.close()
        time.sleep(0.15)

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
            if text and len(text.split()) > 30:
                results.append({"pmid": pmid, "text": text})
        return results
    except Exception as e:
        print(f"    검색 오류: {e}")
        return []


def extract_cuis(text: str, nlp, allowed_cuis: set[str]) -> list[dict]:
    """scispaCy로 CUI 추출 (DISO 필터 + 블랙리스트 적용)."""
    doc = nlp(text)
    found = {}
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if cui in allowed_cuis and score >= SCISPACY_THRESHOLD:
                if cui not in found or score > found[cui]["score"]:
                    found[cui] = {"cui": cui, "score": score, "text": ent.text}
    return list(found.values())


def main():
    print("=" * 80)
    print("49개 질환 PubMed 데이터 수집 + scispaCy CUI 추출")
    print("=" * 80)

    # 데이터 준비
    print("\n[1/4] 데이터 로드...")
    allowed_cuis = load_allowed_cuis()
    diseases = load_ddxplus_diseases()
    print(f"  허용 CUI: {len(allowed_cuis):,}")
    print(f"  DDXPlus 질환: {len(diseases)}")

    # scispaCy 로드
    print("\n[2/4] scispaCy 로드...")
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
    })
    print("  OK")

    # 체크포인트
    checkpoint_file = DATA_DIR / "exp_collect_checkpoint.json"
    all_documents = []
    processed_pmids = set()
    processed_diseases = set()

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
            all_documents = ckpt.get("documents", [])
            processed_pmids = set(d["pmid"] for d in all_documents)
            processed_diseases = set(d["seed_disease"] for d in all_documents)
            print(f"  체크포인트: {len(processed_diseases)} 질환, {len(processed_pmids)} 문서 처리됨")

    # 수집
    print(f"\n[3/4] 질환별 PubMed 수집 ({len(diseases) - len(processed_diseases)}개 남음)...")
    start_time = time.time()

    for idx, (disease_name, disease_cui) in enumerate(sorted(diseases.items())):
        if disease_name in processed_diseases:
            continue

        abstracts = fetch_abstracts(disease_name, n=ABSTRACTS_PER_DISEASE)
        time.sleep(0.3)

        new_count = 0
        for ab in abstracts:
            if ab["pmid"] in processed_pmids:
                continue
            processed_pmids.add(ab["pmid"])

            # CUI 추출
            concepts = extract_cuis(ab["text"], nlp, allowed_cuis)
            cui_list = [c["cui"] for c in concepts]

            all_documents.append({
                "pmid": ab["pmid"],
                "seed_disease": disease_name,
                "seed_cui": disease_cui,
                "text_words": len(ab["text"].split()),
                "n_cuis": len(cui_list),
                "cuis": cui_list,
                "text": ab["text"],  # 평가용 텍스트 보관
            })
            new_count += 1

        elapsed = time.time() - start_time
        done = idx + 1 - len(processed_diseases & set(list(diseases.keys())[:idx+1]))
        total_remaining = len(diseases) - len(processed_diseases) - done
        rate = done / elapsed if elapsed > 0 else 0
        eta = total_remaining / rate if rate > 0 else 0

        print(f"  [{idx+1:2d}/49] {disease_name:40s} 수집={new_count:3d} 누적={len(all_documents):5d} ETA={eta/60:.0f}분")

        # 체크포인트 저장 (10 질환마다)
        if (idx + 1) % 10 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({"documents": all_documents}, f)

    # 최종 저장
    print(f"\n[4/4] 저장...")

    # CUI 쌍 생성
    all_pairs = []
    for doc in all_documents:
        cuis = doc["cuis"]
        for i in range(len(cuis)):
            for j in range(i + 1, len(cuis)):
                a, b = sorted([cuis[i], cuis[j]])
                all_pairs.append({
                    "cui_a": a, "cui_b": b,
                    "pmid": doc["pmid"],
                    "seed_disease": doc["seed_disease"],
                })

    # 통계
    unique_cuis = set()
    for doc in all_documents:
        unique_cuis.update(doc["cuis"])
    unique_pairs = set((p["cui_a"], p["cui_b"]) for p in all_pairs)

    print(f"  총 문서: {len(all_documents)}")
    print(f"  고유 CUI: {len(unique_cuis)}")
    print(f"  고유 CUI 쌍: {len(unique_pairs):,}")
    print(f"  총 CUI 쌍 (중복 포함): {len(all_pairs):,}")

    # 질환별 통계
    disease_stats = defaultdict(lambda: {"docs": 0, "cuis": set()})
    for doc in all_documents:
        disease_stats[doc["seed_disease"]]["docs"] += 1
        disease_stats[doc["seed_disease"]]["cuis"].update(doc["cuis"])

    # 저장
    output = {
        "config": {
            "n_diseases": len(diseases),
            "abstracts_per_disease": ABSTRACTS_PER_DISEASE,
            "scispacy_threshold": SCISPACY_THRESHOLD,
            "allowed_stys": sorted(ALLOWED_STYS),
            "blacklist_cuis": sorted(BLACKLIST_CUIS),
        },
        "stats": {
            "n_documents": len(all_documents),
            "n_unique_cuis": len(unique_cuis),
            "n_unique_pairs": len(unique_pairs),
            "n_total_pairs": len(all_pairs),
        },
        "documents": all_documents,
    }

    with open(DATA_DIR / "exp_documents.json", "w") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"  문서 저장: {DATA_DIR / 'exp_documents.json'}")

    with open(DATA_DIR / "exp_pairs.json", "w") as f:
        json.dump(all_pairs, f)
    print(f"  CUI 쌍 저장: {DATA_DIR / 'exp_pairs.json'}")

    elapsed = time.time() - start_time
    print(f"\n완료! ({elapsed/60:.1f}분)")


if __name__ == "__main__":
    main()
