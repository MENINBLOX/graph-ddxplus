#!/usr/bin/env python3
"""Step 2 실행: gemma4:e4b-it-bf16 + prompt v2로 전체 문서 배치 분류.

471건 문서 × 문서별 CUI 쌍 → ternary classification
결과를 step2_classifications.json에 저장.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from Bio import Entrez

Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"
MAX_PAIRS_PER_DOC = 15  # 문서당 최대 CUI 쌍

DISO_TYPES = {
    "T047", "T184", "T033", "T034", "T191", "T046",
    "T048", "T037", "T019", "T020", "T190", "T049",
}

PROMPT = """You are a biomedical relation extractor. Analyze this PubMed abstract and classify concept pair relationships.

RULES:
1. "present" = text EXPLICITLY states a positive relationship (symptom of, causes, associated with, co-occurs)
2. "absent" = text EXPLICITLY states a NEGATIVE relationship (not seen, absence of, rules out, excluded)
3. "not_related" = both concepts appear but NO explicit relationship between them
4. Do NOT infer relationships not explicitly stated

Text:
{text}

Pairs:
{pairs}

Respond ONLY with JSON array:
[{{"cui_a": "...", "cui_b": "...", "classification": "present|absent|not_related"}}]"""


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, lang, ts, name = p[0], p[1], p[2], p[14]
            if lang == "ENG" and (cui not in names or ts == "P"):
                names[cui] = name
    return names


def call_ollama(prompt: str) -> tuple[str, float]:
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8192},
    }, timeout=180)
    elapsed = time.time() - t0
    return resp.json().get("response", ""), elapsed


def parse_json_response(text: str) -> list[dict]:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def main():
    print("=" * 80)
    print(f"Step 2 실행: {MODEL} 전체 문서 분류")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/3] 데이터 로드...")
    with open(DATA_DIR / "step1_documents.json") as f:
        docs_data = json.load(f)
    with open(DATA_DIR / "step1_cui_pairs.json") as f:
        all_pairs_raw = json.load(f)

    cui_names = load_cui_names()

    # 노이즈 CUI 블랙리스트
    noise_names = ["Symptom", "Other Symptom", "No information available", "Reduced",
        "Test Result", "Increased", "Disease, NOS", "Present", "Well",
        "Expression Negative", "Increased (finding)", "Decreased (finding)",
        "Performance Status", "Normal", "Negative"]
    noise_cuis = set()
    all_cuis = set()
    for p in all_pairs_raw:
        all_cuis.add(p["cui_a"])
        all_cuis.add(p["cui_b"])
    for cui in all_cuis:
        name = cui_names.get(cui, "")
        if any(n in name for n in noise_names):
            noise_cuis.add(cui)

    # 문서별 CUI 쌍 그룹화 + 필터링
    doc_pairs: dict[str, list] = defaultdict(list)
    for p in all_pairs_raw:
        if p["cui_a"] in noise_cuis or p["cui_b"] in noise_cuis:
            continue
        doc_pairs[p["pmid"]].append(p)

    # 초록 텍스트 배치 수집
    print("\n[2/3] 초록 텍스트 수집...")
    documents = docs_data["documents"]
    pmids = [d["pmid"] for d in documents]

    # 배치로 수집 (100개씩)
    pmid_text = {}
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        try:
            handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml")
            records = Entrez.read(handle)
            handle.close()
            for article in records["PubmedArticle"]:
                pmid = str(article["MedlineCitation"]["PMID"])
                abs_parts = article["MedlineCitation"]["Article"].get("Abstract", {})
                pmid_text[pmid] = " ".join(str(t) for t in abs_parts.get("AbstractText", []))
            time.sleep(0.3)
        except Exception as e:
            print(f"  배치 {i} 오류: {e}")
    print(f"  초록 수집: {len(pmid_text)}건")

    # LLM 분류 실행
    print(f"\n[3/3] LLM 분류 ({len(documents)}건)...")
    all_classifications = []
    total_pairs = 0
    total_classified = 0
    errors = 0
    start_time = time.time()

    # 체크포인트 파일
    checkpoint_file = DATA_DIR / "step2_checkpoint.json"
    processed_pmids = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            all_classifications = checkpoint.get("classifications", [])
            processed_pmids = set(c["pmid"] for c in all_classifications)
            print(f"  체크포인트 로드: {len(processed_pmids)}건 이미 처리됨")

    # 파일럿: 질환당 20건 = 100건만 처리
    MAX_PER_DISEASE = 20
    disease_count: dict[str, int] = defaultdict(int)

    for idx, doc in enumerate(documents):
        pmid = doc["pmid"]
        if pmid in processed_pmids:
            continue

        seed = doc.get("seed_disease", "")
        if disease_count[seed] >= MAX_PER_DISEASE:
            continue
        disease_count[seed] += 1

        text = pmid_text.get(pmid, "")
        pairs = doc_pairs.get(pmid, [])

        if not text or not pairs:
            continue

        # 쌍 수 제한
        pairs = pairs[:MAX_PAIRS_PER_DOC]
        total_pairs += len(pairs)

        pairs_text = "\n".join(
            f"- ({cui_names.get(p['cui_a'], p['cui_a'])[:40]}, "
            f"{cui_names.get(p['cui_b'], p['cui_b'])[:40]}) "
            f"[CUI: {p['cui_a']}, {p['cui_b']}]"
            for p in pairs
        )

        prompt = PROMPT.format(text=text[:2500], pairs=pairs_text)

        try:
            response, elapsed = call_ollama(prompt)
            parsed = parse_json_response(response)
            total_classified += len(parsed)

            for item in parsed:
                cls = item.get("classification", "").lower().strip().replace(" ", "_")
                if cls in ("present", "absent", "not_related"):
                    all_classifications.append({
                        "pmid": pmid,
                        "cui_a": item.get("cui_a", ""),
                        "cui_b": item.get("cui_b", ""),
                        "classification": cls,
                        "seed_disease": doc["seed_disease"],
                    })
        except Exception as e:
            errors += 1

        actual_processed = sum(disease_count.values())
        if actual_processed % 5 == 0 or actual_processed == 1:
            elapsed_total = time.time() - start_time
            rate = actual_processed / elapsed_total if elapsed_total > 0 else 0
            total_target = MAX_PER_DISEASE * 5
            remaining = (total_target - actual_processed) / rate if rate > 0 else 0
            print(f"  [{actual_processed:3d}/100] "
                  f"분류={total_classified:,} 오류={errors} "
                  f"속도={rate:.2f}건/s 잔여={remaining/60:.0f}분")

            # 체크포인트 저장
            with open(checkpoint_file, "w") as f:
                json.dump({"classifications": all_classifications}, f)

    # 최종 저장
    elapsed_total = time.time() - start_time
    print(f"\n완료: {elapsed_total/60:.1f}분")

    # 분류 분포
    dist = defaultdict(int)
    for c in all_classifications:
        dist[c["classification"]] += 1
    print(f"  present: {dist['present']:,}")
    print(f"  absent: {dist['absent']:,}")
    print(f"  not_related: {dist['not_related']:,}")
    print(f"  총: {len(all_classifications):,}")

    with open(DATA_DIR / "step2_classifications.json", "w") as f:
        json.dump(all_classifications, f, indent=2, ensure_ascii=False)
    print(f"저장: {DATA_DIR / 'step2_classifications.json'}")


if __name__ == "__main__":
    main()
