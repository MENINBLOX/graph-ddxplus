#!/usr/bin/env python3
"""Step 2 분기: Ollama 모델 + 프롬프트 변형 비교.

qwen3:4b vs llama3.2:3b × 프롬프트 v1(간단) vs v2(상세)
10건 문서 × 상위 10쌍 = 최대 100개 분류 비교.
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
RESULTS_DIR = Path("pilot/results")
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_V1 = """Given the medical text and concept pairs, classify each pair's relationship.

For each pair, output:
- "present": The text states A and B are medically related
- "absent": The text states A is NOT related to B or absent in context of B
- "not_related": No relationship stated between A and B

Text:
{text}

Pairs:
{pairs}

Respond ONLY with JSON array:
[{{"cui_a": "...", "cui_b": "...", "classification": "present|absent|not_related"}}]"""

PROMPT_V2 = """You are a biomedical relation extractor. Analyze this PubMed abstract and classify concept pair relationships.

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


def call_ollama(model: str, prompt: str, temperature: float = 0) -> tuple[str, float]:
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 4096},
    }, timeout=120)
    elapsed = time.time() - t0
    return resp.json().get("response", ""), elapsed


def parse_json_response(text: str) -> list[dict]:
    # /think 태그 제거 (qwen3)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    return []


def main():
    print("=" * 80)
    print("Step 2 분기: Ollama LLM 비교 (qwen3:4b vs llama3.2:3b × prompt v1/v2)")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    with open(DATA_DIR / "step1_documents.json") as f:
        docs_data = json.load(f)
    with open(DATA_DIR / "step1_cui_pairs.json") as f:
        all_pairs_raw = json.load(f)

    cui_names = load_cui_names()

    # 문서별 CUI 쌍 그룹화
    doc_pairs: dict[str, list] = defaultdict(list)
    for p in all_pairs_raw:
        doc_pairs[p["pmid"]].append(p)

    # 테스트 문서 선정: 질환별 2건 (CUI 10개 이상)
    print("\n[2/4] 테스트 문서 선정...")
    test_docs = []
    for disease in ["Pneumonia", "Pulmonary embolism", "GERD", "Panic attack", "Bronchitis"]:
        candidates = [d for d in docs_data["documents"]
                      if d["seed_disease"] == disease and d["n_diso_cuis"] >= 8]
        test_docs.extend(candidates[:2])
    print(f"  테스트 문서: {len(test_docs)}건")

    # 초록 텍스트 수집
    pmids = [d["pmid"] for d in test_docs]
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml")
    records = Entrez.read(handle)
    handle.close()

    pmid_text = {}
    for article in records["PubmedArticle"]:
        pmid = str(article["MedlineCitation"]["PMID"])
        abs_parts = article["MedlineCitation"]["Article"].get("Abstract", {})
        pmid_text[pmid] = " ".join(str(t) for t in abs_parts.get("AbstractText", []))

    # 테스트 케이스 구성
    test_cases = []
    for doc in test_docs:
        pmid = doc["pmid"]
        text = pmid_text.get(pmid, "")
        if not text:
            continue
        pairs = doc_pairs.get(pmid, [])[:10]
        if not pairs:
            continue
        pair_info = []
        for p in pairs:
            pair_info.append({
                "cui_a": p["cui_a"],
                "cui_b": p["cui_b"],
                "name_a": cui_names.get(p["cui_a"], p["cui_a"])[:40],
                "name_b": cui_names.get(p["cui_b"], p["cui_b"])[:40],
            })
        test_cases.append({
            "pmid": pmid,
            "disease": doc["seed_disease"],
            "text": text,
            "pairs": pair_info,
        })

    total_pairs = sum(len(tc["pairs"]) for tc in test_cases)
    print(f"  총 분류 대상: {total_pairs}개 쌍")

    # 비교 실행
    print(f"\n[3/4] 비교 실행...")
    models = ["qwen3:4b-instruct-2507-fp16", "llama3.2:3b"]
    prompts = {"v1": PROMPT_V1, "v2": PROMPT_V2}

    methods = []
    for model in models:
        for pv, pt in prompts.items():
            short_model = model.split(":")[0]
            methods.append((f"{short_model}_{pv}", model, pt))

    all_results = []
    for tc_idx, tc in enumerate(test_cases):
        pairs_text = "\n".join(
            f"- ({p['name_a']}, {p['name_b']}) [CUI: {p['cui_a']}, {p['cui_b']}]"
            for p in tc["pairs"]
        )
        doc_result = {
            "pmid": tc["pmid"],
            "disease": tc["disease"],
            "n_pairs": len(tc["pairs"]),
        }

        for method_name, model, prompt_template in methods:
            prompt = prompt_template.format(text=tc["text"][:2500], pairs=pairs_text)
            try:
                response, elapsed = call_ollama(model, prompt)
                parsed = parse_json_response(response)
                classifications = {}
                for item in parsed:
                    key = f"{item.get('cui_a', '')}_{item.get('cui_b', '')}"
                    cls = item.get("classification", "error")
                    classifications[key] = cls

                doc_result[method_name] = {
                    "classifications": classifications,
                    "n_parsed": len(parsed),
                    "time_s": elapsed,
                }
            except Exception as e:
                doc_result[method_name] = {"error": str(e), "classifications": {}}

        all_results.append(doc_result)
        print(f"  [{tc_idx+1}/{len(test_cases)}] {tc['disease']:20s} pairs={len(tc['pairs'])}")

    # 분석
    print(f"\n[4/4] 분석")
    print("=" * 80)

    method_names = [m[0] for m in methods]

    # 분류 분포
    for mn in method_names:
        dist = {"present": 0, "absent": 0, "not_related": 0, "other": 0}
        total_time = 0
        total_parsed = 0
        for r in all_results:
            if mn not in r or "error" in r[mn]:
                continue
            total_time += r[mn].get("time_s", 0)
            total_parsed += r[mn].get("n_parsed", 0)
            for cls in r[mn]["classifications"].values():
                cls_lower = cls.lower().strip().replace(" ", "_")
                if cls_lower in dist:
                    dist[cls_lower] += 1
                else:
                    dist["other"] += 1
        total_cls = sum(dist.values())
        print(f"\n  {mn}:")
        for cls, cnt in dist.items():
            if cnt > 0:
                print(f"    {cls}: {cnt} ({cnt/max(total_cls,1):.1%})")
        print(f"    파싱 성공: {total_parsed}/{total_pairs}")
        print(f"    총 시간: {total_time:.1f}s ({total_time/max(len(all_results),1):.1f}s/문서)")

    # 방법 간 일치율
    print(f"\n  방법 간 일치율:")
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i+1:]:
            agree = 0
            total = 0
            for r in all_results:
                if m1 not in r or m2 not in r:
                    continue
                c1 = r[m1].get("classifications", {})
                c2 = r[m2].get("classifications", {})
                for key in c1:
                    if key in c2:
                        total += 1
                        v1 = c1[key].lower().strip().replace(" ", "_")
                        v2 = c2[key].lower().strip().replace(" ", "_")
                        if v1 == v2:
                            agree += 1
            rate = agree / total if total > 0 else 0
            print(f"    {m1} vs {m2}: {rate:.1%} ({agree}/{total})")

    # 불일치 상세 (qwen3_v1 vs qwen3_v2)
    print(f"\n  qwen3_v1 vs qwen3_v2 불일치 사례:")
    for r in all_results:
        c1 = r.get("qwen3_v1", {}).get("classifications", {})
        c2 = r.get("qwen3_v2", {}).get("classifications", {})
        for key in c1:
            if key in c2:
                v1 = c1[key].lower().strip().replace(" ", "_")
                v2 = c2[key].lower().strip().replace(" ", "_")
                if v1 != v2:
                    cuis = key.split("_")
                    if len(cuis) >= 2:
                        a_name = cui_names.get(cuis[0], cuis[0])[:30]
                        b_name = cui_names.get(cuis[1], cuis[1])[:30]
                        print(f"    {a_name} - {b_name}: v1={v1}, v2={v2}")

    # 저장
    with open(RESULTS_DIR / "step2_ollama_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {RESULTS_DIR / 'step2_ollama_comparison.json'}")


if __name__ == "__main__":
    main()
