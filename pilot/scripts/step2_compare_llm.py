#!/usr/bin/env python3
"""Step 2 분기: LLM ternary classification 비교.

동일 CUI 쌍 + 문서에 대해 비교:
(A) Claude Haiku - 프롬프트 v1 (간단)
(B) Claude Haiku - 프롬프트 v2 (상세 지시)
(C) GPT-4o-mini - 프롬프트 v1

10건 문서 × 상위 10 쌍 = 최대 100개 분류를 비교.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import anthropic
import openai

DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UMLS_DIR = Path("data/umls_extracted")


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, lang, ts, name = p[0], p[1], p[2], p[14]
            if lang == "ENG" and (cui not in names or ts == "P"):
                names[cui] = name
    return names


# ============================================================
# 프롬프트 변형
# ============================================================

PROMPT_V1 = """You are a medical knowledge extractor. Given a medical text and a list of concept pairs, classify each pair's relationship.

For each pair (Concept A, Concept B), output one of:
- "present": The text states that A and B are medically related (e.g., A is a symptom of B, A causes B, A is associated with B)
- "absent": The text states that A and B are NOT related, or that A is absent in the context of B (e.g., "no fever was observed", "rules out pneumonia")
- "not_related": The text does not state any relationship between A and B

Text:
{text}

Concept pairs to classify:
{pairs}

Respond in JSON format:
[{{"cui_a": "...", "cui_b": "...", "classification": "present|absent|not_related"}}]"""

PROMPT_V2 = """You are a biomedical relation extractor analyzing PubMed literature. Your task is to determine whether a medical text explicitly states a relationship between pairs of medical concepts.

IMPORTANT RULES:
1. Only classify as "present" if the text EXPLICITLY states a positive relationship (symptom of, associated with, causes, complicates, co-occurs with, etc.)
2. Only classify as "absent" if the text EXPLICITLY states a NEGATIVE relationship (not seen in, absence of, rules out, excluded, not associated with)
3. Classify as "not_related" if BOTH concepts appear in the text but NO explicit relationship is stated between them
4. Do NOT infer relationships that are not explicitly stated in the text

Text:
{text}

Concept pairs:
{pairs}

Respond ONLY with a JSON array:
[{{"cui_a": "...", "cui_b": "...", "classification": "present|absent|not_related"}}]"""


def call_claude(prompt: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_gpt(prompt: str, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_llm_response(response_text: str) -> list[dict]:
    """LLM 응답에서 JSON을 추출한다."""
    import re
    # JSON 배열 추출
    match = re.search(r'\[[\s\S]*\]', response_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # JSON object 안에 배열이 있는 경우
    try:
        obj = json.loads(response_text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass

    return []


def main():
    print("=" * 80)
    print("Step 2 분기: LLM ternary classification 비교")
    print("=" * 80)

    # API 키
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not anthropic_key:
        print("ANTHROPIC_API_KEY 없음")
        return
    if not openai_key:
        print("OPENAI_API_KEY 없음")
        return

    # 데이터 로드
    print("\n[1/4] 데이터 로드...")
    with open(DATA_DIR / "step1_documents.json") as f:
        docs_data = json.load(f)
    with open(DATA_DIR / "step1_cui_pairs.json") as f:
        all_pairs = json.load(f)

    cui_names = load_cui_names()

    # 문서별 CUI 쌍 그룹화
    doc_pairs: dict[str, list] = {}
    for pair in all_pairs:
        pmid = pair["pmid"]
        if pmid not in doc_pairs:
            doc_pairs[pmid] = []
        doc_pairs[pmid].append(pair)

    # 테스트 문서 10건 선택 (다양한 질환에서)
    test_docs = []
    for disease in ["Pneumonia", "Pulmonary embolism", "GERD", "Panic attack", "Bronchitis"]:
        disease_docs = [d for d in docs_data["documents"] if d["seed_disease"] == disease and d["n_diso_cuis"] >= 10]
        if disease_docs:
            test_docs.extend(disease_docs[:2])

    print(f"  테스트 문서: {len(test_docs)}건")

    # 각 문서에서 상위 10 CUI 쌍 선택
    print("\n[2/4] 테스트 CUI 쌍 준비...")
    test_cases = []
    for doc in test_docs:
        pmid = doc["pmid"]
        pairs = doc_pairs.get(pmid, [])
        # 빈도 높은 CUI 간 쌍 우선
        selected_pairs = pairs[:10]
        if selected_pairs:
            # 문서 텍스트 가져오기 (초록 사용)
            pair_info = []
            for p in selected_pairs:
                a_name = cui_names.get(p["cui_a"], p["cui_a"])[:40]
                b_name = cui_names.get(p["cui_b"], p["cui_b"])[:40]
                pair_info.append({
                    "cui_a": p["cui_a"],
                    "cui_b": p["cui_b"],
                    "name_a": a_name,
                    "name_b": b_name,
                })

            # 문서의 텍스트를 PubMed에서 다시 가져오지 않고 CUI 목록에서 추론
            # 실제로는 documents.json에 텍스트가 없으므로 다시 가져와야 함
            test_cases.append({
                "pmid": pmid,
                "seed_disease": doc["seed_disease"],
                "pairs": pair_info,
            })

    # 초록 텍스트 다시 가져오기
    print("  초록 텍스트 수집...")
    from Bio import Entrez
    Entrez.email = "max@meninblox.com"
    Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

    pmids = [tc["pmid"] for tc in test_cases]
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml")
    records = Entrez.read(handle)
    handle.close()

    pmid_to_text = {}
    for article in records["PubmedArticle"]:
        pmid = str(article["MedlineCitation"]["PMID"])
        abs_parts = article["MedlineCitation"]["Article"].get("Abstract", {})
        text = " ".join(str(t) for t in abs_parts.get("AbstractText", []))
        pmid_to_text[pmid] = text

    for tc in test_cases:
        tc["text"] = pmid_to_text.get(tc["pmid"], "")

    total_pairs = sum(len(tc["pairs"]) for tc in test_cases)
    print(f"  총 분류 대상: {total_pairs}개 쌍")

    # LLM 비교 실행
    print(f"\n[3/4] LLM 비교 실행 (3가지 방법 × {len(test_cases)} 문서)...")
    all_results = []

    methods = [
        ("Claude_v1", PROMPT_V1, "claude"),
        ("Claude_v2", PROMPT_V2, "claude"),
        ("GPT_v1", PROMPT_V1, "gpt"),
    ]

    for tc_idx, tc in enumerate(test_cases):
        if not tc["text"]:
            continue

        pairs_text = "\n".join(
            f"- ({p['name_a']}, {p['name_b']}) [CUI: {p['cui_a']}, {p['cui_b']}]"
            for p in tc["pairs"]
        )

        doc_results = {
            "pmid": tc["pmid"],
            "disease": tc["seed_disease"],
            "n_pairs": len(tc["pairs"]),
            "text_preview": tc["text"][:200],
        }

        for method_name, prompt_template, provider in methods:
            prompt = prompt_template.format(text=tc["text"][:3000], pairs=pairs_text)

            t0 = time.time()
            try:
                if provider == "claude":
                    response = call_claude(prompt, anthropic_key)
                else:
                    response = call_gpt(prompt, openai_key)
                elapsed = time.time() - t0

                parsed = parse_llm_response(response)
                classifications = {}
                for item in parsed:
                    key = f"{item.get('cui_a', '')}_{item.get('cui_b', '')}"
                    classifications[key] = item.get("classification", "error")

                doc_results[method_name] = {
                    "classifications": classifications,
                    "n_parsed": len(parsed),
                    "time_s": elapsed,
                    "raw_preview": response[:200],
                }
            except Exception as e:
                doc_results[method_name] = {"error": str(e)}

            time.sleep(0.5)

        all_results.append(doc_results)
        print(f"  [{tc_idx+1}/{len(test_cases)}] {tc['disease']:20s} pairs={tc['n_pairs']}")

    # 분석
    print(f"\n[4/4] 분석")
    print("=" * 80)

    # 방법 간 일치율
    for m1, m2 in [("Claude_v1", "Claude_v2"), ("Claude_v1", "GPT_v1"), ("Claude_v2", "GPT_v1")]:
        agree = 0
        total = 0
        for r in all_results:
            if m1 not in r or m2 not in r:
                continue
            if "error" in r[m1] or "error" in r[m2]:
                continue
            c1 = r[m1]["classifications"]
            c2 = r[m2]["classifications"]
            for key in c1:
                if key in c2:
                    total += 1
                    if c1[key] == c2[key]:
                        agree += 1
        rate = agree / total if total > 0 else 0
        print(f"  {m1} vs {m2}: 일치율 {rate:.1%} ({agree}/{total})")

    # 분류 분포
    for method_name in ["Claude_v1", "Claude_v2", "GPT_v1"]:
        dist = {"present": 0, "absent": 0, "not_related": 0, "other": 0}
        total_time = 0
        for r in all_results:
            if method_name not in r or "error" in r[method_name]:
                continue
            total_time += r[method_name].get("time_s", 0)
            for cls in r[method_name]["classifications"].values():
                if cls in dist:
                    dist[cls] += 1
                else:
                    dist["other"] += 1
        total_cls = sum(dist.values())
        print(f"\n  {method_name} 분류 분포:")
        for cls, cnt in dist.items():
            if cnt > 0:
                print(f"    {cls}: {cnt} ({cnt/max(total_cls,1):.1%})")
        print(f"    총 시간: {total_time:.1f}s")

    # 불일치 사례 상세
    print(f"\n  Claude_v1 vs Claude_v2 불일치 사례 (상위 10):")
    disagreements = []
    for r in all_results:
        if "Claude_v1" not in r or "Claude_v2" not in r:
            continue
        if "error" in r["Claude_v1"] or "error" in r["Claude_v2"]:
            continue
        c1 = r["Claude_v1"]["classifications"]
        c2 = r["Claude_v2"]["classifications"]
        for key in c1:
            if key in c2 and c1[key] != c2[key]:
                cuis = key.split("_")
                if len(cuis) == 2:
                    disagreements.append({
                        "pmid": r["pmid"],
                        "cui_a": cuis[0],
                        "cui_b": cuis[1],
                        "v1": c1[key],
                        "v2": c2[key],
                    })
    for d in disagreements[:10]:
        a_name = cui_names.get(d["cui_a"], d["cui_a"])[:30]
        b_name = cui_names.get(d["cui_b"], d["cui_b"])[:30]
        print(f"    {a_name} - {b_name}: v1={d['v1']}, v2={d['v2']}")

    # 저장
    output_file = RESULTS_DIR / "step2_llm_comparison.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
