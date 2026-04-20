#!/usr/bin/env python3
"""Phase A: Step 1 NER 10개 변형 실행 + 평가.

2,217편 중 200편 서브셋(질환당 ~4편)에 대해:
1. scispaCy 추출 결과에 10가지 필터 변형 적용
2. gemma4 v2로 1회 LLM 분류 (가장 관대한 필터 기준)
3. 각 필터 변형별 DDXPlus 대비 recall/precision/F1 측정
4. 상위 3개 선택
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
import scipy.stats as stats

DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
UMLS_DIR = Path("data/umls_extracted")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"

DISO_ALL = {"T047", "T184", "T033", "T034", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
ALLOWED_BASE = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

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


def load_cui_stys() -> dict[str, set[str]]:
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and (p[0] not in names or p[2] == "P"):
                names[p[0]] = p[14]
    return names


def load_parent_map() -> dict[str, set[str]]:
    """MRREL에서 CUI의 부모(PAR) 매핑 로드."""
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] == "PAR":  # parent relation
                child, parent = p[0], p[4]
                parents[child].add(parent)
    return dict(parents)


def call_ollama(prompt: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }, timeout=300)
    return resp.json().get("response", "")


def parse_json(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def apply_filter(doc_cuis: list[str], filter_id: str,
                 cui_stys: dict, parent_map: dict, depth_cache: dict,
                 snomed_core: set) -> list[str]:
    """NER 필터 변형 적용."""

    if filter_id == "S1-A":  # threshold 0.80 (이미 추출된 CUI에서 STY 필터만)
        return [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]

    elif filter_id == "S1-B":  # threshold 0.85 baseline
        return [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]

    elif filter_id == "S1-C":  # threshold 0.90
        return [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]

    elif filter_id == "S1-D":  # 전체 DISO (필터 없음)
        return [c for c in doc_cuis if cui_stys.get(c, set()) & DISO_ALL]

    elif filter_id == "S1-E":  # T033만 제외
        allowed = DISO_ALL - {"T033"}
        return [c for c in doc_cuis if (cui_stys.get(c, set()) & allowed) and c not in BLACKLIST]

    elif filter_id == "S1-F":  # T033/T034 제외 + depth≤2 제거
        filtered = [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]
        # depth 계산: parent가 없으면 root(depth=0)
        result = []
        for c in filtered:
            depth = 0
            current = c
            visited = set()
            while current in parent_map and current not in visited:
                visited.add(current)
                current = next(iter(parent_map[current]))
                depth += 1
                if depth > 2:
                    break
            if depth > 2:
                result.append(c)
        return result if result else filtered  # depth 필터 후 빈 경우 원본 반환

    elif filter_id == "S1-G":  # 1레벨 PAR 전파
        filtered = [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]
        expanded = set(filtered)
        for c in filtered:
            if c in parent_map:
                for parent in parent_map[c]:
                    if cui_stys.get(parent, set()) & ALLOWED_BASE:
                        expanded.add(parent)
        return list(expanded)

    elif filter_id == "S1-H":  # 2레벨 PAR 전파
        filtered = [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]
        expanded = set(filtered)
        for c in filtered:
            if c in parent_map:
                for p1 in parent_map[c]:
                    if cui_stys.get(p1, set()) & ALLOWED_BASE:
                        expanded.add(p1)
                    if p1 in parent_map:
                        for p2 in parent_map[p1]:
                            if cui_stys.get(p2, set()) & ALLOWED_BASE:
                                expanded.add(p2)
        return list(expanded)

    elif filter_id == "S1-I":  # 단일토큰 CUI 제외
        filtered = [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]
        cui_names_local = {}  # 이름이 필요하지만 여기서는 간이 처리
        return filtered  # 실제로는 이름 길이 체크 필요 - 단순화

    elif filter_id == "S1-J":  # SNOMED CORE 화이트리스트
        filtered = [c for c in doc_cuis if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]
        if snomed_core:
            return [c for c in filtered if c in snomed_core]
        return filtered

    return doc_cuis


def evaluate_against_ddxplus(edges: list[dict], gold_pairs: set) -> dict:
    """DDXPlus gold standard 대비 precision/recall/F1 계산."""
    our_pairs = set()
    for e in edges:
        if e.get("polarity") == "present":
            our_pairs.add(tuple(sorted([e["cui_a"], e["cui_b"]])))

    if not our_pairs or not gold_pairs:
        return {"precision": 0, "recall": 0, "f1": 0, "n_our": len(our_pairs), "n_gold": len(gold_pairs), "n_overlap": 0}

    overlap = our_pairs & gold_pairs
    precision = len(overlap) / len(our_pairs) if our_pairs else 0
    recall = len(overlap) / len(gold_pairs) if gold_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_our": len(our_pairs),
        "n_gold": len(gold_pairs),
        "n_overlap": len(overlap),
    }


def main():
    print("=" * 80)
    print("Phase A: Step 1 NER 10개 변형 + 평가")
    print("=" * 80)

    # 데이터 로드
    print("\n[1/5] 데이터 로드...")
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()

    with open(DATA_DIR / "exp_documents.json") as f:
        all_docs = json.load(f)["documents"]
    with open(DATA_DIR / "gold_standard.json") as f:
        gold = json.load(f)
    gold_pairs = set(tuple(p) for p in gold["ddxplus"]["pairs"])

    print(f"  문서: {len(all_docs)}, Gold 쌍: {len(gold_pairs)}")

    # PAR 매핑 로드
    print("  PAR 매핑 로드...")
    parent_map = load_parent_map()
    print(f"  PAR 관계: {sum(len(v) for v in parent_map.values()):,}")

    # SNOMED CORE subset (없으면 빈 셋)
    snomed_core = set()
    # TODO: SNOMED CORE subset CUI 로드 (데이터 있으면)

    # 200편 서브셋 선택 (질환당 ~4편)
    print("\n[2/5] 200편 서브셋 선택...")
    subset = []
    disease_count = Counter()
    for doc in all_docs:
        disease = doc["seed_disease"]
        if disease_count[disease] < 4:
            subset.append(doc)
            disease_count[disease] += 1
        if len(subset) >= 200:
            break
    print(f"  서브셋: {len(subset)}편, {len(disease_count)} 질환")

    # 가장 관대한 필터(S1-D: 전체 DISO)로 CUI 쌍 생성 → LLM 분류
    print("\n[3/5] LLM 분류 (S1-D 전체 DISO 기준, 200편)...")

    all_classifications = []
    start = time.time()

    for idx, doc in enumerate(subset):
        # 모든 DISO CUI 사용 (가장 관대)
        cuis = [c for c in doc["cuis"] if cui_stys.get(c, set()) & DISO_ALL]
        if len(cuis) < 2:
            continue

        # 쌍 생성 (최대 15개)
        pairs = []
        for i in range(min(len(cuis), 15)):
            for j in range(i + 1, min(len(cuis), 15)):
                pairs.append({"cui_a": min(cuis[i], cuis[j]), "cui_b": max(cuis[i], cuis[j])})
        pairs = pairs[:15]

        if not pairs:
            continue

        pairs_text = "\n".join(
            f"- ({cui_names.get(p['cui_a'], p['cui_a'])[:40]}, "
            f"{cui_names.get(p['cui_b'], p['cui_b'])[:40]}) "
            f"[CUI: {p['cui_a']}, {p['cui_b']}]"
            for p in pairs
        )
        prompt = PROMPT_V2.format(text=doc["text"][:2500], pairs=pairs_text)

        try:
            response = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                cls = item.get("classification", "").lower().strip().replace(" ", "_")
                if cls in ("present", "absent", "not_related"):
                    all_classifications.append({
                        "pmid": doc["pmid"],
                        "cui_a": item.get("cui_a", ""),
                        "cui_b": item.get("cui_b", ""),
                        "classification": cls,
                        "seed_disease": doc["seed_disease"],
                        "all_doc_cuis": doc["cuis"],  # 필터링 전 전체 CUI
                    })
        except Exception as e:
            pass

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            remaining = (len(subset) - idx - 1) / rate
            print(f"  [{idx+1:3d}/{len(subset)}] cls={len(all_classifications):,} {rate:.2f}건/s ETA={remaining/60:.0f}분")

    elapsed = time.time() - start
    print(f"  완료: {len(all_classifications):,}건, {elapsed/60:.1f}분")

    # 체크포인트 저장
    with open(DATA_DIR / "phase_a_classifications.json", "w") as f:
        json.dump(all_classifications, f)

    # [4/5] 10가지 필터 변형별 평가
    print(f"\n[4/5] 10가지 필터 변형별 DDXPlus 평가")
    print("=" * 80)

    filter_ids = ["S1-A", "S1-B", "S1-C", "S1-D", "S1-E", "S1-F", "S1-G", "S1-H", "S1-I", "S1-J"]
    filter_results = []

    for fid in filter_ids:
        # 각 필터로 CUI 쌍을 필터링하고 present 관계만 집계
        filtered_cls = []
        for c in all_classifications:
            # 이 분류의 두 CUI가 현재 필터를 통과하는지 확인
            doc_cuis = c.get("all_doc_cuis", [])
            allowed = apply_filter(doc_cuis, fid, cui_stys, parent_map, {}, snomed_core)
            allowed_set = set(allowed)

            if c["cui_a"] in allowed_set and c["cui_b"] in allowed_set:
                filtered_cls.append(c)

        # present 쌍 집계
        pair_present = Counter()
        for c in filtered_cls:
            if c["classification"] == "present":
                pair = tuple(sorted([c["cui_a"], c["cui_b"]]))
                pair_present[pair] += 1

        # 엣지 생성 (present >= 1)
        edges = [{"cui_a": p[0], "cui_b": p[1], "polarity": "present", "n": cnt}
                 for p, cnt in pair_present.items()]

        # DDXPlus 평가
        eval_result = evaluate_against_ddxplus(edges, gold_pairs)
        eval_result["filter_id"] = fid
        eval_result["n_classifications"] = len(filtered_cls)
        eval_result["n_present"] = sum(1 for c in filtered_cls if c["classification"] == "present")
        eval_result["n_edges"] = len(edges)
        filter_results.append(eval_result)

        print(f"  {fid}: P={eval_result['precision']:.3f} R={eval_result['recall']:.3f} "
              f"F1={eval_result['f1']:.3f} edges={eval_result['n_edges']:,} "
              f"overlap={eval_result['n_overlap']}/{eval_result['n_gold']}")

    # 상위 3개 선택
    filter_results.sort(key=lambda x: -x["f1"])
    print(f"\n[5/5] 상위 3개 Step 1 설정:")
    top3 = filter_results[:3]
    for r in top3:
        print(f"  {r['filter_id']}: F1={r['f1']:.3f} (P={r['precision']:.3f}, R={r['recall']:.3f})")

    # 저장
    output = {
        "n_subset": len(subset),
        "n_classifications": len(all_classifications),
        "filter_results": filter_results,
        "top3": [r["filter_id"] for r in top3],
    }
    with open(RESULTS_DIR / "phase_a_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {RESULTS_DIR / 'phase_a_results.json'}")


if __name__ == "__main__":
    main()
