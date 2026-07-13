# 추론 시점 LLM 활용 (CoT/ToT/GoT, rerank)  🟦 보류 (제약상 현재 미사용)

**상태**: 보류 — 현재 strict 제약(추론 시 LLM 미사용, 원칙 #4) 위반이라 비채택. KG-only 부족 시 fallback.

## 아이디어
KG가 top-k 후보를 주면, 추론 시점에 LLM으로 재순위/감별:
- **CoT tie-break** (v87): 동점 후보를 LLM이 명시적 추론으로 정렬 → DDXPlus @1 66.48%.
- **Self-consistency CoT** (v310 N=3 / v318 N=10, majority vote) → **68.40%** (LLM 라인 best).
- **2-stage disambiguation** (v214): KG top-10 → LLM top-1 (86% 목표 라인).
- pairwise tournament, generative DDx, structured CoT 등 다수.

## 제약 위반 내용
- 원칙 #4: "Inference time에 LLM 사용 X — KG + 단일 algorithm만." → 위 전부 추론 시 LLM 사용 = **현재 설계에서 비채택**.
- 단, **train 라벨은 안 씀**(zero-shot)이라 leakage는 아님 → 제약이 완화되면(LLM-at-inference 허용) 유효.
- ⚠️ 구분: **v53 Few-shot Naive Bayes @1 99.43%는 few-shot 예시(train)** 사용 → label leakage = **무효**.

## 위치
교수님 회신: "KG-only 점수 부족하면 CoT/ToT/GoT 시도". 즉 **KG-only 최대화 후 fallback 카드**. 논문에선 "KG-only X% / +LLM rerank Y%"로 보고 가능(추론 LLM 사용 명시).

## 결론
현재는 보류. KG-only(strict) 성능을 먼저 끌어올리고, 부족 시 이 라인을 fallback으로 재개. 관련 스크립트 다수(onlykg_eval_v87/v90, v310 등).
