# Aidea: Value-Weighted Patient Evidence Vector

**상태**: 시스템 design idea로 보존 — 현재 CUI 매핑 구조에서는 평가 효과 없으나, 매핑 재설계 후 또는 다른 benchmark에서 활용 가능.

---

## 핵심 아이디어

Patient evidence vector에서 **각 CUI의 weight를 numeric value로 결정**:

```python
weight = value / max_value      # 0-N scale → [0, 1]
pat_vec[CUI] = weight * (idf[CUI] ** beta)
```

같은 CUI에 여러 token이 weight 추가 시 **max() rule**:
```python
pat_vec[CUI] = max(weight_from_token_1, weight_from_token_2, ...)
```

## 학술적 정당성 (Universal System Design)

DDXPlus 종속이 아닌 **일반 system feature**:
- VAS (Visual Analog Scale, 0-10): pain, anxiety, depression
- PHQ-9: 0-3 per item
- GAD-7: 0-3 per item
- BPI (Brief Pain Inventory): 0-10
- Apgar: 0-10
- ECOG: 0-5
- Glasgow Coma Scale: 3-15

→ "numeric severity scale" = universal medical EHR pattern. 어떤 benchmark/EHR/symptom checker에도 적용 가능.

## 현재 평가 결과 (2026-05-27, v95_full KG, DDXPlus 5K)

| Variant | @1 | Δ vs baseline |
|---|---|---|
| v71 binary (baseline) | 63.22% | — |
| v100 linear (val/10) | 62.00% | -1.22%p |
| v100b denial-only (val=0 → 제외) | 62.52% | -0.70%p |

**모두 regression** — 현재 CUI 매핑 구조에서 효과 없음.

## 효과 없는 이유 (분석)

DDXPlus questionnaire는 **binary parent → numeric follow-up** 구조:
```
douleurxx (binary YES) → C0030193 추가 (weight 1.0)
  ├── douleurxx_intens=4 → C0030193 추가 시도 (weight 0.4) → max() rule, 무시
  ├── douleurxx_carac → C0030193 (1.0)
  ├── douleurxx_endroitducorps → C0030193 (1.0)
  └── ... (모두 binary YES가 dominant)
```

→ 22 CUI 중 단 2개 (Exanthema, Pruritus)만 weight 영향. 효과 미미.

게다가 그 2개도 **CUI 매핑 conflation 때문에 잘못된 방향**:
- `lesions_peau_intens` (통증 정도) → `C0015230 Exanthema` (매핑 자체가 부정확)
- weight 0.3로 down-weight 시 발진 존재 신호 약화 (정확히 반대 효과)

## 미래 활용 시나리오

### 시나리오 1: CUI 매핑 재설계 후
각 sub-question에 specific CUI 부여 시:
- `lesions_peau_intens=3` → `Painful rash` CUI (specific)
- `lesions_peau_elevee=4` → `Raised wheal` CUI (specific)
- 매핑이 깨끗하면 weight 0.3, 0.4 의미 정확

### 시나리오 2: Binary parent 없는 데이터
- EHR clinical notes (LLM-parsed): numeric만 추출되는 경우
- PHQ-9/GAD-7 questionnaire (각 item 0-3 numeric)
- 이 경우 max() rule에 binary가 없으므로 weight가 직접 반영

### 시나리오 3: 다른 benchmark
- SymCat은 disease별 symptom probability — value 자체가 weight (직접 활용 가능)
- 기타 EHR-based benchmark에서 자연스럽게 작동

## 구현 (보존)

`pilot/scripts/v100_value_weighted.py` — system-agnostic implementation:
- `detect_numeric_evidences()`: possible-values가 0-N 정수 리스트인 evidence 자동 감지
- `load_ddxplus_value_weighted()`: weight 계산 + max() 집계
- `score_v100()`: cosine + IDF + v71 negative penalty (weight-aware)

`pilot/scripts/v100_qualitative_one_case.py` — seed=42 case 비교 분석.

## Citation 가능성

논문 작성 시 negative result로도 보고 가치 있음:
> "We explored a value-weighted patient evidence vector where numeric scale values
> (e.g., VAS pain 0-10) directly contribute to CUI weight. The expected benefit
> was preservation of severity gradation lost in binary tokenization. However,
> on DDXPlus this yielded -1.22 pp regression. Analysis revealed the cause as
> structural: DDXPlus questionnaire pre-asks binary parent questions, and CUI
> mappings are shared across binary and numeric tokens, making the max-aggregation
> rule favor binary signals. This negative result demonstrates that
> value-weighted scoring requires upstream CUI mapping that distinguishes
> sub-question semantics, not just downstream weight assignment."

## 관련 ideas (보존)

- **CUI 매핑 재설계 (v100c)**: 각 sub-question에 specific CUI 부여
- **Threshold K cutoff (v100a)**: VAS clinical convention (K=4 = mild/moderate boundary)
- **Linear vs nonlinear weight**: sqrt(val/max), log scale, exponential

세 가지 모두 **매핑 재설계 prerequisite**.

## 결론

**보존 가치**: 
- 깨끗한 system design (universal medical convention 부합)
- Mapping 개선 후 재시도 가치
- LLM 기반/EHR text 기반 future system에 적용 가능
- Negative result 자체가 학술적 발견 (왜 안 되는지 분석 포함)

**현재 (2026-05-27)**: v95_full + v71 binary가 최적. v100 idea는 archive.
