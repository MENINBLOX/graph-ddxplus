# DDXPlus Numeric Scale Evidence — 방법론

## 발견
DDXPlus EVIDENCES는 4개 형태로 구성:
1. **Binary (B)** — Yes/No (예: `douleurxx` "통증 있나요?") — 208개
2. **Multi-choice (M)** — 1개 선택 (예: `douleurxx_carac` pain character) — 5개
3. **Multi-select Categorical (C)** — 복수 선택 (예: body locations) — 10개
4. **Numeric Scale (subset of C)** — 0-10 scale, 의미적으로 별개 — 6개

## 6개 Numeric Scale Evidences

| Evidence ID | 질문 | Base CUI | Value 0-10 의미 |
|---|---|---|---|
| `douleurxx_intens` | 통증 강도 | C0030193 (Pain) | 0=없음 ~ 10=최악 |
| `douleurxx_soudain` | 통증 발현 속도 | C0030193 (Pain) | 0=점진 ~ 10=급격 |
| `douleurxx_precis` | 통증 정확한 위치 | C0030193 (Pain) | 0=확산 ~ 10=정확 |
| `lesions_peau_intens` | 발진 통증 강도 | C0015230 (Eruption), C0030193 | 0-10 |
| `lesions_peau_elevee` | 발진 부어오름 정도 | C0015230 (Eruption) | 0-10 |
| `lesions_peau_prurit` | 가려움 정도 | C0033774 (Pruritus) | 0-10 |

## 문제점
- 모든 0-10 value가 **동일한 base CUI에 매핑됨** — value-specific CUI 없음
- 환자 답 `douleurxx_intens=1` (거의 통증 없음)와 `douleurxx_intens=10` (극심) 동일 처리
- 현재 알고리즘: value 존재 자체를 "Pain CUI 있음" = 양성 evidence로 해석
- 결과: 미약한 통증을 강한 통증과 같은 가중치로 평가 → IDF/cosine 신호 왜곡

## KG 표현 한계
- KG node는 base CUI (C0030193 Pain) 하나만 존재
- "심한 통증 (intensity ≥7)" 같은 별개 phenotype 구분 없음
- → KG 자체로는 numeric scale을 표현할 수 없음
- **Patient evidence 처리 시점에 해결해야 함** (KG 수정 불가)

## 해결 방향 (제안)
- **Threshold-based binary cutoff**:
  - value ≥ K → CUI 추가 (양성)
  - value < K → CUI 추가 안 함 (음성/무응답)
- K는 hyperparameter (lam/tau과 동일 위상, single across all benchmarks)
- 학술적 정당성: clinical VAS (Visual Analog Scale) 통념 (0=no, 1-3=mild, 4-6=moderate, 7-10=severe)
- **금지**: K를 disease label과 correlation으로 학습 = 원칙 3 위반

## 대안 검토
- **Linear weight (value/10)**: continuous, 정보 손실 최소화. 단점: 0과 1 차이를 거의 없게 만듦
- **VAS multi-tier**: 0=0, 1-3=0.2, 4-6=0.5, 7-10=1.0. 임상 표준이지만 hard-coded threshold
- **Threshold sweep K∈{0,1,2,3,4,5}**: 가장 검증 가능. K=0이 현재 SOTA baseline

## 적용 범위
- DDXPlus의 6 numeric evidences에 직접 적용
- SymCat/RareBench/PhenoBrain은 numeric form 없음 (HPO IDs/probabilities)
- MIMIC-RD는 별개 (clinical text)

## 학술적 보고 시 주의
- 단일 threshold K가 모든 benchmark에 영향 없음을 명시
- DDXPlus 특화로 보이지만 numeric form은 다른 medical EHR 데이터에도 일반적
- K 선택 근거 명시 (sweep 결과 + 임상 convention)
