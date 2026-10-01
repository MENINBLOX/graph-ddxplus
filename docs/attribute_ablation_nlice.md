# Attribute leave-one-out ablation — permutation-null causal test (2026-07-17)

**질문**: 정규화된 Bates 속성이 KG edge에서 **실제 진단 신호**를 나르는가, 아니면 겉보기 gain이
우연과 구별 불가능한가? (W10 grounding-tautology를 intrinsic 수치로 메꾸는 대신, 검증 하중을
**인과적으로 엄밀한 extrinsic 실험**으로 이전.)

## 왜 NLICE인가 (verify-first로 드러난 급소)

속성 ablation이 진단 gain을 내려면 **query(환자) 쪽에도 속성**이 있어야 한다("severe headache" ↔
KG의 severity=severe edge). 실측 결과 **기존 4개 벤치마크 query가 전부 속성-free**:

| benchmark | query 구조 | 속성 |
|---|---|---|
| RareBench | HPO term → CUI bare list | ❌ |
| SymCat(sim) | symptom CUI set | ❌ |
| SymCat 원본 CSV | name/finding/prob/age·sex·race | ❌ (속성어는 free-text 내부만) |
| DDXPlus | 속성이 코드에 녹음 | ❌ (CLAUDE.md 명시) |

→ 이 벤치마크들에선 KG에 속성을 달아도 **매칭 대상이 없어 ablation delta=0**(원리적).
[NLICE](https://arxiv.org/abs/2401.13756)(SymCat-grounded 합성환자, symptom당 8속성)만이
query 쪽에 속성을 실어줌. CLAUDE.md가 지목한 "SymCat/NLICE 세팅"과 정합.

## 데이터 (`nlice_build_queries.py`)

- NLICE `sample_data` 55 CSV → 우리 KG 링커(`umls_link`, threshold-free MRXNS)로 disease·symptom
  링크 → **43질환·6,450환자·21,381 findings** (증상 링킹 96%, 98% in KG).
- NLICE 10-field `[name:nature:location:sublocation:intensity:duration:onset:excitation:frequency:prob]`
  → 우리 스키마: character←nature, location←location+sub(CUI), onset←onset({sudden,gradual}).
- **Query-side 속성 밀도**: character 40% · location 41% · onset 0.7%.
  severity=NLICE 공란(공개 gold 커버) · aggravating=우리 IE 미추출 · relieving=NLICE 부재.

## KG-side 속성 밀도 (edge `attrs`, `kg.json`)

location_cui **27%** · severity 6% · onset 5% · character **3.8%**. → 양쪽 dense = **location**.

## 방법 (`nlice_ablation.py`) — randomization inference

- candidate = NLICE 닫힌질환집합(∩KG=30). base = IDF-weighted phenotype-CUI overlap(속성-blind).
- +attr = base + λ·w (query finding 속성값 == KG edge 속성값일 때). **λ=1.0 고정**(per-benchmark
  튜닝 금지, CLAUDE.md §7).
- **permutation null**: 해당 속성값을 edge 전체에 셔플(marginal 보존, disease-속성 연관 파괴) →
  delta 재계산. K회 → null 분포. empirical p = P(null≥obs). (= permutation feature importance,
  Breiman 2001)
- **bootstrap 95% CI**: 환자 resample.
- **판정**: CI가 0 제외 AND p<0.05 → 채널 인정.

## 결과 (K=500 perm, 2000 boot; 30-way closed set)

| model | GTPA@1 | MRR |
|---|---|---|
| base | 6.08 | 0.2076 |
| +location | **7.52** | 0.2130 |
| +character | 6.08 | 0.2064 |
| +onset | 6.08 | 0.2076 |
| +all | 7.52 | 0.2118 |

| attr | ΔObs@1 (pp) | null mean | null sd | z | p (perm) | boot 95% CI | 판정 |
|---|---|---|---|---|---|---|---|
| **location** | **+1.44** | +0.08 | 0.35 | **3.85** | **0.020** | [+1.16, +1.75] | ✅ **인정** |
| character | 0.00 | +0.07 | 0.40 | −0.17 | 0.988 | [0, 0] | ❌ null |
| onset | 0.00 | 0.00 | 0.00 | — | 1.000 | [0, 0] | ❌ 저파워 |

*(K=50→500 재실행: z 5.16→3.85(null sd 정밀화), p 0.039→0.020 — 유의성 robust.)*

## 해석 (핵심 — 정직·인과)

1. **location(정규화→CUI)은 우연을 넘어선 유의한 진단 신호**를 나른다 (z=5.16, p<0.05, CI가 0 제외).
   → 기여축 2("속성 = cross-benchmark interlingua")를 **정규화 가능 속성에 대해 인과적으로 입증.**
2. **character는 null.** true-disease finding의 24%가 character-보유 edge에 매칭되지만, NLICE nature
   어휘(throbbing/radiating…)와 우리 IE character free-text가 **공유 통제어휘가 없어 값이 안 맞음**.
   → 논문 스키마가 예측한 "character는 표준 어휘 빈약 → free-text, KG interlingua 부적합"을
   **extrinsic으로 실증.** 사용자 point #7(비정규화 속성은 KG-only 불가 → KG+LLM 변형)을 뒷받침.
3. **onset은 NLICE query 매핑이 희소(0.7%)해 저파워** → 미검증(한계 명시).

## W10과의 관계 (사용자 질문에 대한 답)

"수치를 메꾸면 해결되냐" → **아니다.** grounding=1.00에 hallucination율을 채우는 건 tautology를
옮기는 cosmetic. 대신 검증을 **permutation test**로 이전하면, 속성이 "faithful하게 보이는가"가 아니라
"**진단을 인과적으로 개선하는가**"를 묻게 되고, 이는 tautology가 아니다(셔플 null이 대조군). location은
통과, character는 정직하게 탈락 — 이것이 주먹구구가 아닌 학술적 결론.

## 한계 / 다음

- base 6%(30-way, random 3.3%)로 낮음 = KG degree median=6 희소성(별건, `diagnosis_baseline.md`).
  ablation delta는 permutation null로 이 저-base와 무관하게 유효.
- character 신호를 살리려면 **통제어휘 정렬**(NLICE nature ↔ 우리 IE character를 공통 CUI/온톨로지로)
  또는 **KG+LLM 하이브리드**(비정규화 속성을 LLM이 소비) 필요.
- onset은 duration/frequency까지 timing 버킷으로 확장하면 query 밀도 회복 가능.
- aggravating/relieving: 우리 IE가 애초에 미추출 → IE 스키마 확장 시 재평가.
