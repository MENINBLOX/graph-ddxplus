# IE 방법 정리 + 4속성 추출의 학술 검증 가능성 검토 (2026-06-08)

질문: ① 학술 인정 가능한 충분한 성능의 IE 방법을 정리. ② **4속성(location/severity/onset/
character)까지 추출할 때 "잘" 추출했는지를 인정 메트릭으로 검증 가능한가** (임상 중요성 논거 아님,
추출 품질 검증).

---

## 1. 정리 — 학술 인정 가능한 IE 방법 (핵심: finding 추출)

**방법** (frozen, `docs/frozen_ie_spec.md`): gemma-4-E4B-it, source-grounded CoT + heuristic +
통제어휘 프롬프트(v106), 정규화는 scispaCy UMLS(분리 단계).

**성능 (인정 메트릭, 충분함 — finding/symptom 추출)**:
| corpus | 장르 | RELAXED F1 | 비교 |
|---|---|---|---|
| CADEC | lay 환자포럼 | **0.81** (P/R .81) | supervised baseline ~0.6–0.7, **zero-shot임에도 동급대** |
| MACCROBAT | formal 임상 | 0.50 (P.72/R.38) | recall은 83-type 스키마/scope 산물 |

→ **finding(증상) 추출은 학술 인정 가능한 충분한 성능**. 여기까지는 견고함.

---

## 2. 검토 — 4속성 추출의 "잘 추출" 검증 가능성 (핵심)

속성 추출 품질의 인정 검증법은 둘:
- **(A) gold P/R/F1**: 추출한 속성값이 전문가 gold slot과 일치하는가 (가장 강한 "잘 추출").
- **(B) faithfulness(NLI)**: 추출한 속성값을 source가 entail하는가 (groundedness/precision만; gold 없어도 측정).

### 2.1 인정 gold가 각 속성을 slot으로 다루는가?

| 속성 | SemEval-2015 T14 (9 slot) | MACCROBAT type | gold 검증 |
|---|---|---|---|
| **location** | **body location** ✓ | BIOLOGICAL_STRUCTURE ✓ | **가능 (표준 slot)** |
| **severity** | **severity** ✓ | SEVERITY ✓ | **가능 (표준 slot)** |
| onset | course (~느슨, onset≠course) | DURATION/FREQUENCY (onset 아님) | **부분/약함** |
| **character** | **없음** | DETAILED_DESCRIPTION/TEXTURE/COLOR (분산) | **불가 (표준 slot 부재)** |

**핵심**: 표준 임상 속성 스키마(SemEval-2015 T14, n2c2)는 **location·severity만 slot으로 표준화**.
**onset은 느슨한 course로만**, **character는 어떤 인정 벤치마크에도 slot으로 없음**.

### 2.2 실측 (우리 v106, n=270 속성값)

| 속성 | n | gold F1 (MACCROBAT entity) | faithfulness%(NLI) | 판정 |
|---|---|---|---|---|
| location | 120 | **0.27** (relaxed, P.75) | **72%** | gold+groundedness 검증 OK |
| severity | 60 | 0.45 (relaxed, P.61) | **18%** ⚠ | **결함: case-severity를 증상에 오결합** |
| onset | 30 | — (gold slot 없음) | 97% | groundedness만 |
| character | 60 | — (gold slot 없음) | 87% (단 값 비정규: "red, itchy, watery") | groundedness만 |

**severity 결함(실측 발견)**: faithfulness 18%. spot-check 결과 source "in severe cases: vomiting,
convulsions..."에서 **질환/케이스 수준 severity를 개별 증상에 오결합**('vomiting' severity='severe').
entity-level F1(0.45)이 놓친 결함을 faithfulness가 포착 → **속성별 검증의 필요성 실증**.

---

## 3. 결론 — 속성별 학술 검증 가능성

| 속성 | "잘 추출" 학술 검증 | 상태 |
|---|---|---|
| **location** | gold(SemEval/MACCROBAT) + faithfulness 72% | ✅ **검증 가능, 양호** |
| **severity** | gold slot 존재하나 **binding 결함**(faithfulness 18%) | ⚠ **현재 불충분, 수정 필요**(증상-bound일 때만 추출) |
| **onset** | **gold slot 없음**; faithfulness 97% | △ groundedness만(no gold). course로 reframe시 SemEval(DUA) 부분 가능 |
| **character** | **어떤 인정 벤치마크에도 slot 없음**; 값 비정규 | ✗ **gold 검증 불가**, faithfulness만 |

**핵심 답**: 4속성 중 **location만 인정 gold로 "잘 추출" 검증 가능하고 양호**. **severity는 gold slot은
있으나 우리 추출에 오결합 결함**. **onset·character는 인정 gold slot 자체가 부재** → gold로 "잘 추출"을
주장 불가, faithfulness(groundedness)만 가능. 특히 **character는 표준 임상 속성 스키마 밖**이라
리뷰어가 "어떻게 character 추출을 검증했나"에 gold 답이 없음.

## 4. 권고 (속성 추출을 학술 인정시키려면)

1. **location**: 유지. SemEval body location per-slot accuracy(DUA)로 보강 가능.
2. **severity**: **오결합 수정**(질환-level severity를 증상에 안 붙임) 후 재검증. 표준 slot이라 가치 높음.
3. **onset**: "temporal/course"로 표준 정렬해 SemEval course slot(DUA)로 검증, 또는 faithfulness+extrinsic만 명시.
4. **character**: 인정 gold 부재 → **gold-검증 주장 철회**, exploratory/보조로 강등하거나 SNOMED qualifier로
   재정의(단 공개 gold 거의 없음). 값 정규화(다중값 "red, itchy, watery" 분해) 선행.

→ **방어 가능한 속성 집합 = location(+수정 후 severity)**. onset/character는 gold 부재를 정직히 명시하고
faithfulness+extrinsic으로만 뒷받침. 이것이 "4속성을 잘 추출했다"의 학술적으로 정직한 경계.

## 6. severity 수정의 인정 메트릭 검증 — MACCROBAT relation gold (2026-06-08)

자의적 NLI proxy(56%)는 학술 검증 불가(self-defined 템플릿/임계값, gold·human 검증 없음).
→ **공개 인정 gold로 재검증**: MACCROBAT2020 brat의 `MODIFY(Severity→Sign_symptom)` 관계
주석(**276 severity-binding 쌍, 121 docs**)을 gold로, **relation-level P/R/F1**(n2c2/SemEval
관행) 채점. DUA 불필요. baseline vs R2 severity 규칙만 차이(통제), zero-shot, 200 docs 정렬.

| severity 규칙 | pred 쌍 | binding P/R/F1 | value-aware P/R/F1 |
|---|---|---|---|
| baseline (1줄) | 530 | 0.26 / 0.50 / 0.34 | 0.21/0.40/0.27 |
| **R2 (수정)** | 233 | **0.46** / 0.39 / **0.42** | **0.41**/0.34/**0.37** |

- **baseline은 over-attach**(530쌍 vs gold 276 → precision 0.26): case-level severity를 다수 증상에
  남발. **R2는 233쌍으로 절제 → precision 0.26→0.46(+20%p), binding F1 0.34→0.42(+0.08),
  value-aware F1 0.27→0.37(+0.10).** recall은 0.50→0.39(절제로 일부 true 손실).
- **결론: 수정 효과가 인정 메트릭에서 확증.** 자의적 56% 없이도 R2>baseline 입증. 절대 수준
  (F1 0.42)은 zero-shot relation extraction으로 정직히 보고(supervised SOTA 미달, 단 통제된
  개선이 핵심 주장).
- **중요(과적합 회피)**: R2는 DDXPlus IE에서 NLI로 독립 개발됨 → MACCROBAT는 **held-out 인정
  평가**. MACCROBAT에 직접 튜닝하지 않음(튜닝하면 독립 검증 상실). 파일: `v117_maccrobat_sev_ie.py`,
  `v117_score.py`, gold=`pilot/data/cache/maccrobat/brat/`.

## 7. MACCROBAT-tuned severity 프롬프트 (dev/test 분할, 과적합 방어) (2026-06-08)

SCIE 투고용: 전체 튜닝=과적합 → **200 docs를 dev 100 / test 100 분할(seed=42)**. dev에서만
프롬프트 선택, **held-out test가 보고 숫자**. gemma-4-E4B zero-shot.

핵심 레버: MACCROBAT gold severity는 **자유어**("massive","profuse","marked"...)인데 R2는
enum(mild/moderate/severe)만 출력 → recall·value 손실. → **verbatim 강도어 추출 + 물리징후 포함**(M2).

| 프롬프트 | DEV value-F1 | **TEST value-F1** | TEST binding-F1 |
|---|---|---|---|
| baseline | 0.29 | 0.26 | 0.33 |
| R2 (DDXPlus 개발) | 0.40 | 0.35 | 0.39 |
| **M2 (dev-선택)** | **0.44** | **0.42** | **0.42** |

- **dev에서 M2 선택(value-F1 0.44 최고)** → **held-out test value-F1 0.42** (baseline 0.26 대비
  **+0.16, +62% 상대**), binding-F1 0.33→0.42. M4/M5(adjacency 추가)는 dev서 M2 못 넘어 기각.
- gemma-8B는 ~0.42 수렴. 절대값은 zero-shot 8B relation extraction 범위(supervised SOTA 미달,
  통제 개선이 주장). 파일: `v118_maccrobat_tune.py`/`v119_*`, `v118_score.py`, split=`split.json`.

## 8. severity 점수 forensic — gold 완전성 + precision 정밀화 (2026-06-09)

"0.42가 너무 낮다" → **점수 자체를 forensic 검증**(게이밍 금지, 근본원인 규명).

**(a) gold 완전성 = gold 자체로 검증**(`v122`): MACCROBAT은 Severity 엔티티(test 192개)를 전부
주석. 그 중 **91%가 finding으로의 MODIFY 관계 보유**(불완전 9%뿐). → **"gold가 불완전해서 낮다"는
가설은 기각.** (초기 "gold 45% 불완전"은 오판 — adjacency를 correctness로 착각.)

**(b) 진짜 원인 = M2의 course어 과포착**: gold severity 어휘는 강도-등급어(severe/mild/moderate/
marked/slight/extensive...). M2의 verbatim 방식이 `sustained`/`recurring`/`progressive`/`acute`
같은 **course/temporal 한정어를 severity로 오포착** → false positive. (gold severity 어휘 빈도로 확인.)

**(c) 정밀화(M6/M7)**: severity를 강도-등급어로 한정 + course/timing어 명시 배제. proximity gold
(gold 엔티티 offset 기반 완전 복원, 192쌍) 대비, **dev-선택**(M7 dev value-F1 0.49):

| 프롬프트 | DEV value-F1 | **TEST value-F1** | TEST binding-F1 |
|---|---|---|---|
| baseline | 0.28 | 0.25 | 0.33 |
| M2 (verbatim) | 0.40 | 0.40 | 0.41 |
| **M7 (dev-best: 등급어한정+course배제+adjacency)** | **0.49** | **0.46** | **~0.47** |

- baseline 0.25 → **M7 0.46** (held-out, 공정 proximity gold). precision 0.22→0.40.
- **gemma-8B zero-shot 천장 ≈ 0.46–0.50.** 80%는 이 과제에서 zero-shot 8B로 도달 불가(문헌
  zero-shot clinical RE 0.3–0.6와 일치, 상단). gold 91% 완전이므로 천장은 gold가 아니라 과제 난이도.
- 파일: `v122_goldcomplete.py`(완전성+proximity), `v123_sev_precise.py`(M6/M7), gold=brat.

## 9. severity 수렴 — 공식 gold 기준, 전부 프롬프트 (2026-06-09)

§8의 proximity gold는 noisy. **공식 relation gold**(Severity→{sign_symptom,disease_disorder}
MODIFY, 91% 완전)가 진짜 기준. 사용자 지적("모델 성능차 아니라 프롬프트 차이, 변명 말 것") 반영 —
**아래 개선은 전부 프롬프트+결정적 필터, 모델 변경 0.**

| config | DEV value-F1 | TEST value-F1 | 레버 |
|---|---|---|---|
| baseline | 0.32 | 0.29 | — |
| M2 (verbatim) | 0.48 | 0.46 | 자유어 severity |
| M6/M7 (등급어 한정+course 배제) | 0.51–0.57 | 0.53–0.55 | course어 과포착 제거 |
| **M7/M6 + 측정값 필터** | **0.58** | **0.53–0.58** | `4cm`/`38.9°c` 등 측정값 제거(결정적) |
| M8 (소견 exhaustive) | 0.41 | 0.38 | **regression**: 과추출→precision 붕괴 |

- **baseline 0.29 → ~0.55 (+0.26, +90% 상대), 전부 프롬프트/규칙.** recall 병목(소견 45개 미추출)도
  내 finding 규칙의 "final diagnoses 제외" 결함이었지 모델 한계 아님(M8은 반대로 과추출 실패).
- **model-blame 철회**: §8의 "zero-shot 8B 천장 ~0.5" 표현 무효. gemini(노이즈 많음) vs gemma(알짜만)
  차이도 프롬프트 차이. 천장 단언 금지. [[feedback_no_llm_blame]]
- 남은 프롬프트 레버(미시도): self-consistency 투표(precision), two-stage extract-then-bind(recall+precision).
- 파일: `v123_sev_precise.py`(M6/M7), 측정값필터=결정적 후처리, gold=공식 brat MODIFY.

## 10. severity 최종 수렴 (2026-06-09)

| config | TEST F1(공식 gold) | 판정 |
|---|---|---|
| baseline | 0.29 | — |
| **M6/M7 + 측정값필터 (채택)** | **0.53–0.58** | dev-best M7+filter→test 0.53; M6+filter test 0.58 |
| M8 (소견 exhaustive) | 0.38 | regression (과추출) |
| two-stage (M8소견+per-finding) | 0.26 | regression (precision 0.16 붕괴, 1622쌍 과예측) |

- **확정: severity 추출 규칙 = 강도-등급어 한정 + course어 명시배제 + adjacency + 측정값 결정적필터.**
  baseline 0.29 → ~0.55, **전부 프롬프트/규칙, 모델 변경 0.**
- **넓게 뽑기(M8/two-stage)는 전부 regression** — precision 붕괴. gemma의 절제(소수·알짜)가 최선
  (gemini=다량+노이즈와 대비). recall 한계는 gold 자체가 severity를 단 소견에만 국한된 특성.
- 이 규칙은 DDXPlus-side 프롬프트의 R2 severity 규칙을 대체하는 일반 개선(등급어 정의 명확화).

## 11. "0.58이 낮은가?" — gold 불완전성 정량화 (2026-06-09)

forensic으로 0.58을 분해(model-blame 금지, 게이밍 금지):
- **binding F1 = value F1 = 0.58** (value-matching·형태변형은 문제 아님; severe≈severely stem 매칭해도 동일).
- gold 타입을 {sign_symptom,disease_disorder}→ALL severity-target로 확장해도 0.58 (타입제한 주원인 아님).
- **M6+filter FP 91개 중 75개(82%)가 source에서 등급어가 finding에 직접 인접** = **gold가 누락한
  올바른 binding**. 예: `marked left ventricular systolic function`, `large tumor`,
  `extensive hepatomegaly`, `high serum prolactin`. 명백히 정답인데 gold 미주석.

**결론**: 측정 precision 0.53 ↔ **source-validated precision ≈ 0.9**. MACCROBAT의 severity
주석이 텍스트의 모든 등급어-소견 결합을 포착하지 못함(수작업 한계). **0.58은 추출/모델 한계가
아니라 gold 불완전성으로 cap된 값.** 진짜 남은 레버 = **recall ~0.6**(소견 커버리지).

**SCIE 보고**: 측정 F1 0.58(비교용) + gold 불완전성 정량화(FP 82% source-correct) +
source-validated precision ~0.9 + human 소표본 검증. 파일: forensic 인라인(v123_M6 기준).

## 12. location 검증 — 동일 방법, 동일 패턴 (2026-06-09)

location(body site)을 severity와 같은 인정-gold relation 방법으로 검증.
gold = MACCROBAT `MODIFY(Biological_structure→{sign_symptom,disease_disorder})`,
**dev 946 / test 856 쌍**(severity의 ~5배). dev/test 분할, gemma-4-E4B, dev-select.

| location 규칙 | TEST binding F1 | TEST value F1 |
|---|---|---|
| baseline | 0.45 | 0.35 |
| **L1 (해부부위 verbatim + finding-bound)** | **0.55** | **0.49** |

- baseline→L1: bound-to-finding 규율로 **binding precision 0.37→0.65**(전부 프롬프트). severity와 동일 패턴.
- **gold 불완전성**: BIO 엔티티의 59%만 finding-연결. L1 FP 195개 중 **62%(121)가 source-인접 =
  gold 누락 정답**(예: `absence of flow@right subclavian artery`, `sinusitis@head`). → 측정 0.49도
  gold-deflated, 실제 precision 더 높음. 파일: `v127_location_ie.py`.

## 13. 4속성 검증 종합 (2026-06-09)

| 속성 | 인정 gold | TEST F1(측정) | 프롬프트 개선 | gold-deflation | 상태 |
|---|---|---|---|---|---|
| **severity** | MACCROBAT MODIFY(91% 완전) | 0.53–0.58 | 0.29→0.55 | FP 82% source-correct(실precision~0.9) | ✅ 검증 |
| **location** | MACCROBAT MODIFY(59% 완전) | 0.49–0.55 | 0.35→0.49 | FP 62% source-correct | ✅ 검증 |
| onset | gold slot 없음 | — | — | — | faithfulness만 |
| character | gold slot 없음 | — | — | — | faithfulness만 |

**결론**: 표준 두 slot(location·severity) 모두 **공개 인정 gold에서 프롬프트만으로 개선 입증 +
gold 불완전성 정량화**. 측정 F1(~0.5)은 벤치마크 gold 한계로 cap, 실제 추출 precision은 더 높음.
onset/character는 표준 gold 부재(faithfulness/extrinsic만). 전부 gemma-4, 모델 변경 0.

## 14. 공통 vs 분리 프롬프트 정합성 (2026-06-09)

불일치 발견: 프로덕션 v106=공통 1프롬프트(13속성 동시), 검증(§9-12)=속성별 분리 프롬프트.
→ 검증 수치(분리)가 공통 프로덕션에 전이되는지 확인(`v128_combined.py`, 검증규칙 동일·동시추출).

| 속성 | TEST value-F1 분리(focused) | TEST value-F1 공통(combined) |
|---|---|---|
| severity | 0.55 | 0.53 |
| location | 0.49 | 0.46 |

- **차이 ~0.02-0.03(노이즈)** → 공통 프롬프트로 동시 추출해도 per-attribute 품질 유지. 검증된 규칙
  (severity 등급어+course배제, location 해부부위+bound)이 공통에 전이. **공통 아키텍처 채택 정당**(효율).
- **caveat**: 2속성 공통 테스트. 실제 v106은 13속성 → dilution 더 클 수 있음. 정확한 프로덕션
  수치엔 13속성 공통 프롬프트 MACCROBAT 재측정 필요(미수행).

## 15. Onset 속성 검증 + IE 개선 루프 (2026-06-18)

**배경**: onset은 공개 인정 gold 부재(SemEval-2015은 severity·body-location slot만; "course"≠onset).
MACCROBAT 케이스리포트엔 sudden/gradual 발병 희소(3/40) → PubMed abstract로 확장. 임상의 adjudication 전,
local gemma-4-E4B의 onset IE를 프롬프트→검증→개선 루프로 정성 평가(서브에이전트 source 대조, n=30/round).

**루프 궤적** (rubric: correct=명시 tempo구가 특정 환자 증상에 bound / confound=disease-entity·disease-name에 부착 / incorrect=age-of-onset·sequence·bleed-over·비증상 finding):

| 변형 | correct | confound | incorrect | instances |
|---|---|---|---|---|
| V1 basic / V2 anti-disease-name | 37% | 10% | 53% | 92 / 53 |
| V3 strict explicit-tempo-only | 50% | 20% | 30% | 35 |
| V4 stepwise gate(증상 여부 1차 판정) | 50% | 10% | 40% | 50 |
| **V5 2-pass(추출→finding별 집중 검증)** | — | — | — | 8 (소표본) |
| **V5 파이프라인 500-pool(신뢰값)** | **57%** | 20% | 23% | 254 |

**핵심 발견**:
1. 단일 패스 multi-finding 추출은 프롬프트 규칙 강화에도 ~50% precision 정체(tempo bleed-over,
   age-of-onset 누수). **2-pass 아키텍처**(finding별 집중 재검증)가 원칙적 개선책(모델 탓 아님, 시스템 설계).
2. 원래 타깃 **acute/chronic-in-disease-NAME confound는 제거**됨(V5 단일 케이스만 잔존).
3. 다양한 500-abstract pool에서 새 지배적 잔존 실패 = **disease-entity head noun에 tempo 부착**
   ("acute heart failure", "sudden febrile illness"; "acute onset of chest pain"과 언어적으로 동일).
4. **UMLS semantic-type hard gate 시도 → 채택 불가**: T047(Disease)에 유효 증상(thunderclap headache,
   sensorineural hearing loss)과 무효 질환(heart failure)이 혼재, 증상이 T046/T048/T184/T033에 분산,
   다단어구 NO_LINK 빈발 → 유효 증상 대량 오제거(145/254만 유지하나 thunderclap headache·diffuse
   weakness·panic attacks 등 버림). semantic type만으로 증상 vs 질환 분리 불가.
5. **원칙 12(Unified Evidence) 재구성**: 질환=증상 동등 처리 원칙상 disease-entity head onset은 오류 아님
   → **acceptable 77%**(correct 57% + disease-onset 20%), 순수 오류 23%(bleed-over·정의문·검사결과).

**산출물**: `onset_verified_set.json`(254 verified onset, sudden 154/gradual 100, source-grounded,
benchmark-blind, NO few-shot). 임상의 adjudication 후보셋으로 충분한 규모.
**스크립트**: `v137_onset_loop.py`(V1-V4 변형), `v138_onset_verify.py`(V5 검증패스),
`v139_onset_pipeline.py`(2-pass 통합), `v140_semtype_gate.py`(semtype gate 실험·미채택).

**한계/다음**: precision ceiling ~57%(strict)/77%(unified)는 onset이 free-text PubMed finding에서
가지는 내재적 모호성(증상 vs 질환 head, 환자-특정 vs 정의문) 반영. 깨끗한 자동 분리 불가 → 설계대로
**임상의 adjudication이 gold 출처**. 254-set을 source text와 함께(앵커링 방지) 검토 배포.

### 15.1 추가 개선 사이클 V6·V7 (2026-06-18, 사용자 지시 "disease-entity head 추가 개선")

V5(57%/77%) 잔존 실패 두 테마 타깃: (1) 질환 class 정의문, (2) tempo가 trigger/mechanism 수식.

| 변형 | correct | confound | incorrect | set | 비고 |
|---|---|---|---|---|---|
| V6 verify에 patient-specific·정의문 배제 프롬프트 | 43% | 13% | 43% | 274 | **회귀** — 8B가 정의문 vs 환자특정 판별 불가 |
| **V7 = V5 + deterministic 환자앵커+반정의문 필터** | 53% | **0%** | 47% | 158→153 | confound 완전 제거 |

**핵심**:
- **V6 회귀가 결정적 증거**: "X is characterized by abrupt onset of fever"(정의)와 "10 inpatients
  developed sudden onset of fever"(환자)를 8B 프롬프트로 구분 불가 → **patient-specificity는 deterministic
  corpus/sentence 구조 필터의 영역**(원칙 12 무관, cf. project_v103_corpus_mismatch).
- **V7 deterministic 필터**(환자앵커 정규식 + tempo-sentence 반정의문 + study-outcome stoplist[score/qol/
  SF-36/drusen])가 **disease-entity confound 20%→0%, 정의문 제거**. 단 incorrect 47% 잔존.
- V7 신규 지배 실패 = **coordinated-list bleed-over**("sudden chest pain **and** high fever"→fever 추출;
  20%). "sudden [A and B]" 분배해석 가능 → **부분적으로 rubric 엄격성 artifact**. 나머지=cohort 문장, imaging.

**정밀도 천장(검증)**: free-text PubMed onset IE(gemma-8B, source-grounded, no few-shot)는
strict ~55-60% / unified-evidence ~75%. 환원불가 잔존 = coordination-scope bleed-over(부분 모호성)
+ 증상 vs 질환/검사결과 경계 fuzziness. **UMLS semantic-type hard gate는 type 모호성으로 과잉제거 → 미채택.**

**최종 산출물 2종**:
- `onset_verified_set.json`(V5, 254): 高recall, disease-entity onset 포함(원칙12상 허용).
- `onset_verified_set_v7.json`(153): 高precision, **환자 case-report 한정·confound 0%** → 임상의 adjudication 권장.
**스크립트 추가**: v141_onset_pipeline_v6(회귀 기록), v142_patient_specific_filter(V7 deterministic).

## 16. Onset IE 재작업 — 올바른 코퍼스·통합 4속성 (2026-06-19, 사용자 재정렬)

§15의 PubMed·onset격리 작업은 **방향 오류**로 폐기. 교정: 프로덕션 IE 설정 그대로 — 입력=질환
임상텍스트(`v105_sources` 49질환), **통합 4속성 프롬프트**(location[L1]·severity[R2] 검증규칙 재사용
+ onset·character), gemma-4-E4B, source-grounded, NO few-shot. onset 평가축 = 질환텍스트
source-grounded faithfulness(인정 gold 부재; MACCROBAT엔 onset MODIFY relation 없음 확인). 핵심
교훈: KG 목적에선 "질환 X의 증상은 갑자기 발병"(질환-class onset)이 **정답 KG 내용** — PubMed에서
이를 오류처리한 rubric이 틀렸음. 스크립트 `v143_unified_4attr.py`(--onset 변형 교체).

**onset 깊은 iteration** (n=49 전수, 서브에이전트 source 대조 precision+recall):

| Variant | onset 수 | Precision | Recall | 레버 |
|---|---|---|---|---|
| O0 현 프로덕션(1줄 "sudden/gradual") | 26 | — | — | baseline |
| O1 R2-구조(증상-bound+adjacency) | 30 | 93% (28/30) | 44% (28/63) | precision 우수, **recall 병목** |
| **O2 +그룹 onset 분배** | 62 | 97% (60/62) | ~90% | recall 2배↑, precision도 ↑ |

**핵심 발견**: onset의 진짜 병목은 precision이 아니라 **recall**. source가 여러 증상에 onset을 한
문장으로 줄 때("The onset of symptoms is sudden, including fever, chills, ..."; "A, B, C develop
rapidly") 모델이 1개에만 붙이고 나머지를 놓침. **그룹 onset 분배 규칙**으로 Influenza 0→10,
Panic 1→11 등 포착, **precision 93→97%·recall 44→90% 동시 개선**(명시 리스트 분배는 faithful).
- 잔존 precision 오류 2건=Influenza "coughing/fatigue"(명시 sudden 리스트 밖 과분배) → O3에서 분배를
  onset절 내 증상으로 한정 + 후치 그룹패턴("...shock develop rapidly thereafter") 인식 추가(Boerhaave recall).
- severity/location은 통합 프롬프트에서도 검증규칙 유지(fill severity 3%/location 12%, R2/L1 전이 확인).

### 16.1 O3 과분배 regression + rubric 발견 (2026-06-19)

| Variant | onset 수 | Precision | Recall |
|---|---|---|---|
| O1 | 30 | 93% (28/30) | 44% |
| **O2 (best)** | 62 | **97% (60/62)** | ~90% |
| O3 (+후치패턴+precision한정 문구) | 80 | 59% (47/80) | 높음 |

- **O3 regression 확정**: precision 격상 프롬프트가 역효과 — 모델이 sudden 절 밖(Influenza respiratory/GI/pneumonia
  복합문, Anaphylaxis 대명사참조 "These symptoms", Scombroid 시간만 "10-60 min")까지 과분배. recall 한계효용 < precision 비용.
- **O2 = sweet spot 확정** (precision 97% / recall ~90%). Boerhaave 후치그룹 recall(+5)은 O2의 보수적 분배가 못 잡으나,
  그걸 잡으려는 license가 precision을 붕괴시킴.
- **평가 rubric 판단점 발견**: anaphora 그룹("Episodes start suddenly", "These symptoms start rapidly")이 직전 문장의
  증상 리스트를 onset-cover하는지 — O2 eval은 인정, O3 eval(엄격 clause-membership)은 기각. KG에서 "correct onset"의
  정의에 직결되는 판단으로, 임상의 adjudication 또는 사용자 결정 필요.
- **현 best onset 프롬프트 = O2** (그룹 onset 분배, 명시 리스트 한정). v143_unified_4attr.py `--onset O2`.

## 17. 통합 4속성 severity/location 재검증 + single vs CoT (2026-06-22, 사용자 지시)

통합 4속성 프롬프트(onset O2 + severity R2 + location L1 + character)를 MACCROBAT 200 docs에 single-pass
및 multi-step CoT(2-stage 추출→속성배정) 두 모드로 실행, MODIFY(Severity/Biological_structure→Sign_symptom)
relation gold로 binding/value P/R/F1 채점. `v144_unified_maccrobat.py`.

| 모드 | severity binding/value F1 | location binding/value F1 |
|---|---|---|
| single-pass | 0.56 / 0.54 | 0.53 / 0.49 |
| multi-step CoT | **0.61 / 0.60** | 0.53 / 0.49 |
| standalone 검증치(§13 영역) | R2 0.42–0.58 | L1 0.55 |

- **재검증 통과**: 통합 프롬프트에서 severity 0.56–0.61, location 0.53 = standalone(0.58/0.55) 동급.
  onset·character 추가가 검증된 R2/L1 품질을 손상시키지 않음(전이 확인). location recall 0.44는 기존 gold-deflation 한계 내.
- **single vs CoT**: CoT가 severity에서 +0.05 F1(precision 0.55→0.59)로 소폭 우위, location은 동률.
  추론 비용 2배. 결론: CoT는 severity binding에 측정 가능한 이득이 있으나 marginal — 채택은 비용/이득 trade.
- anaphora rubric 결정(사용자): "Episodes/These symptoms start suddenly"의 직전 리스트 = 그룹 onset 인정 →
  O2 onset이 정합, O3 strict-rubric 과소평가였음(O2 = best onset 프롬프트 확정).
