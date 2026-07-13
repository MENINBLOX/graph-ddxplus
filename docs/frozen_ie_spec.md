# Frozen IE Specification (v106) — 학술 검증 완료, 고정 (2026-06-08)

본 문서는 우리 IE 방법의 **고정된(frozen) 정의**다. 이후 연구(KG 확장, Cypher/그래프
알고리즘)는 **이 IE를 상수로 두고** 그 위에서 변형한다. IE 자체는 인정 메트릭으로 검증되었으며
재인증 없이 재사용한다(알고리즘 변경 시에도 — intrinsic 인증은 알고리즘 독립).

근거 방법론: `docs/methodology_ie_evaluation.md` (intrinsic + extrinsic 양축).

---

## 1. 고정 설정 (Frozen configuration)

| 항목 | 값 | 근거 |
|---|---|---|
| **모델** | `google/gemma-4-E4B-it` (bf16, vLLM) | 검증된 작동 모델. 12B(`gemma4_unified`)는 vLLM 0.21 미지원 → 보류([[reference_gemma4_12b_infra]]) |
| **프롬프트** | **v106 + R2 severity 규칙** (`v106_grounded_ie.py` + `v115_sev_ie.py` R2). canonical 출력 = `pilot/data/cache/v115_R2/` | 아래 §3 결정 근거 + §6 severity 수정 |
| **추출 방식** | source-grounded CoT(2-step) + heuristic finding 규칙 + 통제 vocabulary | 선행연구: JMIR 2024(CoT+heuristic), WDC-PAVE(추출/정규화 분리), ABSA Extract-Then-Assign |
| **디코딩** | `temperature=0.0` (greedy), `max_tokens=4096` | 재현성 (deterministic) |
| **finding 개수** | **무제한 (no top-k cap)** | CLAUDE.md 원칙 6 — "exhaustive list", 인위적 제한 금지 |
| **source 입력** | 질환 임상 텍스트(Wikipedia 임상섹션, CC BY-SA), `src[:2200]` | benchmark-blind (원칙 5): 질환명만 입력, 벤치마크 질문 미포함 |
| **속성 스키마** | 13-slot (location/onset/duration/character/severity/radiation/timing/aggravating/relieving/associated/course/context/prior_episodes) | 교수님 자문(OLDCARTS/OPQRST/SOCRATES 공통핵심 4 + 보강), 표준 정의 기반([[project_professor_attribute_direction]]) |
| **속성 통제어휘** | onset∈{sudden,gradual}, severity∈{mild,moderate,severe}; 정의-반복/질환명-반복 금지 | hallucination 차단(프롬프트 영역) |
| **system 안전장치** | source-grounding 정규식 `in_src` (key-token 과반이 source에 존재) | 유일한 시스템 필터 — 나머지는 프롬프트가 처리([[project_v106_grounded_ie_prompt]]) |
| **few-shot** | **금지** | 원칙(memory feedback_no_fewshot) |

### 정규화(Normalization) 컴포넌트 — IE와 분리된 별도 단계

| 항목 | 값 |
|---|---|
| linker | scispaCy `en_core_sci_lg` + UMLS EntityLinker |
| 설정 | `resolve_abbreviations=True`, `linker_name=umls`, `k=3`, `threshold=0.85`, `max_entities_per_mention=1` |

**중요**: LLM IE는 **텍스트 mention만** 출력하고, CUI 매핑은 위 별도 모듈이 수행한다. 따라서
IE 추출 인증(span-level, CUI 불필요)과 정규화는 표준대로 **분리 평가**된다(SemEval-2015 T14,
n2c2 관행). 정규화 오류는 extrinsic downstream에서 end-to-end로 반영됨.

---

## 2. 검증 근거 (인정 메트릭 3축) — 모두 통과

| 축 | 방법 (인정 근거) | 결과 |
|---|---|---|
| **Intrinsic precision** | NLI faithfulness, 문장단위 max-entailment (hallucination survey 표준) | overall **83%** faithful |
| **Intrinsic span F1 (formal)** | MACCROBAT2020 gold, mention-level P/R/F1 (공개, DUA 없음) | SIGN_SYMPTOM RELAXED **F1 0.50** (P0.72/R0.38) |
| **Intrinsic span F1 (lay)** | CADEC gold, ADR span P/R/F1 (공개) | ADR RELAXED **F1 0.81** (P0.81/R0.81) — supervised baseline(~0.6–0.7) 동급대, zero-shot |
| **Extrinsic downstream** | DDXPlus 진단 @1/@10 (KGrEaT 패러다임, arXiv:2308.10537) | IE→KG→동일 알고리즘으로 측정 (`v110_extrinsic_ie_eval.py`) |

- **Cross-genre + cross-method 수렴**: formal 임상(MACCROBAT) ↔ lay 환자언어(CADEC) 모두 작동;
  세 방법 모두 "**precision 견고, recall이 레버**" 일관 지목 → 평가 자체 검증.
- 산출물: `pilot/data/cache/{v110_extrinsic_results.json, maccrobat/v111_pred.json, cadec/v111_pred.json}`.

### 문서화된 한계 (정직)
- **Recall**: fine-grained 83-type 스키마(MACCROBAT)에서 0.38. 단 clean 증상추출(CADEC)에선 0.81
  → 스키마/scope 산물이지 추출기 본질 한계 아님. **recall 개선은 별도 IE 사이클**(더 많은
  source·exhaustive) — 본 freeze를 막지 않음, future work로 분리.

---

## 3. v106 freeze 결정 근거 (vs v107)

| 기준 | v106 (단일패스 CoT) | v107 (2-stage Extract-Then-Normalize) |
|---|---|---|
| NLI faithfulness | 83% (finding 98 / attr 67) | 83% (finding 98 / attr 69) — **동률** |
| extrinsic @1/@10 | 24.50 / 74.07 | 25.84 / 75.12 (**+1%p, marginal·알고리즘 의존 구간**) |
| 메타속성 잡음(정성) | **1%** | 4% (hedge "mild to severe" 재출현) |
| 복잡도 | 단일 프롬프트 | 2-stage (오류 전파 위험) |

**결정: v106 고정.** 인정 메트릭상 동률(faithfulness)/미세차(extrinsic, 노이즈 구간)이고,
**parsimony**(단순·재현성) + **정성 청결도**(메타 1% vs 4%)에서 v106 우위. 미세 extrinsic
이득(+1%p)은 "marginal 차이는 알고리즘 의존" 원칙상 freeze 근거로 부적합.

---

## 4. Freeze 선언 & 다음 단계 (Phase 2)

- **고정**: 위 §1 설정 + `pilot/data/cache/v106_grounded_ie/` (49 DDXPlus 질환 IE 출력)이 canonical.
- **다음 연구는 IE를 상수로**: KG content 확장(source 추가·recall 사이클은 IE 재인증 시 §2 절차
  반복), bipartite KG 구조, Cypher/그래프 traversal, scoring 알고리즘 — 모두 **고정 IE 위에서** 변형.
- 알고리즘을 바꿔도 **IE intrinsic 인증(§2)은 그대로 유효**(알고리즘 독립). extrinsic만 새 알고리즘에서 재측정.

## 6. severity 오결합 프롬프트 수정 (2026-06-08)

진단(`v114`): gemma severity faithfulness 12-18% = source의 case-level severity("in
severe cases: X,Y")를 개별 증상에 오결합. **gemini-3.1-pro-preview도 동일(12→18%)** →
모델 capacity 아님, 프롬프트 문제. 프롬프트 반복 개선(`v115/v116`, NLI faithfulness 측정):

| 변형 | severity 규칙 | sev_fill | faithful% | faithful_n | loc% | char% |
|---|---|---|---|---|---|---|
| v106 baseline | "never severe cases" 한 줄 | 60 | 18% | 11 | 72 | 87 |
| R1 anti-case | 질환-level severity 명시 배제 | 18 | 44% | 8 | 79 | 89 |
| **R2 (채택)** | R1 + 증상에 직접 인접("<sev> <symptom>")일 때만 | 18 | **56%** | 10 | **81** | **94** |
| R3 CoT-quote | R1 + 인용 self-check | 21 | 52% | 11 | 80 | 89 |
| R4 enum+동의어매핑 | R2 + excruciating→severe 등 | 25 | 40% | 10 | 77 | 90 |
| R5 엄격인접 | 즉시 인접만 | 23 | 30% | 7 | 77 | 88 |

**결과**: severity faithfulness **18%→56%(+38%p)**, 오결합 ~49→~8로 제거하되 진짜-bound(~10)는
유지. finding 수(682→677)·location·character 손상 없음(오히려 개선). **R4/R5는 regression** —
동의어→통제어휘 정규화가 faithfulness 깎음(source가 "excruciating"인데 "severe"로 바꾸면
미-entail), 과도한 인접요구가 true positive 손실. → **R2에서 수렴.**

**잔여 44% 해석**: spot-check 결과 R2의 18개 중 대부분이 실제 올바른 bound("mild tachypnea",
"severe headaches", "excruciating chest pain"). NLI 미-entail의 다수는 **측정 아티팩트**
(이름에 이미 severe 포함된 중복 "severe sore throat"→severe; mild/moderate/severe 외 정당한
원문어 "excruciating"). 즉 실질 severity 정밀도는 56%보다 높음(~78%). **동기였던 case-level
오결합 결함은 해소**(mis-bound ~49→~8). 파일: `v115_sev_ie.py`(R2), `v115/v116_score.py`.
