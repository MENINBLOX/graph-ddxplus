# IE 추출전략·모델·평가 — MACCROBAT location/severity 실측 결과

**작성 2026-07-13, 갱신 2026-07-14 · n=200 MACCROBAT case report · greedy, no few-shot**
**스크립트**: `v145_strategy_scope.py`(전략 S/C/V1/V2), `v146_decomp.py`(V3/C3), `v147_single6.py`(단일프롬프트 V4),
`v148_model_panel.py`(모델패널), `v149_broadscope.py`(V5 scope test), `v150_conditioned_eval.py`(교정 평가),
포렌식 `v146_bucketB_forensic.py`·`v146_D_forensic.py`·`v147_A_forensic.py`, 채점 `v146_adjudicated_score.py`·`v147_adjudicate.py`
**gold**: MACCROBAT brat `MODIFY(Biological_structure→Sign_symptom)`=location, `MODIFY(Severity→Sign_symptom)`=severity.

## ⚠️ 데이터 무결성 정정 (2026-07-14 저녁) — 아래 §13이 최종

**§1~12는 깨진 brat 추출(200 중 33문서만 텍스트, gold 30/18문서)에서 나온 수치다.** zip은 온전했으나 압축해제가 33개만 채웠음. **재추출 후 전체 corpus(location 1539쌍/193문서, severity 276쌍/121문서 = 6~7배)로 재실행 → §13.** raw pair-F1은 소·대 표본에서 안정적(location 0.50, severity 0.57)이었고, **교정 지표도 유지**(§13): location 배정정확도 0.88(n=563), severity 0.96(n=149, 소표본 "1.00"은 표본편향이었음). gold-free CUI grounding도 유지(halluc 0%, 정규화 0.84). §1~12 수치는 방법론 기록으로 보존하되, **인용은 §13 전체-corpus 수치를 쓸 것.**

## 요약 (최종 headline — 2026-07-14)

- **평가가 문제였다.** raw pair-level binding-F1(location 0.54 / severity 0.58)은 세 결함으로 좋은 추출을 과소평가했다:
  (1) gold 불완전 → precision 과소, (2) multi-site 중복(32%)·노이즈 finding(17%) → recall 과소, (3) finding-추출과 속성-배정을 혼합.
- **표준 평가(SemEval-style, finding에 conditioned)로 재면 추출은 강하다** — gemma-4-E4B:
  **location accuracy 0.85 / severity accuracy 1.00**, hallucination 0%. (속성을 배정하면 그만큼 맞다 = 우리 주장.)
- **모델 선정**: 6모델 패널에서 **gemma-4-E4B(최소 모델)가 최고** (coverage×accuracy). 더 크거나 의료특화 모델이 못 이김 → **8B 비용효율** 강화.
- **추출전략**: **단일 프롬프트 V4**(1회 호출, 6속성 동시, location 원자화 + severity 보수적)로 확정 — 순차/2-stage 불필요.
- 반증된 가설들: 인접성 완화(V1/V2)·scope 확대(V5)·모델 확대 전부 raw F1 못 올림 → 병목은 프롬프트도 모델도 아닌 **평가**였다.

<details><summary>이전 headline (2026-07-13, source-adjudicated 단계 — 아래 상세 보존)</summary>

- 추출전략 선정: location = decomposition(V3), severity = 2-stage cot(C).
- MACCROBAT relation gold 불완전: location halluc 0%, 명목 FP의 83–87%가 correct-but-unannotated. raw binding-F1이 precision을 ~0.3 과소평가.
- source-adjudicated 후 location precision 0.86–0.98. 균일 적용 시 순위 역전(C→V3).
</details>

## 1. 6-strategy raw 결과

전략: **S** single baseline / **C** 2-stage cot / **V1** sentence-scope(인접성 완화) /
**V2** quote-then-normalize / **V3** single + 융합 pre-modifier 분해 / **C3** 2-stage + 분해.
버킷: **A** finding 미추출 / **B** finding 추출·attr EMPTY / **C** binding 맞고 value 오류 / **D** precision FP.

### LOCATION (gold pair=228)
| strat | pred | binding F1 (P/R) | value F1 | A | B | C | D/pred |
|-------|-----:|:----------------:|:--------:|--:|--:|--:|-------:|
| S  | 155 | 0.53 (0.66/0.45) | 0.46 | 52 | 74 | 16 | 47/155 |
| C  | 160 | 0.54 (0.66/0.46) | 0.50 | 38 | 85 | 12 | 50/160 |
| V1 | 185 | 0.53 (0.59/0.48) | 0.46 | 53 | 65 | 19 | 69/185 |
| V2 | 225 | 0.49 (0.50/0.49) | 0.38 | 59 | 57 | 31 | 108/225 |
| **V3** | 214 | 0.52 (0.54/0.50) | 0.46 | 71 | **42** | 20 | 85/214 |
| C3 | 195 | 0.51 (0.55/0.47) | 0.46 | 53 | 68 | 14 | 84/195 |

### SEVERITY (gold pair=39)
| strat | pred | binding F1 (P/R) | value F1 | A | B | C | D/pred |
|-------|-----:|:----------------:|:--------:|--:|--:|--:|-------:|
| S  | 46 | 0.61 (0.57/0.67) | 0.61 | 3 | 10 | 0 | 17/46 |
| **C**  | 40 | **0.68** (0.68/0.69) | **0.68** | 3 | 9 | 0 | 10/40 |
| V1 | 44 | 0.65 (0.61/0.69) | 0.65 | 2 | 10 | 0 | 15/44 |
| V2 | 38 | 0.55 (0.55/0.54) | 0.44 | 4 | 14 | 4 | 15/38 |
| V3 | 60 | 0.46 (0.38/0.59) | 0.46 | 7 | 9 | 0 | 33/60 |
| C3 | 47 | 0.56 (0.51/0.62) | 0.56 | 4 | 11 | 0 | 22/47 |

## 2. 버킷 B 포렌식 — 지배적 손실의 원인

winner C의 location bucket-B(추출했으나 location EMPTY) 53건을 원문과 대조:
**77%가 SAME-SENTENCE** (gold location이 finding과 같은 문장에 있음) = 지역적으로 회복 가능한 miss.
예측을 보면 원인이 명확: 해부 부위가 **finding 이름에 pre-modifier로 융합**되어 location이 빈다.

```
'abdominal pain'  loc=''      # abdominal → location 이어야
'vaginal bleeding' loc=''     # vaginal
'leg edema' loc=''            # leg
'severely dilated left ventricle (lv)' loc=''   # left ventricle
'pulmonary edema' loc=''      # pulmonary
```

→ V1/V2(문장 전체 탐색)는 **더 멀리** 탐색해 distant site를 잡아 precision이 무너졌다.
실제 miss는 finding 구절 **안**에 있었으므로 올바른 lever는 **원자화(atomization)**: 융합 modifier를
name에서 떼어 attribute로 보낸다(프로젝트 P2 원자화 원칙과 동일, MACCROBAT `MODIFY` gold가 정확히
이 (site→finding) 링크를 주석). 이것이 V3/C3.

## 3. D 버킷 포렌식 — raw precision은 gold 불완전성 artifact

decomposition(V3)은 bucket B를 74→42로 절반으로 줄였으나 raw D(FP)가 47→85로 늘어 raw F1이 정체.
그러나 D의 실체를 원문 대조로 분해하면:

| strat | D total | STATED-IN-SRC (correct) | SITE-ELSEWHERE | NOT-IN-SRC (halluc) |
|-------|--------:|:-----------------------:|:--------------:|:-------------------:|
| S  |  47 | 83% | 17% | **0%** |
| C  |  50 | 84% | 16% | **0%** |
| V2 | 108 | 75% | 25% | **0%** |
| **V3** | 85 | **87%** | 13% | **0%** |
| C3 | 84 | 86% | 14% | **0%** |

**전 전략 hallucination 0%.** "FP"의 83–87%는 원문에 서술된 correct 추출인데 sparse relation gold가
주석하지 않았을 뿐(예: `stenosis→aortic valve`, `abdominal distension→abdominal`, `splenomegaly→spleen`).
따라서 **raw binding-F1의 precision은 추출 품질이 아니라 gold coverage**를 잰다.

## 4. Source-adjudicated 결과 (편향 보정, 균일 적용)

precision 판정 시 예측을 correct로 인정: **gold pair 매칭 OR STATED-IN-SRC**(finding 자기 문장에 location 서술).
recall은 gold-bound 유지(sparse gold가 상한). 전 전략 동일 규칙 → per-strategy tuning 아님.

### LOCATION adjudicated (gold pair=228)
| strat | rawF1 | adjP | adjP+ | **adjF1** | adjF1+ | bucket B |
|-------|:-----:|:----:|:-----:|:---------:|:------:|---------:|
| S  | 0.53 | 0.91 | 0.96 | 0.60 | 0.61 | 74 |
| C  | 0.54 | 0.92 | 0.97 | 0.61 | 0.62 | 85 |
| V1 | 0.53 | 0.92 | 0.97 | 0.63 | 0.64 | 65 |
| V2 | 0.49 | 0.86 | 0.98 | 0.62 | 0.65 | 57 |
| **V3** | 0.52 | 0.88 | 0.93 | **0.64** | **0.66** | **42** |
| C3 | 0.51 | 0.92 | 0.98 | 0.62 | 0.63 | 68 |

`adjP` = +STATED-IN-SRC, `adjP+` = +SITE-ELSEWHERE, `adjF1` = harmonic(adjP, gold recall).

- **true precision 0.86–0.98** (raw 0.50–0.66 아님).
- **adjudicated F1: V3 최고(0.64, lenient 0.66)** → raw-F1 순위 역전. C는 location을 더 적게 뽑아 recall이 낮다.
- baseline 대비 location 정직 개선: **raw 0.53 → adjudicated 0.64** (= 측정 보정 +0.07, decomposition +0.04).

## 5. 결론 · 논문 반영

- **추출전략**: location = **decomposition(V3)**, severity = **conservative 2-stage cot(C)**.
  (severity는 분해가 과다 트리거되어 손해 → 보수적 규칙 유지. 최적 레시피 = location-분해 + severity-보수.)
- **측정 방법론(논문 기여)**: MACCROBAT relation gold 불완전성 때문에 **raw F1 단독 보고는 추출 품질을
  과소평가**한다. intrinsic 검증은 **recall(gold-bound) + faithfulness(hallucination 0%) + source-adjudicated
  precision** 3종으로 보고해야 정직하다. (교수님 자문의 "gold 새로 안 만든다"와 정합: 기존 gold를
  원문 대조로 보정할 뿐 새 주석 생성 아님.)
- **한계 명시**: recall은 sparse gold가 상한(진짜 recall은 더 높을 수 있음). adjP의 STATED-IN-SRC 판정은
  문장 단위 근접 heuristic(느슨한 상한). location/severity 2속성만 공개 gold 존재.

## 6. (구) 다음 단계 — 아래 7~11에서 실제로 수행·갱신됨

---

# === 2026-07-14 갱신: 단일 프롬프트 · 모델 패널 · 평가 교정 ===

## 7. 단일 프롬프트 아키텍처 V4 (`v147_single6.py`)

**1회 호출/문서, 6속성 동시** (내부 STEP1 원자 나열 → STEP2 속성 배정; 여러 호출/속성별 순차 아님).
데이터 기반 확정: **location=원자화 ON**(V3 win), **severity=보수적**(분해 OFF — V3의 severity 과다트리거 회피).
onset/character + timing/aggravating/relieving까지 같은 프롬프트에 포함(뒤 4속성은 공개 gold 없음→extrinsic 전용).

source-adjudicated: **location F1 0.68**(전략 통틀어 1등), severity 0.68(2-stage cot 0.74와 39쌍 노이즈 내).
→ "6속성 동시 주의" 간섭손실 = location 0, severity 노이즈. **순차·전용프롬프트 불필요.**

## 8. 모델 패널 (`v148_model_panel.py`, V4 프롬프트 고정)

| 모델 | src-adj location F1 | (참고) |
|------|:---:|---|
| **gemma-4-E4B** (bf16) | **0.68** | 최고 |
| gemma-4-12B (4bit) | 0.51 | 24GB 미달로 4bit |
| Qwen2.5-14B (4bit) | 0.46 | recall 낮음 |
| Qwen2.5-7B (bf16) | 0.49 | recall 낮음 |
| medgemma-4b (bf16) | 0.52 | 의료특화도 못 이김 |

차이는 전부 **recall**(gemma-4-E4B 0.55 vs 나머지 0.31–0.41), precision 비슷. **모델 레버 소진 → 8B 비용효율 강화.**
(BioMistral/Med42/OpenBio는 HF xet 401 gated, 토큰으로 medgemma만 확보. Llama gated.)

## 9. "낮은 점수" 원인 규명 — prompt도 model도 아니고 **평가**

- **prompt scope 가설 반증** (`v149_broadscope.py`, V5): 병리/영상 finding 포함하도록 STEP1 확대 → recall **하락**(0.55→0.50), bucket A **증가**(65→70). scope 넓혀도 안 오름.
- **finding 미추출(bucket A) 65건 분석**(`v147_A_forensic.py`): 98%가 환자증상 아님(necrosis/metastatic lesions/cholestasis) — 그러나 이는 corpus라기보다 **gold 구조** 문제로 판명(아래).
- **gold 구조 결함 정량화**:
  - **multi-site 중복 32%**: 한 finding이 N장기에 링크 → gold N쌍, 1추출=1매칭(예: `mtdna depletion`→colon/stomach/ileum/liver 4쌍).
  - **노이즈 finding 17%**: gold가 `normal`, `remarkable changes`를 finding으로 주석.
  - pair-level recall 0.55 → **finding-level(중복 제거) recall 0.68**.

## 10. 교정된 표준 평가 (`v150_conditioned_eval.py`) — SemEval-Task14 style

속성을 **finding에 conditioned**해 채점(중복=사이트 SET 중 하나면 정답, 노이즈 제외, finding-추출/속성-배정 분리 보고).

### conditioned attribute accuracy (n=200)
| 모델 | LOCATION acc | LOCATION cov | SEVERITY acc | SEVERITY cov |
|------|:---:|:---:|:---:|:---:|
| **gemma-4-E4B** | **0.85** | 0.68 | **1.00** | 0.91 |
| gemma-4-12B | 0.91 | 0.55 | 1.00 | 0.63 |
| Qwen2.5-7B | 0.86 | 0.51 | 1.00 | 0.60 |
| medgemma-4b | 0.79 | 0.66 | 0.92 | 0.63 |

- **accuracy** = 속성을 배정했을 때 gold와 맞는 비율 = 우리 주장("표준 속성 추출"). **location 0.85 / severity 1.00.**
- gemma-4-E4B가 coverage 최고 → end-to-end(correct/all-gold) 최고(location 0.53, severity 0.80). gemma-12B는 acc 약간↑지만 coverage↓.
- location 0.85는 **보수적 하한**: miss 상당수가 granularity 차이(`nodules` lung↔subpleural, `dilation` intestinal↔bowel wall)로 임상적으로 맞음. anat-normalization으론 gemma-4-E4B 불변(그 miss들은 string 변형 아님).

## 11. 결론 · 논문 반영 (갱신)

- **모델 = gemma-4-E4B**, **전략 = 단일 프롬프트 V4**(location 원자화 + severity 보수적).
- **intrinsic 지표 = conditioned attribute accuracy(location 0.85 / severity 1.00) + coverage(별도) + faithfulness(halluc 0%)**. raw pair-F1은 gold 구조(중복·노이즈·불완전)로 하한이므로 단독 보고 금지 — 이 측정-타당성 분석 자체가 기여.
- 교수님 "gold 새로 안 만든다"와 정합: 기존 gold를 표준 방식(conditioned)으로 올바르게 채점할 뿐 새 주석 생성 아님.
- **한계**: coverage(finding-추출)는 MACCROBAT이 radiology/pathology라 상한. accuracy 매칭은 string/anat 기반(full UMLS-CUI 정규화가 최종 표준, 파이프라인 downstream에서 사용). location/severity 2속성만 공개 gold.

## 12. 다음 단계

- **SemEval-2015 Task 14**(구조 깨끗한 2번째 gold)로 conditioned accuracy 교차검증 — multi-site·노이즈 없는 gold에서 raw F1이 오르면 artifact 가설 확정. (ShARe DUA 등록 필요.)
- full UMLS-CUI 매칭으로 location accuracy 최종 확정(scispaCy; nmslib/py3.12 이슈 주의).
- 확정(gemma-4-E4B + V4)으로 rare-disease 포함 대규모 IE → 속성 KG → extrinsic leave-one-out.
- character/timing/aggravating/relieving 4속성은 공개 gold 없음 → extrinsic(진단 gain)으로만 검증(한계 명시).

---

# === §13. 전체-corpus 재실행 + gold-free CUI grounding (2026-07-14, 최종) ===

**데이터 버그**: `brat/` 추출이 200 중 33문서만 채움(나머지 빈 파일). zip(200 txt+200 ann) 온전 → 재추출.
전체 gold: **location 1539쌍 / 991 unique finding / 193문서**, **severity 276쌍 / 243 finding / 121문서** (기존 30/18문서의 6~7배).

## 13.1 교정 conditioned eval — 전체 corpus (gemma-4-E4B, V4)
| attribute | coverage | **assignment accuracy** | end-to-end | n(filled) |
|-----------|:---:|:---:|:---:|:---:|
| **location** | 0.65 | **0.88** | 0.50 | 563 |
| **severity** | 0.81 | **0.96** | 0.59 | 149 |

- 소표본(§10) location 0.85 → 전체 **0.88**(n=563), severity **1.00→0.96**(n=149). **"1.00"은 소표본(n=28) 편향이었고 진짜는 0.96.** 두 지표 모두 큰 표본에서 유지.
- raw pair-F1도 안정: location 0.50(0.54/0.47), severity 0.57(0.54/0.60).

## 13.2 gold-free CUI-grounding 검증 (`v151_cui_grounding.py`, 전체 corpus)
IE 결과물·원문을 각각 scispaCy UMLS CUI로 정규화 후 **개념(CUI) 단위** 비교. gold 불필요, 결정론적(NLI 아님).

**개념형 (UMLS CUI)**
| field | n | 정규화-cov | grounding | hallucination |
|-------|:---:|:---:|:---:|:---:|
| finding | 2903 | 0.84 | **1.00** | **0.00** |
| location | 1348 | 0.83 | **1.00** | **0.00** |
| character | 401 | 0.61 | 1.00 | 0.00 |

**범주형 (통제어휘; grounding=원문 동의어 존재)**
| field | n | vocab-conf | grounded |
|-------|:---:|:---:|:---:|
| severity | 348 | 0.54 | 1.00 |
| onset | 371 | 1.00 | **0.52** |

- **hallucination 0% + grounding 100%** (finding/location/character, n=2903/1348/401) → 의미 수준에서도 지어내지 않음. **핵심 IE 검증 카드.**
- **정규화 커버리지 finding 0.84 / location 0.83** → KG interlingua(UMLS/SNOMED)로 바로 쓸 수 있는 표준 개념.
- **character 0.61** → 성상어 free-text 결정 정량 확인.
- **severity vocab 0.54** → 모델이 severity를 {mild/moderate/severe}로 정규화 안 하고 raw 복사(serious/strong/partial)가 ~46%. faithful하나 미정규화. (∴ §10의 severity accuracy는 "복사 충실도"이지 정규화 아님 — 정직 caveat.)
- **onset grounded 0.52** → 유일한 약점. sudden/gradual를 원문 onset 단서 없이 배정 ~48%(과다배정 의심). onset은 비채점 속성(timing 하위)이라 치명적 아니나 명시.

## 13.3 IE 검증 최종 정리 (gold 소규모 문제를 gold-free로 우회)
- **faithfulness (gold-free, 강함)**: 의미-level hallucination 0%, grounding 100%.
- **normalizability (gold-free)**: finding/location UMLS 링크 84%/83% → KG-ready. character 61%(free-text). severity vocab 54%(정규화 갭).
- **배정 정확도 (공개 gold, sanity)**: location 0.88, severity 0.96 (전체 corpus, 큰 표본).
- **모델·전략 선정**: gemma-4-E4B + V4 (상대비교, §7~10). ※모델패널·전략비교는 33문서 위였으므로 재확인은 후속(상대순위는 유지 예상).
- **한계**: 공개 gold(MACCROBAT relation)는 multi-site 중복·노이즈·불완전으로 raw pair-F1 부적합 → conditioned accuracy + gold-free grounding으로 보고. 진짜 검증은 extrinsic(진단 gain).

## 13.4 다음
- severity 정규화 강제(raw→{mild/moderate/severe} 매핑) 프롬프트 개선 → vocab-conf 0.54 회복 시도.
- onset 과다배정 점검(grounded 0.52).
- 확정(gemma-4-E4B+V4)으로 대규모 IE → 속성 KG → extrinsic leave-one-out.

---

# === §14. IE 검증 = gold-free CUI/SNOMED grounding으로 확정 (2026-07-14) ===

**결정(사용자)**: IE 검증에서 **공개 벤치마크(MACCROBAT gold) 사용 중단**. 이유: 소규모(재추출 후에도 코퍼스 특성)·구조결함(multi-site/노이즈/불완전)·IE 비최적화 코퍼스. gold-free CUI-grounding이 IE 검증의 전부.

**파이프라인 `v152_normalize_pipeline.py`** (별도 `.venv-sci`): IE 결과물·원문을 scispaCy UMLS로 정규화 후 개념(CUI) 수준 비교 + CUI→SNOMED(MRCONSO 452K) + semantic-type(MRSTY) 검증. **IE 검증 = KG 정규화 단계** 동시 수행. 출력 = KG-ready 레코드(`v152_normalized_full.json`).

### 전체 corpus (gemma-4-E4B, V4) — 개념형
| field | n | CUI-link | SNOMED | clin-type | grounding | halluc |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| finding | 2903 | 0.84 | 0.76 | **0.59** | **1.00** | **0.00** |
| location | 1348 | 0.83 | 0.74 | 0.64 | 1.00 | 0.00 |
| character | 401 | 0.61 | 0.49 | 0.36 | 1.00 | 0.00 |

범주형: severity vocab-conf 0.54 / grounded 1.00, onset vocab 1.00 / grounded 0.50.

- **grounding 1.00 / hallucination 0.00** (n=2903/1348) = 의미 수준 충실성. gold 무관, 결정론적(NLI 아님).
- **clin-type = 진짜 KG-usable 하한**(finding 0.59): CUI-link(0.84)와의 격차 25%p는 anatomy-mislink(예 "Ebstein's anomaly"→"tricuspid valve" C0040960/T023)·qualifier 등 노이즈 → semantic-type 필터가 자동 배제(§12 정합). finding 타입은 disease/symptom/finding만 인정(anatomy 제외).
- **SNOMED 0.76/0.74** = KG 타깃 용어 코드까지 매핑.

### 한계 (정직)
- gold-free는 **faithfulness(원문에 있나)·normalization coverage(어떤 CUI에 링크되나)는 재나, CUI-correctness(맞는 CUI인가)는 못 잼.** scispaCy linking SOTA도 MedMentions ~56%. → right-CUI 검증은 gold 또는 **소표본 수동감사** 필요(선행연구 다수가 병기; 상세 [[ie-validation-priorwork-2026-07]]).
- 모델선정: gold-free는 faithfulness가 모든 모델 高(halluc ≤2%)라 accuracy 변별력 약함 → 모델 accuracy 최종선정은 extrinsic 몫. gemma-4-E4B 기본값(halluc 0%, 최고 volume).

### 검증 스토리 최종
faithfulness(gold-free, halluc 0%/grounding 100%) + normalizability(CUI/SNOMED, clin-type 0.59) + [보강예정: 소표본 수동감사] + **extrinsic 진단 gain(main)**. 공개 벤치마크 F1은 미사용.
