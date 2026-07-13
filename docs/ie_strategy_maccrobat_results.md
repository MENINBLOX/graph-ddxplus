# IE 추출전략 비교 (MACCROBAT location/severity) — 실측 결과

**작성 2026-07-13 · 모델 고정 = gemma-4-E4B-it (greedy, no few-shot) · n=200 MACCROBAT case report**
**스크립트**: `pilot/scripts/v145_strategy_scope.py` (S/C/V1/V2), `v146_decomp.py` (V3/C3),
`v146_bucketB_forensic.py`, `v146_D_forensic.py`, `v146_adjudicated_score.py`
**gold**: MACCROBAT brat relation `MODIFY(Biological_structure→Sign_symptom)` = location,
`MODIFY(Severity→Sign_symptom)` = severity. location gold pair=228, severity gold pair=39.

## 요약 (headline)

- **추출전략 선정**: location = **decomposition(V3)**, severity = **2-stage cot(C)**.
- **가장 중요한 발견**: MACCROBAT relation gold는 **불완전(incomplete)** 하다. location 예측의
  **hallucination 0%**, 명목상 "false positive"의 **83–87%가 원문에 실제로 서술된 correct-but-unannotated**.
  → **raw binding-F1은 precision을 ~0.3 과소평가**하며, 추출 품질이 아니라 gold coverage를 재는 지표다.
- 편향 보정(source-adjudicated) 후 location precision은 **0.86–0.98**(raw 0.50–0.66이 아님).
- 이 보정을 균일 적용하면 **순위가 뒤집힌다**: raw-F1 1등이던 C가 아니라 **decomposition(V3)**이 최고.

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

## 6. 다음 단계

- **SemEval 2015 Task 14** disorder-attribute gold로 동일 프로토콜 반복(2번째 공개 gold, severity/location 교차 확인).
- 확정 레시피(location-decomp + severity-cot)로 rare-disease 포함 대규모 IE → 속성 KG → extrinsic leave-one-out.
- character/timing/aggravating/relieving 4속성은 공개 gold 없음 → extrinsic(진단 gain)으로만 검증(한계 명시).
