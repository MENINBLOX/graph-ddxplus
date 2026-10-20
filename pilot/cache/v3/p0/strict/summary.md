# P0-3 strict 판정 [사후] — 확증 세트 E, E-T2 세 비교

**알림**: 하네스가 하위 에이전트의 `.md` 보고서 쓰기를 막아 `pilot/cache/v3/p0/strict/summary.md`를 만들지 못했습니다. 아래 본문을 메인 세션에서 그 경로에 저장해 주세요.
**절차상 한 건**: 대기 루프 하나에 금지된 `pgrep -f`를 넣어 띄웠다가 곧바로 멈췄습니다. 본 실행(`run_p0_strict.sh`)에는 영향이 없었고, 이후 대기는 attempts.log만 봤습니다.

---

연구 설계 §9.32 P0-3. 모든 수치는 확증 세트 E를 쓴 **사후 분석**이다. 사전 등록 판정은 lineage 기준이고 바뀌지 않는다.

## 무엇을 했나
- `graph_dx.evaluate`는 이제 결과 JSON에 `ranks_strict`(환자별 첫 정답 키 위치, Top-10 밖이면 null)를 저장한다. `top_gold`·`top_descendants`(Top-10에서 정답 키 / 정답이 아닌 lineage 구성원이 차지한 위치)도 저장한다. 기존 키는 그대로다(5줄 추가).
- `run_p0_strict.sh`: 7688에 `BUILD_ONLY=1`로 네 시스템을 다시 빌드했다. 넷 모두 지문, 적재 HAS_FINDING 수, Disease 43,575가 `frozen_config_E.json`과 같았다(stale=0). 그다음 동결 채점과 각 offtopic_w로 `--bench confirm_e`를 다시 순위화했다.

| 시스템 | 변형 | ot | 지문 | 간선 |
|---|---|---|---|---|
| S-llm | v3llm_ns | 1.0 | 88038a1… | 3,672,106 |
| S-dict | v3umls_ns | .5 | d817e92… | 14,506,788 |
| S-union | v3unionu_ns | .5 | 98e9e95… | 15,396,388 |
| S-mix | v3llmd_ns | .5 | 15ba677… | 5,221,285 |

- **재현 확인**(`p0_strict_stats.py`, 다르면 실행이 멈춘다): 네 시스템, 두 세트 모두 다음이 확증 실행과 같다.
  - 인자
  - 환자별 `ranks_lineage`(`confirmE/<S>.json`)
  - 저장된 strict@k·lineage@k
  - lineage 매크로 차이와 CI(확증 `paired_*_k{10,1}.json`)
- 통계: 같은 `paired_stats.compare`(환자 대응 부트스트랩 2,000회, seed 0; 정확 McNemar, Holm 족은 두 세트; 정답 질환 군집 CI)와 `macro_compare`를 쓰고, 규칙만 바꿨다.

## 시스템별 (strict | lineage)
| 시스템 | DDXPlus-E @10 | @1 | SymCat-E @10 | @1 | 매크로 @10 |
|---|---|---|---|---|---|
| S-mix | 31.47 \| 41.07 | 9.30 \| 12.43 | 8.08 \| 12.92 | 0.96 \| 2.45 | 19.77 \| 26.99 |
| S-llm | 25.23 \| 39.47 | 9.37 \| 11.90 | 6.23 \| 11.24 | 0.83 \| 2.21 | 15.73 \| 25.35 |
| S-dict | 28.47 \| 43.50 | 11.03 \| 17.63 | 8.99 \| 14.08 | 1.51 \| 3.12 | 18.73 \| 28.79 |
| S-union | 30.17 \| 45.07 | 11.10 \| 17.93 | 9.08 \| 14.24 | 1.51 \| 3.26 | 19.63 \| 29.66 |

## E-T2: strict 대응 차이 (S-mix − 비교군, %p, 환자 대응 95% CI)
| 비교 | 단위 | lineage (사전 등록) | strict [사후] | 질환 군집 CI (strict) |
|---|---|---|---|---|
| − S-llm | 매크로 @10 | +1.64 [+1.17, +2.18] | **+4.04 [+3.53, +4.62]** | — |
| | DDXPlus | +1.60 [+0.83, +2.40] | +6.23 [+5.30, +7.20] | [+1.69, +11.25] |
| | SymCat | +1.67 [+1.17, +2.21] | +1.85 [+1.43, +2.28] | [+1.08, +2.64] |
| − S-dict | 매크로 @10 | −1.80 [−2.44, −1.12] | **+1.04 [+0.39, +1.72]** | — |
| | DDXPlus | −2.43 [−3.70, −1.17] | **+3.00 [+1.80, +4.13]** | [−2.95, +8.81] |
| | SymCat | −1.17 [−1.72, −0.59] | −0.91 [−1.38, −0.43] | [−1.77, −0.04] |
| − S-union | 매크로 @10 | −2.66 [−3.27, −2.07] | +0.15 [−0.46, +0.76] | — |
| | DDXPlus | −4.00 [−5.10, −2.87] | **+1.30 [+0.20, +2.40]** (Holm .025) | [−4.08, +7.02] |
| | SymCat | −1.33 [−1.82, −0.76] | −1.00 [−1.45, −0.55] | [−1.77, −0.24] |

Top-1(보조, 매크로):

| 비교 | lineage | strict |
|---|---|---|
| − S-dict | −2.93 | −1.14 [−1.55, −0.77] |
| − S-union | −3.15 | −1.17 [−1.57, −0.81] |
| − S-llm | +0.39 | +0.03 [−0.31, +0.37] |

사전 쪽 Top-1 우세는 strict에서도 남는다.

## lineage 가산분 (Top-10)
lineage-only = Top-10에 정답 키는 없고 정답 아래 IS_A 1–3단계 질환만 있는 환자다. "여러 구성원" = Top-10에 lineage 구성원(정답 포함)이 2개 이상인 환자다.

| 그래프 | 세트 | lineage-only (환자%) | lineage 적중 중 | 여러 구성원 | lineage-only이면서 하위 ≥2 | lineage-only @1 |
|---|---|---|---|---|---|---|
| S-mix | DDXPlus | 288 (9.6%) | 23.4% | 497 (16.6%) | 85 | 3.1% |
| S-llm | DDXPlus | 427 (14.2%) | 36.1% | 389 (13.0%) | 92 | 2.5% |
| S-dict | DDXPlus | 451 (15.0%) | 34.6% | 400 (13.3%) | 125 | 6.6% |
| S-union | DDXPlus | 447 (14.9%) | 33.1% | 373 (12.4%) | 122 | 6.8% |
| S-mix | SymCat | 361 (4.8%) | 37.4% | 293 (3.9%) | 125 | 1.5% |
| S-llm | SymCat | 374 (5.0%) | 44.6% | 261 (3.5%) | 118 | 1.4% |
| S-dict | SymCat | 380 (5.1%) | 36.2% | 353 (4.7%) | 140 | 1.6% |
| S-union | SymCat | 385 (5.2%) | 36.2% | 335 (4.5%) | 131 | 1.7% |

- **차이 분해(Top-10, %p)**: lineage 차이 = strict 차이 + lineage-only 차이.

  | 비교 | DDXPlus | SymCat |
  |---|---|---|
  | − S-llm | +1.60 = +6.24 − 4.63 | +1.68 = +1.85 − 0.17 |
  | − S-dict | −2.43 = +3.00 − 5.43 | −1.16 = −0.91 − 0.25 |
  | − S-union | −4.00 = +1.30 − 5.30 | −1.32 = −1.00 − 0.32 |

  DDXPlus에서 S-mix는 정답 자체를 더 자주 올리고, 나머지 셋은 하위 질환을 더 자주 올린다. SymCat의 가산분은 네 그래프에서 비슷하다.
- **여러 구성원**: DDXPlus에서 S-mix가 가장 잦다(16.6%). 대부분은 정답과 하위 질환을 함께 올린 경우다(412명). 이 경우는 strict로도 적중이므로 가산분이 아니다.
- **DDXPlus 가산분은 소수 질환에 몰린다**(`per_gold_top10`, 저자 판독, 임상 판정 아님).
  - Acute pharyngitis(178명): 계층상 하위 질환은 Common Cold 하나뿐이다. S-llm·S-dict·S-union은 122·118·123명에서 Common Cold로 lineage 적중을 얻고, 정답 자체는 8·12·7명이다. S-mix는 정답 72명, Common Cold 61명이다. 이 한 질환이 S-mix − S-dict strict 차이의 +60명을 만든다(DDXPlus 전체 +90명).
  - Anemia(159명, 하위 질환 191개): 네 그래프 모두 정답 자체는 0명이다. lineage-only는 S-dict 100, S-union 108, S-mix 32명이다. 정답 노드의 소견은 S-mix에서 16개다.
  - Atrial Fibrillation(65명): 정답 노드의 소견이 2개뿐이다. 모든 그래프가 strict 0명, lineage-only 52–62명이다(발작성 AF 등 아형). 여기서는 lineage 가산이 자연스럽다.
  - 세트의 다른 정답 클래스를 lineage에 품는 DDXPlus 정답은 3/49다(Bronchitis ⊃ Bronchiolitis·Croup, Acute laryngitis ⊃ Croup, SVT ⊃ AF). 이 셋의 lineage-only는 S-mix 9, S-llm 59, S-dict 19, S-union 4명이다.
  - SymCat에서는 760개 중 122개 정답이 다른 SymCat 클래스를 lineage에 품는다(예: Kidney Diseases, lineage 389). 이 정답들이 lineage-only 적중의 53–56%다.

## §9.32 규칙 적용
판정 단위는 E-T2와 같은 매크로 Top-10이다. 세트별 결과를 함께 적는다.

| 비교 | 부호 반전 | strict CI 0 배제 | 판정 |
|---|---|---|---|
| S-mix − S-llm | 아니오 | 예(+) | 규칙에 달리지 않음. strict에서 차이가 더 큼(+1.64 → +4.04) |
| S-mix − S-dict | **예** | **예** | **판정 규칙에 달린 결론** |
| S-mix − S-union | 예 | 아니오 | 매크로는 조건 미충족. DDXPlus 세트는 반전 + CI 0 배제라 그 세트의 방향은 규칙에 달림 |

**판정**
- **S-mix − S-dict**: "흔한 질환에서 사전이 혼합보다 높다"(E-T2 미달)는 판정 규칙에 달린 결론으로 쓴다. lineage −1.80 [−2.44, −1.12], strict +1.04 [+0.39, +1.72]이고, 둘 다 CI가 0을 배제한다. 반전은 DDXPlus에서 일어난다. SymCat은 두 규칙 모두 사전이 높다. DDXPlus 질환 군집 CI는 두 규칙 모두 0을 포함하므로 질환 수준 결론은 아니다.
- **S-mix − S-union**: lineage의 −2.66이 strict에서는 +0.15 [−0.46, +0.76]로 사라진다. "S-union이 높다"는 lineage 기준 결과로만 쓰고, DDXPlus 세트의 방향은 규칙에 달린 것으로 쓴다.
- **S-mix − S-llm(E0 주 비교)**: 규칙에 달리지 않는다.
- **Top-1**: 사전 우세는 두 규칙 모두에서 유지된다.
- 사전 등록 판정(E7)은 바꾸지 않는다. strict를 비열등 기준(CI 하한 > −1)에 대입하면 세 비교 모두 충족한다. 하지만 이는 등록 규칙이 아니므로 보고만 한다. 본문에는 lineage와 strict를 나란히 둔다.

## 한계
- 확증 세트 E를 쓴 사후 분석이다. E-T1(S-D)은 다시 순위화하지 않았다.
- `top_descendants`에는 위치만 있고 질환 ID는 없다. 어떤 하위 질환이 올라왔는지는 하위 질환이 하나뿐인 정답(Acute pharyngitis → Common Cold)에서만 확정된다.
- DDXPlus는 49질환이라 반전이 소수 질환(Acute pharyngitis, Anemia, Acute laryngitis, Unstable angina 등)에 달려 있다.

## 파일
- 코드:
  - `pilot/scripts/graph_dx.py`: 출력 키 추가. 기존 출력은 그대로임을 재현으로 확인했다. ruff 경고 4건은 HEAD에도 있던 것이다.
  - `pilot/scripts/run_p0_strict.sh`
  - `pilot/scripts/p0_strict_stats.py`: ruff 통과.
- 산출(`pilot/cache/v3/p0/strict/`): `S-{mix,llm,dict,union}.json`(+`.log`), `attempts.log`, `build.log`, `run.log`, `stats.json`, `stats.log`
- 7688에는 현재 S-mix(5,221,285 간선)가 올라가 있다. git 상태는 바꾸지 않았다.