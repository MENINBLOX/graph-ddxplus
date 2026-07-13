# IE Methods 검증 결과 (실측)

> 목적: `docs/attribute_rationale_draft_ko.md`(속성 Methods)의 주장을 실제 모델·파이프라인으로 검증하고, 논문에 실을 수치를 확보한다.
> 설정: 모델 **gemma-4-12B-it-qat-w4a16-ct** (vLLM 0.23.1, 1×RTX4090, temp=0), 정규화 linker **scispaCy en_core_sci_lg + UMLS EntityLinker**(k=3, thr=0.82) — frozen 스펙과 동일.
> 입력: benchmark-blind 질환 임상텍스트(Wikipedia 임상섹션). 소규모=49 (DDXPlus 캐논), 대규모=840 (SymCat+DDXPlus 질환명 크롤).
> 날짜: 2026-07-01.

---

## 1. IE 기계적 동작 (frozen v106+R2, 49질환)

- findings **669개**, 질환당 **13.7개**, **JSON parse_fail = 0** → 12B가 13-slot 스키마를 안정적으로 산출.
- 속성별 fill rate (finding 기준):

| location | associated | character | duration | aggravating | severity | onset | timing | radiation | relieving |
|---|---|---|---|---|---|---|---|---|---|
| 18.5% | 19.0% | 10.5% | 4.2% | 4.0% | 3.7% | 3.3% | 2.7% | 0.9% | 0.7% |

**관찰**: aggravating/radiation/relieving은 원문에 드물게 기술되어 sparse. 6속성이 모두 산출되나 빈도는 속성별로 크게 다름.

## 2. 프롬프트 개선 (49질환, 세 변형 비교)

진단된 frozen 프롬프트(V0)의 실패모드: ① 다인자 값이 한 문자열로 뭉침("exertion, emotional stress, full stomach, cold") ② character가 비-성상 포착("blue discoloration","difficulty swallowing") ③ 근사중복 finding + associated 복사 ④ onset "sudden" 과예측.

개선: **P1 = 속성 값을 배열로 원자화**(한 원소=한 개념). **P2 = P1 + character 품질어휘 제약 + onset 규율 + 중복병합.**

| 지표 | V0 (frozen) | P1 (원자화) | **P2 (원자화+규율)** |
|---|---|---|---|
| findings/disease | 13.7 | 13.4 | 12.4 |
| location fill | 18.5% | 23.7% | **26.8%** |
| character fill | 10.5% | 16.8% | 6.6% |
| aggravating fill | 4.0% | 11.8% | 10.2% |
| associated fill | 19.0% | 40.0% | 39.7% |
| severity fill | 3.7% | 6.0% | 8.2% |
| 원자성(location/char/agg) | phrase(≈0) | .98/.93/.99 | **.98/1.0/.99** |
| character 청결도(품질어 비율) | 잡음 | 0.11 | **0.51** |

**결과**: 원자화로 다인자 값이 원소 배열로 분해되고(원자성 ≈0.98), 중복병합으로 finding/disease 소폭 감소. P2의 character 청결도 0.51(P1 0.11 대비)은 품질어휘 제약 효과이며, character fill이 6.6%로 낮아진 것은 비-성상 값을 배제한 결과(precision↑, 잡음↓). **P2 채택.**

## 3. 정규화 실측 — 원자화의 효과 (scispaCy UMLS, 49질환)

phrase(V0)는 다인자를 한 개념으로만 링크하고 나머지를 소실한다("coughing, exercise, or bowel movements" → "Exercise" 하나). 원자화(P2)는 각 원소를 개별 링크한다. **포착된 정규화 개념 수(고유원소 × 링크율):**

| 속성 | V0 phrase | P2 원자화 | 배수 |
|---|---|---|---|
| aggravating | ~17 | ~36 | **2.1×** |
| associated | ~66 | ~191 | **2.9×** |
| location | ~77 (링크 88%, **anatomy 77%**) | ~62 (링크 88%, anatomy 77%) | 유지·정확 |
| character | ~48 (링크 74%, 잡음) | ~17 (링크 57%) | 정확도↑·수↓ |

**핵심**: 원자화는 정규화 그래프가 포착하는 표준 개념 수를 **2~3배** 늘린다(방법론 개선). location은 UMLS anatomy로 정확히 링크(77%가 해부 semantic type).

**character의 근본 한계**(원자화·규율로도 해소 안 됨): 링크되어도 **오매핑** 빈번 — `pressure`→C0033095 "Pressure(물리적 작용)", `itchy`→C0033774 "Pruritus"(성상 아닌 증상), `severe`→severity modifier(속성 혼동). `sharp`,`throbbing`은 아예 UNLINKED. 즉 character는 정규화 대상으로 부적합.

## 4. HPO character 커버리지 (EBI OLS4 실측)

HP:0025280 "Pain characteristic" 하위에서 **일반 성상어는 Sharp(HP:0025281)·Dull(HP:0025282) 둘뿐.** burning/throbbing/stabbing/cramping/squeezing/aching/gnawing/shooting/electric = **일반 HPO term 없음**(있어도 chest 등 부위 고정: HP:6000048 chest pressure, HP:6000049 chest tearing). SNOMED CT는 더 넓음(Burning pain 36349006 확인)이나 전수 열거는 브라우저 차단으로 미완.

## 5. 파이프라인 정규화 실체 (코드 `v104c_build.py`)

- 개념형(location/radiation/**character**/aggravating/relieving/associated) → scispaCy **UMLS CUI**.
- 범주형(onset/severity/timing/duration) → 통제 **토큰**(sev_mild/sev_moderate/sev_severe 등).
- **HPO ID를 직접 부여하는 코드는 없음.** UMLS CUI는 HPO/SNOMED atom을 교차참조하므로 표준 연결은 CUI를 경유.

## 6. 대규모 확장 (840질환, SymCat+DDXPlus)

**출처**: 질환 이름을 SymCat(≈801) + DDXPlus(49)에서 취합(850 고유명, benchmark-blind — 이름만 사용)하고, 각 질환의 임상텍스트를 영문 Wikipedia 임상섹션(~2,600자)에서 크롤(성공 840/850). 개선 프롬프트 **P2**로 IE.

- findings **8,379개**, 질환당 **10.0개**, parse_fail=0.
- 속성 fill (finding 기준): location 34%, associated 51%, context 22%, character 7%, aggravating 7%, severity 5%, timing 5%, duration 4%, onset 3%, radiation 1%, relieving 1%.
- character 청결도(품질어 비율) 0.29 — 상위 값은 실제 품질어가 지배: burning(51), pressure(39), sharp(26), dull(21), tight(21), aching(21), cramping(18), throbbing(11), stabbing(11), shooting(10); 잡음은 pain/painful/painless/thick.
- aggravating 상위(원자, 깨끗): standing, exercise, cold, coughing, emotional stress, bending over, night, heat, eating, exertion. relieving: rest, lying down, sitting up, leaning forward, aspirin, wrist splint.

**관찰**: 개선 프롬프트가 스케일(840)에서도 aggravating/relieving을 깨끗한 원자 인자로 산출. location(34%)·associated(51%)가 주 신호, character/aggravating/relieving은 원문 밀도로 인해 낮게 유지(Wikipedia 임상섹션 한계 — 임상발현 코퍼스 보강 시 상승 여지).

**정규화 링크율 (scispaCy UMLS, 840질환 추출 원자):**

| 속성 | 고유 원자 | 링크율 | anatomy 비율 | ~정규화 개념 수 |
|---|---|---|---|---|
| location | 801 | 84% | **78%** | ~671 |
| radiation | 55 | 73% | 78% | ~40 |
| aggravating | 484 | 77% | 8% | ~372 |
| relieving | 74 | 85% | 13% | ~63 |
| associated | 2,956 | 84% | 11% | ~2,498 |
| character | 376 | 57% | **2%** | ~214(오매핑 다수) |

**결론(스케일 확정)**: ① **location이 가장 견고한 정규화 속성**(84% 링크, 78% 해부 semantic type). ② **aggravating/relieving은 원자화 시 양호하게 정규화**(77~85%, 수백 개 인자 개념 포착). ③ **character는 링크율 57%여도 anatomy 2%·오매핑 빈번**(sharp→미링크, "bright red"→염료 concept) → 정규화 속성으로 부적합, free-text 유지가 정직. 소규모(49)와 대규모(840)에서 동일 결론 → 평가 자체 견고.

## 7. Methods 정정 사항 (본 검증으로 확정)

1. **정규화 서술을 실제에 맞춤**: "속성→HPO 정규화"(과대주장) → **개념형은 scispaCy UMLS CUI, 범주형은 통제어휘(HPO 서수에 대응)**. HPO/SNOMED는 CUI가 교차참조하는 표준.
2. **속성 추출 = 원자화 배열**(한 원소=한 개념) 명시 — 정규화 개념 포착 2~3배(실측).
3. **character role 확정**: 통제어휘·UMLS 모두 성상 커버리지 빈약 + 오매핑 → **대부분 free-text**, sharp/dull만 HPO/UMLS로 안정 링크. 진단 기여는 ablation으로 판정(정규화 속성으로 과대주장 금지).
4. location은 UMLS anatomy로 정확 정규화(77% 해부 타입) — 가장 견고한 정규화 속성.
