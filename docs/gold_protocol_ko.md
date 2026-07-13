# 증상 속성 IE 평가를 위한 임상의 판정(Clinician-Adjudicated) GOLD 주석 프로토콜

**대상 속성 (4종, public gold 부재):** `character`(질/양상), `timing`(onset pace, 발현 속도), `aggravating`(악화 요인), `relieving`(완화 요인)

**목적:** 속성 단위 Information Extraction(IE)의 **정밀도(precision) / 충실도(faithfulness)** 를 임상의 판정 기준으로 정량 평가하고, 논문에 reviewer-defensible한 근거(IAA, 신뢰구간)를 제시한다.

**작성일:** 2026-07 · **판정자(adjudicator):** 임상의(교수) 1인 · **문서 위치:** `docs/gold_protocol_ko.md`

---

## 0. 한 줄 요약 (임상의용 Quick Start)

> 각 행(row)은 `(질환, 원문 근거 문장, 발현소견 finding, 추출된 속성값)` 하나입니다. 원문을 읽고 그 속성값이 **원문에 실제로 있고(grounded) · 올바른 finding에 붙어 있고 · 값이 맞는지**를 판정해 `correct / hallucinated / wrong_attachment / wrong_value` 중 하나로 라벨링합니다. 4개 속성 × 약 100건 × 2인 = 총 약 800건, 예상 소요 각 4~7시간입니다.

---

## 1. 평가 목표와 판정 단위 (Objective & Unit of Adjudication)

### 1.1 평가 대상
본 프로토콜은 **속성 IE의 precision(추출된 값이 옳은 비율)** 만을 측정한다. 이는 자연어처리 IE 평가에서 **faithfulness(원문 충실도)** 에 해당하는 intrinsic 축이다. Recall(원문에 있는데 놓친 값)은 별도의 더 어려운 축이며, 본 프로토콜에서는 선택적으로만 다룬다(§5.3).

### 1.2 판정 단위 (item)
판정의 최소 단위는 아래 4-tuple 하나이다.

```
item = (disease, source_snippet, finding, extracted_value)
```

- `disease` — 질환명 (예: `Cluster headache`)
- `source_snippet` — IE의 근거가 된 원문 발췌 (해당 속성값이 유래했을 만한 1~3문장)
- `finding` — 속성이 부착된 발현소견 (예: `headache`)
- `extracted_value` — 평가 대상 속성값 (예: character=`throbbing`)

### 1.3 라벨 체계 (4-way)
각 item에 대해 아래 4개 중 정확히 하나를 부여한다.

| label | 정의 | 판정 조건 |
|---|---|---|
| **`correct`** | 정답 | 값이 원문에 근거하고(**grounded**) **AND** 올바른 finding에 부착되고(**right attachment**) **AND** 값 자체가 옳음(**right value**) |
| **`hallucinated`** | 환각 | 해당 속성값이 원문 어디에도 없음 (모델이 생성) |
| **`wrong_attachment`** | 부착 오류 | 값은 원문에 있으나 **다른 finding**에 속하는 것을 잘못 붙임 |
| **`wrong_value`** | 값 오류 | 값이 원문에 있고 finding도 맞으나, **속성 정의에 어긋남**(예: character 자리에 severity, timing 자리에 duration) 또는 **의미가 반대/왜곡**(예: relieving 자리에 "완화되지 않음") |

**우선순위 규칙(하나의 item에 복수 결함이 있을 때):** `hallucinated` > `wrong_value` > `wrong_attachment` > `correct`. 즉 원문에 근거 자체가 없으면 무조건 `hallucinated`. 근거는 있으나 값이 속성 정의에 어긋나면 `wrong_value`. 값·정의는 맞으나 finding만 틀리면 `wrong_attachment`.

> **precision 산출:** precision = (`correct` 수) / (전체 판정 item 수). 세 오류 라벨은 모두 오답(non-correct)으로 합산되나, 오류 유형별 분포는 오류 분석(error analysis)에 활용한다.

---

## 2. 표본 설계 (Sampling Design)

### 2.1 모집단
개선된 IE 출력 `pilot/data/cache/ie_scaleup/*.json` (842개 질환 파일, 840개 질환), 원문은 `pilot/data/cache/scaleup_sources/*.json` (`{disease, title, text}`). 각 finding은 배열형 속성(`character`, `aggravating`, `relieving`)과 스칼라형 속성(`onset`, `timing` 등)을 가진다. **"positive instance"** = 해당 속성에 비어있지 않은 값이 1개 이상 존재하는 (finding, value) 쌍.

### 2.2 목표 표본 크기 (Tier = SUPPORTING)
속성 IE는 본 논문의 **주장(headline)이 아니라 진단 성능을 뒷받침하는 보조 근거**이므로, 통계 조사에 근거해 **속성당 양성(positive) 약 100건**을 목표로 한다(하한 50건).

> **Escalation note:** 특정 속성이 향후 **독립적 headline claim**(예: "location 속성 단독으로 SOTA 기여")으로 승격되면, 해당 속성만 **385건/속성**으로 재표집한다. 현재 tier에서는 100건이면 충분하다(근거: §5.2 신뢰구간 계산).

### 2.3 관측된 양성 분포 (2026-07 실측, 840 질환)

| 속성 | 총 positive instances | unique 값 | 상위값(빈도) | 100건 확보 | 비고 |
|---|---|---|---|---|---|
| `character` | 848 | 376 | burning(51), pressure(39), sharp(26), aching(21) | 가능 | 잡음 존재(§3.1) |
| `aggravating` | 1,143 | 484 | standing(31), exercise(23), cold(21) | 가능 | |
| `timing`(onset pace) | 236* | sudden/gradual/rapid | sudden(172), gradual(54), rapid(8) | 가능 | 클래스 심한 불균형 |
| `relieving` | 115 | 74 | lying down(7), rest(7) | **경계선** | 100건 근접, enrichment 명시 필요 |

\* onset-pace 값은 프로즌 IE 스키마에서 `onset` 필드에 담긴다. IE의 원시 `timing` 필드(449건)는 diurnal/course/duration 값(예: night, third week, 12–36 hours)을 혼입하고 있으며, 이는 본 평가의 onset-pace 정의에서 벗어나므로 `wrong_value` 탐지 대상이다(§3.2 참조).

### 2.4 표집 절차 (질환 간 층화, stratified across diseases)
1. 각 속성별로 전체 positive instance 풀을 구성한다: `(disease, finding, value, source_text)`.
2. **질환 층화**: 한 질환이 표본을 과점하지 않도록 **질환당 최대 3~5건** 상한을 둔다. 질환을 무작위 순서로 순회하며 라운드로빈으로 추출해 목표 100건을 채운다.
3. **원문 스니펫 자동 추출**: value 문자열(또는 그 부분어)을 원문 `text`에서 검색해 매칭 문장 ±1문장을 `source_snippet`으로 채운다. 매칭이 없으면 스니펫을 비우고 `[NO_MATCH]`로 표기한다(이 경우 대개 `hallucinated` 후보이나, 판정은 임상의가 원문 전체를 보고 내린다).
4. 재현성을 위해 **고정 시드**(예: `seed=42`)로 표집하고, 표집 스크립트와 산출 시트를 함께 보관한다.

### 2.5 희귀 클래스 처리 (Rare-class caveat)
- **`relieving`(관측 115건)**: 100건 목표가 모집단의 87%를 소모하므로 사실상 **준전수(near-census)**. 이 경우 층화 상한을 완화하고 **가용 전량에 근접해 표집**한다. 시트·논문에 "targeted enrichment(전량 근접 표집)"을 명시한다.
- **`timing`(onset pace) 소수 클래스**: `gradual`(54), `rapid`(8)이 `sudden`(172)에 압도된다. **값별 층화 오버샘플링**으로 `sudden ~60 / gradual ~30 / rapid ~10` 형태로 구성해 소수 클래스 정밀도를 관측 가능하게 한다.
- **양성 < 50건인 속성이 발생하면**: (1) targeted enrichment를 명시적으로 공개하고, (2) **원 개수(raw count) < 30이면 비율 대신 원 개수를 보고**하며(예: "8/9 correct"), Wilson 구간을 함께 제시하되 소표본 한계를 서술한다.

---

## 3. 속성별 판정 가이드라인 (Per-Attribute Operational Guidelines)

> 공통 원칙: **원문(source)이 진실의 유일한 근거**다. 임상 지식으로 "그럴듯하다"고 채우지 않는다. 원문에 없으면 `hallucinated`.

### 3.1 `character` — 증상의 질/양상 (single quality descriptor)
**정의:** 증상(주로 통증·분비물·감각)의 **질을 서술하는 단일 형용사/성상어**. 예: sharp, dull, burning, throbbing, cramping, stabbing, aching, tight, pressure-like, crampy, colicky, thick(분비물), crackling.

**허용(→ 판정 후보 `correct`):**
- ✅ `(Cluster headache, headache, throbbing)` — 원문에 "throbbing/boring pain" 존재 시.
- ✅ `(Acute dystonic reactions, muscle spasms, cramping)` — 경련의 성상.
- ✅ `(Acute pulmonary edema, crackles, crackling)` — 청진음의 성상.

**거부:**
- ❌ **다른 소견을 성상으로 오인**: `productive cough`에서 "productive"는 `cough`의 **별개 finding/속성(가래 동반)** 이지 통증 성상이 아님 → 원 소견이 별개면 `wrong_attachment` 또는 `wrong_value`.
- ❌ **비-성상어(non-quality)**: 색(yellow, red), 다른 증상명(pain을 pain finding의 character로), 심각도어 → `wrong_value`.
- ❌ **severity 누출**: character=`severe`, `mild`는 성상이 아니라 **severity** → `wrong_value` (실측에서 severe 11건, painful/painless 등 관측됨).
- ❌ **tautology**: finding=`pain`, character=`pain`/`painful` → 정보 없음, `wrong_value`.

### 3.2 `timing` = onset PACE (발현 속도: sudden vs gradual)
**정의:** 증상이 **얼마나 빠르게 시작되는가**(발현 속도). 허용값은 본질적으로 `sudden`(급성/돌발), `gradual`(점진), 및 그 동의·정도어(`rapid`, `abrupt`, `acute onset`, `insidious`, `slowly progressive`). **연령(age-of-onset) 아님, 지속시간(duration) 아님, 일중 주기(diurnal) 아님.**

**허용(→ `correct` 후보):**
- ✅ `(Cluster headache, headache, sudden)` — "attacks begin abruptly" 근거.
- ✅ `(COPD exacerbation, shortness of breath, sudden)` — "abrupt worsening" 근거.
- ✅ `(그 외, fatigue, gradual)` — "gradual onset" 근거.

**거부:**
- ❌ **duration/incubation을 pace로 오인**: `12 to 36 hours`, `24 to 72 hours` → 지속·잠복 시간, `wrong_value`.
- ❌ **course/시점을 pace로 오인**: `third week`, `second week`, `prodrome` → 경과 시점, `wrong_value`.
- ❌ **diurnal 패턴**: `night`, `morning`, `around the same hour every day` → 일중 주기(별도 timing 개념), 본 정의에서 `wrong_value`.
- ❌ **age-of-onset**: `childhood`, `after age 50` → `wrong_value`.

> 실측상 IE의 원시 `timing` 필드는 위 거부 사례를 다수 포함(night 18, third week 19, 12–36 hours 11 등). 이는 onset-pace precision 평가에서 **wrong_value로 정직히 계상**되어야 하며, 오히려 속성 스키마 정제 필요성의 근거가 된다.

### 3.3 `aggravating` — 악화/유발 요인
**정의:** 해당 증상을 **악화시키거나 유발/촉발하는** 요인(활동·자세·환경·음식·감정 등).

**허용(→ `correct` 후보):**
- ✅ `(Inguinal hernia, bulge, standing)` — 기립 시 악화.
- ✅ `(GERD류, heartburn, bending over)` — 굴신 시 악화.
- ✅ `(migraine류, headache, glare)` — 눈부심 유발.

**거부:**
- ❌ **원문 부재**: 원문에 그 요인이 없으면(임상적으로 그럴듯해도) `hallucinated` (예: 과도하게 구체적인 `strong blue-spectrum backlights`가 원문에 없으면 hallucination 후보).
- ❌ **완화 요인 혼입**: `rest`, `lying down`이 aggravating에 들어감 → `wrong_value`.
- ❌ **부착 오류**: 요인은 원문에 있으나 **다른 증상**의 악화 요인 → `wrong_attachment`.

### 3.4 `relieving` — 완화 요인
**정의:** 해당 증상을 **완화/경감시키는** 요인(자세·휴식·약물·처치 등).

**허용(→ `correct` 후보):**
- ✅ `(Inguinal hernia, bulge, lying down)` — 누우면 환원/완화.
- ✅ `(Epiglottitis, breathing distress, sitting up)` — 좌위/삼각자세로 완화.
- ✅ `(Pericarditis류, chest pain, leaning forward)` — 전방 굴신 완화.

**거부:**
- ❌ **부정문(negation) 누출**: `does not improve with rest` → 완화 **안 됨**을 서술 → 완화 요인이 아님, `wrong_value` (실측 2건 관측).
- ❌ **약물이 다른 증상 대상**: `paracetamol`이 통증이 아니라 무관 소견에 붙음 → `wrong_attachment`.
- ❌ **악화 요인 혼입** → `wrong_value`.

---

## 4. 주석 워크플로우 (Annotator Workflow)

### 4.1 인력 구성 (n2c2/THYME/ShARe 표준)
- **독립 주석자 ≥ 2인** (그중 **최소 1인은 임상 훈련자**).
- **판정자(adjudicator) 1인** — 두 주석자의 불일치를 최종 해소(임상의/교수).

### 4.2 절차
1. **독립 이중 주석 (independent double annotation):** 두 주석자가 서로의 라벨을 보지 않고 전량(또는 공유 subset)을 라벨링한다.
2. **IAA 산출:** 두 주석자가 공통으로 라벨한 부분집합(권장 **전체의 20% 이상 또는 최소 50건**)에서 4-way 라벨에 대한 **Cohen's κ**(2인) 또는 **Krippendorff's α**를 계산한다.
3. **판정(adjudication):** 불일치 item을 판정자가 원문과 함께 검토해 최종 라벨을 확정한다. 반복되는 불일치 유형은 가이드라인(§3)에 반영해 재주석할 수 있다(iterative refinement).
4. **최종 GOLD 확정:** 판정 결과를 gold로 고정하고, 이를 IE 출력과 대조해 precision을 산출한다.

### 4.3 κ 해석표 (Landis & Koch 1977)

| Cohen's κ | 일치 수준 (Landis–Koch band) |
|---|---|
| < 0.00 | Poor |
| 0.00 – 0.20 | Slight |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | **Substantial** (출판 최소 기준) |
| 0.81 – 1.00 | **Almost perfect** (gold 목표) |

- **출판 하한:** κ ≥ 0.61 (substantial). 미달 시 가이드라인 정제 후 재주석.
- **gold 목표:** κ ≥ 0.80.
- α(Krippendorff) 사용 시 동일 밴드로 해석하되 관례상 α ≥ 0.667(수용), α ≥ 0.80(신뢰)을 병기한다.

---

## 5. 지표와 보고 템플릿 (Metrics & Reporting)

### 5.1 주지표: 속성별 precision + Wilson 95% CI
각 속성에 대해:

```
precision = correct / n
```

신뢰구간은 **Wald(정규근사) 금지**, **Wilson score interval**(Wilson 1927)로 보고한다. 소표본·극단 비율에서 Wald보다 커버리지가 우수하다(Brown, Cai & DasGupta 2001, *Statistical Science*). 필요 시 bootstrap(BCa) 95% CI를 병기한다.

**보고 표 템플릿:**

| Attribute | n (positives) | correct | precision | Wilson 95% CI | κ (IAA) | 비고 |
|---|---:|---:|---:|---|---:|---|
| character | 100 | — | 0.__ | [__, __] | 0.__ | |
| timing (onset pace) | 100 | — | 0.__ | [__, __] | 0.__ | sudden/gradual/rapid 층화 |
| aggravating | 100 | — | 0.__ | [__, __] | 0.__ | |
| relieving | ~100 | — | 0.__ | [__, __] | 0.__ | near-census, enrichment 명시 |

**오류 분해(선택):** `hallucinated / wrong_attachment / wrong_value` 각 개수를 부표로 제시해 오류의 성격(환각 vs 부착 vs 정의 위반)을 구분한다.

### 5.2 표본 크기 정당화 (Sample-size justification)
n ≈ 100에서 관측 precision이 양호할 때 Wilson 95% CI 반폭(half-width)은 대략:
- p ≈ 0.90 → 약 **±5.8%p** (CI ≈ [0.825, 0.945])
- p ≈ 0.85 → 약 **±7%p** (CI ≈ [0.767, 0.909])
- p ≈ 0.75 → 약 **±8.4%p**

즉 **n ≈ 100이면 ±7–10%p 정밀도**로 SUPPORTING tier 주장을 뒷받침하기에 충분하다. headline claim(±5%p 미만, n ≈ 385) 승격 시에만 재표집한다.

### 5.3 Recall 한계 명시 (필수)
본 프로토콜은 precision(추출값의 정답률)만 측정한다. **Recall(원문에 존재하나 IE가 놓친 속성값)은 별개의 더 어려운 축**이며, 원문 전체에서 gold 속성값을 완전 열거해야 하므로 본 tier에서는 측정하지 않는다. 논문 한계(Limitations)에 다음을 명시한다: "제시된 수치는 faithfulness/precision이며 recall/completeness는 평가하지 않았다." 선택적으로, 소수 질환에 대해 원문에서 gold 속성값을 완전 주석해 recall을 탐색적으로 보고할 수 있다.

---

## 6. 주석 시트 사양 (Annotation Sheet Spec)

### 6.1 컬럼 정의 (CSV/XLSX)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | int | 행 고유번호 |
| `attribute` | enum | `character` / `timing` / `aggravating` / `relieving` |
| `disease` | str | 질환명 |
| `finding` | str | 속성이 부착된 발현소견 |
| `extracted_value` | str | 평가 대상 속성값 |
| `source_snippet` | str | 근거 원문 발췌(±1문장). 매칭 없으면 `[NO_MATCH]` |
| `label` | enum | `correct` / `hallucinated` / `wrong_attachment` / `wrong_value` |
| `note` | str | (선택) 판정 근거·비고 |

> 이중 주석을 위해 주석자별로 `label_a`, `label_b`, 판정 `label_final` 3개 컬럼을 두거나, 주석자별 시트를 분리하고 병합 시 IAA를 계산한다.

### 6.2 임상의용 지시문 (시트 상단에 삽입)

```
[주석 지침 — 6줄]
1) 각 행의 원문(source_snippet)만을 근거로 판정합니다. 임상 지식으로 값을 보완하지 마세요.
2) 원문에 그 값이 없으면 → hallucinated.
3) 값은 원문에 있으나 다른 소견(finding)에 속하면 → wrong_attachment.
4) 값이 있으나 속성 정의에 어긋나거나(예: character에 severity, timing에 duration/야간) 의미가 반대면 → wrong_value.
5) 값이 원문에 있고 · 올바른 finding에 붙고 · 정의에 맞으면 → correct.
6) 결함이 여러 개면 우선순위: hallucinated > wrong_value > wrong_attachment.
   판정 근거가 애매하면 note에 간단히 남기세요(예: "severity지 성상 아님").
```

---

## 7. 작업량 추정 (Effort Estimate)

| 항목 | 수량 |
|---|---|
| 속성 수 | 4 |
| 속성당 item | ~100 |
| 주석자 | 2 (독립 이중) |
| **총 주석 결정 수** | **4 × 100 × 2 ≈ 800건** |
| 주석자 1인당 부담 | 400건 |
| item당 소요(원문 발췌 사전 삽입 가정) | 30–60초 |
| **주석자 1인 소요** | **약 3.5–7시간** (1~2세션) |
| IAA subset(20%, ≈80건 공유) + 불일치 판정 | 판정자 약 1–2시간 |

결론: 총 800건 규모로 **소규모·실행가능(feasible)**하며, 사전 스니펫 자동 삽입(§2.4-3)으로 판정 부담을 최소화한다.

---

## 8. 인용한 규범 (Key Norms Cited)

- **IAA 밴드:** Landis, J.R. & Koch, G.G. (1977). "The measurement of observer agreement for categorical data." *Biometrics*, 33(1), 159–174. (κ ≥ 0.61 substantial, ≥ 0.81 almost perfect)
- **κ / α:** Cohen, J. (1960) *Educ. Psychol. Meas.* 20(1):37–46 (Cohen's κ); Krippendorff, K. *Content Analysis* (Krippendorff's α, 다수 주석자·서열 대응).
- **이중 주석 + 판정자 표준:** n2c2 2018 shared task — Henry, S. et al. (2020), "2018 n2c2 shared task on adverse drug events and medication extraction," *JAMIA* 27(1):3–12 (double annotation + adjudicator). 유사 임상 NLP 코퍼스 표준: THYME, ShARe/CLEF.
- **비율 신뢰구간:** Wilson, E.B. (1927), *JASA* 22:209–212 (Wilson score interval); Brown, L.D., Cai, T.T. & DasGupta, A. (2001), "Interval Estimation for a Binomial Proportion," *Statistical Science* 16(2):101–133 (Wald 지양, Wilson/Agresti-Coull 권장).

---

## 부록 A. 표집 스크립트 산출물 (재현성)

표집은 고정 시드로 아래를 산출해 시트에 채운다: `(id, attribute, disease, finding, extracted_value, source_snippet)`. 입력은 `pilot/data/cache/ie_scaleup/*.json`(속성값), `pilot/data/cache/scaleup_sources/*.json`(원문). `source_snippet`은 `extracted_value`를 원문에서 검색해 매칭 문장 ±1문장으로 자동 채우고, 무매칭은 `[NO_MATCH]`로 표기한다. 스크립트·시드·산출 CSV를 리포지토리에 함께 보관해 재현성을 확보한다.
