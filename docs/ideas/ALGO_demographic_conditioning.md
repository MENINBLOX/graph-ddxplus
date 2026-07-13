# Demographic(age/sex) conditioning  ⚪ 미검증/marginal

**상태**: 일부 시도(v67 AGE channel), 효과 marginal — 재검증 필요

## 아이디어
DDXPlus는 age/sex 제공. 일부 질환은 연령/성 특이(bronchiolitis=영아 vs bronchitis=성인). benchmark-blind disease-demographic prior(문헌/UMLS에서, 라벨 X)로 tie-break.
- 스크립트: v67 AGE 채널 통합.

## 근거
역학(epidemiology). 임상에서 연령·성은 강한 prior.

## 상태
v67에서 AGE 채널 추가했으나 효과 marginal/mixed. demographic prior를 **benchmark-blind하게(질환 역학을 IE/문헌에서)** 도출하는 게 관건(train 라벨 사용 금지).

## 할 것
deep crawl IE에서 질환별 연령/성 분포 추출 → prior. chief anchoring과 달리 demographic은 변별 신호가 명확해 재시도 가치.
