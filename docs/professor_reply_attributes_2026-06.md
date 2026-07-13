# 교수님 자문 답변 기록 — Evidence 속성 설계 (2026-06)

> 본 문서는 evidence 속성(attribute) 설계에 대한 교수님 자문 답변을 원문 그대로 기록하고,
> 현재 연구 단계에서의 적용 방안을 정리한다.

## 1. 배경 (자문 질문 요지)

- DDXPlus 환자 케이스는 하나의 evidence(증상)에 여러 속성이 동반된다.
  - 예/아니오 · 보기 선택(위치·색·성상) · 0~10 숫자 척도(강도·가려움).
  - 예: "통증" 하나에 위치(복부)/강도(4/10)/발현속도(6/10)/성상(쥐어짜는)이 함께 붙음.
  - "부종"도 유/무가 아니라 위치("양쪽 뺨")가 angioedema를 가리키는 핵심 정보.
- 현재 KG는 "phenotype CUI 유/무"만 표현 → 속성 정보가 소실됨.
- 질문: (1) location/severity/onset/character 4속성 선택의 학술적 정당성, (2) 어떤 속성값을 쓸지,
  (3) DDXPlus 종속/leakage를 피하는 방법.
- OLDCARTS / CPX / DDXPlus 속성 비교 테이블을 첨부하여 질의.

## 2. 교수님 답변 (원문 충실 기록)

### 2.1 속성 4가지(location/severity/onset/character) 선택의 정당성
- 자의적 선택이 아님을 논문에서 강조할 것.
- **OLDCARTS, OPQRST, SOCRATES, 국시 CPX** 병력청취 항목을 나란히 놓으면 네 프레임워크의
  **공통 핵심이 location, severity, onset, character**. "어디가/얼마나/언제부터/어떻게 불편하냐"는 임상 병력청취의 기본.
- 즉 이 4속성은 특정 벤치마크가 아니라 **임상 병력청취의 보편 기본**.
- 근거 문헌: **Panju, JAMA 1998** (흉통 감별 고전). 같은 "흉통"이라도 성상이 찌르는 듯하면
  심근경색 LR이 0.3으로 하락, 양팔로 방사되면 예측도 상승. → **같은 증상이라도 속성이 진단을 가른다.**
- 흔한 시점 분류 예: 식후 가슴통증→역류성식도염, 운동/중량물/계단 중 가슴통증→협심증·심근경색.
  (정답은 없고 국시 요약본·syllabus 수준. 정량화·표준화는 어려움.)
- 불완전하면 **radiation(방사), frequency/timing, aggravating/alleviating** 추가 가능.
  **duration은 onset/timing과 유사 개념이라 별개일 필요 없음.**

### 2.2 어떤 속성값을 쓸까 (leakage 회피)
- DDXPlus 코드를 KG에 그대로 넣으면 **벤치마크 종속 + leakage 소지**.
- 속성 정의 출처를 DDXPlus가 아니라 **상위 보편 표준(OLDCARTS 같은 임상 프레임워크 + HPO/SNOMED 온톨로지)**에 둘 것.
- 방향: **표준에서 속성 정의 → KG에 삽입 → 평가 시점에 각 벤치마크 코드를 표준으로 변환.**
- 구체:
  - (a) 속성 vocabulary와 값 등급을 **HPO/SNOMED ID로 고정**.
  - (b) 벤치마크 고유 코드는 KG가 아니라 **평가 어댑터에서만** 표준으로 매핑.
  - (c) DDXPlus 값 분포(0~10 표집 구간 등)는 **KB에 학습시키지 않음**.

### 2.3 NLICE 인용 (방법론 근거)
- NLICE: SymCat **binary만 Top-1 58.8%** → 증상마다 속성(Nature, Location, Intensity, Chronology, Excitation)
  부여 시 **Top-1 82%**까지 상승, 노이즈 교란에도 안정적.
- 단순 비교가 아니라 **방법론 근거로 인용**. 우리 기여 = 그 속성을 **표준 온톨로지에 정규화하여 5개 벤치마크를 가로질러 작동**하게 함.

### 2.4 HPO 1차 표준
- KG가 HPO ID + CUI 정규화를 축으로 하므로 HPO의 다음을 1차 표준으로:
  - **Clinical modifier subontology HP:0012823**
  - **Onset HP:0003674**
  - **Severity 계열 HP:0012824 등**
- **character가 문제**: sharp/burning/throbbing을 HPO·SNOMED가 체계적으로 완비하지 않음.
  → **매핑 가능한 건 별도 phenotype term으로, 나머지는 속성 문자열로 보존** (논문에 명시).

### 2.5 구조/값에 대한 지침
- 색·세부 병변 위치처럼 CUI로 깔끔히 안 잡히는 표현을 노드로 만들지 않고 **속성으로만 보존 → 좋음**.
- **numeric severity(0~10)를 edge weight로 쓰는 것은 임상적 의미가 약함** (많이 아프면 응급, 진단을 가르지는 않음). → 지양.
- **속성은 별도 노드로 승격하지 말고 disease–phenotype edge에 붙는 qualified statement로 둘 것** (그래프 비대화 방지).

### 2.6 연구의 정체성 (종합)
- NLICE 방법을 차용하되 **표준에서 정의를 끌어오는 원칙**만 지키면, 강점은 SOTA 점수가 아니라:
  1. **5개 벤치마크를 가로지르는 robustness**
  2. **traceable provenance** (출처 추적 가능성)
  3. **8B로 돌아가는 비용 효율**
- **속성을 표준에 정규화하는 작업이 cross-benchmark 강점의 핵심.**

## 3. 핵심 take-away (의사결정)

1. **4속성 = location / severity / onset / character** 확정. 출처는 OLDCARTS/OPQRST/SOCRATES/CPX 공통핵심 + HPO.
   필요시 radiation/timing/aggravating-relieving 보강. duration은 onset으로 흡수.
2. **속성은 disease–phenotype edge의 qualified statement** (노드 승격 금지, 그래프 비대화 방지).
3. **표준(HPO) 정규화**: HP:0012823(modifier) / HP:0003674(onset) / HP:0012824(severity).
   character는 매핑 가능분만 phenotype term, 나머지는 문자열 보존.
4. **leakage 분리**: 속성 정의·값등급은 표준 ID로 KG에 고정. 벤치마크 코드↔표준 변환은 **평가 어댑터 전용**.
   DDXPlus 값분포는 KB 미학습.
5. **numeric severity → edge weight 폐기** (임상적 무의미).
6. **목표 재정의**: SOTA가 아니라 cross-benchmark robustness + provenance + 8B 비용효율. 속성 정규화가 그 핵심.
7. **인용**: NLICE(SymCat 58.8→82%), Panju JAMA 1998(흉통 성상/방사 LR).
