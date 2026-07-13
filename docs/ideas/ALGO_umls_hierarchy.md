# UMLS 계층 coarse→fine 진단  ⚪ 미검증

**상태**: 미검증. content 벽 가능성 있으나 grouping은 깨끗.

## 아이디어
@1 실패 대부분이 같은 군 내 혼동(bronchitis vs bronchiolitis). UMLS MRHIER/MRREL(PAR/CHD)로 질환을 군집화(benchmark-blind) → 군 단위 1차 분류(고정확) → 군 내 변별. 구조적 prior가 cluster 혼동 완화.

## 근거
임상 진단 schema(호흡기냐 심장이냐 먼저). UMLS 계층 = benchmark-blind.

## 리스크
group 내 변별은 여전히 noisy KG 의존 → 메타결론상 한계 가능. grouping이 impossible 후보 pruning으로 @1 소폭 도움은 가능.

## 할 것
48 DDXPlus 질환의 UMLS 상위(PAR) 군집 도출 → 2-stage 점수. content 교정 후 우선순위.
