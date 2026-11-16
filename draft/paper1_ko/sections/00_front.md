# DDXPlus의 공개 목록은 대화형 과제에서 정답 질환을 얼마나 맞히는가

**How Much of DDXPlus Do Its Public Lists Solve?**

## Abstract

DDXPlus는 대화형 감별진단 에이전트를 평가하는 합성 환자 벤치마크이고, 49개 질환마다 증상과 병력의 목록을 확률 없이 공개한다. 이 연구는 이 공개 목록만으로 정답 질환을 얼마나 맞히는지 쟀다. DDXPlus의 공식 규칙에서 에이전트는 환자의 나이, 성별, 초기 evidence 1개를 받고 최대 30개까지 질문한다. 이 규칙에서 공개 목록만 쓰는 목록 기반 Bayes 방법과 목록 대신 train 환자의 빈도를 쓰는 빈도 기반 방법을 평가했고, 문헌의 Bayes 실험설계 기준선인 BED를 다시 구현했다. 설정은 validation 세트에서 정하고 test 세트 134,529명에서 평가했다. train, validation, test 세트 어디에도 정답 질환의 목록 밖 양성 evidence를 가진 환자는 없었다. 목록 기반 방법은 환자의 98.44%(95% 신뢰구간 98.38–98.51)에서 정답 질환을 맞혔고, 초기 evidence를 포함해 평균 10.35개의 evidence를 모았다. 종료 임계 0.99에서 목록 대신 빈도를 쓰면 수집 evidence가 3.08개 줄었고, 정확도는 0.16%p 높아졌지만 그 차이의 부호는 종료 임계에 따라 바뀌었다. 응답의 값까지 쓰면 정확도가 0.70%p 더 높아지고 수집 evidence가 2.17개 더 줄어, 99.30%와 5.10개가 되었다. BED 재구현은 99.47%와 5.48개로 보고값 99.47%와 5.47개를 재현했다. 다만 BED의 시뮬레이터는 범주형과 다중선택 evidence에 기록된 기본값을 양성으로 답하는데, 공식 규칙에서 이 답은 "아니오"다. test 환자의 38.95%가 이런 답을 한 번 이상 받았다. validation 세트에서 이 답이 알려 주는 정보를 빈도 기반 방법에 주면 정확도가 0.10%p 높아지고 수집 evidence가 0.30개 줄었다. 공개 목록만으로 DDXPlus 환자의 98%에서 정답 질환이 식별된다. DDXPlus의 대화형 결과를 보고할 때는 같은 규칙에서 돌린 목록 기반 기준선의 결과와 시뮬레이터의 응답 규칙을 함께 적어야 한다.

**Keywords:** Diagnosis, Differential; Diagnosis, Computer-Assisted; Bayes Theorem; Knowledge Bases; Benchmarking
