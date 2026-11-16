## 3. Results

### 1) 자료의 성질

정답 질환의 목록 밖에 양성 evidence를 가진 환자는 train, validation, test 세트 어디에도 없었다. 양성 evidence를 모두 담는 목록이 정답 질환 하나뿐인 환자는 validation 세트에서 92.35%, test 세트에서 92.40%였다.

반대로 목록에 있는 evidence가 그 질환의 환자에게 늘 나타나지는 않았다. 질환과 evidence의 쌍으로 센 목록 항목은 888개다. 이 가운데 46개는 train 세트에서 한 번도 양성이 아니었다. 그 질환의 train 환자 가운데 evidence가 양성인 비율이 5% 미만인 항목은 10.36%였고, 99% 이상인 항목은 22.64%였다. train과 validation 세트의 기록에서 범주형과 다중선택 evidence는 정답 질환의 목록에 있으면 값이 기본값이어도 모두 기록되어 있었고, 목록에 없으면 기록되어 있지 않았다. validation 세트의 급성 부비동염 환자 1,866명 가운데 622명(33.33%)은 양성 evidence가 모두 만성 부비동염의 목록 안에 있었고, 불안정형 협심증 환자 2,748명 가운데 56명(2.04%)은 양성 evidence가 모두 안정형 협심증의 목록 안에 있었다.

validation 세트에서 정답 감별진단의 1위가 정답 질환인 환자는 73.38%였다. 정답 감별진단의 확률 가운데 평균 70.13%는 환자의 양성 evidence를 모두 담지 못하는 목록의 질환에 놓여 있었다.

### 2) test 세트의 정확도와 수집 evidence 수

validation 세트에서 P1의 아홉 후보는 97.72–98.42%의 정확도를 보였고, α 0.5와 ε 10⁻⁴가 선택되었다. 선택된 ε가 후보의 끝 값이어서 함께 평가한 ε 10⁻⁵의 세 조합은 97.79–98.41%였다(Supplementary Table S1). Table 1은 이 설정으로 test 세트 134,529명을 평가한 결과다. test 세트의 모든 환자가 평가에 포함되었다.

P1은 환자의 98.44%(95% 신뢰구간 98.38–98.51)에서 정답 질환을 맞혔고, 초기 evidence를 포함해 평균 10.35개(10.31–10.39)의 evidence를 모았다. 질문이 30개에 이르러 멈춘 환자는 3.27%였다. 나이, 성별, 초기 evidence, 기록된 evidence와 그 값이 모두 같은 환자가 train 세트에 없는 132,540명에서 P1의 정확도는 98.42%(98.36–98.49)였고, StreamBench와 같은 1,764명에서는 98.11%(97.36–98.65)였다(Supplementary Table S3).

P1과 P2 사이의 짝 비교는 다음과 같았다(Supplementary Table S4). P2 이진 응답은 P1보다 정확도가 0.16%p(0.09–0.22) 높았고, 수집 evidence가 3.08개(3.05–3.11) 적었다. 나이와 성별을 쓰는 P2 이진 응답은 P2 이진 응답보다 정확도가 0.06%p(0.03–0.08) 높았고, 수집 evidence가 0.15개(0.14–0.16) 적었다. P2 값 단위 응답은 P2 이진 응답보다 정확도가 0.70%p(0.65–0.76) 높았고, 수집 evidence가 2.17개(2.15–2.20) 적었다. 나이와 성별을 쓰는 P2 값 단위 응답은 P2 값 단위 응답보다 정확도가 0.02%p(0.00–0.03) 낮았고, 수집 evidence가 0.13개(0.12–0.13) 적었다.

종료 임계를 0.9에서 0.999까지 바꾸면 P1의 정확도는 98.11%에서 98.49%로, 수집 evidence는 9.02개에서 13.25개로 달라졌다(Supplementary Table S5). P2 이진 응답과 P1의 정확도 차이는 임계 0.9에서 −1.31%p, 임계 0.999에서 +0.37%p였고, 수집 evidence는 네 임계 모두에서 P2 이진 응답이 3.08–4.34개 적었다(Supplementary Table S4).

하위 질문의 답을 채우는 규칙을 적용하면 test 세트에서 정확도가 P1에서 0.11%p(0.08–0.15), P2의 네 변형에서 0.10–0.14%p 낮아졌고, 수집 evidence는 다섯 방법에서 0.06–0.43개 줄었다(Supplementary Table S4).

질환별로 보면 P1의 정확도가 가장 낮은 질환은 급성 부비동염으로 67.20%였다. 이 질환의 정확도는 P2 이진 응답에서 67.14%, P2 값 단위 응답에서 84.36%였다.

질문을 하지 않는 기준선의 정확도는 가장 흔한 질환을 답할 때 6.50%, 초기 evidence만 쓸 때 25.29%였다. 223개 evidence 모두의 답을 질문 없이 주면 목록 대조는 95.93%, P1의 확률은 99.19%, P2의 확률은 이진 응답에서 99.28%, 값 단위 응답에서 99.70%였다(Supplementary Table S6).

{{TABLE1}}

### 3) 질문 수에 따른 정확도

Figure 1은 질문 수에 따른 정확도다. 질문 30개 뒤의 정확도는 P1 98.49%, P2 이진 응답 98.88%, P2 값 단위 응답 99.63%, BED 재구현 99.71%였다. 정확도가 95%에 처음 이르는 질문 수는 P1 17개, P2 이진 응답 11개, P2 값 단위 응답 6개, BED 재구현 8개였다. P2 이진 응답과 P1의 정확도 차이는 질문 5개에서 17.05%p(16.79–17.30)였고, 질문 30개에서 0.39%p(0.33–0.44)였다(Supplementary Table S4). P1의 질문 선택만 바꾸면 질문 30개 뒤의 정확도가 무작위 질문에서 88.45%, 빈도순 질문에서 83.57%였다(Supplementary Table S7).

**Figure 1.** Accuracy for the true disease on the test set by number of questions asked after the initial evidence. P1 and P2 ask exactly the given number of questions without the stop conditions. BED asks up to the given number and asks no further question once no question has a positive utility. The P2 variants shown do not use age and sex. Supplementary Table S7 gives the plotted values at 0 to 6, 8, 10, 15, 20, 25 and 30 questions. P1, list-based method; P2, frequency-based method; BED, Bayesian experimental design.

![Figure 1](figures/figure1_accuracy_by_questions.png)

### 4) BED 재구현과 응답 규칙

validation 세트에서 BED 코드는 정확도 99.442%와 수집 evidence 5.472개를, 재구현은 99.442%와 5.476개를 냈다. test 세트에서 재구현의 정확도는 99.47%(99.43–99.51), 수집 evidence는 5.48개(5.45–5.50)였다. CASANDE 논문이 보고한 BED의 값은 99.47%와 5.47개다 [@casande]. CASANDE 논문의 BED 보고값 가운데 Supplementary Table S2에 함께 실은 세 지표도 재구현의 값과 같았다.

BED 재구현이 test 세트에서 종료할 때까지 물은 범주형과 다중선택 질문은 315,692개였다. 이 가운데 193,028개는 공식 규칙으로 "아니오"였고, 그 가운데 60,261개에 BED의 시뮬레이터는 양성으로 답했다. 이 답은 모두 기본값이 기록된 evidence에 대한 것이었다. 이런 답을 한 번 이상 받은 환자는 52,404명으로 test 세트의 38.95%였다.

같은 환자에서 P2 값 단위 응답과 비교하면, BED 재구현은 정확도가 0.16%p(0.12–0.21) 높았고 수집 evidence가 0.38개(0.36–0.40) 많았다. 질문 수의 상한을 맞춰 비교하면 질문 5개에서는 P2 값 단위 응답이 6.25%p, 10개에서는 0.71%p 높았고, 20개에서는 BED 재구현이 0.09%p, 30개에서는 0.08%p 높았다(Supplementary Table S4).

응답 규칙의 효과는 validation 세트에서 쟀다(Supplementary Table S8). P2 값 단위 응답의 정확도는 99.26%, 수집 evidence는 5.10개였다. 기록 여부를 함께 받는 P2 기록 포함 응답은 정확도가 0.10%p(0.06–0.14) 높았고 수집 evidence가 0.30개(0.29–0.31) 적어, 99.36%와 4.79개였다. BED 재구현은 validation 세트에서 99.44%와 5.48개였다. P2 값 단위 응답보다 정확도가 0.18%p(0.14–0.22) 높았고, P2 기록 포함 응답보다는 0.08%p(0.05–0.11) 높았다. 수집 evidence는 P2 기록 포함 응답보다 0.68개(0.66–0.70) 많았다.

### 5) 요소 분석과 환자층별 정확도

Table 2는 validation 세트에서 P1의 요소를 하나씩 바꾼 결과다. 질문에 대한 "예"만 반영하면 정확도가 86.64%였고, "아니오"만 반영하면 57.94%였다. 초기 evidence만 쓴 정확도는 25.22%였다(Supplementary Table S6). ε를 10⁻²에서 10⁻⁹까지 바꾸면 정확도는 98.17–98.42%였고, ε가 0.1이면 94.95%였다. 질문 선택을 바꾸면 정확도가 83.30–91.11%였다. 하위 질문을 상위 질문이 "예"일 때만 물으면 정확도가 0.24%p 높았고, 하위 질문을 묻지 않으면 0.20%p 높았으며, 답을 채우는 규칙을 적용하면 0.10%p 낮았다. 사전확률을 train 세트의 질환 빈도로 바꾸면 정확도가 0.25%p 낮았다.

validation 세트에서 양성 evidence를 모두 담는 목록이 정답 질환 하나뿐인 환자는 122,321명(92.35%)이었다. 이 환자들에서 P1의 정확도는 99.30%였고, 나머지 10,127명에서는 87.76%였다. P1이 잃은 점수의 59.28%가 나머지 환자에서 나왔다. 나머지 환자에서의 정확도는 P2 이진 응답 90.45%, P2 값 단위 응답 95.61%였고, 정답이 하나로 정해지는 환자에서는 각각 99.25%와 99.57%였다(Supplementary Table S9).

{{TABLE2}}

### 6) 문헌 보고값과의 비교

Table 3은 DDXPlus에서 보고된 값과 평가 조건이다. test 세트 전체에서 초기 evidence 뒤 최대 30개 질문으로 평가한 연구에서, 정답 질환으로 학습한 모형의 정확도는 91.81–99.21%, 평균 수집 evidence 수는 6.73–19.53개로 보고되었다 [@ddxplus; @casande]. 이 모형들은 train 환자로 학습하며, Diaformer를 뺀 세 모형은 나이와 성별도 입력으로 받는다. CASANDE 논문은 이 값들을 세 번 실행한 평균으로 보고했고, 함께 적은 95% 신뢰구간 값은 정확도에서 0.54–1.95%p였다 [@casande]. 응답 방식은 연구마다 달랐다. 공개 목록만 쓰는 대화형 기준선은 1,000명과 질문 상한 10에서 39.6–60.4%로 보고되었다 [@cotcagent]. LLM을 쓴 연구는 평가한 환자 수와 질문 상한이 공식 규칙과 달랐고, MEDDxAgent와 StreamBench는 시작 입력과 응답 방식도 달랐다 [@cardi; @meddxagent; @streambench].

{{TABLE3}}
