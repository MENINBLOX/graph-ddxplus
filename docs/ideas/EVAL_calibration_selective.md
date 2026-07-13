# Calibration(ECE) + 선택적 예측  🟡 미검증 (유망)

**상태**: 미검증, content-noise와 무관하게 유효. 저비용.

## 아이디어
@1을 SOTA로 못 올려도, **"확신할 때 정확도(selective accuracy)"와 calibration**은 임상 신뢰성 contribution. confidence = top1−top2 margin(또는 softmax). accuracy-vs-confidence 곡선, ECE, selective-risk(risk-coverage) 보고. "불확실하면 referral" 시나리오.

## 근거
교수님 제안 평가지표(ECE, calibration). 논문 contribution (iv) 임상 실증 가능성.

## 왜 유망
기존 점수만 재활용 → KG content noise 무관. @1 경쟁이 아니라 신뢰성 축으로 차별화.

## 할 것
margin/entropy로 confidence 정의 → ECE·risk-coverage curve·selective accuracy 측정·plot. 5 benchmark 공통.
