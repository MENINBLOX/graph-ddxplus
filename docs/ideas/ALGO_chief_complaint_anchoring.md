# 주소증(chief complaint) anchoring  🔴 negative

**상태**: 검증됨 negative (2026-06-01)

## 아이디어
진단은 주소증(왜 왔는가)에서 출발. DDXPlus `INITIAL_EVIDENCE` 제공. 점수 = `P(D|주소증)^γ × 증상매칭`. 주소증과 양립 안 되는 질환 down-weight.

## 근거
Problem representation, 임상 추론(주소증이 differential anchor).

## 검증
deep120 KG, DDXPlus 5K: base 33.33 → γ=0.5:30.75 → γ=1:27.91 (단조 하락). 원인: IE-KG가 "어느 질환이 이 주소증으로 prominent"를 신뢰성 있게 인코딩 못 함(주소증이 true 질환 프로필에 약/부재 → 오히려 penalty).

## 결론
KG content noise 문제. 과거 v72 INITIAL_EVIDENCE boost도 marginal. IE 교정 후 재검증.
스크립트: `pilot/scripts/v103_eval_algoideas.py --idea chief`
