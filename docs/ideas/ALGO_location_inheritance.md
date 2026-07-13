# Location encounter-anchor soft 상속  ⚪ 미검증 (Stage 2)

**상태**: 미검증. 인터랙티브 Stage 2(IL 절약)용.

## 아이디어
location은 진료 단위 공유 맥락(다리 부종 환자 → 다른 증상도 "다리에" anchor). severity/onset/character는 per-symptom(상속 불가). encounter anchor = 명시 위치들의 가중합(주소증에 높게), 위치 미명시 증상에 soft prior(γ<1, override 가능)로 상속.

## 근거
임상 review of systems는 regionally anchored. spatial(location) inheritable vs temporal/qualitative 비inheritable 구분.

## 왜 미검증
static eval에선 위치가 주어져 효과 제한. **Stage 2 인터랙티브 질문선택에서 IL(질문수) 절약**으로 실효. applicability/discriminativeness × infogain으로 질문 우선순위.
