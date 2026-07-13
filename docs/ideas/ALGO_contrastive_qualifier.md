# Semantic qualifier를 변별 신호로  🔴 negative

**상태**: 검증됨, DDXPlus negative (2026-06-01)

## 아이디어
@10=79%인데 @1=34% = 유사질환 변별 실패. semantic qualifier(acute/chronic 등)가 "differentiating power"(Bordage)를 주니, 후보 간 값이 갈리는 속성에 가중.
- A1: attribute value-IDF (환자 값이 특이하면 up-weight)
- A2: cross-disease entropy (한 증상에 질환들 간 속성값 spread 크면 up-weight)

## 근거
Bordage & Lemieux (Acad Med 1991): 우수 진단가일수록 semantic qualifier 多 사용.

## 검증
appl 대비 A1/A2 모두 +0.0~0.04%p (노이즈 수준). 원인: DDXPlus 속성 coarse + 변별 핵심속성(onset) KG에 sparse → contrastive 신호 부재.

## 결론
이론 타당하나 속성 신뢰도 부족으로 DDXPlus negative. 속성 풍부+신뢰 KG(RareBench, IE 교정 후) 재검증.
