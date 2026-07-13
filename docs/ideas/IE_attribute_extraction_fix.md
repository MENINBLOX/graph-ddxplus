# IE 속성추출 편향 교정  📋 선행과제

**상태**: 검증으로 확정된 최우선 선행과제 (2026-06-01)

## 문제
data-derived attribute applicability ↔ 임상 semiology gold 상관 **r=−0.27** (음의 상관). location이 모든 증상에 과충전(dyspnea 0.93, fever 0.86), severity/onset/character는 과소충전. = PubMed가 어떤 증상이든 해부 서술하며 위치 흘림.

## 할 것
1. **location 과추출 억제**: 증상이 국소화 가능한 종류(pain/swelling/rash/lesion)일 때만 location 부여. 전신/감각 증상(fever/dyspnea/fatigue/nausea)엔 organ 언급돼도 금지.
2. **severity/onset/character 추출 강화**: post-validation 과drop 점검·완화, case-report 비중↑, 명시 지침 보강.
3. **character_dist enum 강제**: 비-성상 텍스트("clinical and radiographic signs") 제거.

## 검증 게이트
semiology gold 상관 **r>0 (목표 >0.5)** 회복 → 그 후에야 속성 가중·알고리즘 라인 재검증 공정.
스크립트: `pilot/scripts/v103_eval_applicability.py`
