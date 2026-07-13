# KG noise hub CUI 제거  🟡 부분

**상태**: 부분 적용 (v94), 원칙 #11 연계

## 아이디어
demographic/temporal/qualifier CUI(Woman·year·Severe·History·Age·"Symptoms" 등)가 phenotype으로 inject되어 **IDF discrimination 약화 + hub(deg>1000)** 형성. HAS_PHENOTYPE에서 제거.
- 스크립트: `v94_kg_cleanup.py`, 진단 `kg_topology_inspect.py`.

## 근거
CLAUDE.md 원칙 #11(topology quality): noise hub 검출(Woman/Age/Severe = noise) → 즉시 제거. Singleton/hub 모니터링.

## 상태
일부 적용했으나 16개 noise hub 완전 제거는 미완(pending). 효과는 소폭/중립 추정. **deep KG 재빌드 시 noise hub 필터를 build_kg에 내장**하면 일관 적용.

## 할 것
v103_build_kg_cui.py에 stop-CUI/semantic-type 필터 내장 → topology_inspect로 hub 재확인.
