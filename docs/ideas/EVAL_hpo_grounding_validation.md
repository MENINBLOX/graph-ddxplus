# HPO modifier/frequency로 정당화·검증  🚫 blocked

**상태**: blocked (로컬 데이터 없음 + leakage 제약)

## 아이디어
- **C (검증)**: data-derived applicability ↔ HPO clinical-modifier 구조 일치로 "임상 applicability 복원" 입증. + 교수님 제안 "메인 KG vs +HPO union" ablation.
- **D (정당화)**: 우리 frequency_in_abstracts = P(E|D)를 HPO phenotype frequency(Obligate/Frequent/Occasional)의 데이터 복원으로 제시·검증.

## Blocker
- HPO 주석 파일(phenotype.hpoa/hp.obo) 로컬에 없음 (다운로드 필요).
- **HPO disease-phenotype 주석 = RareBench 라벨 leakage** (원칙: 절대 금지). 검증은 **비-벤치마크 질환**으로만, ablation은 HPO를 generic 자원(synonym/modifier 구조)으로만 사용해야 함.

## 할 것 (leakage-safe)
hp.obo의 modifier subontology 구조(disease 주석 X)만 받아 modifier 적용가능성 비교. frequency 검증은 비-벤치마크 질환 샘플로.
