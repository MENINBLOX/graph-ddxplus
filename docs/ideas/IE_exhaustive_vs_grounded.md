# IE 방식: LLM-exhaustive vs source-grounded  🟢 핵심 설계축

**상태**: 둘 다 검증됨 — trade-off 확정 (설계 결정축)

## 두 방식
### exhaustive LLM IE (self-knowledge) — v95_full
- LLM이 disease명만 받아 "exhaustive list of symptoms" 생성(자기지식, hallucination 포함).
- 결과: disease당 264 phen, 환자 CUI coverage 97.5%, **DDXPlus @1 62%** (현 production).
- 약점: source 미인용 → 검증 불가(학술적 약점).

### source-grounded IE — v103
- 근거 문장 인용(evidence_span) 강제 → 무환각. PubMed/MedlinePlus 텍스트에서만 추출.
- 약점: sparse(초기 12 phen) → deep crawl로 179편 사용 시 coverage 24.9→61.4%, @1 18→34, @10 79.
- 강점: traceable provenance(논문 contribution 축).

## 결론 (핵심 trade-off)
**무환각·검증가능(grounded) ↔ coverage·점수(exhaustive)**. grounded가 학술적으로 옳으나 coverage 회복이 과제(deep crawl + MedlinePlus). 논문은 "grounded만으로 X%, 표준자원 추가 시 Y%"(교수님 제안 ablation) framing.
관련: 메모리 project_v103_pubmed_only_ceiling, project_v95_full_sota.
