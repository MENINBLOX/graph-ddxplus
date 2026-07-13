# SEED/CRAWL: deep PubMed crawl  🟢 채택

**상태**: 검증·채택 (2026-05-31)

## 핵심
- Seed = **benchmark-blind UMLS DISO pool** (`medkg_umls_seed.py`: TUI 6종 + clinical SAB + ENG). 운영 manifest `v103strict_seed_manifest.tsv` (34,895). benchmark disease는 자연 subset (절대 benchmark에서 추출 X — 위반).
- **Deep crawl**: 질환당 abstract 20편(중앙값 13) → **depth=300, ~179편**. crawler `medkg_pubmed_deepcrawl.py`(retry+batched efetch, NCBI key).
- 효과: DDXPlus raw-text symptom coverage 24.9%→61.4%, **@1 18→34%, @10 79%**.

## 남은 idea
- depth=300에도 적게 회수되는 질환(whooping cough 16, min 1) → **alt-query/synonym 확장 크롤** 보강.
- crawl 우선순위는 ORDER만(blind 유지).

## ✅ IE 완료 후 필수 확인 (TODO)
- **Evidence(증상/검사) coverage 100% 도달 여부 검증** (CLAUDE.md 원칙 #1: Disease 100% + Evidence 100% 필수).
  - 현재(부분 KG, deep120·일부 질환만 IE 완료): DDXPlus 환자 CUI coverage **~61%**.
  - blind IE 전체(34,895) 완료 시점에 **5개 benchmark 각각의 (disease, evidence) CUI가 100% cover**되는지 측정.
  - 미달 시: 부족 evidence CUI를 채울 source/alt-query 추가 (원칙 #1 사전계획대로).
  - 검증: benchmark별 `환자 evidence CUI ∩ KG phenotype CUI` 비율.
