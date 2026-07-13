# 아이디어 인덱스

## 네이밍 규칙 — 연구 단계별 prefix
`{STAGE}_{상세}.md`. 한 아이디어당 한 파일. 각 파일: 아이디어 / 근거 / 검증결과 / 결론 / 스크립트.

| STAGE | 단계 | 범위 |
|---|---|---|
| **SEED_** | 시드 선별 / 코퍼스 크롤 | UMLS DISO seed, PubMed deep crawl |
| **IE_** | 정보추출 | 속성·관계 추출, recursive, source-grounding, 증상/검사 분리 |
| **KG_** | KG 구축 | phenotype→CUI 매핑, edge 가중, 정규화 |
| **ALGO_** | 진단 알고리즘 | scoring(cosine/NB), prior, 변별, 계층, 속성가중 |
| **EVAL_** | 평가 / 검증 | benchmark별 평가법, 지표(ECE), 외부검증·ablation |

상태: 🟢positive / 🔴negative / 🟡미검증(유망) / ⚪미검증 / 🚫blocked / 📋선행과제 / 🟦보류(제약상)

## 목록
| 파일 | 아이디어 | 상태 |
|---|---|---|
| SEED_deep_crawl.md | deep PubMed crawl (20→179편) | 🟢 채택 (@1 18→34) |
| IE_attribute_extraction_fix.md | location 과추출 억제 등 IE 편향 교정 | 📋 선행과제 |
| IE_attribute_expansion.md | laterality·radiation·시간패턴 추가 | ⚪ 미검증 |
| IE_recursive_enrichment.md | 꼬리물기(recursive) — profile 풍부화 vs phen-phen edge | 🟢 (a)채택 / ⚪ (b)재시도 가능(원칙 제약 삭제) |
| IE_symptom_test_separation.md | 일반증상 vs 검사결과 분리 (SOAP S/O) | 🟡 부분탐색, 서비스분리·차등가중 미완 |
| KG_cui_mapping.md | phenotype name→CUI 매핑(현 51%) | 🔴 n-gram 확장 @1 regression |
| KG_noise_cleanup.md | noise hub CUI(Woman/Age/Severe) 제거 | 🟡 부분 적용 (원칙 #11) |
| IE_exhaustive_vs_grounded.md | LLM-exhaustive vs source-grounded IE | 🟢 trade-off 확정 (설계축) |
| ALGO_cluster_discriminative.md | 질환-쌍 cluster differential CUI 가중 | 🔴 regression (v68/v96/v98) |
| ALGO_demographic_conditioning.md | age/sex demographic prior | ⚪ marginal (v67), 재검증 |
| ALGO_llm_inference_rerank.md | 추론 시 LLM(CoT/self-consistency) 재순위 | 🟦 보류 (원칙 #4 위반) — KG-only 부족 시 fallback |
| ALGO_attribute_applicability.md | 데이터기반 속성 applicability 가중 | 🔴 @1 gain은 artifact (외부검증 FAILED) |
| ALGO_contrastive_qualifier.md | semantic qualifier를 변별 신호로 | 🔴 negative |
| ALGO_chief_complaint_anchoring.md | 주소증 prior/gate | 🔴 negative |
| ALGO_naive_bayes.md | KG frequency 기반 Bernoulli NB | 🔴 negative |
| ALGO_umls_hierarchy.md | UMLS 계층 coarse→fine | ⚪ 미검증 |
| ALGO_location_inheritance.md | location encounter-anchor soft 상속 | ⚪ 미검증 (Stage 2 IL) |
| ALGO_value_weighted_evidence.md | numeric VAS를 evidence 가중 | 🔴 과거 regression, 보존 |
| EVAL_calibration_selective.md | calibration(ECE) + 선택적 예측 | 🟡 미검증, content-noise 무관 유효 |
| EVAL_hpo_grounding_validation.md | HPO modifier/frequency로 정당화·검증 | 🚫 blocked (데이터+leakage) |

## 메타결론 (2026-06-01)
ALGO 라인(applicability/contrastive/chief/NB)이 **전부 negative**. 공통 원인: 현재 v103 IE-KG content가 noisy → 절대값 의존 방법이 noise 증폭. cosine+IDF(robust)만 버팀. GT-KG 98% vs 우리 49% 발견과 일치.
→ **lever는 ALGO가 아니라 IE content 교정** (`IE_attribute_extraction_fix.md`). EVAL_calibration은 content 무관하게 병행 가능.
