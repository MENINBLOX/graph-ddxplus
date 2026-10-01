# IE 파이프라인 (production) — consolidated reference

**최종 정리 2026-07-15.** disease → phenotype + 임상 표준 속성 추출 → UMLS CUI/SNOMED 정규화
→ 속성-KG의 disease→phenotype edge(qualified statement). **벤치마크 gold 미사용, gold-free 검증.**

---

## 0. 한눈에 — end-to-end stages

| # | Stage | Script | Input → Output | Env |
|---|-------|--------|----------------|-----|
| S0 | Disease seed | `data/medkg/seeds/combined_seed.jsonl` | — → 38,906 diseases (CUI+name, rare-inclusive: OMIM/ORPHANET/HPO/RareBench) | — |
| S1 | Source (scale) | `v156_build_sources.py` (pubmed∪sections) · crawl: `medkg_pubmed_crawl_fast.py`(+`_synonym_retry`) | disease → 증상-풍부 텍스트 blob `{cui}.json` | — |
| S2 | **IE** | `v153_production_ie.py --backend vllm --source prod --resume` | disease + source → atomic findings + 6 raw attrs | `.venv` (GPU, vLLM) |
| S3 | **Normalize + Validate** | `v155_normalize_umls.py` (threshold-free; +`attr_normalize`) | IE + source → KG-ready records | `.venv` (plain py3) |
| S4 | **KG build** | `v157_build_kg.py` | KG records → bipartite 속성-KG + §12 topology | — |

산출물: `pilot/cache/prod/sources/{cui}.json`(S1) · `ie_pred_full.json`(S2) · `kg_records.json`(S3) · `kg.json`(S4).

**Scale status (2026-07-16)**: source coverage 38%→**92.5%**(rare 95.4%) via pubmed 전건 재크롤+synonym-retry
(`docs/source_coverage_audit.md`). vLLM 동작 확정(WSL2 3-flag, transformers 대비 ~10x) → 38k IE ≈ 10~20h
1-run(checkpoint/resume). S3→S4 32-disease 검증 완료(name CUI 0.97, KG 402 edges, noise 9 제거).

---

## 1. S2 — IE (추출)

- **모델 = gemma-4-E4B** (오픈 패널 선정: gemma-4-12B·Qwen2.5-7B/14B·medgemma-4b 전부 열세; 차이는 recall).
- **프롬프트 = 단일 V4** (1 call/doc, 6속성 동시, 내부 STEP1 원자화(site 분해)→STEP2 배정). **benchmark-blind, few-shot 금지, temp=0.**
- **6 후보 속성**: onset · severity · location · character · timing · aggravating · relieving.
- severity/onset은 **원문 표현 그대로(raw) 추출**(faithful) — 정규화는 S3에서 결정론적으로.
- disease-oriented framing("이 질환 환자가 겪는 증상"), 질환명·원인·치료·역학·정상소견 제외.

## 2. S3 — 정규화 + gold-free 검증

**개념형 (finding / location / character) — threshold-FREE (`umls_link.py`)**
- **UMLS 공식 정규화 인덱스(MRXNS_ENG, SPECIALIST Lexicon norm)에 EXACT match.** 유사도 임계치 미사용 = MetaMap/QuickUMLS/cTAKES 계열 표준 사전기반 매핑이라 학술 방어 완결. (이전 scispaCy CandidateGenerator+임계치 0.70/0.88은 magic number라 폐기.)
- 쿼리 term에 **결정론적 형태소 축약** 적용 후 매칭: 굴절정규화(단복수·동사원형 deverb), severity/onset 형용사 strip, 수량접두어 strip("amount of cough"→cough), head noun-phrase("medical shock"→shock). 전부 exact-match, 임계치 없음.
- 다중 CUI **disambiguation = semantic-type 우선순위**(Sign/Symptom>Finding>…>Neoplasm), 결정론적.
- **CUI → SNOMED CT** (MRCONSO, `cui_snomed.tsv` 452K) + semantic type(MRSTY).
- **clin-type 필터**: finding은 disease/symptom/finding 타입만 인정(anatomy/qualifier 오링크 배제) = §12 노이즈 필터.
- 실측(15 diseases): coverage **99%**(exact 85% + head-noun/deverb/latin-irregular). KG-유입 링크 ~97% 정확. 남은 오류는 링킹 아닌 **상류 IE 추출 품질**(vague/anatomy term).
- **왜 100% 불가 + 논문 정책**: 남은 1%는 (b)조합형 패러프레이즈("sensation of heart beating irregularly"=palpitations, IE가 서술 추출) + (c)UMLS 복합어-only gap("hypersecretion"). 형태소로 회복 불가 — **100% 강제 = 제거한 유사도 임계치 부활**. 논문: threshold-free 99% 보고(강제 100% 아님) + linkable-coverage ~100% 보조 + 잔여는 **free-text+source provenance 노드로 보존**(traceability 축) + 유형 특성화. 문헌 대비 EL SOTA ~56%(MedMentions)라 threshold-free 99%가 강함.

**범주형 (severity / onset)** — `attr_normalize.py`, 결정론적
- severity → {mild, moderate, severe}: 형용사 · **NRS 0-10**(1-3/4-6/7-10) · grade I-IV. 비-grade(increased/worsened 등) 거부.
- onset → {sudden, gradual}: acute/rapid→sudden, insidious/chronic→gradual.
- **LLM에 정규화 안 시킴** — raw는 faithful 추출, 매핑은 결정론적(0-10 대응 가능, garbage 거부).

**검증 (gold-free, 벤치마크 0)**
- **faithfulness** = grounding(IE의 CUI ∈ 원문 CUI 집합, 개념수준 · string 무관) + hallucination(원문에 없는 개념). **결정론적, NLI/LLM-judge 아님.**
- **normalizability** = CUI-link · SNOMED · clin-type.
- 원문 grounding으로 검증 → gold label 불필요.

---

## 3. 실측 (소규모 15 diseases, 증상-풍부 소스)

| field | n | CUI | SNOMED | clin-type | grounding | halluc |
|-------|---|-----|--------|-----------|-----------|--------|
| finding | 240 | **1.00** | 0.93 | 0.80 | **1.00** | **0.00** |
| location | 64 | **1.00** | 0.95 | 0.88 | **1.00** | **0.00** |
| character | 33 | 0.94 | 0.79 | 0.52 | 1.00 | 0.00 |

(CandidateGenerator 전환 전 NER-기반: finding CUI 0.85/SNOMED 0.79/clin-type 0.59 — 흔한 증상 미링크 15%. 전환 후 **미링크 0건**.)

severity/onset: 실제 grade는 ~100% 통제어휘 매핑, 비-grade는 정확히 거부. IE density 16.0 findings/disease.

**적용 최적화**: ① name-cleaning(CUI 정확성) ② 증상-풍부 소스(location 0.80→0.88) ③ self-contained grounding.

---

## 4. 핵심 설계 결정 (rationale)

- **벤치마크 gold 폐기** (사용자 확정): MACCROBAT 등은 소규모·구조결함(multi-site 중복·노이즈·불완전)·IE 비최적화. raw pair-F1·conditioned accuracy·coverage 등 gold-대조 지표 전부 미사용.
- **gold-free CUI-grounding 채택**: 선행연구의 "LLM-free reference-free fact-checking" 계열(개념수준 변형). LLM-as-judge 회피 = same-family 편향 문헌이 지지.
- **UMLS/SNOMED는 벤치마크 아님** = KG 정규화 용어체계(interlingua). 이 검증 = KG 정규화 단계 그 자체.
- **severity 결정론적 정규화**: 스키마 통제어휘 강제 + 0-10 NRS 대응(교수님/CLAUDE.md 매핑).

## 5. 한계 (정직)

- gold-free는 **faithfulness·normalization coverage는 재나 right-CUI(맞는 CUI인가)는 못 잼** (scispaCy linking SOTA도 MedMentions ~56%). → 내부 sanity로만 확인, 논문엔 한계로 명시(수동감사 "검토 데이터"로 표기 금지).
- 모델 accuracy 최종선정은 gold-free 변별력 약함 → **extrinsic(진단 gain)의 몫.**
- character는 free-text(clin-type 0.18) — 정규화 부적합, 한계 명시.

## 6. 남은 것

- **✅ Source coverage**(해결): 38%→**92.5%**(rare 95.4%). pubmed 전건 재크롤(26,875 회수)+synonym-retry
  (3,775 회수)+hpoa. 잔여 7.5%=ICD 행정버킷(문헌無)→한계 명시. `docs/source_coverage_audit.md`.
- **✅ Inference 속도**(해결): vLLM WSL2 3-flag 동작 → transformers 대비 ~10x. 38k ≈ 10~20h 1-run.
  (과거 "flashinfer 막힘"의 진짜 원인 = `VLLM_USE_FLASHINFER_SAMPLER=0` 누락. memory `server-env-2026-07`.)
- **🔴 다음 = extrinsic 진단 평가**(논문 하중): 완성 KG(`kg.json`) → 단일 algorithm 진단(GTPA@1) +
  **속성 leave-one-out ablation**(#8 IDF/IC 가중, `docs/evidence_weighting_dedup.md`) + cross-benchmark
  (DDXPlus/SymCat/RareBench…). 정규화 안 되는 속성(character)은 KG+LLM 변형에서 활용. 진단 harness 를
  새 속성-KG 스키마에 연결하는 작업이 남음.

## 7. 참조

- 상세 실험 로그·검증 방법론: `docs/ie_strategy_maccrobat_results.md`(§13 전체-corpus, §14 gold-free 확정).
- 선행연구 대조: memory `ie-validation-priorwork-2026-07`.
- 스크립트: S2 `v153_production_ie.py` · S3 `v154_prod_normalize.py`(+`v152_normalize_pipeline.py`, `attr_normalize.py`) · 검증데모 `v151_cui_grounding.py`.
- 환경: `.venv`(transformers/GPU) · `.venv-sci`(scispaCy/UMLS) — memory `server-env-2026-07`.
