# 논문 초안 (working draft)

> 구조 결정: **메인 논문 1편(자동 감별진단 시스템) + IE 산출물 dataset/resource companion**. (IE 단독 논문 분리는 기각 — 근거: `docs/paper1_necessity_debate.md`.)
> 본 초안은 메인 논문의 골격. [TODO]는 아직 측정/작성 전.

---

## 제목 (후보)
"Traceable Zero-Shot Differential Diagnosis across Benchmarks via an Attribute-Normalized Knowledge Graph built by a Local 8B LLM"

## Abstract (skeleton)
- 문제: 자동 감별진단 KG는 (a) 증상 유무만 표현해 속성 정보를 잃고, (b) 벤치마크별로 따로 만들어 종속·비전이.
- 방법: benchmark-blind 질환 지식 텍스트에서 **로컬 8B LLM**으로 finding + **임상 표준 속성**(Bates seven attributes)을 추출, disease–phenotype edge의 **qualified statement**로 부착하고 **표준 온톨로지(HPO/UMLS/SNOMED)에 정규화**. 단일 KG·단일 알고리즘으로 5개 벤치마크 평가.
- 결과: [TODO 수치] DDXPlus/SymCat/RareBench cross-benchmark; 외부 큐레이션 0으로 X%, 표준자원 union 시 Y%.
- 기여: 표준-정규화 속성을 **benchmark-agnostic interlingua**로 써서 단일 KG가 이종 벤치마크를 가로질러 작동(NLICE의 닫힌 SymCat 어휘는 전이 불가). traceable provenance + 8B 비용효율.

## 1. Contributions (주장 = 속성 선택이 *아님*)
1. **Benchmark-blind, source-grounded 속성 IE**: 로컬 8B LLM이 질환 지식 텍스트에서 finding+속성을 추출, **인정 gold(MACCROBAT/SemEval) + 전문가 adjudication**으로 IE 품질 검증. (산출물=공개 dataset)
2. **표준 정규화 속성 = cross-benchmark interlingua**: 속성을 HPO/UMLS/SNOMED에 정규화 + 평가단계 adapter로 각 벤치마크 코드를 표준에 매핑 → **단일 KG가 5벤치 작동**. (선행: SemEval-2015·Phenopackets·OMOP·NLICE를 명시 인용하고 *cross-benchmark 전이 delta*를 주장.)
3. **Cross-benchmark robustness + provenance + 8B 비용효율**: SOTA 점수가 아니라 전이·추적성·비용이 강점(교수님 정의).

## 2. Related Work [TODO 확장]
- LLM 임상 IE / 속성 정규화: SemEval-2015 T14, CUILESS2016, CLINES, OntoGPT/SPIRES, PheNormGPT.
- 표준 interlingua: OMOP CDM, GA4GH Phenopackets, HPO 허브.
- 속성-증강 진단: **NLICE**(SymCat 58.8→82%, 단 SymCat 내부 어휘) — 우리의 직접 foil.
- KG 진단 / 벤치마크: DDXPlus, SymCat, RareBench, MEDDxAgent.

## 3. Methods

### 3.1 KG 구축 + IE (benchmark-blind)
- 입력: 질환 임상텍스트(교과서/Wikipedia 임상섹션) + UMLS DISO 시드 PubMed. 외부 큐레이션 KG 미흡수, raw text→우리 IE만.
- 모델: gemma-4-E4B(local, bf16, vLLM), temp=0, **few-shot 금지**, source-grounded CoT. finding 무제한.
- 정규화: scispaCy + UMLS/HPO linker (IE의 mention 출력과 **분리**된 단계).

### 3.2 속성 스키마 (← ① 확정, `docs/attribute_selection_final.md`)
> **한 문단 + 표 1.** 속성 선택은 정당성 토대이지 기여가 아님.

증상 특성화의 교과서 표준인 **Bates seven attributes**를 채택한다. KG 슬롯 = **location(+radiation을 `radiates-to` 방향 qualifier로), character, severity, timing{onset·duration·frequency}, aggravating, relieving**. associated는 공존 finding으로 구조 표현, disease-phenotype frequency는 edge prior로 분리. 각 속성은 표준 온톨로지(HPO HP:0012824/0011008/0025204/0025254/0025280, UMLS Anatomy, SNOMED)에 정규화 가능한 범위에서 매핑하고, 통제어휘가 없는 character 일부는 문자열로 보존한다. 병력청취 mnemonic(OLDCARTS/OPQRST/SOCRATES/LOCATES/COLDSPA)과 NLICE(계산 프레임워크)는 보강 근거이며, Bates·COLDSPA를 포함해도 합의 코어가 유지됨을 robustness check로 보고한다(표 1).

**[표 1]** Bates 속성 × mnemonic 합의 + 온톨로지 타깃 + KG 슬롯 (= `attribute_selection_final.md` §2-3).

### 3.3 속성을 qualified edge로 + 표준 정규화 (② 기여)
- disease–phenotype edge에 속성을 **qualified statement**로 부착(노드 승격 금지, 그래프 비대화 방지; Phenopackets 정합). numeric severity edge-weight 미사용.
- **평가 어댑터**: 각 벤치마크 고유 코드(DDXPlus douleurxx_*, SymCat NLICE 문자열, RareBench HPO)를 **평가단계에서만** 표준으로 매핑 → KG는 benchmark-blind 유지(leakage 분리).

### 3.4 진단 알고리즘 (단일, 모든 벤치마크 동일)
- [기존 KG-NB / cosine+IDF / 속성-aware 매칭 기술]. inference에 LLM 미사용(KG+단일 algorithm).

## 4. Evaluation

### 4.1 IE 품질 (intrinsic, B) — "추출이 충실한가"
**축별 검증 등급표(표 2, [TODO 작성])**:
- **T1 (인정 gold)**: location, severity → MACCROBAT MODIFY / SemEval slot, P/R/F1. + gold-deflation **전문가 재판정**.
- **T2 (gold 부재)**: onset, character, aggravating, relieving, timing → source-grounded faithfulness + **소규모 전문가(임상의) adjudication gold**.
- 현재 수치(잠정): location/severity 통합 프롬프트 F1 0.56–0.61(MACCROBAT); onset 자체점검 정밀도/재현율(임의 gold, 비검증) — 전문가 adjudication 필요.

### 4.2 Cross-benchmark 전이 (extrinsic, ③) — "어떤 속성이 전이되는가"
- **Oracle/IE 2-tier**: gold 속성(oracle)으로 속성의 진단 정보 측정 → IE 추출로 달성치.
- **저-multiplicity ablation**: leave-one-out + add-one (2N회, 전조합 아님). **전부 보고.**
- **전이 검정**: dev 벤치에서 정하고 **held-out 벤치에서 확인**.
- **baseline**: bare-CUI KG / location-only / 동일 8B end-to-end LLM / NLICE.
- **외부 anchor**: 측정 기여 vs Panju LR(수렴 검증).
- 범용 vs 벤치마크별 KG의 %p 차이 보고(교수님 지침). 메인=IE만, ablation=+Orphanet/HPO union.

### 4.3 지표 [TODO]
GTPA@1/@5, MRR, calibration(ECE), disease-stratified accuracy, cross-benchmark transfer, trace fidelity.

## 5. Results [TODO]

## 6. Discussion / Limitations
- 기여=robustness/provenance/비용효율(SOTA 아님).
- 한계: IE recall(fine schema 산물), onset/character gold 부재(전문가 adjudication 의존), duration numeric 미표현, character 비정규화 잔존.

## 7. Companion dataset (별도 release)
- register-general·vocab-decoupled, onset·character 포함 소규모 전문가-gold, falsifiable deflation-adjudication, 진단 논문 cross-citation (조건: `docs/paper1_necessity_debate.md`).

---
### 부속 문서
- ① 속성 선택 확정: `docs/attribute_selection_final.md`
- 정규화/ablation 설계: `docs/attribute_normalization_and_ablation.md`
- IE 속성 검증 기록: `docs/ie_attribute_validation_review.md`
- 논문 구조 결정(공방): `docs/paper1_necessity_debate.md`
