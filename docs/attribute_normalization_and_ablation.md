# 속성 ② 표준 정규화 + ③ 변별 ablation — 조사·설계 (2026-06-25)

문헌 조사(웹) 기반. 두 질문: ② 속성을 표준에 어떻게 정규화하나, ③ 어떤 속성이 진단에 기여하는지 어떻게 정직하게 보이나.

## ② 속성별 정규화 타깃 (조사 결과)

| 속성 | 표준 타깃 | 정확 ID/커버리지 | clean? | fallback |
|---|---|---|---|---|
| **severity** | HPO Severity **HP:0012824** | Borderline HP:0012827<Mild 0012825<Moderate 0012826<Severe 0012828<Profound 0012829 (5점 ordinal) | **clean** | 없음 (SNOMER severity는 2025.11 deprecate) |
| **location** | UMLS Anatomy(T023/T029) / SNOMED Body structure 123037004 | scispaCy UMLS linker + TUI 사후필터(내장 필터 없음). T023/T029는 일관성 낮아 **단일 anatomy 버킷으로 병합** | partial | laterality는 HPO modifier(Unilateral/Bilateral)로 분리, 미연결시 문자열 |
| **onset-pace** | HPO Clinical course **HP:0011008** (Acute 0011009/Subacute 0011011/Chronic 0011010/Insidious 0003587) | ⚠️ age-of-onset(HP:0003674)와 **다른 가지**. 단 symptom-pace vs disease-course-pace 미분리 | partial | sudden→Acute, gradual→Insidious로 매핑 + binary pace flag 별도 보존 |
| **aggravating/triggers** | HPO **HP:0025204 "Triggered by"**(자식 33: 운동 0025377/한랭 0025206/음식 0033793...) + **HP:0025285 "Aggravated by"** | ⚠️ "postprandial" 정확term 없음(음식섭취로 근사), 비통증 positional 부재. SNOMED엔 aggravating attribute 없음 | partial | HPO child 있으면 매핑, 없으면 trigger ENTITY를 UMLS CUI로(운동 C0015259) + local relation |
| **character** | HPO Pain characteristic **HP:0025280**(Sharp 0025281/Dull 0025282/Tender) + 개별 phenotype(Productive cough HP:0031245) | ❌ 통합 quality subontology 없음. 흔한 quality 10개 중 ~40-50%만 term(throbbing/burning/aching/cramping 없음) | **no** | HP:0025280 child→개별 phenotype term→SNOMED/UMLS qualifier→문자열 순 |

**요약: clean 1(severity) / partial 3(location·onset·aggravating) / no 1(character).** 모든 속성에 fallback 층 필요. ⭐ **교정**: aggravating은 HPO HP:0025204로 생각보다 잘 정규화됨(character보다 나음).

## ② 진짜 기여는 "정규화"가 아니라 "cross-benchmark interlingua" (선행연구 결론)

- **이미 확립(우리 novelty 아님)**: 속성 slot 정규화=SemEval-2015 T14(location→UMLS, severity/course; 2015), CUILESS2016, CLINES/OntoGPT/PheNormGPT(2024-25). 속성=first-class 정규화 필드=GA4GH Phenopackets(severity/onset/modifiers←HPO). source→standard interlingua+adapter=OMOP CDM. 속성이 진단 향상=NLICE(+23%p).
- **진짜 빈칸(novelty)**: **NLICE의 속성값은 SymCat 내부 free-text 문자열**(Throbbing/Right_Lower_Quadrant/Worsening_after_meals), 온톨로지 코드 아님 → **SymCat에서만 작동, 전이 불가**. "속성을 표준 interlingua로 정규화해 **단일 KG/단일 알고리즘이 DDXPlus+SymCat+RareBench를 가로질러** 작동"한 선례 없음. **이 cross-benchmark 전이가 측정 가능한 미청구 기여.**
- **프레이밍**: "우리가 속성 정규화를 발명"(X, thin) → "**속성-정규화 KG = benchmark-agnostic interlingua**, NLICE의 닫힌 SymCat 어휘는 전이 불가하나 우리는 표준 정규화로 5벤치 전이됨을 실증"(O). 베이스라인=NLICE(신호 선례)·MEDDxAgent(I/O 정규화)·OMOP/Phenopackets(구조 선례) 명시 인용 후 transfer delta를 주장.
- **메커니즘(=엔지니어링)**: 위 표 per-attribute 타깃 + eval-time **benchmark adapter**(DDXPlus douleurxx_endroit→UMLS anatomy, soudain 0-10→sudden/gradual, SymCat Excitation→HPO trigger). adapter는 KG 아닌 평가단계만 → benchmark-blind 유지(교수님 leakage 분리).

## ③ "13속성 전조합 → top 고르기"의 문제 + 정직한 대안

**사용자 안(2^13 조합 search→top 3-5→해몽)의 치명결함**:
1. 2^13=8192 × 5벤치 = **다중비교 과적합**. "best combo"는 noise-선택. FATAL.
2. eval 벤치마크에서 best 고르기 = **leakage**, strict zero-shot·robustness 주장 붕괴.
3. 사후 해석 = reverse-engineering(=꿈보다 해몽 직감 정확).

**대안 (낮은 multiplicity·사전지정·전이검증)**:
1. **세트를 이론(①)으로 사전 고정** — 벤치 성능 보기 전. (anti-post-hoc)
2. **2-tier 평가로 신호/추출 분리**:
   - **Tier A — Oracle(gold 속성값)**: 속성의 내재적 진단 정보 측정(IE noise 無). "신호 있나"를 깨끗이(character artifact 재발 방지). ※ 분석 도구일 뿐, KG는 benchmark-blind 유지.
   - **Tier B — IE 추출**: 달성치. A−B gap = IE 개선여지.
3. **저-multiplicity ablation(2^N 아님)**: leave-one-out(N회: 각 속성 한계기여) + add-one-to-base(N회: 단독기여) = **2N회**(N=6→12회). **전부 보고**, cherry-pick 금지.
4. **cross-benchmark 전이를 검정**: dev 벤치에서 정하고 **held-out 벤치에서 확인**. "이 벤치 best combo"가 아니라 "이론 세트가 전이됨"이 주장.
5. **속성별 정보이론 변별도**(intrinsic·benchmark-blind 분석): 각 (symptom,attr)에서 disease별 속성값 분포 차이(mutual information / LR-spread) → "evidence별 적절 속성" 맵. 서술적 분석이지 해몽 아님.
6. **외부 anchor(Panju LR)**: 측정 기여가 published LR(location/radiation 강, character 유의미)과 일치하면 **수렴 검증** → 해몽이 triangulation으로 격상.

**왜 해몽이 아닌가**: 세트는 임상이론으로 사전고정, 모든 속성 기여를 정직히 보고(약한 것 포함), 선택은 dev→held-out 확인, 외부 LR로 anchor. 서술적("임상근거 속성이 X만큼 기여·전이")이지 역설계 아님. 13개 추출은 **분석용**(oracle ablation에 다 포함)은 OK이나 **최종 스키마는 이론고정 세트** — search가 스키마를 고르게 두지 말 것.

**출처**: HPO OLS(HP:0012824/0011008/0025204/0025280); SemEval-2015 T14(Elhadad); CUILESS2016(Osborne JBS 2017); CLINES(medRxiv 2025-12); OntoGPT/SPIRES(Caufield Bioinformatics 2024); Phenopackets(Jacobsen Nat Biotechnol 2022); OMOP CDM(OHDSI); NLICE(arXiv 2401.13756); RareBench(Chen KDD 2024); Panju JAMA 1998.
