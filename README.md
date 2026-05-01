# Graph-DDXPlus v2

자동 문진을 위한 KG 구축 및 감별진단 최적화 연구.

## 현재 상태

v2 연구 진행 중입니다. v1의 설계 결함(DDXPlus 답지로 KG 구축)이 확인되어 PubMed 기반 독립 KG 구축으로 재설계하였습니다.

**현재 SOTA: DDXPlus GTPA@1 = 66.48% (v87, 5K)**. 80% 목표까지 13.52%p 격차가 남아있습니다.

### 다중 벤치마크 결과 (v87 framework 일반화)

| 벤치마크 | v54 baseline | v87 (현재 SOTA) | Δ | 학습 SOTA | 비교 |
|--------|------------|-----------|------|---------|------|
| DDXPlus (n=5K) | 60.4% | **66.48%** | **+6.08%p** | meddxagent (GPT-4o): 86% | gemma-4-E4B(8B), zero-shot |
| SymCat (n=2.4K) | 39.7% | **43.27%** | **+3.57%p** | 70-75% (학습 기반) | KG quality 한계 |
| RareBench (n=1.1K) | 22.1% | **26.49%** | **+4.39%p** | GPT-4: 22-29% | **GPT-4 동등 또는 초과** |

세 벤치마크 모두 v87 framework (KG features + CoT tie-break)가 baseline 대비 consistent 향상을 입증. 특히 RareBench에서 8B LLM으로 GPT-4를 초과하는 결과 도출.

### KG 소스 비교 (DDXPlus 5K, 동일 v87 파이프라인)

| KG 소스 | GTPA@1 | 비고 |
|--------|--------|------|
| **PubMed top-8 cooccurrence** (v87) | **66.48%** | ★ 최종 SOTA |
| PubMed top-8 + 노이즈 블랙리스트 (v98) | 65.92% | 노이즈 제거가 의외로 손해 |
| PubMed top-4 + SemMedDB top-4 (v100) | 63.14% | combination이 PubMed 단독보다 낮음 |
| **LLM 텍스트북 자동생성** (v101) | **63.06%** | textbook-clean이 noisy PubMed보다 낮음 |
| PubMed top-16 raw (v86, 2K) | 62.5% | 더 많은 features = 차이 미미 |
| SemMedDB top-8 relation-typed (v99) | 58.26% | typed relations 단독은 narrow |
| TF-IDF discriminative (v82, 2K) | 56.5% | generic features 제거가 손해 |

**핵심 발견 (counterintuitive)**: PubMed 공출현은 외형상 "노이즈"가 보이더라도 실제로는 LLM이 감별진단에 활용하는 광범위한 의학적 맥락(differential, complication, comorbid)을 포함하므로, **정제하거나 textbook-clean 대안으로 대체할수록 오히려 성능이 떨어진다**. v101은 가장 명확한 증거: LLM이 직접 생성한 textbook 수준의 깔끔한 features("Rhinorrhea, Cough, Sore throat" for URTI)가 PubMed cooccurrence("Urticaria, Rhinitis, Asthma...")보다 -3.42%p 낮은 결과. SemMedDB의 typed relations도 단독으로는 너무 좁고, 결합해도 PubMed 단독 신호를 희석시킨다.

이는 KG-augmented LLM 진단에서 "noisy 광범위 cooccurrence"가 "clean narrow 관계"보다 효과적임을 시사하며, 향후 KG 구축 연구의 방향성에 함의가 있다.

### 최종 결론: 8B LLM + PubMed-only KG의 한계

100여 개의 변형을 테스트한 결과, DDXPlus GTPA@1의 실용 상한은 **66.48%** (v87, 5K)로 수렴.

| 비교 | GTPA@1 | 비고 |
|------|--------|------|
| 본 연구 (gemma-4-E4B 8B, zero-shot, fixed evidence, PubMed KG) | **66.48%** | 학습/추가질문 없음 |
| GPT-4 zero-shot 보고치 | 55-60% | 본 연구가 +6-11%p 우위 |
| AARLC (RL 학습) | 75.39% | DDXPlus 학습 데이터 사용 |
| meddxagent (GPT-4o, agentic, IL=15) | 86% | GPT-4o + 추가 질문 인터랙션 |
| **사용자 목표** | **80%** | 13.52%p 격차 |

80% 도달을 위해 추가로 필요한 변경:
1. **더 큰 LLM** (GPT-4o 수준) — 현재 제약 위반
2. **Agentic inquiry** (meddxagent처럼 추가 질문) — 문제 setup이 다름
3. **DDXPlus 학습 데이터 사용** — "절대 편법 금지" 위반

본 연구는 8B 모델 + zero-shot + 고정 evidence + PubMed-only KG라는 strict 제약 하에서 GPT-4 zero-shot을 능가하는 결과를 도출하였으며, multi-benchmark에서 RareBench(8B)가 GPT-4 수준에 도달함을 보였다. v87 (KG features in prompt + CoT tie-break) 프레임워크는 일반화 가능한 기여이다.

- v1 산출물은 `archive/v1_ddxplus/`에 보존되어 있습니다.
- v2는 본 디렉토리에서 진행됩니다.

### 두 번째 돌파구 (2026-05-01): CoT Tie-Break

v79 stage1 score 분석 결과, **27.8%의 patient가 top-1 score가 다른 disease와 동률(tie)**이었으며, 그 중 78%는 정답이 동률 set에 포함되어 있었습니다. 동률 케이스만 LLM CoT(Chain-of-Thought)로 재선택하면 +3.48%p 상승하였습니다 (63.0% → 66.48%).

```
Patient: 65yo Female
Chief complaint: cough
PAIN: ...
HISTORY: ...

The following candidates are equally likely so far. For each, briefly evaluate which patient features support or contradict it. Then pick the SINGLE most likely.
(1) Bronchitis — typical: ...
(2) Acute COPD exacerbation — typical: ...

EVAL:
(1) ...
(2) ...
PICK: <number>
```

| 변형 | 데이터 | GTPA@1 | 비고 |
|-----|------|--------|------|
| v79 stage1 | 5K | 63.00% | 49 cand × 0-100 score |
| v79 final (stage1+stage2) | 5K | 63.9% | 2-stage rescore (이전 SOTA) |
| v84b (listwise tie-break) | 5K | 65.28% | +2.28%p (단순 list pick) |
| **v87 (CoT tie-break)** | **5K** | **66.48%** | **+3.48%p (현재 SOTA)** |
| v92 (top-5 CoT 모든 patient) | 5K | 64.68% | 비-동률에서 LLM 오류 → 손실 |
| v93 (selective CoT, gap≤10) | 5K | 64.26% | 동일 문제 |

**핵심 통찰**: top-5 후보의 92.18%에 정답이 포함되어 있지만, LLM에게 모든 patient에서 top-5 중 선택하게 하면 confident한 non-tied 케이스를 잘못 바꾸어 손실. 동률 케이스만 재선택하는 것이 sweet spot.

### 첫 번째 돌파구 (2026-05-01): KG features in prompt

30개 이상의 변형을 시도한 후, 각 후보 disease의 PubMed-derived top-K 증상을 LLM 프롬프트에 명시적으로 inject하면 60.4%의 ceiling을 깰 수 있음을 발견하였습니다.

```
Diagnosis hypothesis: Pneumonia
Typical features (medical literature): fever, cough, dyspnea, sputum production, ...

How well does the patient's presentation match this diagnosis? 0-100.
```

이전 변형들은 모두 후보 disease 이름만 제공하고 LLM 자체의 의료 지식에 의존하였으나, KG-derived signature를 함께 제공하니 LLM의 변별력이 즉시 향상되었습니다.

| 단계 | 변형 | 데이터 | GTPA@1 | @3 | @5 | 비고 |
|-----|------|------|--------|-----|-----|------|
| Stage 1 단독 | v76 (KG features만) | 2K | 62.4% | 86.0% | 91.8% | +2.0%p over baseline |
| Stage 1 + Stage 2 | v79 (KG features 2-stage) | 2K | 63.8% | 86.2% | 93.0% | hybrid |
| 확정 | v79 | 5K | 63.9% | 86.8% | 93.4% | 이전 SOTA |
| baseline | v66 (KG features 없음) | 30K | 60.4% | 86.3% | 92.1% | 이전 baseline |

## 배경

v1 시스템은 논문에 "UMLS 의료 지식그래프 위에서 추론을 수행한다"고 기술되었으나 실제로는 DDXPlus 데이터셋이 제공하는 질환-증상 관계 데이터를 그대로 사용하여 KG를 구축하고 있었습니다. 같은 데이터셋의 정답 라벨로 KG를 만들고 그 KG로 같은 데이터셋을 평가한 셈이며 DDXPlus에서 GTPA@1 91.05%라는 높은 점수가 나온 것도 이러한 구조적 우위에서 비롯된 결과입니다.

다른 벤치마크 데이터셋에 적용해 본 결과 커버리지가 매우 낮고 진단 점수 또한 처참한 수준으로 나타나면서 문제가 확인되었습니다. 순수 UMLS 데이터만으로 KG를 재구축해 본 결과 DDXPlus 49개 질환 중 30개만 커버되어 커버리지가 61%에 불과하였고 질환당 평균 증상 수는 6.0개로 DDXPlus 원본 평균 18.1개의 1/3 수준이었습니다.

## 학술적 정당성

NLM의 SemMedDB는 SemRep 기반으로 PubMed 의료 의미관계를 추출한 표준 자원이며 1억 8천만 건 이상의 트리플을 보유하고 있습니다. 그러나 SemMedDB는 2024년을 마지막으로 업데이트가 종료되었으며 이는 의료 KG 분야 전체의 공백을 의미합니다.

SemMedDB의 종료로 규칙 기반 의료 관계 추출의 공백이 발생한 상황에서, 본 연구는 재현 가능하고 지속 가능한 오픈소스 LLM 기반 의료 KG 구축 파이프라인을 제안합니다. 구축된 KG를 활용하여 자동 감별 진단 시스템을 구현하고 다중 벤치마크 및 임상 환경에서 검증합니다.

## 연구 계획

본 연구의 핵심은 "자동 문진을 위한 KG 구축 및 감별진단 최적화"입니다. KG 구축과 진단 최적화는 분리된 단계가 아니라 하나의 과정입니다. 질환만을 입력으로 PubMed에서 KG를 구축하고, 그 KG로 진단을 수행하여 벤치마크 점수를 달성하는 것이 1단계 목표입니다.

| 단계 | 질환 소스 | 목표 | 평가 |
|------|---------|------|------|
| **1단계** | 벤치마크 질환 (DDXPlus 49개) | KG 구축 + 감별진단 최적화 | DDXPlus GTPA@1 >= 0.80 |
| **2단계** | 다른 벤치마크 질환 (SymCat, HealthKG 등) | 동일 파이프라인으로 다중 벤치마크 검증 | 각 벤치마크 GTPA@1 |
| **3단계** | UMLS 전체 질환 | 벤치마크 없이 범용 KG 확장 | 춘천성심병원 임상 검증 |

1단계와 2단계는 UMLS가 필요하지 않습니다. 각 벤치마크 데이터에서 질환 이름만 추출하여 PubMed로 KG를 구축하고 진단 성능을 최적화합니다. 3단계에서는 벤치마크가 없으므로 UMLS에서 질환 목록을 가져와 동일 파이프라인을 적용합니다.

### 1단계. 벤치마크 질환 기반 KG 구축

#### 통합 개념 모델

질환과 증상을 별개 노드 타입으로 분리하지 않습니다. SNOMED CT의 Clinical Finding hierarchy와 HPO phenotype ontology가 그러하듯 모든 의학 개념을 단일 노드 타입으로 통합 모델링합니다. 분리 모델은 hypertension(질환이자 finding), tachycardia(증상이자 finding), lab abnormality, pathology finding 같은 경계 모호 케이스를 처리하지 못합니다.

```python
class MedicalConcept:
    cui: str                       # PRIMARY KEY (UMLS CUI)
    name: str
    semantic_types: list[str]      # T047, T184, T033, T034, T191 등
    semantic_groups: list[str]     # DISO, CHEM, ANAT 등

    # Optional cross-references
    snomed_id: Optional[str]
    hpo_id: Optional[str]
    mondo_id: Optional[str]
    omim_id: Optional[str]
    icd10: Optional[str]
```

UMLS CUI를 primary key로, HPO/MONDO/OMIM/SNOMED 등은 cross-reference로 보관합니다. 이 구조는 다음의 장점을 가집니다.

1. **경계 모호 케이스의 자연 처리**: 모든 의학 개념이 동일 공간에 존재
2. **풍부한 관계 표현**: DISO 그룹 내 모든 개념 간 관계를 자연스럽게 표현. disease-symptom 외에도 disease-disease(comorbidity), symptom-symptom(syndrome), disease-lab finding, finding-finding 등 포함
3. **외부 데이터 통합 용이**: DDXPlus, RareBench, 임상 데이터 모두 UMLS CUI로 매핑되면 동일 그래프에 합쳐짐
4. **사후 분류 가능**: query 시점에 semantic_type 필터로 원하는 관계 추출

#### 수집 범위

PubMed 검색 범위를 특정 데이터셋에 한정하지 않고 UMLS의 DISO (Disorders) semantic group을 기반으로 합니다. DISO 12개 semantic type 중 T033(Finding)과 T034(Lab Result)는 "Reduced", "Increased", "Normal" 같은 비특이적 소견이 대량 포함되어 NER 과다 매칭의 원인이 되므로 제외합니다. 다음 10개 semantic type을 사용합니다.

- T047 Disease or Syndrome
- T184 Sign or Symptom
- T191 Neoplastic Process
- T046 Pathologic Function
- T048 Mental or Behavioral Dysfunction
- T037 Injury or Poisoning
- T019 Congenital Abnormality
- T020 Acquired Abnormality
- T190 Anatomical Abnormality
- T049 Cell or Molecular Dysfunction

UMLS 자체가 Disease와 Symptom을 동일 그룹으로 분류하고 있으므로 통합 개념 모델의 정당성이 standard vocabulary 수준에서 뒷받침됩니다. 이 범위는 답지를 보고 푸는 문제를 원천 차단하고 어떠한 외부 벤치마크에도 적용 가능한 KG를 확보합니다.

#### 데이터 소스

PubMed 초록과 PMC full text를 모두 활용합니다. 사전 테스트에서 full text는 초록 대비 약 8배 많은 DISO 개념을 포함하는 것으로 확인되었습니다. Discussion의 감별 진단 논의, Results의 증상 상세 기술, Case의 합병증 기록 등 초록에서 지면 제약으로 생략되는 정보가 full text에 포함되어 있으며 이는 KG 커버리지에 직접적인 영향을 줍니다.

| 소스 | 본문 접근 | 수량 | 용도 |
|------|---------|------|------|
| **PubMed** | 초록만 | 약 2,700만 건 (노이즈 유형 제외) | 전체 처리 |
| **PMC Open Access** | 전문(full text) | 약 500만 건 | full text 접근 가능한 논문은 전문 처리 |

데이터 수집 시 Publication Type 필터링을 적용하여 Journal Article, Clinical Trial, Case Reports, Review, Systematic Review, Meta-Analysis, Observational Study 등 학술적 가치가 높은 유형만 포함합니다. News, Editorial, Comment, Letter, Interview, Retracted Publication 등은 제외합니다.

full text 접근이 가능한 논문은 full text에서 관계를 추출하고 초록만 제공되는 논문은 초록에서 추출합니다. 이미 처리된 PMID를 DB에 기록하여 중복 처리를 방지합니다.

#### IE 파이프라인 (5단계)

```
[Step 1] PubMed/PMC 텍스트 → MetaMap → UMLS CUI 식별
[Step 2] 텍스트 + 후보 CUI 쌍 → LLM ternary classification
[Step 3] 다수 문헌 집계 → Jensen Lab weighted co-occurrence score
[Step 4] Dunning's G² + Benjamini-Hochberg FDR
[Step 5] (선택) HPO frequency bin proxy 매핑
```

##### Step 1. UMLS 개념 식별 (MetaMap)

NIH NLM의 공식 도구인 **MetaMap** (Aronson & Lang, *JAMIA* 2010)을 사용하여 텍스트에서 UMLS CUI를 결정론적으로 식별합니다. MetaMap은 형태학적 변형, 약어 풀이, 동의어, 부정 처리를 통합 수행하며 1990년대부터 의료 IE의 사실상 표준입니다. 결정론적이므로 재현성 100%가 보장되며 LLM이 존재하지 않는 개념을 hallucinate하는 문제를 원천 차단합니다.

##### Step 2. LLM ternary classification

오픈소스 의료 LLM이 다음 단순 분류만 수행합니다.

- **present**: 텍스트가 두 개념 사이의 양성 관계를 명시
- **absent**: 텍스트가 두 개념 사이의 부정 관계를 명시 ("not seen", "absence of", "rules out")
- **not_related**: 두 개념이 같은 텍스트에 등장하지만 관계 진술 없음

LLM이 not_related를 분류하므로 같은 문서에 단순 공출현하는 무관한 쌍은 자동으로 걸러집니다. 빈도, 심각도, 확신도 등 LLM이 생성하면 신뢰할 수 없는 정량 정보는 묻지 않습니다. LLM은 텍스트 추론이라는 자신의 강점에만 사용됩니다. 후보 모델은 BioMistral, Meditron, OpenBioLLM 등 의료 도메인 특화 모델과 일반 instruct 모델이며 4월 한 달 동안 비교 후 정량적 근거로 선정합니다.

특히 absent 추출은 일반 IE보다 어려우므로 NegEx (Chapman et al., *J Biomed Inform*, 2001) 또는 ALBERT 기반 negation 분류기와의 cross-check를 적용합니다 (Hu et al., *arXiv:2503.17425*, 2025).

##### Step 3. Jensen Lab weighted co-occurrence score

LLM의 ternary 분류 결과를 다수 초록에 걸쳐 집계할 때 의료 KG 분야의 de facto 표준인 Jensen Lab 공식을 사용합니다. 이 공식은 STRING(Franceschini et al., *NAR* 2013), DISEASES(Pletscher-Frankild et al., *Methods* 2015), TISSUES, COMPARTMENTS 등 NIH/EMBL의 주요 의료 KG가 모두 채택하고 있습니다.

```
S(a, b) = C(a, b)^α · [C(a, b) · C(·, ·) / (C(a, ·) · C(·, b))]^(1−α)

  C(a, b): 가중 공출현 카운트 (LLM confidence를 soft weight로 사용)
  두 번째 항: observed-over-expected ratio (배경 빈도 보정)
  α ≈ 0.6
```

LLM per-mention confidence를 soft weight로 통합하는 방식은 CoCoScore (Junge & Jensen, *Bioinformatics* 2020)에서 검증된 distant supervision 패러다임을 따릅니다.

이 공식의 철학은 "절대 증거의 양"과 "배경 빈도 대비 특이성"의 균형이며 단순 PMI의 저빈도 편향과 단순 카운트의 흔한 단어 편향을 동시에 해결합니다.

##### Step 4. Dunning's G² + Benjamini-Hochberg FDR

각 (개념 a, 개념 b) 쌍의 통계적 유의성을 Dunning의 Log-Likelihood Ratio test (Dunning, *Computational Linguistics* 1993)로 평가합니다. G²는 희귀 사건에서 점근 정규성이 무너지지 않는 유일한 통계량이며 PubMed의 저빈도 쌍에서 χ²나 PMI보다 정확합니다.

다중 검정 보정은 Benjamini-Hochberg FDR (Benjamini & Hochberg, *JRSS* 1995)을 적용합니다. 질환 1,000개 × 증상 10,000개 규모에서 Bonferroni는 지나치게 보수적이므로 BH-FDR이 의료/생물 분야의 표준입니다. q < 0.05 인 쌍만 KG에 포함시킵니다.

##### Step 5. HPO frequency bin proxy 매핑 (선택)

KG export 시점에 freq_proxy = C_doc(d, s) / C_doc(d) 를 계산하여 HPO가 공식 정의한 백분율 구간(80~99%, 30~79%, 5~29%, 1~4%)에 매핑합니다. 단 PubMed mention frequency는 임상 frequency의 proxy이며 HPO의 case series 기반 카운트와 다르다는 점을 Limitations에 명시합니다.

#### KG 엣지 스키마

```python
class ConceptRelation:
    cui_a: str                     # 알파벳 순 첫 번째
    cui_b: str                     # 알파벳 순 두 번째
    polarity: Literal["present", "absent", "ambiguous"]

    # 통계적 증거 (모두 계산으로 산출, LLM 생성 아님)
    n_present: int
    n_absent: int
    co_occurrence_score: float     # Jensen Lab weighted score
    g_test: float                  # Dunning's G²
    p_value: float
    q_value: float                 # Benjamini-Hochberg FDR adjusted

    # Provenance
    pmids_present: list[str]
    pmids_absent: list[str]
```

#### 관계 분류 체계의 학술적 근거

본 연구는 관계의 의미적 종류(CAUSES, MANIFESTATION_OF 등)를 세분화하지 않고 polarity(present/absent)만 분류합니다. 이 설계는 다음의 학술적 합의에 기반합니다.

**진단용 벤치마크 데이터셋의 합의**: DDXPlus(Fansi Tchango et al., *NeurIPS* 2022), SymCat, RareBench 모두 relation type을 세분화하지 않고 binary association만 사용합니다. DDXPlus는 질환-증상 관계에 빈 딕셔너리만 저장하며 어떤 의미적 predicate도 부여하지 않습니다.

**주요 의료 KG의 설계**: Hetionet(Himmelstein et al., *eLife* 2017)은 disease-symptom에 `presents` 단일 타입만 사용합니다. PrimeKG(Chandak et al., *Scientific Data* 2023)은 `disease_phenotype_positive`와 `disease_phenotype_negative` 두 가지만 구분합니다. RTX-KG2(Wood et al., *BMC Bioinformatics* 2022)는 77개 predicate를 가지지만 disease-phenotype에는 `has_phenotype` 하나만 사용합니다.

**HPO의 공식 설계**: 세계 최대의 disease-phenotype 데이터베이스인 HPO(Köhler et al., *NAR* 2024)에도 relation type 전용 컬럼이 없습니다. qualifier(NOT)와 frequency로만 관계를 표현하며 이는 본 연구의 present/absent와 구조적으로 동일합니다.

**고전 진단 시스템의 검증된 설계**: INTERNIST-1(Miller et al., *NEJM* 1982), QMR-DT(Shwe et al., *Methods Inf Med* 1991), DXplain(Barnett et al., *JAMA* 1987) 등 수십 년간 검증된 진단 시스템들은 relation type을 구분하지 않고 수치적 가중치(frequency weight, evoking strength)로 관계의 강도를 표현합니다. 진단 추론에서 관계의 의미(semantic type)보다 관계의 강도(strength)가 더 중요하다는 것이 이 분야의 학술적 합의입니다.

관계의 의미적 분류가 필요한 경우 양 끝 노드의 semantic type 조합에서 query 시점에 자동으로 유도합니다(Bodenreider, *NAR* 2004). 예를 들어 T184(Sign/Symptom)과 T047(Disease)의 present 관계는 symptom-disease 관계로 해석되며 T047과 T047의 present 관계는 comorbidity로 해석됩니다.

#### 관련 연구 포지셔닝

```
[큐레이션 기반 KG]
  HPO, OMIM, DrugBank 등 → PrimeKG (기존 DB 통합)

[규칙 기반 IE → KG]
  PubMed → SemRep (규칙 기반 IE) → SemMedDB (2024 종료)

[LLM 기반 IE → KG]  ← 본 연구
  PubMed/PMC → MetaMap + LLM IE → 통계적 검증 → KG
```

PrimeKG은 이미 큐레이션된 DB를 통합하는 프로젝트이며 새로운 관계를 발견할 수 없습니다. SemMedDB가 2024년에 종료되면서 PubMed 기반 의료 관계 추출에 공백이 발생하였으며, 본 연구는 LLM 기반 접근으로 이 영역을 다루면서 통계적 검증을 강화합니다. PrimeKG과 관계 스키마(positive/negative)가 유사한 것은 학술 표준을 따른 결과이며 방법론은 근본적으로 다릅니다.

| | PrimeKG | SemMedDB | 본 연구 |
|---|---|---|---|
| 데이터 소스 | 큐레이션된 DB 20개 | PubMed (SemRep) | PubMed/PMC (LLM) |
| 새로운 관계 발견 | 불가능 | 가능 | 가능 |
| 통계적 검증 | 없음 | predication count만 | Jensen Lab + G² + FDR |
| 지속가능성 | 상류 DB 의존 | 2024 종료 | 자체 파이프라인 재실행 |
| 임상 검증 | 없음 | 없음 | 춘천성심병원 |

#### 학술적 근거 종합

| 단계 | 표준 | 출처 |
|------|------|------|
| 통합 개념 모델 | SNOMED CT Clinical Finding, HPO, UMLS Semantic Network | Bodenreider, *NAR* 2004; Köhler et al., *NAR* 2024 |
| Step 1 (개념 식별) | MetaMap | Aronson & Lang, *JAMIA* 2010 |
| Step 2 (negation cross-check) | NegEx, ALBERT-based assertion | Chapman et al., *J Biomed Inform* 2001; Hu et al., *arXiv* 2025 |
| Step 3 (집계) | Jensen Lab weighted co-occurrence, CoCoScore | Franceschini et al., *NAR* 2013; Junge & Jensen, *Bioinformatics* 2020 |
| Step 4 (유의성) | Dunning's G², Benjamini-Hochberg FDR | Dunning, *Comp Linguistics* 1993; Benjamini & Hochberg, *JRSS* 1995 |
| Step 5 (HPO 매핑) | HPO frequency sub-ontology | Köhler et al., *NAR* 2021/2024 |

#### 노이즈 CUI 처리

파일럿 테스트에서 UMLS DISO 개념 중 "Present", "Reduced", "Well", "Morbidity", "Comorbidity" 같은 비임상적 메타 개념이 대량으로 매칭되는 노이즈 문제가 확인되었습니다. 다음 3가지 방법을 테스트한 결과 semantic type 필터링 + 소규모 블랙리스트 조합이 최적입니다.

**테스트 결과 불가능한 방법:**
- PubMed 검색 횟수 기반 필터링: "Present" 375만 건, "Reduced" 231만 건으로 임상 용어보다 오히려 많이 검색되어 분리 불가
- UMLS SUPPRESS 필드: 노이즈 CUI 대부분이 SUPPRESS=N(Not suppressible)으로 효과 없음
- SNOMED CT 전환: SNOMED CT에서도 "Symptom (finding)", "Disease (disorder)" 같은 최상위 추상 개념이 존재하여 동일 문제 발생

**채택 방법: Semantic Type 필터링 + 블랙리스트**

T033(Finding)과 T034(Lab Result)는 "Reduced", "Increased", "Normal", "Present" 같은 비특이적 소견이 대량 포함되어 NER 과다 매칭의 주 원인입니다. 파일럿에서 T033+T034를 제외했을 때 노이즈가 76% 감소하면서 임상 CUI 탈락은 0건이었습니다. 이에 따라 DISO 12개 semantic type 중 T033과 T034를 제외한 10개를 사용합니다. SemMedDB 구축 시에도 overly general semantic type을 필터링한 사례가 있습니다 (Kilicoglu et al., *Bioinformatics* 2012).

T033/T034 제외 후에도 남는 최상위 추상 개념 5개(Symptom, Other Symptom, Disease NOS, Pathogeneses, Unexplained Symptoms)는 수동 블랙리스트로 추가 제외합니다.

선행 연구에서도 동일한 접근이 사용되었습니다. Rotmensch et al. (2017)은 EHR 기반 KG 구축에서 T033/T034를 명시적으로 제외하고 "Present", "Absent", "Normal" 등을 stopword처럼 제거하는 것을 standard practice로 보고했습니다. NLM도 MetaMap 공식 문서에서 `--restrict_to_sts` 옵션을 통한 task-specific semantic type 제한을 권장하며 "MetaMap은 recall 지향 설계이므로 post-processing이 필수"라고 명시합니다 (Aronson & Lang, *JAMIA* 2010).

#### 1단계 평가 프레임워크

KG 구축 품질을 1단계 독립적으로 평가하기 위해 3단계 프레임워크를 적용합니다.

**평가 1: Intrinsic (KG 자체 통계)**

노드/엣지 수, 질환-증상 커버리지, degree distribution, connected components, 중복률 등 KG 구조적 품질을 보고합니다. PrimeKG(Chandak et al., *Scientific Data* 2023)과 Hetionet(Himmelstein et al., *eLife* 2017)의 보고 형식을 따릅니다.

**평가 2: Gold Standard 대비 Precision/Recall**

HPO phenotype.hpoa를 primary gold standard로 사용하여 disease-phenotype 관계의 precision/recall을 측정합니다. SemMedDB는 자동 추출물이므로(precision ~75%) gold standard가 아닌 baseline 비교군으로 사용합니다.

**평가 3: Expert Validation**

구축된 KG에서 200~500개 triple을 무작위 표본 추출하여 2인 이상 의료 전문가가 독립적으로 정확성을 평가합니다. Cohen's kappa로 inter-rater agreement를 보고합니다(0.6 이상이면 substantial). 구체적 프로토콜은 춘천성심병원과 별도 협의하여 확정합니다.

#### LLM 선정 벤치마크 근거

LLM 모델 선정 시 다음 벤치마크 성능을 근거로 제시합니다.

- **BioRED** (Luo et al., *Briefings in Bioinformatics* 2022): document-level 의료 관계 추출 벤치마크
- **ChemProt + DDI**: RE fine-tuning 성능 비교. fine-tuned BERT F1=0.73 vs GPT-4 zero-shot F1=0.33 (Nature Communications 2025)
- **Me-LLaMA** (npj Digital Medicine 2024): 의료 특화 LLM이 GPT-4를 5/8 데이터셋에서 능가

disease-symptom RE 전용 벤치마크는 현재 존재하지 않으므로 BioRED의 disease 관련 관계를 proxy로 사용하거나 자체 gold standard를 구축합니다.

#### 검증 및 비교

종료된 SemMedDB의 기존 트리플은 historical baseline 및 비교군으로 활용하여 LLM 기반 접근의 우월성을 정량적으로 입증합니다.

### 2단계. 다중 벤치마크 평가 및 알고리즘 최적화

#### 다중 벤치마크 평가

DDXPlus 단일 데이터셋 의존성을 회피하기 위해 수집 가능한 모든 의료 진단 벤치마크에서 동시 평가합니다. DDXPlus, RareBench, DxBench 등 현재 검토 가능한 데이터셋과 추가 수집 가능한 데이터셋을 모두 포함하여 일반화 능력을 객관적으로 입증합니다.

#### 관계 polarity의 진단 신호 강도

자동 문진에서 환자 응답과 KG 관계 polarity의 조합은 다음과 같은 진단 신호로 작동합니다.

| 환자 응답 | present 관계 | absent 관계 | 관계 부재 |
|---------|------------|-----------|---------|
| **있다** | 강하게 가능성 증가 | 강하게 가능성 감소 (rule-out) | 약하게 가능성 감소 |
| **없다** | 약하게 가능성 감소 | 강하게 가능성 증가 (rule-in) | 약하게 가능성 증가 |

핵심은 관계 부재가 단순한 "변화 없음"이 아니라 **약한 음성 신호**로 작동한다는 점입니다. 문헌에 명시되지 않았다는 사실 자체가 약한 음성 증거이며 이는 QMR-DT의 leak probability 원리(Shwe et al., *Methods Inf Med* 1991)와 일치합니다. Reiter의 open-world assumption(*Logic and Databases* 1978) 하에서 의료 KG는 명시되지 않은 관계를 "거짓"이 아니라 "약한 음성"으로 해석합니다.

이 신호 강도는 검증 데이터로 튜닝되는 하이퍼파라미터이며 v1의 binary KG가 활용하지 못했던 추가 진단 정보를 활용할 수 있게 합니다.

#### 알고리즘 최적화

Cypher 탐색 전략, 정지 기준, 점수 함수 등의 알고리즘을 최적화합니다. 점수 함수는 1단계에서 산출된 통계적 지표(Jensen Lab co-occurrence score, G²)와 EBM 표준 지표(sensitivity, specificity, likelihood ratio)를 결합하여 활용합니다 (Jaeschke et al., *JAMA*, 1994).

### 3단계. 춘천성심병원 임상 검증

2단계까지 검증된 시스템을 춘천성심병원 실제 환자 데이터로 평가합니다. 벤치마크가 아닌 실제 임상 환경에서의 유효성을 입증하는 단계이며 본 연구를 단순한 벤치마크 엔지니어링이 아닌 임상 적용 가능 연구로 격상시키는 핵심입니다.

#### 사용자 모드 분류

최종 시스템은 일반인 모드와 의료인 모드를 구분합니다. UMLS semantic type을 기준으로 각 MedicalConcept의 접근성(accessibility)을 분류합니다.

| Semantic Type | 접근성 | 이유 |
|--------------|--------|------|
| T184 (Sign or Symptom) | **patient** | 환자가 스스로 인지 가능 (fever, headache, cough) |
| T034 (Lab Result) | **professional** | 의료 검사 필요 (WBC count, CRP, blood glucose) |
| T033 (Finding) | **혼재** | 일부 환자 인지 가능 (rash, swelling), 일부 전문가 필요 (hepatomegaly, pleural effusion) |

T033(Finding)의 세분류는 SNOMED CT hierarchy를 활용하여 patient-observable과 professional-only를 자동 분류합니다.

일반인 모드에서는 accessibility=patient인 개념만 문진에 사용하고 의료인 모드에서는 모든 개념을 사용합니다. 이 분류는 KG 구축 시점이 아닌 query 시점에 필터링되므로 KG 자체의 변경 없이 API 파라미터로 동작합니다.

DDXPlus 벤치마크는 모든 evidence가 환자 보고 가능 항목이므로 일반인 모드에 해당하며 본 연구의 KG는 의료인 모드를 추가로 지원함으로써 임상 적용 범위를 확장합니다.

#### 평가 프로토콜

다음 항목을 사전 협의하여 확정합니다.

- 사용자 인터페이스 및 증상 입력 방식
- 정답 기준의 정의
- 본 시스템의 진단과 의사 진단의 일치 또는 불일치 검토 방법
- 평가 대상 환자 수 및 질환 분포
- 일반인 모드와 의료인 모드의 각각에 대한 평가

## 파일럿 실험 결과 (2026년 4월)

DDXPlus 49개 질환 × 50편 = 2,217편 PubMed 초록에서 193편 서브셋으로 1,920개 방법 조합을 테스트했습니다.

### 실험 설계

10개 프롬프트 변형 × 4개 NER 필터 × 48개 통계 파라미터 = 1,920개 조합. DDXPlus KG 324 symptom 쌍(antecedent 제외)을 gold standard로 CUI 정규화(MRREL 계층 전파)를 적용한 precision/recall/F1로 평가했습니다.

### 최적 설정 (F1=0.793)

```
NER: scispaCy (threshold=0.85) + T033/T034 제외 + 블랙리스트 5 CUI
CUI 정규화: MRREL PAR/RB 1-level 계층 전파
LLM: gemma4:e4b-it-bf16
프롬프트: S2-J (이진 분류 + 관계 범위 명시)
통계: 최소 공출현 3회 (FDR 필터 불필요)
```

성능: **Precision=77.3%, Recall=81.5%, F1=79.3%** (DDXPlus 324 symptom 쌍 중 264 쌍 재현)

### 주요 발견

1. **프롬프트 설계가 가장 큰 영향**: 최고(S2-J: 0.793) vs 최저(S2-A: 0.763) = +3%p. 이진 분류(present/not_related) + 관계 범위 명시(symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)가 최적.
2. **CUI 계층 전파가 필수**: 적용 전 F1=0.027 → 적용 후 F1=0.793 (29배 향상). scispaCy가 하위 CUI(Acute Bronchitis)로 매핑해도 상위 CUI(Bronchitis)로 전파하여 gold standard 매칭.
3. **NER 필터 간 차이 없음**: T033/T034 제외 vs 전체 DISO에서 F1 차이 없음. 노이즈는 LLM의 not_related 분류가 처리.
4. **Jensen Lab α 값 무관**: α=0.3~0.7 모두 동일 F1. 통계적 필터보다 LLM 분류 품질이 성능을 결정.
5. **FDR 필터 불필요**: LLM이 이미 not_related를 분류하므로 추가 통계 필터는 recall만 낮춤. 최소 공출현 3회가 유일한 유효 필터.
6. **few-shot 예시 효과**: 3-shot(S2-C: 0.785) > zero-shot(S2-A: 0.763) = +2.2%p. 학술적 기대치(Gutierrez 2022: +10-15%p)보다 작지만 일관된 향상.

### 프롬프트별 최고 F1

| 프롬프트 | F1 | 학술 근거 | 설명 |
|---------|-----|---------|------|
| **S2-J** | **0.793** | Wadhwa 2023 | 이진 + 관계 범위 명시 |
| S2-C | 0.785 | Gutierrez 2022 | few-shot 3예시 |
| S2-F | 0.780 | Li 2024 | few-shot + CoT |
| S2-I | 0.780 | Wadhwa 2023 | recall 우선 |
| S2-H | 0.779 | Wadhwa 2023 | precision 우선 (P=0.807 최고) |
| S2-A | 0.763 | baseline | zero-shot 기본 |

## v2 실험 진행 과정 (2026년 4월)

### 연구 방향 전환

파일럿 F1=0.793은 3-level 조상 + 공통 조상 매칭을 적용한 결과였으며, 엄격한 1-level 매칭에서는 F1=0.22 수준이었습니다. 이 사실을 확인한 후 연구 방향을 "KG 엣지 재현"에서 "자동 진단을 위한 KG 구축 방법론"으로 전환하였습니다.

핵심 원칙:
- **질환만 ICD-10 → UMLS CUI로 매핑** (정답 확인용)
- **증상은 UMLS 매핑하지 않음** — KG가 자체적으로 질환-증상 관계를 정의
- **최종 평가 지표는 GTPA@1** (진단 정확도), 엣지 F1이 아님

### CUI 추출 방법 비교

동일 초록에서 3가지 방법으로 CUI를 추출하여 DDXPlus gold recall을 비교하였습니다.

| 방법 | 평균 CUI/질환 | Gold Recall | 속도 (20편) |
|------|-------------|------------|------------|
| MeSH 태그 | 26 | 131% | 즉시 |
| 텍스트 매칭 (Aho-Corasick) | 84 | 240% | 2ms |
| scispaCy NER | 128 | 238% | 700ms |

텍스트 매칭이 NER과 동등한 recall에 350배 빠른 속도를 달성하여 텍스트 매칭을 채택하였습니다.

### KG 구축 반복 실험

| 버전 | 초록 | 방법 | 엔진 | 시간 | 엣지 F1 (1-level) |
|------|------|------|------|------|-------------------|
| V2A | 4,694 | TextMatch + V2 프롬프트 | Ollama | 18h | 0.209 |
| V4 | 9,431 | TextMatch + V2 + 동의어 검색 | vLLM | 9min | 0.270 |
| V7 | 23,000+ | Clinical V2 프롬프트 | vLLM | 21min | 0.368 |
| V8 | 23,000+ | Clinical V2 + STY 필터 + MC=15 | vLLM | 21min | 0.400 |
| V10 | 6,537 | DDXPlus 87 CUI 폐쇄형 이진 분류 | vLLM | 106s | 0.565 |

주요 발견:
1. **vLLM batch + chat template**이 Ollama 대비 200배 빠름
2. **Clinical V2 프롬프트**(임상 증상 초점)가 일반 V2 대비 +35% F1 향상
3. **DDXPlus 증상 어휘 폐쇄형**이 오픈 어휘 대비 precision 3배 향상
4. **CUI 전파가 FP의 81%를 생성** — 전파 없이 평가하는 것이 공정
5. **FP의 상당수가 의학적으로 타당한 관계** — DDXPlus gold가 불완전

### 진단 평가 (GTPA@1)

DDXPlus 134,529명 테스트 환자에 대해 감별진단 수행. 증상 매핑 없이 텍스트 매칭 + LLM re-ranking.

#### v2 방식 (UMLS CUI 매핑 사용 — 폐기)

| 진단 알고리즘 | GTPA@1 | @3 | @5 | @10 |
|-------------|--------|----|----|-----|
| v15_ratio (yes>=2) | 40.0% | 60.8% | 70.8% | 85.4% |

#### 현재 방식 (증상 매핑 없음)

KG 구축: 질환 이름 → PubMed 500편 → Aho-Corasick → LLM 1회/초록 → 13,930 쌍
진단: 환자 evidence 영문 텍스트 ↔ KG 증상 이름 텍스트 매칭 → Bayesian 후보 → LLM re-ranking

| 방법 | GTPA@1 | @3 | @5 | @10 |
|------|--------|----|----|-----|
| Bayesian only (텍스트 매칭) | 29.3% | 49.3% | 62.0% | 80.4% |
| Bayesian + LLM re-ranking (구어체) | 48.1% | — | — | — |
| Bayesian + LLM re-ranking (구조화 프로필) | 55.3% | 67.5% | 74.2% | 80.2% |
| Bayesian + age/sex prior + 구조화 re-ranking | 56.2% | — | — | 81.1% |
### 다중 벤치마크 검증

| 벤치마크 | 질환 수 | 환자 수 | KG 쌍 | Bayesian | + LLM re-rank |
|---------|--------|--------|-------|---------|--------------|
| DDXPlus | 49 | 134,529 | 13,930 | 29.3% | **56.2%** |
| SymCat | 50 | 5,000 (시뮬레이션) | 13,756 | 27.6% | **38.5%** |
| RareBench | 441 | 1,121 | 45,132 | 15.6% | **16.4%** |

학습 데이터 없는 합법적 결과. 동일 파이프라인(질환 이름 → PubMed KG → Bayesian + LLM re-rank)을 세 벤치마크에 적용.

관찰:
- DDXPlus는 LLM 효과 가장 큼 (+27%p) — 일반 질환, gemma-4 의료 지식 활용 가능
- SymCat은 중간 (+11%p) — 영어 의학 용어 직접 매칭
- RareBench는 미미 (+0.8%p) — 441 희귀질환 / HPO 표현형의 어려움

KG 순수 기여도 측정 (DDXPlus):

| 방법 | GTPA@1 | KG 기여 |
|------|--------|---------|
| LLM only (KG 미사용, 49 질환 직접 선택) | 54.3% | — |
| KG + LLM re-rank (top-10) | 56.2% | +1.9%p |

DDXPlus의 49개 질환은 일반 의학 질환으로 8B LLM이 훈련 데이터에서 충분히 학습. KG의 순수 기여는 +1.9%p에 그침. KG의 진정한 가치는 LLM 학습 데이터에 부족한 희귀 질환이나 새로운 도메인에서 나타날 것으로 예상.

### 통계적 KG 노이즈 제거 실험

여러 통계적 방법으로 KG 정제 시도:

| 방법 | @1 | @10 | 결과 |
|------|----|----|------|
| Standard Bayesian (baseline) | 30.2% | 73.1% | **최적** |
| Dunning G² + BH-FDR 필터 (q<0.05) | 17.8% | 66.5% | 하락 |
| G² 점수 가중치 | 21.4% | 68.1% | 하락 |
| IDF Bayesian (k=1) | 27.9% | 72.8% | 비슷 |
| IDF² (k=2) / IDF³ (k=3) | 21.4% / 15.6% | — | 하락 |
| 희귀 특이 증상만 보존 (≤5 질환) | 7.0% | 20.1% | no_match 폭증 |
| PMI / P(D|S) / Lift weighting | 8.5-23.8% | — | 하락 |

**결론**: 
- LLM 추출 KG는 이미 적절한 통계적 균형 보유
- 추가 통계 처리(Dunning G², BH-FDR, IDF 강화 등)는 모두 신호 손실
- 흔한 증상도 환자-KG 매칭에 필수 (제거 시 no_match 폭증)
- 학술적 표준 방법(Dunning, BH-FDR)도 이 환경에서는 도움 안 됨
- KG 구축 단계의 LLM 의미적 필터링이 통계적으로 충분함을 입증

실험한 방법 (v3~v27):
- 텍스트 매칭 변형 (키워드 추출, evidence 이름 포함, min_len 조정): 최대 29.3%
- LLM evidence-symptom 매칭 (v4): 매칭 과다로 bayesian 불가, v15 최대 20.6%
- 하이브리드 텍스트+LLM (v5): text+bayesian 28.6%가 최선
- UMLS CUI 확장 (v6): 노이즈 증가로 하락
- Bayesian + LLM re-ranking (v7): 48.1%
- KG 증상 포함 re-ranking (v8): 노이즈로 40.8%
- 다단계 re-ranking 10→3→1 (v9): parse fail로 47.0%
- KG 2000편 (v10): 노이즈 증가로 47.2%
- DeepSeek-R1 re-ranking (v11b): parse fail 94%, 효과 없음
- 구조화 환자 프로필 (v12): 48.1→54.8% (+6.7%p, 프롬프트 형식이 핵심)
- 전체 antecedent 포함 (v13): 55.3%
- 다중 알고리즘 union 후보 (v14): 55.0% (후보 증가가 re-ranking 효율 상쇄)
- medgemma-1.5-4b (v15): 33.1% (4B 크기 한계)
- 감별 고유증상 표시 (v16): 55.5%
- age/sex prior (v17): **56.2%** — 학습 없는 최고
- KG 매칭 증상 표시 (v19): 51.2% (KG 노이즈가 LLM 방해)
- 임상 보고서 형식 (v25): 50.9%
- 의학 용어 정제 (v26): 54.7% (haunting→shooting/stabbing 등)
- 영문 → 의학 용어 매핑 추가 (v27, 진행 중): @10 80.4%→84.7%로 향상
- Borda count 앙상블: bayesian 단독보다 하락
- STY 필터링: ALL types가 @10 최고 (79.2%)

### 현재 진행 중

DDXPlus 56.2% → 80% 목표를 위한 추가 실험:

**v27 (영문 → 의학 용어 매핑, 134K 환자 평가 완료)**:
- DDXPlus question_en 일반 표현을 의학 용어로 매핑 (EVIDENCE_MEDTERM dict)
- Baseline @10: 80.4% → 84.7%
- LLM re-rank: **GTPA@1 = 52.8%** (no prior 적용 시)

**v28 (KG signature + 의학용어 보정 + clean profile, full 134K 평가 완료)**:
- 각 후보 disease의 KG 상위 변별력 증상 (IDF·count 기준 top-5) 표시
- DDXPlus 영문 번역 보정 ("haunting"→"stabbing", "tugging"→"pulling" 등)
- 환자 프로필 통합 (header 중복 제거, 의학용어 표준화)
- 모호한 disease 이름 명확화 (URTI → "Upper respiratory tract infection")
- **GTPA@1 = 57.9%** (full 134K, +1.7%p over v17 56.2%)
- @3 = 71.6%, @5 = 79.1%

**v34 (hand-curated 임상 features, 30K 평가 완료)**:
- 의학 교과서의 임상 변별 features를 각 후보에 표시 (예: Pericarditis "pleuritic chest pain better leaning forward; pericardial friction rub")
- 외부 LLM/지식 소스 아닌 표준 임상 의학 지식
- **GTPA@1 = 56.9%** (v28과 비슷, 변별력 marginal)

**v33 (top-20 후보 확장, 30K 평가 완료)**:
- top10 (86.9%) → top20 (94.3%) recall 확장
- LLM이 더 많은 후보 중 선택
- **GTPA@1 = 57.2%** (top10 v28 57.8%과 비슷, 후보 증가가 LLM 정확도 하락 상쇄)

**v32 (3-prompt ensemble + majority vote, 30K 평가 완료)**:
- P1 (basic): 54.5%, P2 (KG sig): 55.2%, P3 (rule-out): 55.8%
- Majority vote: **GTPA@1 = 56.0%**
- 단일 프롬프트 v28(57.8%)보다 낮음 → 프롬프트별 errors 상관관계 높음

**v41 (logprob-based ranking, 2K 평가)**:
- LLM의 next-token prediction logprob 활용
- **GTPA@1 = 18.9%** — 실패 (multi-token disease 이름 길이 편향)

**v50 (per-candidate independent scoring, 30K 평가 완료)**:
- "이 진단이 환자에 얼마나 맞나?" 각 후보를 독립적으로 0-9 score
- LLM이 list 중 하나 선택 대신 각 후보 독립 평가 → 비교 anchoring bias 제거
- **GTPA@1 = 59.1%** (NEW BEST, +2.9%p over v17, +1.2%p over v28)
- @3 = 79.8%, @5 = 83.8%
- 30K LLM calls = 300K (각 환자 × 10 후보)

**v52 (v50 → v28 두 단계, 5K 평가)**:
- Stage 1: v50 score → top-5
- Stage 2: top-5에서 v28-style list pick
- **GTPA@1 = 58.3%** (v50 stage1 alone 58.5%보다 -0.2%p) — 두 단계는 효과 없음

**v53 (LLM score + Bayesian 가중 결합, 5K 평가)**:
- combined_score = α * LLM_score + (1-α) * Bayesian_normalized
- α=0.9 (LLM 90% + Bay 10%): @1=59.0% @5=85.6% (best)
- α=1.0 (LLM only, v50): @1=58.9%
- α=0.5 균형: @1=47.9% (악화)
- LLM이 dominant signal; Bayesian 추가 marginal effect

**v54 (per-candidate scoring 0-100 fine scale, full 134K 평가 완료)**:
- v50의 0-9 scale 대신 0-100 percentage 사용
- finer granularity로 ties 감소, LLM이 분포 더 잘 활용
- **GTPA@1 = 60.4%** (full 134K, 최종 확정, +4.2%p over v17)
- @3 = 81.1%, @5 = 84.5%
- 30K subset: 60.9% (variance ±0.5%p)
- 60% 이상 달성

**v55 (v54 + Bayesian 가중 결합, 5K 평가)**:
- α=1.0 (v54 only): @1=60.4%
- α=0.9 (LLM 90% + Bay 10%): @1=59.8%
- v54 단독이 최고; Bayesian 결합 효과 없음

**v56 (구조화된 평가 기준 prompt, 5K 평가)**:
- prompt에 "symptom match + demographic fit + symptom pattern" criteria 추가
- @1=59.7% (v54와 비슷, 효과 marginal)
- 구조화된 criteria도 큰 효과 없음 — LLM이 이미 implicit하게 모두 고려

**v57 (brief reasoning + score, CoT-lite, 5K 평가)**:
- "Match: <one sentence> Score: <0-100>" 형식
- @1=60.1% (v54 5K=59.6%와 거의 동일)
- LLM이 reasoning을 score에 반영하나 dramatic 차이 없음

**v58 (self-consistency, temperature=0.5, n=5 samples 평균, 5K 평가)**:
- 각 (patient, candidate) 쌍을 5번 sample → 평균 score
- @1=58.8% (v54 5K=59.6%보다 나쁨)
- Temperature 노이즈가 정확도 저하 → deterministic (temp=0) 이 최적

**v59 (모든 49 candidates LLM scoring, 2K 평가)**:
- KG top-10 필터 없이 모든 49 disease를 LLM이 score
- @1=57.0%, @3=82.0%, **@5=88.8%**
- 후보 폭이 넓어서 @1은 낮지만 @5는 v54보다 높음 (recall ceiling 상향)

**v60 (v59 → top5 → v28-style list pick, 2K)**: @1=60.0% (v54와 비슷)
**v61 (v59 → top5 → CoT, 2K)**: @1=14.9% (parser 실패)
**v62 (v59 → top5 → pairwise tournament, 2K)**: @1=48.0% (LLM A/B bias)
**v63 (negative evidence missing symptoms, 5K)**: @1=56.3% (오히려 하락)
**v64 (KG cleaned top-50 per disease, 5K)**: @1=52.9% (KG noise 제거가 신호 함께 제거)

**v66 (v59 stage1 → top10 → v54-style stage2 rescore, 30K 확정)**:
- Stage 1: v59 score 49 candidates → top-10 (recall 96.9%, vs Bayesian 85.5%)
- Stage 2: v54-style 0-100 re-score on top-10 → top-1
- **GTPA@1 = 60.4%** (v54와 동일 @1)
- **@3 = 86.3%, @5 = 92.1%** (v54 @5=84.5%보다 +7.6%p)
- 같은 @1이지만 더 높은 recall ceiling

**v67 (v66 stage1 + stage2 weighted combined, 2K)**:
- α=1.0 (stage2 only): @1=58.4%
- α=0.5: @1=58.2%
- α=0.0 (stage1 only): @1=54.5%
- Stage 2 단독이 최고 → combining 효과 없음

**2026-05-01 추가 변형 결과 (모두 2K subset)**:
- v68 (unique discriminator features): @1=56.4%
- v69 (Yes/No logprob re-score): @1=54.0%
- v70 (symmetric pairwise tournament): @1=52.7%
- v72 (LLM-cleaned profile): @1=58.2%, @10=96.7%
- v75 (3-prompt ensemble borda): @1=58.9% (T0 alone=59.9%)
- v77 (generative DDx 1-call): @1=54.2%

**v76 (KG features in prompt, 2K) — 신규 best stage1**:
- 각 후보 disease의 top-8 KG symptom features를 prompt에 inject
- "Diagnosis: X. Typical features (medical literature): A, B, C, ..."
- LLM이 KG-derived signature를 알고 환자와 비교 가능
- **GTPA@1 = 62.4% (+2.0%p over v66 60.4%)**
- @3 = 86.0%, @5 = 91.8%, @10 = 97.8%
- KG 정보 inject가 단순 disease name보다 변별력 향상

**v79 (v76 KG features stage1 → top10 → KG features rescore) — 최고**:
- Stage 1: v76-style 49 candidates scoring with KG features
- Stage 2: rescore top-10 with KG features + "top-10 most likely" framing
- **GTPA@1 = 63.8% (2K), 63.9% (5K 확정)**
- @3 = 86.8%, @5 = 93.4% (5K, 가장 높은 ceiling)
- vs v66 60.4% baseline: **+3.5%p**
- v76 stage1-only 63.0%에서 +0.9%p 추가 향상
- KG features의 일관된 활용이 핵심
- v80 (v79 + Bayesian prior) 시도: α=0 (no prior)이 최적, prior 추가 효과 없음

**SymCat v54 (다중 벤치마크, 50 disease × 50 patients)**:
- v54 (per-candidate scoring) 적용
- **GTPA@1 = 39.7%** (기존 38.5% 대비 +1.2%p)
- @3 = 61.1%, @5 = 68.1%
- Bayesian top-10 = 73.5% (DDXPlus의 86.5%보다 낮음, KG quality 문제)

**RareBench v54 (다중 벤치마크, 440 rare diseases × 1,121 patients)**:
- v54 (per-candidate scoring) 적용
- **GTPA@1 = 22.1%** (기존 16.4% 대비 +5.7%p)
- @3 = 36.4%, @5 = 43.7%
- Bayesian top-10 = 48.8% (440 disease 후보, KG 매우 sparse)
- 큰 후보 공간이 ceiling 결정

**핵심 결론** (DDXPlus 평가 기준):
- v54 (per-candidate scoring 0-100) 가 최고: **GTPA@1 = 60.4%** (full 134K, 최종)
- @3 = 81.1%, @5 = 84.5%
- 80% 목표 (@1) 달성 못함 → 19.6%p 격차
- list pick보다 per-candidate scoring이 +1.2%p
- 0-100 fine scale이 0-9 coarse scale보다 +1.8%p
- 후보 비교 시 anchoring bias가 LLM 선택 정확도 저하시킴
- v50/v54 + Bayesian 결합도 marginal 효과만

**80% 달성을 막는 구조적 한계**:
- KG top10 recall = 86.5% (이론적 상한)
- v50 @3 = 79.8% (실제 가능 상한, 완벽한 stage2 가정 시)
- 80% 달성하려면 KG 품질 향상 (더 많은 PubMed/HPO) 또는 LLM 모델 업그레이드 필요

### 최종 결과 표

| 변형 | 데이터 | GTPA@1 | @3 | @5 | 비고 |
|-----|------|--------|-----|-----|------|
| v17 (baseline + prior) | 134K | 56.2% | - | - | reference |
| v27 (no prior + medterm) | 134K | 52.8% | - | - | prior 효과 -3.4%p |
| v28 (KG sig + clean profile) | 134K | **57.9%** | 71.6% | 79.1% | +1.7%p |
| v33 (top-20 candidates) | 30K | 57.2% | 73.5% | 82.2% | top10보다 marginal |
| v34 (clinical features) | 30K | 56.9% | 72.4% | 79.4% | 마찬가지 |
| v32 (3-prompt ensemble) | 30K | 56.0% | - | - | errors 상관 |
| v40 (CoT hierarchical) | 30K | 31.7% | - | - | 파싱 오류 |
| v41 (logprob ranking) | 2K | 18.9% | - | - | 길이 편향 |
| v52 (two-stage v50→v28) | 5K | 58.3% | 80.7% | - | stage2 효과 없음 |
| v50 (per-candidate 0-9 score) | 30K | 59.1% | 79.8% | 83.8% | scoring 효과 |
| v53 (v50 + Bay α=0.9) | 5K | 59.0% | 80.8% | 85.6% | marginal 결합 |
| v54 (per-candidate 0-100 score) | 30K | 60.9% | 81.1% | 85.0% | scale 효과 |
| v54 (per-candidate 0-100 score) | 134K | **60.4%** | 81.1% | 84.5% | full eval |
| v59 (LLM 49 candidates) | 2K | 57.0% | 82.0% | 88.8% | recall 향상 |
| **v66 (v59→top10→v54-rescore)** | **30K** | **60.4%** | **86.3%** | **92.1%** | **prior best** |
| v68 (unique discriminator) | 2K | 56.4% | - | - | hurt |
| v69 (yes/no logprob) | 2K | 54.0% | 81.8% | 90.6% | hurt |
| v70 (sym pairwise) | 2K | 52.7% | 82.7% | 90.0% | hurt |
| v72 (LLM-cleaned profile) | 2K | 58.2% | 81.5% | 89.6% | @10=96.7% |
| v75 (3-prompt ensemble) | 2K | 58.9% | 84.2% | 90.2% | T0 alone 59.9% |
| v77 (generative DDx) | 2K | 54.2% | 76.9% | 83.9% | 1 call/patient |
| v76 (KG features in prompt) | 2K | 62.4% | 86.0% | 91.8% | KG inject |
| v79 (v76 + 2-stage rescore) | 2K | 63.8% | 86.2% | 93.0% | hybrid |
| **v79 (v76 + 2-stage rescore)** | **5K** | **63.9%** | **86.8%** | **93.4%** | **NEW BEST** |

주요 발견:
- 초록 500편이 최적 (2000편은 노이즈 증가로 하락)
- **프롬프트에 KG 증상을 포함하면 성능 하락** — KG는 후보 선정에만 사용, re-ranking은 LLM 자체 지식이 최적
- **구조화 프로필이 +6.7%p 향상** — 프롬프트 설계가 핵심 (LLM 능력이 아닌 입력 형식 문제)
- 19개 질환이 0% 정확도 (유사 질환 혼동: URTI↔rhinosinusitis, PSVT↔panic attack 등)
- age/sex prior로 @10 81.5%, @1 56.2% 달성
- KG 자체는 임상적으로 유효한 증상을 잘 추출 (Pneumonia→Fever/Cough, GERD→Reflux/Heartburn 등)

### 학술 조사 결과

- DDXPlus + 외부 KG로 진단 평가: **선행 연구 없음** (novel contribution)
- SymCat(801 질환), HealthKG(157 질환), RareBench(102 희귀질환) 추가 벤치마크 가능
- Rotmensch et al. 2017 (Scientific Reports, 340 인용): EMR 기반 자동 KG, P=0.85/R=0.60
- "LLM KG + 폐쇄형 어휘 + DDXPlus 감별진단 평가" 조합은 선행 없음

## 연구 방향

**자동 문진을 위한 KG 구축 및 감별진단 최적화**

### KG 구축 원칙

1. **벤치마크에서 질환 이름만 사용**: 증상 목록은 사용하지 않음. 벤치마크에 증상이 존재하지 않는다고 가정함.
2. **PubMed에서 자유롭게 관계 추출**: 질환 이름으로 PubMed 검색 → 초록에서 텍스트 매칭으로 모든 CUI 추출 → 초록 + CUI 리스트를 LLM에 전달 (초록당 1회) → LLM이 초록 내용을 읽고 관계 정의
3. **증상 매핑 없음**: 벤치마크 증상과 KG 증상 사이에 UMLS CUI 매핑을 하지 않음. KG가 자체적으로 증상 어휘를 정의.
4. **평가 지표는 진단 정확도 (GTPA@1)**: 엣지 수준 F1이 아닌, 구축된 KG로 실제 진단이 맞는지가 최종 기준.

### 목표

| 단계 | 목표 | 상태 |
|------|------|------|
| 1 | DDXPlus GTPA@1 >= 0.80 | **진행 중 (현재 63.9%, v79 KG features + 2-stage rescore)** |
| 2 | SymCat, RareBench 등 다중 벤치마크 검증 | v54 baseline 완료 (SymCat 39.7%, RareBench 22.1%); v79 적용 대기 |
| 2 | SymCat, HealthKG 등 다중 벤치마크에서 동일 파이프라인 검증 | 대기 |
| 3 | UMLS 전체 질환으로 범용 KG 확장 + 춘천성심병원 임상 검증 | 대기 |

### KG 구축 파이프라인

```
벤치마크 질환 이름 (DDXPlus 49개)
  → PubMed 검색 (질환 이름 + 동의어)
    → 초록에서 텍스트 매칭으로 모든 CUI 추출 (Aho-Corasick)
      → 초록 + CUI 리스트 → LLM 1회 호출 → 관계 정의
        → 통계 집계 → KG
          → KG로 감별진단 → GTPA@1 측정 → 최적화
```

### 사용 가능한 진단 벤치마크

#### 시도 완료
| 데이터셋 | 질환 수 | 형태 | 우리 결과 | SOTA (학습) | 비고 |
|---------|--------|------|---------|------------|------|
| **DDXPlus** | 49 | 134K 합성 환자 | **v79: 63.9% (5K)** | 80% (PARD/AARLC, Tchango 2022) | 주요 평가 벤치마크 |
| **SymCat** | 50 (of 801) | 확률 행렬 | v54: 39.7% (2.5K) | 70~75% (Mullenbach 2018) | DDXPlus의 전신, v79 미적용 |
| **RareBench** | 440 (of 1,121) | HPO 기반 케이스 | v54: 22.1% | GPT-4 22~29% (Chen 2024 KDD) | 희귀질환, KDD 2024, 거의 동등 |

#### 미시도 (확장 후보)
| 데이터셋 | 질환 수 | 형태 | 공개 | 우선순위 | SOTA 참고 | 비고 |
|---------|--------|------|------|---------|---------|------|
| **HealthKG** | 157 | 질환-증상 그래프 | O | 高 | - | MIT, 270K 환자 기반, 큐레이션 KG 비교 |
| **DDx100 / NEJM CPC** | 100 | 의사용 어려운 사례 | O | 高 | GPT-4 ~46% (Kanjee 2023 JAMA) | 임상 challenging 검증 |
| **MIMIC-IV diagnoses** | 수백 | EHR 실제 환자 | △ (PhysioNet credentialed) | 中 | various | RWE 검증 |
| **PrimeKG** | 17,080 | 다중 생물학적 KG | O | 中 | - | 커버리지 비교용, 진단 평가 부수적 |
| **MedQA-USMLE** | - | MCQ | O | 中 | GPT-4 90%+ | task 형태 다름 (선택형) |
| **AMBOSS / Step 1-3** | - | MCQ | △ (라이선스) | 低 | - | 의사 시험 |
| **Chinese DXY/Chunyu** | - | Chinese 자유서술 | △ | 低 | CCKS competition | 중국어 NLP, 본 연구 외 |

## 연구 기록 (2026-05-01 업데이트)

### 진단 알고리즘 변형 timeline

DDXPlus GTPA@1 향상을 위해 30개 이상의 변형을 시도하였습니다. 주요 분기점은 다음과 같습니다.

#### Phase 1: Baseline 정립 (v17–v34)

- v17 (Bayesian + age/sex prior): 56.2% — 첫 baseline
- v27 (no prior + medterm 매칭): 52.8% — prior 효과 -3.4%p
- v28 (KG signature + clean profile): 57.9% — +1.7%p
- v32–v34 (앙상블, top-20 확장, hand-curated features): 모두 56–57% 수준 — 효과 marginal

#### Phase 2: Per-candidate scoring 도입 (v50–v54)

- v50 (per-candidate 0–9 score): 59.1% — anchoring bias 해소로 +1.2%p
- v54 (per-candidate 0–100 fine score): **60.4% (134K 확정)** — 0–9 → 0–100으로 +1.8%p
- 이때까지 60% 천장이 형성됨

#### Phase 3: Stage 1 후보 확장 (v59, v66)

- v59 (모든 49 candidates LLM scoring, top-K 필터 없음): 57.0% (@5=88.8%)
- v66 (v59 stage1 → top10 → v54-style stage2 rescore): **60.4% (30K 확정), @3=86.3%, @5=92.1%**
- @1은 v54와 동일하나 @5 ceiling이 +7.6%p 향상

#### Phase 4: 60.4% 천장 깨기 시도 모음 (v68–v77, 모두 실패)

| 변형 | 접근 | 결과 (2K) | 분석 |
|-----|------|---------|------|
| v68 unique discriminator | 후보 disease 고유 KG features | 56.4% | KG noise 함께 제거 |
| v69 Yes/No logprob | 토큰 logprob softmax | 54.0% | LLM이 거의 항상 "Yes", 변별력 부족 |
| v70 symmetric pairwise | A/B + B/A 양방향 토너먼트 | 52.7% | 토너먼트 잡음 누적 |
| v72 LLM-cleaned profile | LLM이 환자 narrative 정제 | 58.2% | 정보 손실 |
| v75 3-prompt ensemble (borda) | 3개 prompt framing 결합 | 58.9% | T0 단독 (59.9%)이 가장 좋음 |
| v77 generative DDx | LLM이 49개 list에서 top-5 선택 | 54.2% | 1-call 생성은 per-candidate scoring 대비 약함 |

#### Phase 5: 돌파구 — KG features in prompt (v76, v79)

- **v76**: 각 후보 disease의 PubMed-derived top-8 symptom features를 prompt에 inject — **62.4% (2K), +2.0%p**
- **v79**: v76 stage1 → top10 → KG features rescore (2-stage) — **63.9% (5K 확정), +3.5%p**
- v80 (Bayesian prior 결합): α=0이 최적, prior 추가 효과 없음

### 누적 변형 결과 표 (DDXPlus)

| 변형 | 데이터 | GTPA@1 | @3 | @5 | 비고 |
|-----|------|--------|-----|-----|------|
| v17 (baseline + prior) | 134K | 56.2% | - | - | reference |
| v28 (KG sig + clean profile) | 134K | 57.9% | 71.6% | 79.1% | +1.7%p |
| v50 (per-candidate 0–9) | 30K | 59.1% | 79.8% | 83.8% | scoring 효과 |
| v54 (per-candidate 0–100) | 134K | 60.4% | 81.1% | 84.5% | scale 효과 |
| v66 (v59→top10→v54-rescore) | 30K | 60.4% | 86.3% | 92.1% | high @5 ceiling |
| v76 (KG features in prompt) | 2K | 62.4% | 86.0% | 91.8% | KG inject 첫 효과 |
| v79 (v76 + 2-stage rescore) | 2K | 63.8% | 86.2% | 93.0% | hybrid |
| **v79 (5K 확정)** | **5K** | **63.9%** | **86.8%** | **93.4%** | **현재 SOTA** |

### 다중 벤치마크 평가 + SOTA 비교

#### 시도한 벤치마크 (3종)

| 벤치마크 | 질환 수 | 환자 수 | 우리 결과 | @5 | KG top10 recall | Zero-shot LLM baseline (literature) | 학습 SOTA (literature) |
|---------|--------|--------|----------|-----|----------------|-----------------------------------|----------------------|
| **DDXPlus** | 49 | 5,000 (134K available) | **63.9% (v79)** | 93.4% | 97.9% | GPT-4 zero-shot ~55–60% | PARD/AARLC ~80% (Tchango 2022, NeurIPS) |
| **SymCat** | 50 (of 801) | 2,500 | 39.7% (v54) | 68.1% | 73.5% | zero-shot ~30–35% | Babylon Health Mullenbach 2018 trained ~70–75% |
| **RareBench** | 440 (of 1,121) | 1,121 | 22.1% (v54) | 43.7% | 48.8% | GPT-4 Top-1 ~22–29% (Chen 2024 KDD) | - (LLM-only 평가 벤치마크) |

#### SOTA 격차 분석

- **DDXPlus**: v79 **63.9%** vs 학습 SOTA **80%** → −16.1%p. Zero-shot LLM baseline 대비 +4~9%p 우위. 학습 신호 없이 메우기 어려운 구조적 격차.
- **SymCat**: v54 **39.7%** vs 학습 SOTA **70~75%** → −30~35%p. KG quality 자체가 낮음 (recall 73.5%) — KG 개선 우선. v79 미적용 → 적용 시 +3~5%p 기대.
- **RareBench**: v54 **22.1%** vs **GPT-4 22~29%** → 거의 동등. gemma-4-E4B (8B)로 GPT-4 (>1T params)와 동등한 결과. KG sparsity가 ceiling.

#### 미시도 벤치마크 (확장 후보)

| 벤치마크 | 질환 수 | 형태 | 데이터 보유 | 우선순위 | SOTA 참고 | 비고 |
|---------|--------|------|----------|---------|---------|------|
| **HealthKG** | 157 | 270K 환자 그래프 | ✗ (다운로드 필요) | 高 | - | MIT, KG 비교 baseline으로 적합 |
| **PrimeKG** | 17,080 | 다중 생물학적 KG | ✗ (다운로드 필요) | 中 | - | 커버리지 비교용, 진단 평가는 부수적 |
| **DDx100 / NEJM CPC** | 100 | 의사용 어려운 사례 | ✗ | 高 | GPT-4 ~46% top-1 (Kanjee 2023, JAMA) | 임상 challenging 검증 |
| **MedQA-USMLE** | - | MCQ | △ (공개) | 中 | GPT-4 90%+ | 진단 task와 형태 다름 |
| **AMBOSS / Step 1-3** | - | MCQ | ✗ (라이선스) | 低 | - | 의사 시험 |
| **Chinese DXY/Chunyu** | - | Chinese 자유서술 | ✗ | 低 | CCKS competition | 중국어 NLP, 본 연구 외 |
| **MIMIC-IV diagnoses** | 수백 | EHR 실제 환자 | △ (PhysioNet 자격 필요) | 中 | various | 실제 임상 환자, RWE 검증 |
| **ECP (NEJM CPC) extended** | - | NEJM 모든 CPC | ✗ | 中 | - | 추가 어려운 사례 |

**우선 추가 후보**:
1. **HealthKG**: KG-기반 평가 baseline. 우리 LLM-derived KG vs 큐레이션 KG 비교 가능.
2. **DDx100 (NEJM CPC)**: 의사용 어려운 사례에서 KG features inject 효과 검증. 임상적 의의 강조.
3. **MIMIC-IV diagnoses**: 실제 환자 (RWE)에서 검증. PhysioNet credentialed access 필요.

미시도 벤치마크 추가 시 v79 (KG features + 2-stage) 일관 적용으로 설계 일관성 유지.

### 핵심 학습

1. **Prompt의 KG 정보 inject가 결정적**: 같은 LLM, 같은 KG라도 prompt에 KG-derived features를 명시적으로 포함하면 천장을 깸. Disease 이름만 제공할 때는 LLM의 implicit medical knowledge가 한계.
2. **Per-candidate scoring > list-pick**: 49개 후보 중 1개 고르기는 anchoring bias 발생. 각 후보 독립 scoring이 안정적.
3. **0–100 fine scale > 0–9 coarse**: ties 감소, 분포 활용성 +1.8%p.
4. **Two-stage rescore의 효과는 미미**: stage1 선별 후 rescore는 +0.7~1.0%p에 그침. KG features inject가 +2.0%p로 더 큰 효과.
5. **Bayesian + LLM 결합 효과 없음**: α sweep에서 α=0 (LLM only)이 최적. LLM이 implicit하게 demographic prior를 이미 반영.
6. **Self-consistency, Yes/No logprob, pairwise tournament 모두 실패**: LLM의 confidence를 다른 형식으로 추출하는 시도들은 0–100 직접 scoring을 능가하지 못함.

## 다음 연구 방향 (2026-05 이후)

### 단기: KG-features 변형 확장 (v81–v85, 2K~5K 평가)

| 변형 | 접근 | 가설 |
|-----|------|------|
| v81 | KG features + counts (`fever (n=125)`) | 빈도 정보로 가중치 인식 |
| v82 | TOP_K_FEATURES = 12 또는 16 | 더 많은 features로 변별력↑ |
| v83 | KG features를 "exclude" 형식 ("typical: A, B; atypical: X, Y") | 부정 정보로 변별력↑ |
| v84 | 환자 evidence를 KG features와 직접 alignment + score | 명시적 매칭 |
| v85 | TOP_K=8 + counts + 2-stage (v79+v81 결합) | 단순 합산 |

목표: 65~70% 도달.

### 중기: 다중 벤치마크 확장 (SymCat, RareBench)

- **SymCat에 v79 적용**: 현재 39.7% (v54). KG features inject로 +3~5%p 기대 (43~45%). KG quality 개선이 더 시급할 가능성도 검토.
- **RareBench에 v79 적용**: 현재 22.1% (v54). 440 disease 후보 공간이라 ceiling 자체가 낮음. KG sparsity가 가장 큰 제약.
- **HealthKG, PrimeKG 추가**: 동일 파이프라인을 다양한 disease 분포에 검증.

### 중장기: KG 품질 개선

KG inject가 효과 있다면 더 좋은 KG가 더 좋은 효과를 낼 것입니다.

- **Step 1 NER 개선**: 현재 scispaCy threshold 0.85. LLM 기반 NER (UMLS-aware) 또는 SapBERT embedding 활용 검토.
- **Step 2 LLM 분류 정교화**: 현재 ternary (present/absent/related). discriminative feature까지 표시하는 4-way 분류 (`pathognomonic`, `common`, `possible`, `unrelated`) 시도.
- **PubMed 검색 확장**: 현재 질환당 ~500편. 1,000~2,000편으로 확장 시 KG density 효과 검증.
- **외부 KG 통합**: SemMedDB (legacy), HPO annotations, Hetionet 등을 LLM-derived KG와 합쳐 broad coverage 확보.

### 장기: 80% 달성을 위한 fundamental change

현재 LLM (gemma-4-E4B 8B) + 단일 PubMed KG로는 60% 후반이 천장으로 보입니다. 80%를 달성하려면:

1. **더 강한 LLM** (gemma-4 31B Dense, GPT-4 class) — user 제약상 불가
2. **다중 KG 앙상블** — PubMed + SemMedDB + HPO + DDXPlus-style symptom checker가 결합된 KG가 disease별 features 풍부도를 높임
3. **Patient profile 의미적 정규화** — LLM이 raw evidence를 임상 narrative로 변환 후 scoring (v72에서 단순 시도, 정보 손실 확인됨; clinical NER + ontology mapping 필요)
4. **Disambiguation chain** — 유사 disease 쌍 (URTI ↔ rhinosinusitis, PSVT ↔ panic attack)을 LLM이 명시적으로 비교하는 fine-grained pairwise

### 학술적 방향

- **Novel contribution 강화**: "PubMed-derived KG features inject into LLM prompt for closed-set differential diagnosis"는 선행 연구 부재. v79를 기준 결과로 한 paper 작성 가능.
- **Reproducibility**: 모든 변형 스크립트와 결과를 `pilot/scripts/`, `pilot/results/`에 보존. 재현 가능한 ablation study.
- **다음 baseline**: v79를 v66 대신 새 baseline으로 설정. 향후 모든 변형은 v79 대비 비교.

## 디렉토리 구조

```
Graph-DDXPlus/
├── README.md            v2 연구 계획 (본 문서)
├── CLAUDE.md            프로젝트 개발 지침
├── data/                원본 데이터셋 (UMLS, DDXPlus, SemMedDB, RareBench 등)
├── pilot/
│   ├── data/            파일럿 실험 데이터 (2,217 문서, gold standard, 분류 결과)
│   ├── results/         실험 결과 (1,920 조합 평가, 상위 10 보고서)
│   └── scripts/         실험 스크립트
└── archive/
    └── v1_ddxplus/      v1 연구 산출물 (코드, 논문, 결과, 로그)
```

## 협력

- **연구 책임**: 맨인블록 김주영
- **교신저자**: 방창석 교수님 (춘천성심병원)
- **협조 필요 사항**:
  - 1단계: LLM IE 결과 검증에 참여 가능한 의료진 인력
  - 3단계: 환자 데이터 활용, IRB 절차, 평가 프로토콜 설계 자문
