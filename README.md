# Graph-DDXPlus v2

UMLS 기반 데이터셋 독립 의료 지식그래프와 자동 감별 진단 시스템 연구.

## 현재 상태

연구 재설계 진행 중입니다. v1 연구의 설계 결함이 확인되어 한토픽 투고분은 철회되었으며 처음부터 다시 시작합니다.

- v1 산출물은 `archive/v1_ddxplus/`에 보존되어 있습니다.
- v2는 본 디렉토리에서 새로 구축됩니다.

## 배경

v1 시스템은 논문에 "UMLS 의료 지식그래프 위에서 추론을 수행한다"고 기술되었으나 실제로는 DDXPlus 데이터셋이 제공하는 질환-증상 관계 데이터를 그대로 사용하여 KG를 구축하고 있었습니다. 같은 데이터셋의 정답 라벨로 KG를 만들고 그 KG로 같은 데이터셋을 평가한 셈이며 DDXPlus에서 GTPA@1 91.05%라는 높은 점수가 나온 것도 이러한 구조적 우위에서 비롯된 결과입니다.

다른 벤치마크 데이터셋에 적용해 본 결과 커버리지가 매우 낮고 진단 점수 또한 처참한 수준으로 나타나면서 문제가 확인되었습니다. 순수 UMLS 데이터만으로 KG를 재구축해 본 결과 DDXPlus 49개 질환 중 30개만 커버되어 커버리지가 61%에 불과하였고 질환당 평균 증상 수는 6.0개로 DDXPlus 원본 평균 18.1개의 1/3 수준이었습니다.

## 학술적 정당성

NLM의 SemMedDB는 SemRep 기반으로 PubMed 의료 의미관계를 추출한 표준 자원이며 1억 8천만 건 이상의 트리플을 보유하고 있습니다. 그러나 SemMedDB는 2024년을 마지막으로 업데이트가 종료되었으며 이는 의료 KG 분야 전체의 공백을 의미합니다.

본 연구는 종료된 SemMedDB를 대체할 수 있는 재현 가능하고 지속 가능한 오픈소스 LLM 기반 의료 KG 구축 파이프라인을 제안하고 이를 활용하여 데이터셋 독립적인 자동 감별 진단 시스템을 구현하고 임상 환경에서 검증합니다.

## 연구 계획

본 연구의 목표는 다음과 같이 우선순위가 정해져 있습니다.

| 우선순위 | 단계 | 목표 |
|---------|------|------|
| **1차** | 1단계 | UMLS 기반 데이터셋 독립 KG 구축 |
| **2차** | 2단계 | 구축된 KG로 자동 진단 벤치마킹 최적화 |
| **3차** | 3단계 | 춘천성심병원 환자 데이터로 임상 검증 |

1차 목표인 KG 구축이 본 연구의 핵심이며 학술적 contribution의 중심입니다. 2차 목표는 1차에서 확보한 KG의 실용성을 입증하는 단계이고 3차 목표는 벤치마크 수준을 넘어 실제 임상 환경에서의 유효성을 검증하는 단계입니다.

### 1단계. UMLS 기반 데이터셋 독립 KG 신규 구축

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

PrimeKG은 이미 큐레이션된 DB를 통합하는 프로젝트이며 새로운 관계를 발견할 수 없습니다. 본 연구는 2024년에 종료된 SemMedDB의 후속으로서 SemRep(규칙 기반)이 수행하던 역할을 LLM으로 대체하면서 통계적 검증을 강화합니다. PrimeKG과 관계 스키마(positive/negative)가 유사한 것은 학술 표준을 따른 결과이며 방법론은 근본적으로 다릅니다.

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

## 다음 목표

### Phase 1: UMLS DISO 전체 KG 구축

UMLS DISO에서 노이즈를 제거한 전체 CUI를 대상으로 KG를 구축합니다. 파일럿에서 검증된 최적 설정(S2-J 프롬프트, CUI 1-level 전파, 최소 공출현 3회)을 적용합니다.

노이즈 제거 방법은 여러 접근을 병행합니다.
- T033/T034 semantic type 제외 (Kilicoglu et al. 2012, Rotmensch et al. 2017)
- 수동 블랙리스트 5 CUI
- Information Content 기반 overly general 개념 제외 (Sánchez et al. 2011)
- SNOMED CT CORE Problem List Subset 화이트리스트 교집합

### Phase 2: 다중 벤치마크 평가

구축된 KG를 DDXPlus 외 다른 벤치마크에서도 평가합니다.
- DDXPlus (49 질환, 324 symptom 쌍)
- RareBench (11,150 질환, HPO annotation)
- SemMedDB (189,879 disease-symptom 엣지, baseline 비교)
- HPO phenotype.hpoa (전문가 큐레이션 gold standard)

### Phase 3: GPU 서버에서 대규모 처리

별도 GPU 서버에서 vLLM 기반 배치 추론으로 대규모 데이터를 처리합니다.

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
