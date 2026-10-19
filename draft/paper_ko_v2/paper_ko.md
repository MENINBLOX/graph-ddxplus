# 열린 집합 감별진단을 위한 출처 추적형 문헌 그래프: 텍스트 검색과의 상보성과 근거 품질 신호

> 국문 초안 v2.2(2026-10-12). 재설계 연구의 clean v2.1c 그래프 결과와 baseline·비교 지식원·유의성 검정(`docs/research_design.md` §9.14–9.25)을 반영해 직접 개정했다. `[ref: …]`는 미해결 인용이며, 표 11의 외부 보고 수치는 인용 전 원문 대조가 필요하다.

## 초록

진단 보조 시스템이 근거 문헌을 제시하려면, 질환과 소견의 관계가 원 문헌의 문장까지 연결된 지식 계층이 필요하다. 본 연구는 저사양 LLM으로 공개 문헌에서 추출한 질환–소견 그래프가 이러한 지식 계층으로서 무엇을 기여하는지를, 추론 시점 LLM 없이 열린 집합 감별진단으로 평가한다. 사람 질환 43,575개의 이름으로 PMC 오픈액세스 리뷰·증례보고와 StatPearls·GeneReviews를 수집하고, GPU 1장에서 구동하는 Qwen3-4B-Instruct-2507로 소견을 원문 표현 그대로 추출해 UMLS로 정규화했다. 그 결과 소견 간선 1,184,344개와 근거 언급 1,648,509개로 이루어진 Neo4j 그래프를 얻었으며, 모든 간선은 근거 문장과 원 문헌에 연결된다. 설정은 DDXPlus validate 분할에서만 고르고 9개 시험 데이터셋에 그대로 적용했다. 같은 문헌의 BM25 텍스트 검색과 비교하면 그래프는 DDXPlus와 RAMEDIS에서 높았지만 HPO 기반 희귀질환 데이터셋 다수에서는 낮았다. 두 계층을 순위 결합하면 그래프 단독 대비 Top-10이 9개 중 8개 데이터셋에서 올랐다(DDXPlus 29.1%→36.3%, Phenopacket-store 11.1%→16.1%). LLM 없이 MEDLINE 전체의 MeSH 공출현으로 만든 그래프를 같은 질의로 평가하면 9개 모두에서 LLM 추출 그래프가 높았다. 출처 정보는 근거 품질의 신호로도 작동했다. 감사에서 근거 문장의 지지율은 전체 29.5%였으나, 질환을 주제로 다루는 문헌에서 온 간선은 38.3%, 그렇지 않은 간선은 11.9%였다. 독립 큐레이션(HPO 주석)과의 일치율도 지지 문헌이 1편에서 5편 이상으로 늘면 13.3%에서 63.2%로 높아졌다. 이 결과는 출처 추적형 문헌 그래프가 텍스트 검색을 대체하기보다 보완하는 지식 계층이며, 간선 단위 출처가 근거를 선별하는 신호로 쓰일 수 있음을 보여 준다.

## 1. 서론

감별진단을 자동화하려면 질환과 소견의 관계를 담은 지식원이 필요하다. 한쪽 극단에는 HPO 주석처럼 큐레이션된 질환–표현형 주석이 있다 [ref: HPO annotations]. 이 자원은 정확하지만 희귀질환을 중심으로 구성되어 있고 사람이 계속 유지해야 하며, 흔한 질환에는 같은 수준의 자원이 드물다. 다른 쪽 극단에는 넓은 의학 지식을 갖췄지만 진단할 때마다 호출 비용이 드는 대형 LLM이 있다. 대형 LLM의 근거는 모델이 생성한 텍스트이므로 결론을 뒷받침한 원 문헌을 결정론적으로 확인하기 어렵다 [ref: LLM diagnosis]. 최근의 진단 시스템은 LLM에 검색·도구 계층을 결합하는 방향으로 발전하고 있다 [ref: DeepRare, MEDDxAgent, MedRAG]. 이 구조에서 지식 계층의 질은 최종 진단과 그 근거 제시의 상한을 정한다. 임상 의사결정 보조에서는 결론의 근거를 원 문헌까지 확인할 수 있어야 한다는 요구도 있다 [ref: explainability in clinical decision support].

검색 계층으로 가장 널리 쓰이는 것은 문헌을 잘게 나눈 텍스트 검색이다 [ref: MedRAG]. 텍스트 검색은 구축이 간단하지만 문단 단위로 결과를 돌려주므로, 어떤 소견이 어느 문장에서 어느 질환의 근거가 되었는지를 구조적으로 제시하지 않는다. 반면 문헌에서 질환–소견 관계를 추출해 문장 단위 출처와 함께 그래프로 구성하면 근거를 소견 단위로 제시할 수 있다. 그러나 이러한 그래프가 같은 문헌의 텍스트 검색에 비해 진단 후보를 얼마나 잘 회수하는지, 둘이 대체 관계인지 보완 관계인지, 그리고 출처 정보가 근거의 질을 가려내는 데 쓰일 수 있는지는 측정되지 않았다. 평가 조건에도 공백이 있다. 후보를 벤치마크 목록으로 좁히지 않는 열린 집합에서, 인코딩이 서로 다른 여러 벤치마크를 하나의 지식원과 하나의 설정으로 평가한 사례는 드물다 [ref: cross-benchmark evaluation].

본 연구는 이 질문들을 네 가지로 정리한다. RQ1은 출처 추적형 문헌 그래프가 같은 문헌의 텍스트 검색, 문헌 공출현 그래프, 큐레이션 지식과 비교해 9개 데이터셋의 열린 집합 감별진단에서 어떤 정확도를 보이며 텍스트 검색과 결합하면 어떻게 달라지는가다. RQ2는 그래프 구조를 이용한 다단계 감별의 질문 수 대비 정확도다. RQ3은 근거 문장이 간선을 실제로 뒷받침하는지와, 출처 정보가 그 지지 여부를 가려내는 신호가 되는지다. RQ4는 IE 모델 교체가 진단 정확도를 바꾸는지에 대한 간접 검증이다.

지식 구축 단계에서는 질환명만으로 공개 문헌을 수집하고, 소비자용 GPU 1장에서 구동하는 저사양 LLM이 문헌에 쓰인 표현 그대로 소견과 관련 요인을 추출한다. 추출 결과는 UMLS 개념으로 정규화해 그래프 DB(Neo4j)에 적재하고, 모든 소견 간선에 근거 문장과 원 문헌(PMID/PMCID 또는 StatPearls·GeneReviews 장 ID)을 연결한다. 각 근거에는 그 문헌이 해당 질환을 주제로 다루는지도 함께 기록한다. 진단 단계에서는 환자 소견을 그래프에 질의해 후보 질환의 순위를 내고, 근거 경로 질의 하나로 소견, 근거 문장, 논문까지 반환한다. 같은 출처 구조는 간선을 뒷받침하는 문헌의 수와 주제성을 질의 시점에 집계하게 해 주므로, 근거의 질을 점수에 반영하는 기능도 그래프 질의로 표현된다.

점수에 영향을 주는 정보는 네 가지로 한정한다. 공개 문헌에서 추출한 결과, 개념 어휘와 그 계층, 벤치마크 근거 코드를 표준 개념으로 옮기는 어휘 사상, 평가 대상 환자 본인의 소견이다. 질환–증상 주석, 벤치마크 정답 라벨, 벤치마크 집단 통계는 점수 계산에 쓰지 않는다. 추론 시점 LLM을 쓰지 않는 것도 같은 목적의 통제다. 생성기를 결합하면 생성기의 능력과 지식 계층의 질이 섞이므로, 지식 계층의 기여를 따로 측정하려면 검색 단계만으로 평가해야 한다. 큐레이션된 HPO 주석은 점수에 쓰지 않고, 비교 지식원과 간선 품질의 참조로만 쓴다. 설정은 공식 검증 분할인 DDXPlus validate 분할에서만 선택한다.

이 연구의 기여는 다섯 가지다. 첫째, 저사양 LLM 추출과 그래프 DB로, 모든 소견 간선이 원 문헌의 문장에 연결된 감별진단 그래프(후보 질환 43,575개, 소견 간선 1,184,344개)와 그 구축 파이프라인을 제시한다. 둘째, 같은 문헌의 텍스트 검색(BM25, MedCPT), 문헌 공출현 그래프, 기존 지식 그래프(PrimeKG), 큐레이션 주석을 같은 열린 집합과 같은 프로토콜로 비교해, LLM 추출 그래프가 공출현보다 일관되게 높고 텍스트 검색과는 상보적임을 대응 검정과 함께 보고한다. 셋째, 의료진 감사, LLM 판정, 큐레이션 대조의 세 방향에서 근거 품질을 측정하고, 문헌의 주제성과 지지 문헌 수가 근거 지지 여부를 가르는 신호임을 보인다. 넷째, 흔한 질환 데이터로 고른 점수 형태 기능이 희귀질환으로 전이되지 않는 반면 출처 기반 기능은 방향이 대체로 일관됨을 ablation과 대응 검정으로 보고한다. 다섯째, IE 모델 교체 실험, 그래프 기반 다단계 감별, 누수 점검을 포함한 코드, 설정, 감사 시트, 측정 기록을 공개한다.

## 2. 관련 연구

DDXPlus 등 공개 벤치마크에서 지도학습과 강화학습 방법은 학습 라벨을 활용해 높은 정확도를 보고한다 [ref: AARLC, DDxT]. DDXPlus는 진단 정확도와 함께 진단에 이르기까지 던진 질문 수(IL)도 평가한다 [ref: DDXPlus]. 대형 LLM으로 증례를 진단하는 zero-shot 방법과, 도구 호출·검색·증례 데이터베이스를 결합한 agentic 방법(예: DeepRare, MEDDxAgent)도 보고되었다 [ref: DeepRare, MEDDxAgent]. 이 방법들은 추론 시점에 LLM을 호출하며, 다수는 후보 질환을 벤치마크 목록으로 한정한 조건에서 평가되었다. 본 연구는 이들의 내부 구성 요소인 지식·검색 계층을 따로 평가하므로, 보고된 수치는 조건과 함께 참조로만 제시한다(표 11).

의료 질의응답의 검색 증강 생성(RAG)에서는 PubMed, StatPearls, 교과서를 BM25나 MedCPT 같은 검색기로 찾는 구성이 표준 비교 조건으로 쓰인다 [ref: MedRAG/MIRAGE, MedCPT]. 본 연구는 같은 검색기를 같은 문헌에 적용해 진단 후보 회수를 비교한다. 희귀질환 진단에서는 환자의 HPO 표현형과 큐레이션된 질환 주석의 유사도로 순위를 매기는 표현형 매칭이 쓰인다 [ref: Phenomizer, Exomiser, PhenoBrain]. 문헌과 여러 데이터베이스를 통합한 의료 지식 그래프도 구축되어 왔고 [ref: SemMedDB, PrimeKG, iKraph, PubTator3], 최근에는 LLM 에이전트로 문헌에서 대규모 지식 그래프를 구축하고 RAG에 활용한 연구도 있다 [ref: MedKGent]. 근거 출처를 진단 설명에 연결하거나, 구축 시점의 근거 품질 점수를 하류로 전파하는 연구도 보고되었다 [ref: provenance gap in clinical AI; construction-time quality propagation]. 본 연구는 질환별 문헌에서 소견을 원문 그대로 추출해 문장 단위 출처를 보존하고, 그 출처 정보 자체를 품질 신호로 측정하며, 이를 열린 집합의 다중 벤치마크 진단에서 텍스트 검색·공출현·큐레이션 지식과 같은 조건으로 비교한다는 점에서 다르다.

LLM을 이용한 임상 정보추출에서는 원문에 없는 내용이 생성 결과에 섞일 위험이 있다 [ref: LLM hallucination in IE]. 본 연구는 원문 표현을 그대로 옮기라는 추출형 지시와, 원문에서 실제로 찾은 구절만 남기는 규칙으로 모든 간선을 원문 문장에 결정론적으로 연결한다. 그러나 원문에 있는 구절이라도 그 문장이 해당 질환의 소견을 서술한다는 보장은 없으므로, 지지 여부를 사람 감사와 큐레이션 대조로 따로 측정한다.

## 3. 방법

### 3.1 설계 경계

진단 점수에 영향을 줄 수 있는 정보의 범위를 미리 정하고, 그 밖의 정보는 파이프라인 어디에서도 점수에 쓰지 않는다(표 1). 점수에 쓸 수 있는 정보는 공개 문헌에서 추출한 결과, UMLS·HPO·MONDO의 개념–개념 관계로 이루어진 어휘와 계층, 벤치마크 근거 코드를 표준 개념으로 옮기는 어휘 사상(크로스워크), 평가 대상 환자 본인의 입력 네 가지다. HPO annotation, Orphanet, OMIM clinical synopsis 같은 질환–표현형 주석은 질환과 소견의 연결을 직접 담고 있으므로 주 시스템의 점수에 쓰지 않는다. 이들은 3.7절의 비교 지식원과 3.8절의 품질 평가에서 별도 시스템과 참조로만 쓴다. 벤치마크가 배포한 질환–증상 정보, 학습 라벨, 집단 통계는 쓰지 않는다. 벤치마크 정답 목록을 후보 질환으로 쓰거나 정답 질환에 대해서만 문헌을 깊게 수집하는 것도 금지한다(닫힌 집합 보조 분석은 4.4절에 별도로 표시한다). 설정 선택에는 시험(test) 분할을 쓰지 않는다. 벤치마크 증례의 원출처 논문이 수집 문헌에 섞이는 것을 막기 위해 Phenopacket-store가 기록한 원출처 논문(PMID 1,733편)을 수집에서 제외하고, 원출처를 알 수 없는 데이터셋에는 데이터 기반 누수 점검을 적용한다(3.9절).

Table 1. Boundary of information that may influence scores of the proposed system

| Allowed | Forbidden |
|---|---|
| Extraction results from public literature | Disease–phenotype annotations (HPO annotation, Orphanet, OMIM clinical synopsis)^a^ |
| Concept vocabularies and hierarchies (concept–concept relations in UMLS, HPO, MONDO) | Disease–symptom information distributed with benchmarks |
| Vocabulary mapping (crosswalk) from benchmark evidence codes to standard concepts | Benchmark training labels |
| Input of the patient under evaluation | Benchmark population statistics |
| | Using benchmark gold-diagnosis lists as candidates^b^ |
| | Collecting literature in depth only for gold diagnoses |
| | Using test data for configuration selection |
| | Source papers of benchmark cases (1,733 PMIDs recorded by Phenopacket-store) |

^a^ Used only as separate comparator systems (Section 3.7) and as a quality reference (Section 3.8), never in the proposed system's scores. ^b^ Except the auxiliary closed-set analysis, reported separately (Section 4.4).

### 3.2 질환 목록과 문헌 수집

후보 질환 집합은 두 의학 용어 체계를 합쳐 구성했다. UMLS만 쓰면 최근에 정의된 희귀질환이 빠지고, MONDO만 쓰면 흔한 질환이 빠지기 때문이다. 문헌 수집 대상은 UMLS 질환 개념 35,590개에 UMLS에 대응 개념이 없는 MONDO 질환 16,304개를 더한 51,894개다. 이 가운데 MONDO의 '비인간 동물 질환'(MONDO:0005583) 하위 8,319개는 사람의 진단 대상이 아니므로 제외했으며, 최종 후보 질환은 43,575개다. 노드 키는 UMLS CUI를 기본으로 하고 CUI가 없는 질환에만 MONDO ID를 썼다. 질환 계층은 MONDO is_a와 UMLS 질환 계층을 합쳐 45,118쌍으로 구성했다.

문헌 검색의 질의어로는 질환명과 그 이름 변형, 표준 용어 동의어만 사용하였다. 질환당 수집 상한은 PMC 오픈액세스 리뷰 15편, 증례보고 10편, 논문당 본문 12,000자로 두고 모든 질환에 동일하게 적용하였다. StatPearls와 GeneReviews는 NCBI LitArch 배포본에서 제목 일치로 해당 질환에 연결하였다. 논문마다 PMID, PMCID, 제목, 연도, 라이선스를 저장하였다. 요청 실패는 '문헌 없음'과 구분하여 기록하고 재시도하였다. 20개 이상의 질환에 동시에 연결된 문헌은 특정 질환의 근거로 보기 어려워 제외했으며, 이 기준은 코퍼스만으로 사전에 정하였다. PMC 검색은 본문 전체를 대상으로 하므로 다른 질환을 다루면서 해당 질환명을 지나가듯 언급한 문헌도 검색된다. 이를 구분하기 위해 문헌 제목이 그 질환의 이름 형태(질의에 쓴 이름, 변형, 동의어) 중 하나를 포함하면 그 문헌을 해당 질환의 '주제 문헌'으로 표시했다. 이 규칙도 코퍼스만으로 정의했다.

### 3.3 정보추출과 정규화

정보추출에는 Qwen3-4B-Instruct-2507을 주 모델로, gemma-4-E4B를 비교 모델로 둔다. 주 모델은 시험 평가 전에 라벨 없는 시범 추출(질환 40개)의 질환당 고유 개념 수를 기준으로 정했다(4.2절). 추론은 RTX 4090(24GB) 1장에서 vLLM 배치 추론으로 수행하고, 디코딩은 탐욕적 방식으로 고정한다. 모델은 질환명과 논문 본문만 입력받아, 한 번의 통과에서 질환의 임상 소견(증상, 징후, 검사·영상 소견, 합병증)과 관련 요인(위험인자, 노출, 병력, 가족력, 약물, 연령·성별 경향)을 추출한다. 프롬프트는 각 항목을 원문 표현 그대로 6단어 이하의 구절로 쓰게 하고, 부정된 항목과 치료, 다른 질환의 소견은 제외하도록 지시한다. 예시는 넣지 않으며 추출 개수도 제한하지 않는다. 긴 논문은 6,000자 창으로 나누고 인접 창을 400자씩 겹친다. 추출된 구절은 원문 창에서 발견될 때만 남기고, 구절이 들어 있는 문장을 근거 문장으로 저장한다. 표의 행처럼 문장 경계가 없어 600자를 넘는 근거는 구절을 중심으로 600자를 잘라 저장한다.

추출 구절은 scispaCy로 후보 개념을 생성한 뒤 정확 일치 후보만 채택하는 임계값 없는 UMLS 링커로 정규화한다 [ref: scispaCy]. 의미유형은 UMLS MRSTY에서 읽고, 허용 의미유형(증상·징후·소견·질환·검사결과·노출·약물 등)에 속하지 않으면 연결을 버린다. 검사·시술·진단 절차·의료 활동(T058–T061)은 소견이 아니므로 제외한다. 개념 이름이나 동의어가 구절 안에 연속으로 나타나지 않거나 구절 내용어의 50% 미만만 덮으면, 그리고 부정 표지가 있거나 구절이 8단어를 넘으면 연결을 버린다.

### 3.4 그래프 모델

정규화된 추출 결과는 네 종류의 노드와 일곱 종류의 관계로 이루어진 그래프에 적재한다(표 2). Mention 노드는 추출 구절, 근거 문장, 그리고 그 문헌이 해당 질환의 주제 문헌인지(topic)를 담으며 출처의 기본 단위가 된다. HAS_FINDING 간선은 해당 연결을 뒷받침하는 언급 수(n_mentions), 문헌 수(n_docs), 주제 문헌 수(n_topic_docs)를 가진다. 적재 후에는 모든 HAS_FINDING 간선이 Mention으로 뒷받침되는지와 Disease 노드 수가 후보 질환 수와 같은지를 검사하고, 어긋나면 적재를 중단한다. 따라서 진단 근거는 언제나 경로 질의로 근거 문장과 원 문헌까지 반환된다.

Table 2. Graph schema

| Element | Type | Properties | Role |
|---|---|---|---|
| Disease | Node | name, number of findings, collected text size | Candidate disease |
| Concept | Node | semantic type, IDF | Normalized finding concept |
| Doc | Node | PMID, PMCID, title, year, license | Source article or book chapter |
| Mention | Node | extracted span, evidence sentence, topic flag | Unit of provenance |
| (Disease)-[:HAS_FINDING]->(Concept) | Relationship | n_mentions, n_docs, n_topic_docs | Disease–finding association with support counts |
| (Mention)-[:OF]->(Disease) | Relationship | – | Links a mention to its disease |
| (Mention)-[:MENTIONS]->(Concept) | Relationship | – | Links a mention to its finding concept |
| (Mention)-[:FROM]->(Doc) | Relationship | – | Links a mention to its source document |
| (Concept)-[:IS_A]->(Concept) | Relationship | – | Finding concept hierarchy |
| (Disease)-[:IS_A]->(Disease) | Relationship | – | Disease hierarchy |
| (Concept)-[:IS_DISEASE]->(Disease) | Relationship | – | Links a finding concept to the corresponding disease |

### 3.5 진단 질의

단일 진단은 전체 질환을 후보로 두는 열린 집합에서 하나의 Cypher 질의로 순위를 낸다. 질의는 환자 소견 개념을 IS_A로 한 단계 확장하고(확장 개념에는 감쇠 가중), HAS_FINDING으로 각 질환과 대조해 일치 개념의 IDF를 합산한다. 각 간선의 기여에는 두 가지 출처 가중을 곱한다. 지지 문헌 수 가중 n_docs^β와, 주제 문헌이 없는 간선(n_topic_docs = 0)에 곱하는 가중 γ다(β = 0, γ = 1이면 출처 가중이 없다). 합산 점수는 질환 프로파일 크기 n에 대한 길이 정규화를 거치되, 문헌이 적은 질환의 과대평가를 막기 위해 유효 길이 n' = n + μ(1 − c)·L̄를 쓴다(c = s/(s + s_med), s는 질환별 수집 원문 문자 수, s_med는 그 중앙값, L̄는 평균 프로파일 길이). 통합 근거는 환자 소견 개념이 질환 노드이기도 할 때 IS_DISEASE와 질환 계층을 따라 해당 질환과 하위 질환을 지지한다. 출처 가중은 간선을 그래프에서 지우지 않는다. γ = 0이어도 해당 간선과 근거 문장은 추적 질의로 조회된다.

다단계 감별은 그래프에 질의해 추가 소견을 차례로 묻고 그 응답으로 순위를 갱신한다. 초기 근거(DDXPlus는 주소, Phenopacket-store는 첫 소견)로 순위를 낸 뒤, 상위 후보들의 HAS_FINDING 이웃 소견 가운데 순위 기반 신념을 가장 고르게 나누는 소견을 그래프 질의로 골라 묻는다. 응답은 기록에 있으면 '있음', 명시적으로 없으면 '없음'(DDXPlus 문진 중 기록되지 않은 항목, Phenopacket-store의 excluded 소견), 그 밖은 '알 수 없음'이다. '없음' 소견을 가진 후보는 감점 가중치만큼 순위를 낮춘다. 개방 행동 공간은 증상·징후·소견·질환 의미유형의 모든 개념을, 벤치마크 행동 공간은 해당 벤치마크가 기록하는 개념만 질문 대상으로 삼는다 [ref: DDXPlus]. DDXPlus 시험 분할과 Phenopacket-store에서 각각 1,000명을 시드 0으로 추출해 질문 예산 0·5·10·20회 후의 순위를 평가한다.

### 3.6 벤치마크와 평가 프로토콜

시험에는 9개 데이터셋을 사용하였다(표 3). DDXPlus는 시험 분할에서 시드 0으로 3,000명을 추출하였다. SymCat은 801개 질환마다 10명씩 생성한 뒤 증상이 없는 표본과 정답 사상이 안 된 질환을 제외하여 7,481명을 사용하였다. 희귀질환은 RareBench의 RAMEDIS(624명), LIRICAL(370명), HMS(88명), MME(40명), 북경협화병원의 실제 증례 PUMCH_L(988명), PUMCH-ADM(75명), 출판 증례 Phenopacket-store 0.1.27(10,310명)을 사용하였다.

Table 3. Benchmarks (encoding, patients, source type)

| Benchmark | Encoding | Patients | Source type |
|---|---|---|---|
| DDXPlus (test) | Coded questionnaire answers with values | 3,000 (sampled, seed 0) | Synthetic |
| SymCat | Symptom list | 7,481^a^ | Synthetic |
| RareBench RAMEDIS | Phenotype terms | 624 | RareBench subset |
| RareBench LIRICAL | Phenotype terms | 370 | RareBench subset |
| RareBench HMS | Phenotype terms | 88 | RareBench subset |
| RareBench MME | Phenotype terms | 40 | RareBench subset |
| PUMCH_L | Phenotype terms | 988 | Real hospital cases (PUMCH) |
| PUMCH-ADM | Phenotype terms | 75 | Real hospital cases (PUMCH) |
| Phenopacket-store 0.1.27 | Phenotype terms | 10,310 | Published case reports |

^a^ Generated as 801 diseases × 10 patients; samples with no randomly drawn symptom and diseases without a gold-label mapping were excluded.

근거 전처리는 결정론적 어휘 사상으로 수행하였다. DDXPlus는 응답 값을 반영해 기본값 응답을 근거에서 제외하고 통증 부위 응답을 부위별 통증 개념으로 합성하였다. 정답 질환은 영문 질환명의 UMLS 정확 일치(5자 이하 대문자 약어 제외)를 우선하고 없으면 ICD-10 경로로 사상하였다. 희귀질환 정답은 OMIM·Orphanet 식별자에서 UMLS 또는 MONDO 교차참조로 사상하였다.

정답 판정의 주 기준은 lineage로, 예측 질환이 정답 개념이거나 그 하위 3단계 이내 개념이면 정답으로 인정하고 상위 개념은 인정하지 않는다. 보조 기준 strict는 정확 일치만 인정한다. 지표는 Top-1, Top-3, Top-10이며, 95% 신뢰구간은 환자 단위 부트스트랩(1,000회)으로 구한다. 설정은 DDXPlus validate 분할(3,000명)에서만 lineage Top-1(동률 시 Top-3)을 기준으로 좌표 탐색해 고른다. 탐색 범위는 길이 지수 α {0.2, 0.25, 0.35, 0.5}, μ {0, 1, 2, 4}, 계층 확장 감쇠 {0, 0.3, 0.5, 0.7}, 통합 근거 가중치 {0, 0.5, 1, 2}, β {0, 0.5, 1, 2}, γ {1, 0.5, 0.25, 0}이다. 다단계 감별의 감점 가중치 {0, 0.5, 1}과 후보 수 {10, 20, 40}는 단일 진단 설정을 고정한 뒤 validate 500명, 벤치마크 행동 공간, 질문 20회에서 고른다. 비교 시스템(3.7절)도 각자의 설정을 같은 validate 분할과 같은 기준으로 고른다.

시험 분할 사용 이력은 다음과 같다. 그래프 구축 초기 두 차례는 그래프 초기화가 메모리 한도로 일부 실패해 이전 적재의 노드와 간선이 남은 상태에서 시험 분할이 평가되었고, 두 평가는 무효로 처리했다(4.1절). 이후 그래프 변경(비인간 질환 제외, 검사·시술 의미유형 제외, 주제 문헌 표시)은 시험 결과가 아니라 출처 감사에서 도출했으며, 설정은 매번 validate에서만 다시 골랐다. 그래프와 텍스트 검색의 순위 결합(3.7절)은 텍스트 검색 baseline의 시험 결과를 본 뒤 추가한 사후 분석이며, 결합 가중치는 validate에서만 골랐다.

### 3.7 비교 시스템

**같은 문헌의 텍스트 검색.** 그래프와 같은 수집 문헌(일반 문헌과 비인간 질환 제외) 122,661편을 1,000자 단위로 나눈 655,965개 청크를 색인했다. 이는 그래프에 근거가 적재된 문헌(88,266편)보다 많아 비교 시스템에 유리하다. 검색기는 BM25(영어 어간 추출, 불용어 제거)와 MedCPT(의료 문헌 검색용 dense retriever, 질의·문서 인코더 분리) [ref: MedCPT]를 사용했다. 환자 소견 개념은 UMLS 선호명으로 바꿔 질의하며, 그래프와 같은 정보만 입력한다. 검색된 청크의 점수는 그 문헌이 연결된 질환들에 집계해 열린 집합 순위를 낸다. 질의 방식(소견 전체를 한 질의로, 또는 소견별 질의 후 합산), 검색 깊이 {100, 1,000}, 집계 방식 {최댓값, 합, 역순위 결합(RRF)}은 validate에서 고른다.

**같은 질의, 다른 지식원.** 정확도의 차이가 질의 방식 때문인지 지식 때문인지를 가르기 위해, 기존 지식원을 같은 스키마로 별도 Neo4j 인스턴스에 적재하고 같은 Cypher 질의로 평가했다. 지식원은 세 가지다. MEDLINE MeSH 공출현(2024년 요약, 공동 색인 인용 수를 지지 수로 사용)은 LLM 없이 생의학 문헌 전체의 색인에서 얻는 질환–소견 연관이다. PrimeKG의 disease_phenotype_positive 관계 [ref: PrimeKG]와 HPO 주석 [ref: HPO annotations]은 큐레이션 지식이다. 이 지식원들은 근거 문장을 갖지 않는다. 설정은 각각 validate에서 고른다.

**순위 결합.** 그래프 순위와 텍스트 검색 순위를 역순위 결합한다. 점수는 (1 − w)/(60 + 그래프 순위) + w/(60 + 텍스트 검색 순위)이며, w ∈ {0, 0.25, 0.5, 0.75, 1}을 validate에서 고른다. 결합 결과에서 그래프가 순위를 낸 질환은 그래프의 근거 경로를 그대로 가진다.

### 3.8 출처 품질 평가

출처를 식별할 수 있다는 것과 근거 문장이 간선을 실제로 뒷받침한다는 것은 다르다. 첫째, 소견 간선 200개를 시드 고정 무작위로 뽑아 근거 언급 하나와 함께 제시하고, 근거 문장이 그 질환의 소견·징후·검사 이상·합병증·위험인자·병력을 서술하는지(supports), 서술하지 않는지(not_supported), 서술하지만 추출 개념이 달라졌거나 소견이 아닌 개념(약물·검사·방법 등)으로 연결되었는지(wrong_concept)를 판정했다. 판정은 연구팀과 협력 병원 의료진이 수행했고, LLM(Claude)이 같은 표본을 서로의 판정을 보지 않고 따로 판정해 일치도를 쟀다. 둘째, 판정자에 의존하지 않는 대조로, HPO 주석(phenotype.hpoa, 2026-02-16 판)이 있는 질환에서 그래프 간선의 소견 개념이 큐레이터가 같은 질환에 부여한 표현형과 일치하는 비율을 쟀다. 비교 대상은 소견 개념이 HPO에 존재하는 간선으로 한정하고, 우연 일치 수준은 질환을 무작위로 뒤바꾼 20회 순열의 평균으로 구했다. HPO 주석은 불완전하므로 이 일치율은 간선 정밀도의 하한이다. 두 평가는 점수와 설정 선택에 쓰지 않았다. 감사 초기 검토에서 확인된 실패 유형(비인간 질환, 검사·시술 개념, 다른 질환을 다루는 문헌)은 그래프 구축 규칙의 수정에 반영했다.

### 3.9 통계와 누수 점검

같은 환자에 대한 두 시스템의 비교에는 환자 단위 대응 부트스트랩(2,000회)으로 Top-k 차이의 95% 신뢰구간을 구하고, 불일치 환자에 대한 정확 McNemar 검정을 적용한 뒤 9개 데이터셋에 Holm 보정을 했다. 데이터셋 전반의 방향 일관성은 차이의 부호에 대한 양측 부호검정으로 평가했다. 원출처 PMID가 없는 출판 증례 데이터셋의 누수는 데이터 기반으로 점검했다. 환자마다, 정답 질환의 근거로 그래프에 적재된 단일 문헌이 환자 소견 개념의 50% 이상(최소 5개)을 포함하면 원출처 증례일 가능성이 있는 환자로 표시했다. 이 기준은 정확도를 보기 전에 정했으며, 표시된 환자를 제외한 정확도를 민감도 분석으로 함께 보고한다.

## 4. 결과

### 4.1 코퍼스와 그래프

수집 대상 51,894개 질환 중 24.8%는 문헌을 얻지 못했다. 질환–문헌 연결 304,428건과 고유 문헌 136,373건이 수집되었고, 누수 차단으로 원출처 PMID 중 검색 결과에 나타난 269편을 제외하였다. 수집 원문은 15.6억 자이며, 추출 언급 812만 개 중 32.5%가 개념에 연결되었다. 최종 그래프는 후보 질환 43,575개(소견이 있는 질환 27,586개), 개념 56,306개, 문헌 88,266건, 소견 간선 1,184,344개, 근거 언급 1,648,509개로 이루어진다(표 4). 근거 언급의 62.3%는 주제 문헌에서 나왔고, 주제 문헌이 하나 이상인 간선은 61.4%였다.

구축 중 무증상 실패 하나를 발견해 수정했다. 기존 그래프를 지우는 일괄 삭제가 메모리 한도로 일부 배치에서 실패했으나 서버 경고로만 남아 적재가 정상 종료된 것처럼 보였다. 그 결과 제외 대상 질환 2,682개와 근거 언급이 없는 이전 간선 62,497개가 남아 있었다. 이 실패는 "모든 간선은 근거 언급을 가진다"는 불변식을 질의로 확인하는 과정에서 드러났다. 삭제 실패와 불변식 위반 시 적재를 중단하도록 수정한 뒤 그래프를 다시 구축했고, 이후 모든 결과는 수정된 그래프에서 얻었다.

Table 4. Corpus and graph statistics (primary IE model Qwen3-4B-Instruct-2507).

| Stage | Item | Value |
|---|---|---|
| Corpus | Diseases queried | 51,894 |
| Corpus | Diseases without retrieved documents | 24.8% |
| Corpus | Disease–document links | 304,428 |
| Corpus | Unique documents | 136,373 |
| Corpus | Documents excluded by leakage control | 269 |
| Corpus | Total text (characters) | 1.56 B |
| Extraction | Extracted mentions | 8.12 M |
| Extraction | Mentions linked to concepts | 32.5% |
| Graph | Candidate diseases (non-human animal diseases removed) | 43,575 |
| Graph | Diseases with findings | 27,586 |
| Graph | Concepts | 56,306 |
| Graph | Documents | 88,266 |
| Graph | Generic documents excluded (linked to ≥20 diseases) | 1,119 |
| Graph | Finding edges | 1,184,344 |
| Graph | Evidence mentions | 1,648,509 |
| Graph | Mentions from topic documents | 62.3% |

### 4.2 IE 모델 시범과 설정 선택

라벨 없는 시범 추출(질환 40개, 317개 창)에서 gemma-4-E4B는 창당 15.3개 개념, 원문 일치율 96.3%, 질환당 고유 개념 31.3개를, Qwen3-4B는 창당 29.8개, 원문 일치율 76.3%, 질환당 고유 개념 43.7개를 추출했다. 처리 시간은 비슷했고(약 220초), 주 모델은 질환당 고유 개념이 많은 Qwen3-4B로 정했다.

DDXPlus validate에서 기본 설정의 lineage Top-1/3/10은 12.6/17.0/26.9%였다. 선택 설정은 α 0.5, μ 2, 감쇠 0(계층 확장 없음), 통합 근거 2, β 0, γ 0이며 Top-1/3은 15.9/21.2%였다. 주제 밖 간선 가중을 0으로 내린 단계에서 Top-1이 15.5%에서 15.9%로, Top-3이 19.6%에서 21.2%로 올랐고, β는 Top-1을 높이지 못해 0으로 남았다. α와 통합 근거 가중치는 탐색 범위의 상한에 놓였다. 다단계 감별 설정은 감점 가중치 0, 후보 40(validate Top-1/3 5.2/8.4%)이 선택되었다. 비교 시스템의 선택 설정은 BM25가 소견별 질의·깊이 1,000·최댓값 집계(validate 10.7/18.1%), MedCPT가 소견별 질의·깊이 1,000·RRF(8.0/11.4%), 순위 결합이 w = 0.25(17.7/25.5%)였다.

### 4.3 열린 집합 진단 정확도와 비교 시스템

표 5는 그래프의 시험 결과를, 표 6은 비교 시스템을 함께 보여 준다. 그래프의 Top-10은 DDXPlus 29.1%, PUMCH_L 39.0%, RAMEDIS 36.2%, PUMCH-ADM 32.0%였고, LIRICAL, HMS, Phenopacket-store, SymCat은 15% 미만이었다. 정답 판정을 strict로 바꾸면 Top-10은 DDXPlus 23.6%, PUMCH_L 34.2%, Phenopacket-store 10.3%로 낮아졌다.

Table 5. Open-set diagnostic accuracy of the literature graph on test datasets (lineage; % with patient-bootstrap 95% CI, B = 1,000). Candidates: 43,575 diseases.

| Benchmark | n | Top-1 | Top-3 | Top-10 |
|---|---|---|---|---|
| DDXPlus test | 3,000 | 14.4 [13.1, 15.6] | 19.6 [18.1, 21.0] | 29.1 [27.4, 30.8] |
| RareBench RAMEDIS | 624 | 3.5 [2.2, 5.1] | 21.8 [18.6, 24.8] | 36.2 [32.5, 39.9] |
| RareBench LIRICAL | 370 | 4.9 [2.7, 7.0] | 8.9 [5.9, 12.2] | 13.5 [10.0, 17.0] |
| RareBench HMS | 88 | 3.4 [0.0, 6.8] | 6.8 [2.3, 12.5] | 12.5 [6.8, 20.5] |
| RareBench MME | 40 | 5.0 [0.0, 12.5] | 15.0 [5.0, 27.5] | 17.5 [7.5, 30.0] |
| PUMCH_L | 988 | 7.0 [5.4, 8.6] | 19.8 [17.4, 22.4] | 39.0 [35.9, 41.9] |
| PUMCH-ADM | 75 | 12.0 [5.3, 20.0] | 22.7 [13.3, 32.0] | 32.0 [21.3, 42.7] |
| Phenopacket-store | 10,310 | 3.6 [3.2, 4.0] | 6.7 [6.2, 7.1] | 11.1 [10.5, 11.7] |
| SymCat | 7,481 | 3.4 [3.0, 3.8] | 5.6 [5.1, 6.1] | 10.7 [10.0, 11.4] |

같은 문헌의 텍스트 검색과 비교하면, BM25는 9개 중 6개 데이터셋에서 그래프보다 Top-10이 높았으나 방향의 일관성은 유의하지 않았다(부호검정 p = 0.51). Holm 보정 후 유의한 차이는 그래프 우세 RAMEDIS(36.2% 대 12.8%)와 BM25 우세 PUMCH_L(49.2% 대 39.0%), Phenopacket-store(15.6% 대 11.1%)였고, DDXPlus는 그래프가 2.6%p 높았다(Holm 보정 p = 0.056). Top-1에서는 그래프가 DDXPlus(14.4% 대 10.3%)와 SymCat에서, BM25가 PUMCH_L(26.3% 대 7.0%)에서 유의하게 높았다. MedCPT는 대부분의 데이터셋에서 두 시스템보다 낮았다.

두 계층을 순위 결합하면 그래프 단독 대비 Top-10이 9개 중 8개에서 올랐고(부호검정 p = 0.039), Holm 보정 후 DDXPlus(+7.2%p), LIRICAL(+6.2%p), PUMCH_L(+5.1%p), Phenopacket-store(+5.0%p), SymCat(+1.1%p)에서 유의했다. Top-1은 7개에서 오르고 하락은 없었다(p = 0.016). BM25 단독과 비교하면 결합은 DDXPlus(+9.8%p), RAMEDIS(+25.3%p), SymCat(+1.4%p)에서 유의하게 높고 PUMCH_L(−5.2%p)에서 유의하게 낮아, 어느 쪽이 일관되게 우세하지는 않았다(5:4).

같은 Cypher 질의에 다른 지식원을 넣으면, LLM 추출 그래프는 MEDLINE MeSH 공출현 그래프보다 9개 데이터셋 모두에서 Top-10이 높았다(예: PUMCH_L 39.0% 대 13.1%, Phenopacket-store 11.1% 대 3.4%). PrimeKG와는 결과가 갈렸다(RAMEDIS, LIRICAL, HMS, Phenopacket-store는 PrimeKG가 높음). HPO 주석은 희귀질환 데이터셋에서 매우 높았으나(Phenopacket-store Top-1/Top-10 36.5/60.5%, MME 45.0/77.5%), 흔한 질환에서는 문헌 그래프보다 낮았다(DDXPlus 16.1% 대 29.1%, SymCat 6.5% 대 10.7%). 희귀질환 데이터셋의 증례와 HPO 주석은 같은 출판 증례에서 큐레이션된 경우가 많아, 이 비교는 HPO 주석에 유리하다.

Table 6. Comparison under the same open-set protocol (lineage Top-1 / Top-10, %, test). Each system's configuration was selected on DDXPlus validate. Graph+BM25 is a post-hoc analysis.

| Dataset | Literature graph (ours) | BM25 | MedCPT | Graph+BM25 | MeSH co-occurrence | PrimeKG | HPO annotations^a^ |
|---|---|---|---|---|---|---|---|
| DDXPlus | 14.4 / 29.1 | 10.3 / 26.5 | 7.7 / 19.3 | 15.7 / 36.3 | 10.3 / 24.6 | 8.4 / 15.5 | 5.0 / 16.1 |
| RAMEDIS | 3.5 / 36.2 | 2.6 / 12.8 | 1.1 / 16.7 | 5.6 / 38.1 | 0.5 / 2.2 | 24.0 / 40.5 | 30.9 / 53.0 |
| LIRICAL | 4.9 / 13.5 | 8.1 / 17.8 | 1.4 / 8.9 | 7.6 / 19.7 | 1.1 / 3.5 | 11.3 / 21.6 | 37.3 / 63.2 |
| HMS | 3.4 / 12.5 | 8.0 / 22.7 | 2.3 / 8.0 | 3.4 / 17.1 | 1.1 / 4.5 | 4.5 / 14.8 | 10.2 / 31.8 |
| MME | 5.0 / 17.5 | 2.5 / 22.5 | 0.0 / 2.5 | 5.0 / 15.0 | 0.0 / 2.5 | 2.5 / 7.5 | 45.0 / 77.5 |
| PUMCH_L | 7.0 / 39.0 | 26.3 / 49.2 | 1.0 / 15.3 | 15.1 / 44.0 | 1.7 / 13.1 | 3.7 / 15.6 | 20.2 / 53.6 |
| PUMCH-ADM | 12.0 / 32.0 | 14.7 / 42.7 | 0.0 / 16.0 | 14.7 / 40.0 | 5.3 / 10.7 | 6.7 / 24.0 | 21.3 / 53.3 |
| Phenopacket-store | 3.6 / 11.1 | 4.2 / 15.6 | 0.7 / 6.6 | 5.9 / 16.1 | 1.0 / 3.4 | 3.7 / 13.3 | 36.5 / 60.5 |
| SymCat | 3.4 / 10.7 | 2.2 / 10.4 | 2.5 / 10.7 | 3.9 / 11.8 | 3.2 / 8.1 | 2.6 / 5.8 | 2.4 / 6.5 |
| *Sentence-level provenance* | *yes* | *chunk* | *chunk* | *yes (graph-ranked)* | *no* | *no* | *no* |

^a^ Curated from published cases that overlap with the rare-disease benchmarks; shown as a reference for the knowledge gap, not as a competing system.

누수 점검에서 원출처 증례로 의심되는 환자는 RAMEDIS, HMS, MME에서 0명, LIRICAL에서 8명(370명 중), Phenopacket-store에서 113명(10,310명 중 1.1%)이었다. 이들을 제외하면 그래프의 Top-10은 LIRICAL 13.5%→11.6%, Phenopacket-store 11.1%→10.3%로 소폭 낮아졌다. Phenopacket-store는 원출처 PMID를 수집에서 제외했음에도 같은 증례가 다른 문헌(재보고, 리뷰의 증례표 등)을 통해 남아 있었다.

### 4.4 닫힌 집합 보조 분석

후보를 각 벤치마크의 정답 라벨 공간으로 한정하면 그래프의 Top-1/Top-10은 DDXPlus(48개 질환) 30.8/81.3%, RAMEDIS(121개) 43.3/58.0%, LIRICAL(341개) 21.9/41.4%, HMS(101개) 18.2/56.8%, MME(25개) 35.0/72.5%, PUMCH_L(400개) 54.4/86.7%, PUMCH-ADM(107개) 64.0/88.0%, Phenopacket-store(884개) 14.4/27.9%, SymCat(760개) 5.2/17.7%였다. 이 조건은 정답 목록을 후보로 쓰므로 주 결과가 아니며, 라벨 공간의 크기가 문헌 보고와 달라 외부 수치와 직접 비교할 수 없다. 표 11에 외부 보고 수치를 조건과 함께 참조로 정리했다.

### 4.5 그래프 기능 ablation

동결 설정에서 그래프 기능을 하나씩 바꾸어 시험 Top-10을 다시 측정했다(표 7, 사후 분석). 출처 기능인 주제 문헌 가중을 끄면(γ = 1) Top-10이 9개 중 7개에서 떨어졌으나 방향 일관성은 유의하지 않았고(p = 0.18), Holm 보정 후 개별 유의 하락은 DDXPlus, Phenopacket-store, SymCat이었다. Top-1에서는 8개에서 떨어졌다(p = 0.039). 커버리지 스무딩을 끄면 7개에서 떨어졌고(p = 0.070), PUMCH_L(−16.9%p)을 포함한 5개에서 개별 유의했다. 점수 형태를 조정하는 통합 근거와 길이 정규화는 방향이 갈렸다. 두 기능은 DDXPlus, RAMEDIS, SymCat에서는 유의하게 기여했지만, 끄면 PUMCH_L과 Phenopacket-store 등 HPO 기반 희귀질환 데이터셋에서 유의하게 올랐다(길이 정규화를 끈 PUMCH_L 39.0%→51.0%). 계층 확장을 켜면 PUMCH_L(+5.4%p)과 Phenopacket-store(+0.9%p)에서 유의하게 올랐다.

Table 7. Effect of changing one graph feature from the frozen configuration (lineage Top-10, %, test). * Holm-adjusted McNemar p < 0.05 versus frozen. Sign test row: datasets where the changed configuration is higher : where frozen is higher.

| Dataset | Frozen | +hierarchy expansion | −unified evidence | −coverage smoothing | −topic weighting (γ = 1) | −length normalization |
|---|---|---|---|---|---|---|
| DDXPlus | 29.1 | 29.0 | 26.5* | 22.4* | 27.4* | 25.5* |
| RAMEDIS | 36.2 | 36.4 | 30.9* | 30.8* | 35.3 | 26.0* |
| LIRICAL | 13.5 | 13.8 | 19.5* | 14.6 | 13.0 | 17.0 |
| HMS | 12.5 | 13.6 | 17.1 | 5.7 | 14.8 | 28.4* |
| MME | 17.5 | 17.5 | 22.5 | 17.5 | 20.0 | 20.0 |
| PUMCH_L | 39.0 | 44.3* | 43.4* | 22.1* | 38.8 | 51.0* |
| PUMCH-ADM | 32.0 | 33.3 | 37.3 | 26.7 | 29.3 | 36.0 |
| Phenopacket-store | 11.1 | 12.1* | 14.4* | 10.0* | 10.2* | 16.2* |
| SymCat | 10.7 | 10.4 | 7.6* | 8.2* | 9.9* | 9.0* |
| *Sign test (direction)* | | *6:2, p = 0.29* | *6:3, p = 0.51* | *1:7, p = 0.070* | *2:7, p = 0.18* | *6:3, p = 0.51* |

### 4.6 다단계 감별

DDXPlus 벤치마크 행동 공간에서는 질문 예산이 늘수록 정확도가 단조 증가해 20회 질문 후 Top-1 6.8%, Top-10 15.8%에 이르렀다(표 8). 개방 행동 공간에서는 20회 후 Top-10이 9.0%였다. Phenopacket-store에서는 Top-10이 5.7%에서 6.3–6.5%로 소폭 올랐다. 개방 행동 공간의 질문 중 환자 기록으로 답할 수 없는 비율은 DDXPlus validate 200명에서 77.8%, Phenopacket-store 200명에서 94.6%였다. 20회 질문 후의 Top-10(15.8%)은 전체 기록 단일 진단(29.1%)의 절반을 조금 넘었다.

Table 8. Multi-turn diagnosis (lineage Top-1 / Top-10, %, 1,000 test patients; budgets 0, 5, 10, 20 questions).

| Dataset, action space | 0 | 5 | 10 | 20 |
|---|---|---|---|---|
| DDXPlus, benchmark action space | 0.9/2.9 | 2.5/8.9 | 3.7/11.3 | 6.8/15.8 |
| DDXPlus, open action space | 0.9/2.9 | 1.5/4.1 | 2.0/6.7 | 2.9/9.0 |
| Phenopacket-store, benchmark action space | 1.3/5.7 | 1.3/6.3 | 1.2/6.2 | 1.6/6.3 |
| Phenopacket-store, open action space | 1.3/5.7 | 1.3/6.3 | 1.2/6.2 | 1.2/6.5 |

### 4.7 추적성과 근거 품질

근거 언급 1,648,509개 중 93.3%는 PMID로, 1.1%는 PMCID로, 5.6%는 StatPearls·GeneReviews 장 ID로 원 문헌이 식별되었다. 근거 경로 질의는 후보 질환, 소견, 근거 문장, 문헌을 하나의 경로로 반환한다. 예컨대 연골무형성증은 소견 "flattened nasal bridge"와 원문 문장을 거쳐 PMID 32029970(Clinical Practice Guidelines for Achondroplasia)으로 이어지며, 이 간선은 주제 문헌 3편의 지지를 받는다.

감사(200개 간선)에서 감사자 판정 기준 supports는 59개(29.5%, Wilson 95% CI 23.6–36.2), not_supported 99개, wrong_concept 42개였다. LLM 판정과의 일치는 supports 여부 기준 89%(Cohen's κ 0.75), 세 범주 기준 70%(κ 0.48)였고, 불일치의 대부분은 약물·검사·방법 같은 비소견 개념의 범주 분류 차이였다. 근거 문헌이 해당 질환을 주제로 다루는 간선의 supports는 38.3%(51/133)로 주제 밖 간선(11.9%, 8/67)의 3배였고, 지지 문헌이 2편 이상인 간선은 52.9%(9/17), 1편인 간선은 27.3%(50/183)였다. 주된 실패는 다른 질환(대개 인접 아형·증후군)을 다루는 문헌에서 온 간선, 비소견 개념으로의 연결, 감별진단으로만 언급된 경우였다.

큐레이션 대조(표 9)에서 HPO 주석이 있는 질환의 비교 가능한 간선 194,318개 중 큐레이터 표현형과 정확히 일치한 비율은 16.4%로 무작위 대응(2.6%)의 6.3배였다. 일치율은 지지 문헌 수에 따라 1편 13.3%에서 5편 이상 63.2%로 단조 증가했고, 주제 문헌이 있는 간선(19.3%)이 주제 밖 간선(11.5%)보다 높았다.

Table 9. Evidence quality by provenance signal: clinical audit (n = 200) and agreement with HPO curated annotations (exact concept match; lower bound on edge correctness).

| Edge group | Audit: supported (%) | HPO agreement (%) | HPO edges |
|---|---|---|---|
| All | 29.5 (59/200) | 16.4 | 194,318 |
| Random disease permutation | – | 2.6 | 194,318 |
| ≥1 topic document | 38.3 (51/133) | 19.3 | 121,862 |
| Off-topic documents only | 11.9 (8/67) | 11.5 | 72,456 |
| Supported by 1 document | 27.3 (50/183) | 13.3 | 168,512 |
| Supported by ≥2 documents | 52.9 (9/17) | 31.7–63.2^a^ | 25,806 |

^a^ 2 documents 31.7%, 3–4 documents 46.3%, ≥5 documents 63.2%.

### 4.8 IE 변경의 영향

IE 모델만 gemma-4-E4B로 바꾼 그래프는 소견 간선 832,671개로 Qwen3 그래프보다 작았으나 HPO 큐레이션 일치율이 18.2%로 높았다(무작위 2.8%). validate에서 따로 고른 설정(α 0.5, μ 4, 감쇠 0.3, 통합 근거 2, β 0.5, γ 0)의 validate Top-1은 13.5%로 Qwen3(15.9%)보다 낮았지만, 시험 Top-10은 9개 중 8개에서 Qwen3보다 높았다(표 10; PUMCH_L 39.0%→51.6%, PUMCH-ADM 32.0%→50.7%). gemma 그래프에서 고른 지지 문헌 수 가중(β 0.5)을 끄면 9개 모두에서 Top-10이 떨어졌다. RAMEDIS의 Top-1 차이(3.5%→15.1%)는 단일 질환의 입도 차이로, 페닐케톤뇨증 환자 148명에게 Qwen3 그래프가 상위 개념인 고페닐알라닌혈증을 1위로 낸 데서 비롯되었다(lineage 규칙은 상위 개념을 인정하지 않음).

Table 10. IE model replacement (lineage Top-1 / Top-10, %, test). Each graph uses its own configuration selected on DDXPlus validate.

| Dataset | Qwen3-4B-Instruct-2507 | gemma-4-E4B |
|---|---|---|
| DDXPlus | 14.4 / 29.1 | 13.0 / 31.9 |
| RAMEDIS | 3.5 / 36.2 | 15.1 / 34.6 |
| LIRICAL | 4.9 / 13.5 | 6.8 / 18.9 |
| HMS | 3.4 / 12.5 | 5.7 / 21.6 |
| MME | 5.0 / 17.5 | 0.0 / 25.0 |
| PUMCH_L | 7.0 / 39.0 | 11.8 / 51.6 |
| PUMCH-ADM | 12.0 / 32.0 | 13.3 / 50.7 |
| Phenopacket-store | 3.6 / 11.1 | 5.8 / 14.6 |
| SymCat | 3.4 / 10.7 | 3.8 / 12.0 |
| *Edges / HPO agreement* | *1,184,344 / 16.4%* | *832,671 / 18.2%* |

Table 11. Published results on the same benchmarks, shown for reference only (conditions differ; compare Top-k only with the same k).

| Benchmark | Method | Metric | % | Conditions |
|---|---|---|---|---|
| DDXPlus | AARLC [ref] | Top-1 | 99.2 | 49-class closed set, supervised RL, interactive evidence |
| DDXPlus | GPT-4o → MEDDxAgent [ref] | Top-1 | 69 → 86 | 49-candidate list given, LLM at inference, 100 cases |
| RareBench (5 sets) | GPT-4 zero-shot [ref] | Top-1/10 | 32.3/58.9 | open generation, LLM at inference |
| PUMCH_L | PhenoBrain ensemble [ref] | Top-1/10 | 31.5/63.0 | 9,260 rare diseases, curated annotations, no LLM |
| PUMCH-ADM | PhenoBrain ensemble [ref] | Top-1/10 | 45.3/69.3 | same |
| RAMEDIS / LIRICAL | PhenoBrain API [ref] | Top-1 | 20.7 / 27.1 | HPO input, curated annotations |
| Phenopacket-store | Exomiser [ref] | Top-1/10 | 35.5/58.5 | HPO only, curated annotations |
| — | Ours, open set | Top-1/10 | Table 5 | 43,575 diseases, literature only, no LLM at inference |
| — | Ours, closed set (auxiliary) | Top-1/10 | Section 4.4 | benchmark label space, literature only |

## 5. 논의

이 연구의 첫째 결과는 출처 추적형 문헌 그래프가 같은 문헌의 텍스트 검색을 대체하지 않는다는 것이다. 정확도만 보면 BM25는 HPO 기반 희귀질환 데이터셋 다수에서 그래프보다 높았고, 그래프는 문진형 흔한 질환(DDXPlus)과 대사질환(RAMEDIS)에서 높았다. 그러나 두 계층을 결합하면 그래프 단독보다 거의 모든 데이터셋에서 정확도가 올랐고, BM25 단독보다도 흔한 질환과 대사질환에서 크게 높았다. 이는 두 계층이 서로 다른 신호를 회수한다는 뜻이다. 텍스트 검색은 소견 표현이 문헌에 그대로 나오는 경우를 넓게 잡고, 그래프는 정규화된 개념과 계층으로 표현이 다른 근거를 잇는다. LLM 없이 MEDLINE 색인에서 얻은 MeSH 공출현 그래프는 같은 질의에서 9개 모두 문헌 추출 그래프보다 낮았으므로, 그래프의 기여는 구조 자체보다 LLM이 문장 수준에서 추출한 질환–소견 관계에서 나온다고 볼 수 있다.

둘째, 그래프가 텍스트 검색과 다르게 제공하는 것은 근거의 단위와 품질 신호다. 텍스트 검색은 문단을 돌려주지만, 그래프는 후보마다 어떤 소견이 어느 문장과 어느 논문에서 왔는지를 경로로 돌려준다. 이 경로는 생성된 설명이 아니라 저장된 결정론적 연결이므로 같은 입력에는 같은 근거가 재현된다. 더 중요한 점은 출처가 근거 품질을 가린다는 것이다. 전체 간선의 지지율은 29.5%에 그쳤지만, 주제 문헌에서 온 간선은 38.3%, 지지 문헌이 여러 편인 간선은 52.9%였고, 독립 큐레이션과의 일치율도 같은 방향(13.3%→63.2%)이었다. 의료진 감사, LLM 판정, 큐레이션 대조의 세 방향 측정이 같은 결론을 가리킨다. 이 신호는 별도 자원 없이 Mention–Doc 경로를 집계하는 질의로 얻어지며, 검색 증강 진단에서 생성기에 넘길 근거를 고르거나 근거의 신뢰도를 함께 표시하는 데 쓸 수 있다.

셋째, 흔한 질환 데이터로 고른 설정의 전이는 기능의 성격에 따라 달랐다. 점수 형태를 조정하는 통합 근거와 길이 정규화는 흔한 질환에서 유의하게 기여했지만 HPO 기반 희귀질환에서는 유의하게 해로웠고, 이 양상은 두 IE 모델에서 반복되었다. HPO 기반 데이터셋의 환자 기록은 소견이 수십 개씩이고 정답 질환의 문헌 프로파일도 넓은 편이어서, 프로파일 크기에 벌점을 주는 정규화가 정답의 순위를 끌어내렸을 가능성이 있다. 페닐케톤뇨증 대신 프로파일이 짧은 상위 개념이 1위에 오른 RAMEDIS 사례도 같은 방향이다. BM25가 선택한 점수 형태(소견별 최댓값의 합)에 길이 정규화가 없다는 점, 그리고 그래프에서 길이 정규화를 끄면 PUMCH_L에서 BM25와 비슷해진다는 점(51.0% 대 49.2%)도 이 해석과 맞는다. 반면 출처 기반 기능인 주제 문헌 가중은 방향이 대체로 일관되었으나(Top-1 8/9, p = 0.039), Top-10의 개별 유의는 3개 데이터셋에 그쳤다. 공식 validation 분할이 흔한 질환에만 있는 조건에서는 데이터셋과 무관하게 정의되는 출처 신호에 기반한 기능이 점수 형태 기능보다 안전한 선택이다.

넷째, 희귀질환에서 문헌 그래프와 큐레이션 지식의 격차는 크다. 같은 질의에 HPO 주석을 넣으면 Phenopacket-store Top-1이 36.5%로, 보고된 Exomiser 수치(35.5%)와 비슷했다. 따라서 희귀질환 정확도의 병목은 질의 방식이 아니라 지식에 있다. 다만 희귀질환 증례와 HPO 주석은 같은 출판 증례에서 나온 경우가 많아 이 격차는 과대평가되었을 수 있다. 흔한 질환에서는 반대로 문헌 그래프가 큐레이션보다 높았으므로, 문헌 그래프는 큐레이션이 덮지 않는 범위를 보완하는 자원으로 볼 수 있다.

다섯째, IE 모델 교체는 어떤 점수 기능보다 큰 차이를 냈고, DDXPlus validate는 IE 모델 선택에 반대 방향의 신호를 주었다. 반면 벤치마크를 쓰지 않는 IE 품질 신호(원문 일치율, 큐레이션 일치율)는 시험 결과와 같은 방향이었다. 그래프 전체를 바꾸는 선택에서는 단일 벤치마크 정확도보다 출처 수준의 품질 지표가 전이를 더 잘 예측할 수 있다. 주 모델은 시험 전에 정한 대로 Qwen3를 유지했다.

구축 과정에서는 오류 없이 결과만 축소·오염되는 실패가 여러 번 확인되었다. 짧은 질환명의 질의 누락, DDXPlus 기본값 응답의 오사상, ICD 경로의 정답 오사상, 의미유형 표 누락은 validate 분석에서, 그래프 초기화의 부분 실패는 출처 불변식 검사에서 발견했다. 열린 집합 평가에서는 이런 실패가 최종 정확도에만 흔적을 남기므로, 그래프 수준의 불변식을 적재 단계에서 검사하는 것이 중요하다. 그래프 DB에서는 이 불변식을 짧은 질의로 표현해 매 적재마다 검사할 수 있다.

다단계 감별은 DDXPlus에서 질문을 거치며 정확도를 높였으나, 개방 행동 공간의 질문 대부분은 기록으로 답할 수 없었다. 그래프 이웃에서 질문 후보를 고르는 방식은 가능하지만, 순위 기반 신념은 수만 개 후보의 차이를 거칠게만 표현하므로 질문 선택과 신념 갱신의 확률적 정교화가 다음 과제다. 그래프가 텍스트 검색과 결합할 때 정확도가 오른다는 결과는, 이 지식 계층을 생성기와 결합하는 검색 증강 진단에서 근거 단위 출처와 품질 신호를 함께 제공하는 방향을 시사한다. 그 종단 간 효용은 이 연구의 범위 밖이다.

### 5.1 한계

첫째, 설정은 DDXPlus validate에서만 선택했으므로 희귀질환에 최적화되지 않았고, 선택값 일부는 탐색 범위의 상한에 놓였다. 둘째, 시험 분할은 무효 처리된 두 차례 평가에서 이미 열람되었고, 순위 결합은 텍스트 검색 결과를 본 뒤 추가한 사후 분석이다. 셋째, 감사자는 한 연구팀과 한 병원의 의료진으로 의료진 전체를 대표하지 않으며, 표본 200개로는 신뢰구간이 넓고 지지 문헌 2편 이상 간선은 17개뿐이다. 넷째, 근거 지지율은 29.5%로 낮으며, 다른 질환을 다루는 문헌과 비소견 개념으로의 연결이 주된 실패였다. 주제 문헌 가중은 점수에서 이를 완화하지만 간선을 없애지는 않는다. 다섯째, 큐레이션 대조와 HPO 주석 비교는 희귀질환에 치우쳐 있고 증례와의 순환성이 있다. 여섯째, 정규화 연결률이 32.5%에 그쳤고 수집 대상 질환의 24.8%는 오픈액세스 문헌을 확보하지 못했다. 일곱째, 원출처 PMID 제외 후에도 Phenopacket-store 증례의 1.1%가 다른 문헌을 통해 남아 있었다(민감도 분석으로 보고). 여덟째, HMS, MME, PUMCH-ADM은 표본이 작고, SymCat은 합성 환자이며, RAMEDIS 결과는 단일 질환(페닐케톤뇨증)에 크게 좌우된다. 마지막으로, 생성기를 결합한 종단 간 효용은 평가하지 않았다.

## 6. 결론

이 연구는 저사양 LLM으로 공개 문헌에서 질환–소견 관계를 추출해 모든 간선이 원문 문장과 논문에 연결된 그래프를 구축하고, 이를 추론 시점 LLM 없이 열린 집합 감별진단의 지식 계층으로 평가하였다. 같은 문헌의 텍스트 검색과 비교하면 그래프는 흔한 질환과 대사질환에서, 텍스트 검색은 HPO 기반 희귀질환에서 높았으며, 두 계층의 결합은 그래프 단독보다 9개 중 8개 데이터셋에서 정확도를 높였다. LLM 추출 그래프는 MEDLINE MeSH 공출현 그래프보다 9개 모두에서 높았다. 간선을 뒷받침하는 문헌의 주제성과 수는 의료진 감사와 독립 큐레이션 모두에서 근거 지지 여부를 가르는 신호였다. 따라서 출처 추적형 문헌 그래프는 텍스트 검색을 대체하는 검색기가 아니라, 근거를 소견 단위로 제시하고 그 품질을 가려내는 보완 계층으로서 가치가 있다. 후속 연구를 위해 코드, 동결 설정, 감사 시트, 측정 기록을 공개한다.
