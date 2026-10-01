# Method

## 1. 개요와 문제 정의

본 연구는 표준-정규화된 증상 속성을 부착한 단일 지식 그래프(knowledge graph, KG)로 이종(異種) 감별진단 벤치마크를 엄격한 zero-shot(strict zero-shot) 조건에서 진단한다. 여기서 "엄격한 zero-shot"은 다음을 의미한다. 첫째, KG는 질환 *이름*만으로 구축하며 어떤 벤치마크의 학습 레이블이나 disease–symptom 매핑 데이터도 사용하지 않는다. 둘째, 대규모 언어모델(LLM)은 KG 구축(정보추출) 단계에서만 사용하며 이는 교과서로부터 지식을 추출하는 것과 동일한 위상을 갖는다[1]. 진단 시점에는 KG와 단일 알고리즘만 동작하고 추론시 LLM을 호출하지 않는다. 셋째, 정보추출 프롬프트에는 벤치마크의 질문이나 증상 목록을 포함하지 않으며 few-shot 예시나 인위적 finding 개수 제한을 두지 않는다. 넷째, 모든 벤치마크에 동일한 알고리즘과 동일한 초매개변수를 적용하며, 벤치마크마다 달라지는 것은 원자료를 공통 표현으로 변환하는 전처리뿐이고 스코어링 알고리즘 자체는 고정된다.

표기를 정의한다. KG를 이분(bipartite) 그래프 $G=(D\cup P,\,E)$로 두며, $D$는 질환 개념(UMLS CUI) 집합, $P$는 표현형(phenotype) 개념 집합이다. 간선 $(d,p)\in E$는 역문서빈도 가중치 $w(p)=\log\!\big(|D|/\mathrm{df}(p)\big)$를 가지며[7] $\mathrm{df}(p)$는 $p$를 갖는 질환의 수이고, 각 간선은 선택적으로 정규화된 수식자(qualified attributes) $a(d,p)$를 가질 수 있다(§2.4). 환자 질의 $q$는 근거 개념 집합 $E_q$로 표현되며 §3.2의 통합 근거 원칙에 따라 표현형과 병력(disease-history) 개념을 구분하지 않는다. 일부 벤치마크는 finding별 속성값을 추가로 제공한다(§4).

## 2. 지식 그래프 구축

### 2.1 질환 시드와 근거 텍스트

희귀질환을 포함한 대규모 스코프를 확보하기 위해 UMLS의 Disorders 의미군(semantic group)에 속하는 질환 개념 35,193개를 시드로 취한다[5]. 각 시드 질환에 대해 자유텍스트 근거를 조립하며, 구체적으로 PubMed 초록(질환명 및 MRCONSO 영문 동의어로 질의)과 임상 요약 문서(StatPearls, Wikipedia, MedlinePlus의 "Signs and symptoms" 절)를 증상 단서 점수로 순위화하여 단어 예산 내에서 취한다. KG 구축에 앞서 근거 커버리지를 실측하며, 재수집 후 KG에 사용 가능한 근거의 커버리지는 90.7%(희귀 하위집합 92.3%)였고 잔여분은 전용 문헌이 존재하지 않는 행정용 ICD 코드가 대부분이어서 한계로 명시한다. 한편 OMIM/ORPHANET을 HPO로 사상한 구조화 자원(예: `phenotype.hpoa`)은 disease–symptom 매핑에 해당하므로 KG 구축에 사용하지 않으며 커버리지 참고 용도로만 활용한다.

### 2.2 Zero-shot 속성 정보추출

단일 GPU에서 구동 가능한 로컬 소형 언어모델(Gemma E4B 계열 instruction-tuned 체크포인트, 유효 매개변수 약 4B; temperature=0의 greedy decoding)로 각 질환의 근거 텍스트에서 finding과 그 속성을 단일 패스로 추출한다. 입력은 질환명과 근거 텍스트뿐이며, 프롬프트는 (i) 해부학적 부위를 명칭에서 분리하여 location으로 보내는 원자화 규칙, (ii) 근거가 명시한 속성만 채우도록 하는 제약, (iii) 개수 제한 없는 소진적(exhaustive) 추출 지시를 포함한다. 추출 스키마는 증상 특성화의 교과서 표준인 Bates의 seven attributes[1]에 앵커하며, 속성을 별도 노드로 승격하지 않고 표현형에 부착된 수식자로 두는 방식은 GA4GH Phenopacket 스키마[2]와 정합한다. 후보 속성은 Table 1과 같다. 방사(radiation)는 독립 축이 아니라 location의 방향 보존 관계로 표현하고, duration은 timing 하위의 자유텍스트로 두되 스코어링에는 사용하지 않는다.

> **[흐름 — 2-pass 추출로의 변경. 수율 표는 결과 확정 후 작성]**
>
> - **문제**: 하나의 프롬프트가 소진적 finding 목록과 속성 충실도를 동시에 최적화하지 못한다. 속성 규칙을 강화하면 속성 수율은 오르지만 finding 수가 감소한다(측정: −12%, 동시에 timing 수율 반감). 주의(attention)가 두 목표 사이에서 경합한다.
> - **방법**: 두 단계로 분리한다. Pass 1은 검증된 레시피 그대로 소진적 finding 목록만 추출한다. Pass 2는 **동일 근거 텍스트**와 pass 1의 finding 목록을 함께 주고 각 finding의 속성만 채운다. Pass 2는 finding을 추가·삭제할 수 없으므로 코퍼스는 불변이고 속성 셀만 증가한다.
> - **정의의 경계 명시**: 속성 추출의 정밀도는 정의의 모호성에서 무너진다. (i) 악화요인은 *요인*이어야 하며 경과 서술이 아니다("progressively worsening dyspnea"는 악화요인이 아니고 "dyspnea on exertion"은 악화요인이다). (ii) 완화요인은 증상 삽화를 완화·중단시키는 것이며 질환의 근치 치료가 아니다(항생제·수술·"치료 시작"은 제외). 이 경계를 명시하기 전 완화요인 정밀도는 19%였고 명시 후 71%였다.
> - **측정 방법에 대한 주의**: 키워드 감사는 속성 존재를 신뢰성 있게 재지 못한다. 느슨한 패턴은 경과어를 세어 과대추정하고, 명시적 트리거 동사를 요구하는 엄격한 패턴은 임상 산문의 실제 어법을 놓쳐 과소추정한다. 따라서 수율은 **정성 감사**로 검증한다.

**Table 1.** Candidate symptom attributes, value types, and normalization targets.

| Attribute | Bates axis | Value type | Normalization target |
|---|---|---|---|
| Location | Location | concept (multi-valued) | UMLS CUI; SNOMED CT Body structure |
| Character | Quality | free text | none (no controlled standard) |
| Severity | Quantity | ordinal | {mild, moderate, severe} (HPO Severity, HP:0012824) |
| Timing | Timing | ordinal + text | onset {sudden, gradual} (HPO Temporal pattern, HP:0011008); duration free text |
| Aggravating / Relieving | Aggravating / relieving | concept (multi-valued) | UMLS CUI; HPO *Aggravated/Ameliorated by* (HP:0025285/HP:0025254) |

*Notes.* 속성은 원문이 명시할 때만 부여되므로 대체로 sparse하다. 개념형 값은 UMLS로 링크하고[5] 해부학적 부위는 SNOMED CT Body structure 코드를 추가로 부여한다[6]. Table 1은 후보이며, 스코어링에 실제로 사용하는 속성은 평가 단계에서 결정한다(§4).

### 2.1a 근거 텍스트의 *종류* 선택 (source-type selection by attribute yield)

> **[흐름 — 수율 표와 최종 코퍼스 구성은 결과 확정 후 작성]**
>
> - **관찰**: 근거의 *양*이 아니라 *종류*가 어떤 속성을 얻을 수 있는지를 결정한다. 질환 중심 텍스트(StatPearls·리뷰·초록)는 어떤 finding이 나타나는지를 적지만 그것이 *어떻게* 나타나는지(악화/완화 요인, 시간 양상)는 임상의에게 자명하다는 이유로 생략한다.
> - **가설 검정 설계**: 동일 IE 레시피를 서로 다른 source type에 적용하고 속성별 수율(추출 finding 중 해당 속성을 가진 비율)을 나란히 비교한다. 비교 대상 = 질환 중심 텍스트 / 증상 조직형 교과서(semiology) / 개별 환자 서사(PMC 오픈액세스 증례보고).
> - **선택 규칙**: 편의가 아니라 **측정된 속성 수율**로 코퍼스를 채택한다(§1의 사전 커버리지 검증을 속성 축으로 확장).
> - **한계 명시**: 증례보고는 비전형·중증 사례로 치우치는 **보고 편향**이 있으므로 교과서(전형)와 조합하며, 단독 사용하지 않는다.
> - **제약 준수**: 질의는 질환명만 사용하고(§5 benchmark-blind), 자유텍스트만 사용한다(§3).

### 2.3 결정론적 정규화

개념형 속성(location, character, name)은 문턱값 없는(threshold-free) 방식으로 UMLS에 링크한다. SPECIALIST Lexicon의 정규화 문자열(MRXNS_ENG)에 대한 정확 일치와 결정론적 형태소 축소를 결합하고(SPECIALIST Lexical Tools[22]; MetaMap/QuickUMLS 계열[8]), 임상적으로 유효한 의미유형으로 중의성을 해소한 뒤 SNOMED CT로 사상한다. 범주형 속성은 통제어휘로 사상하는데, severity는 수치 등급(NRS 0–10)을 0–3/4–6/7–10 규칙으로 {mild, moderate, severe}에, onset은 {sudden, gradual}에 정규화하며 이는 각각 HPO의 Severity(HP:0012824)와 Temporal pattern(HP:0011008) 서수 term에 대응한다[4]. 임상텍스트 속성 정규화의 선행 연구도 통제 대상(body location·severity·course)이 있는 슬롯만 표준 개념에 정규화한 바 있다[3]. 본 연구도 이 관행을 따라 통제 대상 어휘가 존재하는 속성만 정규화하며, character는 그러한 통제 표준이 없어 자유텍스트로 보존한다. 이 결정은 character를 스코어링에서 배제한다는 뜻이 아니라(§4.3) 표준 정규화를 강제하지 않는다는 뜻이다.

### 2.4 그래프 조립

질환에서 표현형으로 향하는 단방향 `HAS_PHENOTYPE` 이분 KG를 구성하며, 통합 근거 원칙(§3.2)에 따라 질환 개념도 다른 질환의 표현형 대상으로 등장할 수 있다. 속성은 노드 승격 없이 간선의 수식 진술로 부착하며 이는 Phenopacket 및 HPO Clinical modifier 모델과 일치한다[2][4]. 인구통계·시간·수식어에 해당하는 잡음 의미유형은 정지-의미유형 필터로 제거하고 허브 표현형은 IDF에 의해 자연히 감쇠시킨다. 기본 KG는 질환 35,193개, 표현형 18,866개, 간선 약 2.3×10⁵개로 구성되며 간선의 27%가 location 수식자를 갖는다. 정보추출 프롬프트를 소진성·속성-완전성 방향으로 강화한 변형 KG는 간선 약 3.1×10⁵개, location 주석 38%로 속성이 더 조밀하다. 본 연구는 진단 성능(§6.2)을 기본 KG에서, 속성의 인과적 검증(§6.1)을 두 판본 모두에서 보고한다. 밀도 증가는 attribute-free 진단에는 이득이 없거나 미세한 손해였고 속성 채널의 검정 가능성을 높였으므로, 두 판본을 용도별로 구분하여 사용하되 §6.1에서 두 밀도의 결과를 함께 제시하여 선택적 보고가 아님을 보인다.

## 3. Zero-shot 진단 알고리즘

본 절의 모든 구성요소는 UMLS 및 어휘 구조와 KG만을 사용하므로 §1의 제약을 위반하지 않으며 단일 초매개변수로 전 벤치마크에 적용된다.

### 3.1 질환 계보 해석

우리 KG는 희귀질환 포함 시드로 구축되어 질환을 구체적 하위유형 개념으로 인덱싱하는 반면, 벤치마크 질환은 상위(umbrella) 명칭인 경우가 많아 이를 그대로 정규화하면 표현형 프로필이 비었거나 잘못된 개념에 링크되어 진단이 구조적으로 실패한다. 이를 교정하기 위해 후보 질환 $c=(\mathrm{name},\mathrm{cui})$를 KG 계보 군집으로 해석한다.

$$\kappa(c)=\Big(\{\mathrm{cui}\}\;\cup\;\mathrm{Lex}(\mathrm{name})\;\cup\;\mathrm{Desc}_{\mathrm{MRREL}}(\mathrm{cui})\Big)\cap D_{\mathrm{prof}}.$$

여기서 $\mathrm{Lex}(\mathrm{name})$은 KG 질환명이 name의 내용어 토큰을 진부분집합으로 포함하는(즉 더 구체적인) 질환 집합으로 어휘적 특수화는 용어체계에서 대체로 is-a 관계에 해당하며, $\mathrm{Desc}_{\mathrm{MRREL}}$은 UMLS MRREL의 CHD/RN 관계를 통한 하위개념이고, $D_{\mathrm{prof}}$는 표현형 프로필을 가진 질환으로의 제한이다. 후보의 병합 프로필은 $\Phi(c)=\{p\mapsto\max_{d\in\kappa(c)}w(p):(d,p)\in E\}$로 둔다. UMLS 계층은 벤치마크 자료가 아니므로 이 해석은 zero-shot 제약과 양립한다.

### 3.2 통합 근거 스코어링

환자 근거는 표현형과 병력 개념을 구분하지 않는다. 예컨대 대사질환 환자의 병리진단적(pathognomonic) 소견인 "hyperphenylalaninemia"는 우리 KG에서 페닐케톤뇨증(PKU)의 상위 질환 노드이다. 따라서 각 근거 $e$가 후보 $c$에 기여하는 정도를 표현형 정합과 질환-계층 정합의 최댓값으로 정의한다.

$$\mu(e,c)=\max\big(m_{\mathrm{ph}}(e,c),\;m_{\mathrm{dz}}(e,c)\big),\qquad m_{\mathrm{dz}}(e,c)=\mathrm{IC}_D(e)\cdot\mathbb{1}\!\left[e\in\mathrm{Rel}_D(\kappa(c))\right],$$

여기서 $\mathrm{Rel}_D$는 질환 계층에서의 조상·자손 관계, $\mathrm{IC}_D(e)=\log\!\big(|D|/(1+|\mathrm{desc}(e)\cap D|)\big)$는 근거 개념의 정보량으로 구체적(희귀) 상위개념일수록 커서 변별적이다[21].

### 3.3 표현형 정합: 정보량, 계층, 임베딩 정렬

표현형 정합 $m_{\mathrm{ph}}$는 IDF 가중 중첩을 기반으로 하되 어휘 격차를 두 방식으로 보정한다. 첫째, UMLS 표현형 계층을 따라 근거와 프로필을 조상으로 감쇠 확장하여 개념 세분성(granularity) 차이를 잇는다. 둘째, 프로필에 정확 일치가 없는 근거를 개념-특화 생의학 인코더의 코사인 유사도로 가장 가까운 KG 표현형에 연성 정합(soft match, $\mathrm{sim}\ge\tau$)한다. 인코더는 동일 프로토콜로 비교한 다섯 후보(SapBERT[10], CODER, BioLORD[11], PubMedBERT, 일반도메인 MiniLM) 중 종합 성능이 가장 우수한 BioLORD-2023-C를 사용하며, 이 비교에서 개념-특화 인코더가 일반도메인 및 비개념 생의학 인코더를 상회한 사실은 인코더 선택의 근거가 된다(§5.3).

### 3.4 채널별 분리 정규화

질환-계층 사전확률 채널과 표현형 유사도 채널은 기수(cardinality)가 다르므로 각각을 자신의 기수로 정규화한다. 두 채널을 합쳐 프로필 크기로 정규화하면 공유 상위개념 사전확률에 의해 프로필이 얕은 근연(近緣) 질환이 정답을 이겨 Top-1 정확도가 저하되기 때문이다. 따라서

$$S_{\mathrm{base}}(c,q)=\frac{\sum_e m_{\mathrm{dz}}(e,c)}{|\mathrm{dsig}(c)|^{\beta}}+\frac{\sum_e m_{\mathrm{ph}}(e,c)}{|\Phi(c)|^{\beta}},\qquad \beta=0.5,$$

로 두며 $\mathrm{dsig}(c)$는 후보의 질환-계층 관계 집합이다.

### 3.4a 문서화 깊이 보정 (coverage-aware length smoothing)

> **[흐름 — 세부·수식·표는 결과 확정 후 작성]**
>
> - **문제**: 위 정규화는 $|\Phi(c)|$가 작다는 것을 "이 질환은 표현형이 좁다"로 읽지만, 본 KG에서는 대개 "이 질환에 대해 읽은 텍스트가 적다"를 뜻한다(중앙값 차수 7). 즉 **간선의 부재가 음성 증거로 취급**되고, 얕게 문서화된 질환이 자신의 결측 때문에 이득을 본다.
> - **근거(측정)**: (i) 질환의 2.8%만 근거를 보강했는데 보강되지 않은 환자들의 @3까지 이동 → 문서화가 df/IDF를 통해 전역으로 누출됨. (ii) 벤치마크별 정규화 지수의 최적값이 상충(공통질환은 낮은 값, 희귀질환은 높은 값) → 단일 상수로 해소 불가.
> - **방법**: 질환별 문서화 깊이를 KG에 1급 속성으로 기록하고(근거 문자수·추출 finding 수), 커버리지 $\gamma_c\in[0,1)$를 포화함수로 정의해 분모에 $\mu(1-\gamma_c)\overline{|\Phi|}$를 더한다. 정보검색의 Dirichlet 평활과 동형이며 문턱값을 도입하지 않는다.
> - **위치 설정**: 새 스코어링 기법이 아니라 **교란 통제**로 서술한다. 문서화 깊이를 보정하지 않으면 속성 ablation(§4)의 이득이 문서화 차이로 설명될 수 있기 때문이다.
> - **정직성**: $\mu$를 테스트 점수로 고르지 않는다. 검증셋 또는 원리적 근거로 고정하고 민감도를 함께 보고한다.

### 3.5 잠재의미 융합

희소한 프로필이 명시적 간선으로 담지 못한 연관을 저차원 표현으로 보완한다. 질환×표현형 행렬 $M\in\mathbb{R}^{|D|\times|P|},\,M_{dp}=w(p)$를 계수 $k=256$의 절단 특이값분해로 분해하여 질환·표현형의 잠재의미(latent semantic) 임베딩을 얻고[12], 질의 잠재벡터를 $q_{\mathrm{LSA}}=\widehat{\sum_{e\in E_q\cap P}w(e)V_e}$, 후보 잠재벡터를 $U_c=\widehat{\mathrm{mean}_{d\in\kappa(c)}U_d}$로 두어($\hat{\cdot}$은 L2 정규화) $s_{\mathrm{LSA}}(c)=\cos(U_c,q_{\mathrm{LSA}})$를 계산한다. $S_{\mathrm{base}}$와 $s_{\mathrm{LSA}}$를 질의별 최소-최대 정규화한 뒤 후기 융합한다.

$$S(c,q)=(1-\lambda)\,\widetilde S_{\mathrm{base}}(c,q)+\lambda\,\widetilde s_{\mathrm{LSA}}(c,q),\qquad \lambda=0.4.$$

가중치 $\lambda$는 격자탐색에서 0.4일 때 흔한 질환에서 최대 이득을 주면서 희귀질환 성능을 보존했으며(0.5 이상에서 희귀질환이 저하) 단일 전역값으로 고정한다.

### 3.6 판별적 속성 채널과 후보 동일성 병합

벤치마크가 finding별 속성을 제공하는 경우(§4) 정합된 finding $f$의 속성 $A_f$와 간선 속성 $A_e=a(d,p)$를 비교하여 일치에 보상을, 통제어휘의 모순에 벌점을 부여한다.

$$\delta(f,c)=\!\!\sum_{t\in\{\mathrm{loc},\mathrm{sev},\mathrm{ons}\}}\!\!\big[\lambda_a\,\mathbb{1}[\mathrm{agree}_t]-\mu_a\,\mathbb{1}[\mathrm{contradict}_t]\big]\,w(p)\;+\;\lambda_a\,\mathbb{1}[\mathrm{char\ overlap}]\,w(p).$$

severity와 onset은 통제어휘이므로 모순 벌점이 명확하고, location은 개념 및 계층 일치로 판정하며, character는 자유텍스트이므로 표면형 불일치를 모순으로 간주하지 않고 일치 보상만 부여한다($\lambda_a=\mu_a=1$로 고정). 이 항은 표현형 채널에 합산된다. 또한 한 벤치마크가 동일 질환을 복수 코드로 제공하는 경우(예: RareBench의 OMIM·ORPHA·CCRD) 그 벤치마크가 선언한 동일성만 병합하여 후보를 군집으로 순위화하며, UMLS 계층으로 후보를 병합하지는 않는데 이는 임상적으로 구별되어야 할 질환을 뭉쳐 성능을 인위적으로 부풀리기 때문이다.

## 4. 속성의 외재적 검증

정규화된 속성이 zero-shot KG 진단을 실제로 개선하는지 통제 실험으로 검정한다. 검증 무대는 NLICE[13]로, SymCat 통계에 기반한 합성 환자에 증상별 속성(Nature/Location/Intensity/Chronology/Excitation)을 부여한 데이터이며 우리가 사용하는 공개 벤치마크 중 질의 측면에서 분리 가능한 속성을 제공하는 유일한 자료이다(§5.1).

### 4.1 설계

후보 집합, 환자, 기저 스코어러를 고정한 채 속성 채널만 켜고 끄는 개체내(within-subjects) 설계를 취한다. 외부 NLICE 연구는 이진(SymCat) 표현과 속성(NLICE) 표현을 서로 다른 질환 수(각각 801개와 55개)에서 평가하여 표현 효과와 레이블 공간 축소 효과가 교락(confounding)되어 있으나, 본 설계는 후보 집합을 동일하게 유지하므로 이 교락이 없는 더 통제된 표현 ablation이다.

### 4.1a 속성별 검정 가능성과 벤치마크 상보성

단일 벤치마크로는 속성 스키마 전체를 검정할 수 없다. 채널이 기여하려면 **환자 기록과 KG 간선이 같은 소견에 대해 같은 속성을 모두 진술**해야 하는데, 이 "기회(opportunity)" 수를 두 벤치마크에서 실측하면 상보적으로 분포한다(정답 질환의 소견 기준, 각각 28,554건과 28,527건):

| 속성 | NLICE | DDXPlus |
|---|---|---|
| location | 6,792 | 3,503 |
| character | 2,514 | 998 |
| severity | **0** | 801 |
| onset | 198 | 645 |
| timing | 387 | **0** |
| aggravating | 1,360 | **0** |

NLICE는 severity·relieving을 질의 측에서 비우고, DDXPlus는 timing·aggravating을 아예 묻지 않는다. 따라서 어느 한 벤치마크에서 관측된 $\Delta=0$은 **채널에 신호가 없다는 증거가 아니라 그 벤치마크가 그 속성을 묻지 않는다는 사실**이며, 두 무대를 합쳐야 스키마가 커버된다. 본 연구가 속성을 cross-benchmark interlingua로 규정하는 근거가 여기에 있으며, 각 속성은 자신이 검정 가능한 무대에서 독립적으로 평가한다.

### 4.1b 채널 결합과 선택도

한 소견이 여러 속성에서 동시에 일치할 때 채널별 이득을 **합산하면** 그 소견 하나가 채널 수에 비례한 가중을 받아, 따로 정합된 여러 소견을 압도한다. 실측에서 이 결합은 다속성 모델이 최선의 단일 채널보다 나쁜 역전을 낳았다(DDXPlus 4속성 46.33 대 onset 단독 49.17). 따라서 소견당 속성 이득은 **평균**으로 결합하여 "이 소견의 수식어가 일치한다"는 진술이 채널 수와 무관하게 한 번만 계상되도록 한다.

또한 속성값의 **선택도**가 채널마다 다르다. NLICE에서 일치가 보상하는 후보 수는 character 3.0개(정답 포함 65.4%), location 2.0개(42.4%)인 반면 aggravating은 6.3개(35.1%)로, 후자의 통제 어휘(`movement`, `exertion`)가 후보 집합의 6분의 1을 한꺼번에 포괄한다. 값별 역문서빈도로 이득을 조정하는 방안을 검정하였으나, aggravating 어휘는 값들이 **균일하게** 광범위하여 일치 전체가 같은 상수로 축소될 뿐 후보 간 순위가 바뀌지 않았다($\Delta$ 불변). 이 채널의 음의 기여는 §6에 음성 결과로 보고한다.

### 4.2 통계적 검정

각 속성 채널의 leave-one-attribute-in 이득 $\Delta$GTPA@1을 측정한 뒤, 해당 속성값을 KG의 모든 간선에 걸쳐 무작위 치환하여(주변분포 보존, 질환–속성 연관 파괴) 귀무 분포를 생성한다. 이는 치환 특징 중요도(permutation feature importance)에 해당한다[9]. $K=200$회 반복으로 얻은 귀무 $\Delta$ 분포에 대해 경험적 $p=P(\text{null}\ge\text{obs})$와 $z=(\text{obs}-\overline{\text{null}})/\mathrm{sd}$를 계산하고 환자 부트스트랩($B=2000$)으로 이득의 95% 신뢰구간을 구한다. 채널은 $p<0.05$이면서 신뢰구간이 0을 배제할 때에만 인정한다.

속성 채널이 활성화되는 판본은 속성을 조밀하게 추출한 KG(간선 location 주석 38%)이며, 평가는 NLICE에서 우리 KG로 해석되는 폐집합 29개 질환의 6,450명 환자에 대해 수행하였다. 정규화(location)와 비정규화(character), 희소(onset) 속성을 대비하기 위해 세 채널을 각각 leave-one-attribute-in으로 평가하였고, 각 채널의 이득에 대한 귀무 분포와 신뢰구간을 위 절차로 산출하였다. 결과는 §6에 보고한다.

## 5. 평가 프로토콜

### 5.1 벤치마크와 전처리

원자료의 형태가 벤치마크마다 다르므로 공통 표현(근거 개념과 finding별 정규화 속성)으로의 전처리만 벤치마크별로 상이하고 스코어링 알고리즘은 동일하다. DDXPlus[20](49-way, 흔한 질환)는 근거 코드를 기저 개념으로 사상하고 병력(antecedent)을 질환 개념으로 통합 근거에 투입한다. 다만 속성(carac/endroit/intens/soudain)은 통증·발진에만 부착되어 커버리지가 6–12%에 그치고 환자가 다수 부위를 동시에 보고하여 변별력이 낮으므로(onset만 순변별 +7.8%) 이 벤치마크에서는 속성 채널을 비활성화하고 한계로 보고한다. RareBench[19](RAMEDIS/LIRICAL/MME/HMS)는 HPO를 개념으로 사상하는데, HPO term이 수식어와 개념을 단일 복합 개념으로 합치므로 기저 개념에 수식자를 분리 부착하는 우리 간선과 정합되지 않아 속성 채널을 비활성화한다. SymCat은 속성이 증상 명칭에 어휘화되어 분리 불가능하며 증상 빈도는 disease–symptom 확률 표에 해당하므로 사용하지 않는다. NLICE는 §4의 검증 무대로 10-필드 속성을 우리 스키마로 사상한다.

### 5.2 개념 정규화와 커버리지

각 벤치마크의 환자 근거는 이종 인코딩을 UMLS CUI라는 단일 어휘로 사상하여 KG와 매칭하며, 이 사상은 독자 스키마가 아니라 표준 자원 또는 벤치마크가 제공하는 파일에 기반한다. RareBench는 표현형을 HPO 코드로, 질환을 OMIM/ORPHANET 코드로 인코딩하므로 HPO·OMIM이 UMLS의 source vocabulary인 점을 이용해 결정론적 크로스워크로 CUI를 얻는다[5]. DDXPlus는 공개된 근거 스키마를 CUI로 사상한 파일(기저 근거 204/223)과 ICD-10을 경유한 질환 사상(49/49)을 함께 제공한다[20]. SymCat은 증상을 SNOMED CT(역시 UMLS source vocabulary)에 연결하므로 이를 CUI로 사상하며[6], NLICE는 SymCat 파생 개념을 동일 방식으로 사상한다.

정규화 자체가 병목이 아님을 실측으로 확인하였다. Table 5는 (i) 원자료 개념이 CUI로 정규화되는 비율과 (ii) 그 CUI가 우리 KG 어휘에 존재하는 비율을 환자 수준으로 보고한다. 개념 정규화는 환자 수준 91.8–100%로 거의 완전하다(DDXPlus의 잔여분은 다치형 속성-값 복합 코드로, 개념이 아니라 속성이므로 속성 채널이 별도 처리한다). 잔여하는 KG-매칭 격차(환자 수준 72–93%)는 정규화 실패가 아니라 KG의 콘텐츠 커버리지 한계이다. 미매칭 근거를 분해하면 우리 KG가 다른 질환에 대해서는 보유하나 정답 질환 프로필에만 부재하는 개념이 60–76%(정답 프로필의 불완전성), KG 전체에 부재하는 개념이 17–33%였다. 전자의 근본 원인은 정보추출의 과소추출이 아니라 질환 설명 텍스트 자체가 해당 표현형을 언급하지 않는 source 콘텐츠 부재이며, 이는 더 공격적인 재추출이 진단에 net-negative였다는 관찰과 정합한다(§6.3). 따라서 이 격차의 개선 레버는 정규화나 추출 강도가 아니라 더 풍부한 free-text source의 보강이다(§6.3).

**Table 5.** Concept-normalization and KG-match coverage (patient-level mean %). *evidence→CUI* = fraction of a patient's raw evidence concepts that normalize to a UMLS CUI via the standard/provided mapping; *CUI in KG* = fraction that additionally exists in our KG vocabulary and can therefore match.

| Benchmark | evidence→CUI | CUI in KG |
|---|---|---|
| DDXPlus (n=30,000) | 91.8 | 79.2 |
| SymCat | 100.0 | 93.5 |
| RareBench:RAMEDIS | 100.0 | 72.3 |
| RareBench:LIRICAL | 99.8 | 88.1 |
| RareBench:MME | 100.0 | 84.3 |
| RareBench:HMS | 99.7 | 84.2 |

### 5.3 지표

주 지표는 최상위 순위 정확도 GTPA@1이며 GTPA@{3,5,10}과 MRR을 함께 보고한다. 벤치마크 간 일관성(cross-benchmark robustness)과 속성 ablation의 이득을 GTPA@1과 함께 공동 1차 지표로 삼는다. 정답을 원문보다 위로 올릴 수 없는 구조인 DDR/DDP/DDF1은 사용하지 않는다.

### 5.4 SOTA 비교의 세팅 분리

외부 방법과의 비교는 세팅을 명시하여 분리한다. 우리와 세팅이 동일한 비교군은 추론시 LLM을 사용하지 않는 knowledge-based 표현형 유사도 방법(HPO 기반)뿐이며, 지도학습(벤치마크 분포로 학습), agentic LLM(도구·검색·문진 반복), zero-shot LLM(추론시 LLM 사용)은 세팅이 다르므로 방법론적 상한으로만 인용한다. RareBench 하위집합의 per-dataset 비교 수치는 여러 패러다임을 단일 프로토콜로 재실행한 DeepRare Extended Data Table 2를 출처로 한다[14]. 외부 지표 인용 시에는 GTPA@1 정의 차이(DDxT/DDXPlus 계열의 집합 포함 대 MEDDxAgent의 진정한 top-1), NLICE의 58.8→82.0 이득에 내재한 질환 수 교락, SymCat·NLICE에 zero-shot 기준선이 부재하다는 점을 각주로 명시한다. 결과 비교표는 §6에 제시한다.

## 6. 결과

### 6.1 속성의 인과적 기여

Table 2는 §4의 통제 ablation 결과이다. 정규화된 location 속성만이 유의한 이득을 보였으며(치환 $z=9.35$, $p=0.005$, 부트스트랩 95% 신뢰구간이 0을 배제), 동일 이득이 전체 진단 파이프라인에서도 재현되었다(기저 GTPA@1 8.17→+location 12.78, +4.6pp). character와 onset은 귀무였다.

**Table 2.** Attribute ablation on NLICE (base = binary phenotype-CUI overlap; closed 29-way set, 6,450 patients). Δ = GTPA@1 gain over base; permutation null $K=200$, bootstrap $B=2000$.

| Attribute | Δ GTPA@1 (pp) | null (mean±sd) | $z$ | $p$ (perm) | bootstrap 95% CI | credited |
|---|---|---|---|---|---|---|
| **Location** (→ UMLS CUI) | **+3.64** | −0.02 ± 0.39 | **9.35** | **0.005** | **[+3.19, +4.11]** | ✔ |
| Character (free text) | 0.00 | 0.00 | — | 1.000 | [0, 0] | ✗ |
| Onset (controlled) | 0.00 | 0.62 ± 1.01 | −0.61 | 1.000 | [0, 0] | ✗ |

이 결과는 세 가지 해석을 지지한다. 첫째, 정규화된 속성은 지도학습 없이도 zero-shot KG 진단을 유의하게 개선하며, 이는 속성이 공통 어휘로 정규화될 때 벤치마크를 가로지르는 상호언어로 작동함을 시사한다. 둘째, 정규화 가능한 속성만 전이되었다. character가 귀무인 것은 변별력의 부재가 아니라 자유텍스트 표면형이 출처 간에 정렬되지 않는 비정규화성에서 기인하며(§2.3), 따라서 character는 스코어링에서 배제하지 않고 표면 일치 보상만 부여하였다. 셋째, 이득은 조밀한 간선 주석에 의존하였다. 희소 KG(간선 location 27%)에서 location 이득은 +1.44pp($p\approx0.02$)였으나 조밀 KG(38%)에서 +3.64pp($p=0.005$)로 강화되었다. character, onset, aggravating, relieving은 자유텍스트이거나 희소하여 외재적으로 미검증으로 남으며 이를 한계로 명시한다.

### 6.2 벤치마크 진단 성능과 SOTA 대비

Table 3은 최선 구성(단일 KG, 질환 계보 해석, 통합 근거, 분리 정규화, BioLORD 정렬, 잠재의미 융합)의 재현된 per-dataset 결과와 여러 패러다임의 외부 수치[14]를 함께 제시한다.

**Table 3.** Per-dataset Recall@1 (%) across paradigms on RareBench subsets, with our zero-shot KG (Recall@1 / Recall@3; production enriched KG, §6.3). Numbers other than ours are from DeepRare Extended Data Table 2 (HPO-input only; "DeepRare" = DeepSeek-V3 backbone) [14], verified against the source table. KB = knowledge-based (HPO).

| Dataset | Ours (@1 / @3) | PubCaseFinder (KB) | PhenoBrain (KB) | GPT-4o (0-shot LLM) | DeepRare (agentic) |
|---|---|---|---|---|---|
| MME (n=40) | **70.0 / 80.0** | 40.0 | 25.0 | 2.5 | 70.0 / 77.5 |
| HMS (n=87) | **28.7 / 52.9** | 7.3 | 22.4 | 47.7 | 57.0 |
| LIRICAL (n=370) | 28.9 / 42.7 | 37.8 | 27.1 | 31.1 | 56.0 |
| RAMEDIS (n=624) | 23.7 / 55.5 | 25.8 | 20.7 | 35.5 | 72.6 |

제안 방법은 전 벤치마크에서 절대 SOTA(지도·agentic)에 근접하지 않았다. 그러나 세팅이 동일한(추론시 LLM 없음) knowledge-based 비교에서 초희귀·신규 질환 집합 MME는 Recall@1 70.0%로 KB 최고(40.0)와 zero-shot LLM(2.5)을 크게 상회하고 agentic DeepRare(70.0)와 동률에 이르렀으며 Recall@3 80%는 DeepRare(77.5)를 넘어섰고, HMS는 Recall@1 28.7%로 KB 최고(22.4)를 상회하였다. RAMEDIS(23.7)와 LIRICAL(28.9)의 Recall@1은 동일-세팅 KB 비교군(PubCaseFinder 25.8 / PhenoBrain 20.7~27.1)과 동급이며 Recall@3(55.5 / 42.7)에서는 이를 상회하였다. 이들 데이터셋에서 zero-shot LLM·agentic·지도 방법의 상위 성능은 학습된 disease–symptom 매핑이나 추론시 LLM을 요구하는 다른 세팅에서 얻어진 것이다. 흔한 질환 증상체커 SymCat은 자체 disease–symptom 빈도표(§3에서 금지)에 기반한 방법만이 상위 성능을 보고하며 동일-세팅(zero-shot, no-inference-LLM) 공개 기준선이 부재하므로, 본 연구의 Recall@1 7.1/@3 15.2는 해당 세팅의 초기 기준선으로 보고한다. DDXPlus에서는 GTPA@10이 재현적으로 83.9%로 zero-shot LLM의 Top-5(80.4%[15])를 상회하였다. 표본이 작은 MME(n=40)의 Recall@3은 재실행 간 80–82%로 변동하였다.

### 6.3 근거 텍스트 커버리지 보강 (GeneReviews + StatPearls)

각 벤치마크의 낮은 절대 성능의 원인을 forensic으로 규명한 결과, 이는 스코어링이나 정보추출의 결함이 아니라 근거 텍스트 자체의 커버리지 한계였다. 희귀질환(RAMEDIS·LIRICAL)에서는 환자 표현형 명칭의 상당 부분(RAMEDIS 표본에서 약 75%)이 PubMed 유래 근거 텍스트에 애초에 등장하지 않았고, 흔한 질환 증상체커(SymCat)에서는 정답 질환이 KG에 존재함에도(49/50이 표현형 간선 보유) 환자의 일상어 호소(예: "groin mass", "ache all over", "coughing up sputum")와 특정 CUI 기반 IE 프로파일 사이의 어휘·granularity 불일치로 환자 근거와 정답 프로파일의 중앙값 겹침이 1, 44.8%가 겹침 0이었다. 두 경우 모두 §3 제약(disease–symptom 매핑 금지) 아래에서 유일하게 남는 개선 경로는 더 풍부한 free-text 근거의 재-IE이다.

이에 두 개의 벤치마크-blind free-text 자원(질환명 시드, 벤치마크 질문·증상 목록 미포함)을 각각의 질환군에 맞추어 재-IE하였다. (i) 희귀질환은 생화학 지표까지 서술하는 GeneReviews(NCBI Bookshelf) 888개 챕터에서 표현형 서술 절만 유지하고 병리진단적 검사·대사 지표를 명시적으로 요구하는 소진적 추출 프롬프트로 재-IE하여(질환당 중앙값 36개 finding, PubMed의 6–10개 대비 상승) 592개 질환 근거를 얻었고, (ii) 흔한 질환은 StatPearls 596개 챕터의 "History and Physical" 절에서 동일 방식으로 재-IE하여(질환당 중앙값 33개 finding) 360개 질환 근거를 얻었다. 후자는 PubMed 블롭보다 2.5–5배 길 뿐 아니라 topic drift(예: 한 폐렴 블롭이 COVID-19 영상의학 논문)를 교정하여 일상어 호소 어휘를 회복시켰다. 두 자원을 finding 수준에서 순차 union 병합(GeneReviews +18,991, StatPearls +11,234 신규 finding)한 뒤 단일 KG를 재구축하였다.

**Table 4.** Cumulative effect of content enrichment (Recall@1 / @3 / @10; same single-KG pipeline and fixed configuration as Table 3). GeneReviews targets rare diseases; StatPearls targets common diseases. Each source only touches its own disease class, so the single KG covers all benchmarks (§2).

| Dataset | Base KG (PubMed) | + GeneReviews | + GeneReviews + StatPearls |
|---|---|---|---|
| LIRICAL (n=370) | 23.0 / 37.0 / 53.2 | 28.4 / 41.6 / 58.1 | **28.9 / 42.7 / 57.8** |
| MME (n=40) | 62.5 / 80.0 / 95.0 | **70.0 / 80.0 / 92.5** | **70.0 / 80.0 / 90.0** |
| RAMEDIS (n=624) | 22.9 / 51.3 / 62.2 | 23.9 / 55.3 / 67.8 | 23.7 / **55.5 / 68.4** |
| HMS (n=87) | 29.9 / 55.2 / 75.9 | 24.1 / 51.7 / 73.6 | 28.7 / 52.9 / 74.7 |
| DDXPlus (n=3000) | 38.3 / 55.3 / 81.0 | 38.3 / 55.5 / 80.9 | 37.1 / **57.7 / 83.9** |
| SymCat (n=2426) | 4.4 / 14.0 / 30.1 | 4.4 / 14.0 / 30.1 | **7.1 / 15.2 / 32.7** |

GeneReviews 보강은 생화학이 진단을 좌우하는 대사·유전 질환 집합에서 Recall@1을 LIRICAL +5.4pp, MME +7.5pp 끌어올렸고—후자는 agentic DeepRare의 HPO-only Recall@1(70.0)과 동률—RAMEDIS에서는 정답 질환을 상위권으로 밀어 Recall@3·@10을 개선하였다. StatPearls 보강은 흔한 질환 프로파일에 일상어 호소 어휘를 더하여 최약 벤치마크 SymCat의 Recall@1을 4.4→7.1(+61% 상대)로, DDXPlus의 Recall@3·@10을 +2.4pp·+2.9pp 올렸으며, GeneReviews 단계에서 밀도 부작용으로 하락했던 HMS의 Recall@1을 24.1→28.7로 회복시켰다(HMS는 흔한 질환과 일부 겹침). 각 자원이 자기 질환군만 건드리므로 다른 질환군의 성능은 보존되어 단일 KG 원칙(§2)이 유지된다. 다만 조밀화의 비용으로 DDXPlus Recall@1은 −1.2pp 소폭 하락하였다(상위권 hub 표현형 증가). HMS 밀도 부작용의 기전은 §6.3의 forensic으로 확인된다: HMS 정답 질환이 GeneReviews 보강 질환과 겹치는 비율은 1%에 불과하므로 하락은 정답 프로파일의 직접 오염이 아니라 전역 정보량(IC)·잠재의미 통계 이동에 의한 것이며, StatPearls의 흔한 질환 보강이 이 전역 통계를 재조정하여 회복시켰다.

보강의 상한도 명시한다. 각 벤치마크 정답 CUI 중 보강이 직접 도달한 비율은 1–3%에 그쳤다. RAMEDIS 정답 480개 중 224개가 KG 질환 노드로 존재하나 GeneReviews 보강을 받은 것은 14개였는데, 이는 상당수 정답이 특정 질환이 아니라 일반 개념 CUI(예: "Muscle degeneration", "Gastrointestinal inflammation")로 인코딩되어 특정 질환 근거가 CUI 수준에서 매칭될 수 없기 때문이다. 이 격차는 우리 IE의 한계가 아니라 벤치마크 정답 인코딩의 한계이며, 이를 넘으려면 학습된 disease–symptom 매핑이 필요하므로 §3 제약 아래에서는 도달 불가능한 경계다. 따라서 문헌 커버리지 보강은 §3-safe 경로에서 얻을 수 있는 이득의 상한 부근에 있으며, 우리는 base KG와 보강 KG 양쪽을 투명하게 보고한다.

### 6.4 흔한 질환과 희귀 질환의 알고리즘적 긴장

단일 알고리즘 제약(§7) 아래에서 흔한 질환과 희귀 질환은 상반된 스코어링 선호를 보였다. 우리는 두 개의 전역 하이퍼파라미터를 통제 실험하였다. (i) 프로파일 길이 정규화 지수를 낮추면(0.5→0.3) DDXPlus Recall@1이 +2.7pp 오르나 MME는 70→52.5로 붕괴하였고, (ii) 임베딩 soft-match를 어휘-공백 근거에서 전체 근거로 확장하면(align_all) SymCat·DDXPlus Recall@1이 오르나 MME는 다시 70→62.5로 하락하였다. 후자에 대해 근거의 정보량으로 soft-match를 threshold-free하게 게이트하는 기법(특이 근거는 억제, 일상어 근거만 확장)을 도입하여 MME 하락을 62.5→65로 완화하였으나 희귀 벤치마크의 하락을 완전히 제거하지는 못했다. 기전은 명확하다: 희귀질환 환자는 특이 HPO 근거가 특정 표현형에 정확히 매칭되므로 길이 정규화와 sharp exact 매칭에서 이득을 얻는 반면, 흔한 질환 환자는 소수의 일반어 근거를 가지므로 fuzzy 매칭과 약한 길이 패널티를 선호한다. 단일 전역 설정은 양쪽을 동시에 최적화할 수 없다. 우리는 희귀질환 성능(속성 기여와 fair-setting 비교가 가장 유리한 지점, 특히 MME의 agentic-SOTA 동률)을 보존하는 방향으로 설정을 고정하고, 이 긴장을 per-benchmark 튜닝(§7 위반)으로 우회하지 않았음을 명시한다.

### 6.5 근거 파이프라인의 무결성 (integrity of the evidence pipeline)

> **[흐름 — 결함 목록·수정 전후 수치는 결과 확정 후 확정 작성]**
>
> - **주장**: 본 연구에서 보고하는 성능 변화의 상당 부분은 알고리즘이 아니라 **파이프라인이 조용히 버리던 근거를 회수한 것**에서 나온다. 이는 문헌 기반 KG 구축 전반에 해당하는 재현성 이슈이므로 결과의 일부로 보고한다.
> - **결함의 공통 성질**: 네 건 모두 예외를 발생시키지 않고 *그럴듯한* 산출물을 냈다(길이 상한에 의한 본문 절단, 질환 제목의 총칭 개념 붕괴, 출력 상한 초과 시 창(window) 단위 폐기, 후처리기의 속성 스키마 축소). 종단 점수만으로는 탐지되지 않으며, 원자료를 직접 열어야 드러났다.
> - **대응**: 단계 사이의 보존 성질을 명시적 검사로 고정한다(입력 대비 출력 텍스트량, 입력 문서 수 대비 산출 질환 수, 모든 창의 파싱 성공, 프롬프트가 요구한 속성 필드의 잔존, §12 차수 분포). 검사기는 **기여가 아니라 산출물 신뢰의 근거**로서 resource/companion 논문의 품질보증 절에 둔다.
> - **정직성**: 알려진 미해결 검사(예: 질환의 41%가 표현형 ≤5개, §12 기준 초과)는 통과시키지 않고 실패로 남겨 현재의 병목을 드러낸다.

## 7. 재현성과 구현

정보추출은 greedy decoding(temperature=0)이므로 표본추출 seed가 존재하지 않으며, 잔여 비결정성은 GPU 커널, 배치·패딩 구성, 라이브러리·자료형·모델 판본에서 비롯하므로 결정론적 알고리즘 설정과 고정 배치·고정 판본으로 통제한다. 진단 알고리즘 자체는 결정론적이고 환자 시뮬레이션은 고정 seed(42)를 사용하므로, 아래 명령으로 Table 2·3의 수치가 재현된다(표본이 작은 MME의 Recall@3만 재실행 간 80–82%로 소폭 변동).

Table 3(프로덕션 enriched KG 진단 성능)과 Table 4의 최종 열은 `python3 pilot/scripts/onlykg_eval_unified2.py --kg pilot/cache/prod/kg_merged2.json --benchmarks rarebench,ddxplus,symcat --ppd 50 --hier 2 --lsa 1 --lsa_w 0.4 --sapbert 1 --emb_model biolord --tau 0.78`로, Table 4의 base 열은 동일 명령에서 `--kg pilot/cache/prod/kg.json`으로, Table 2(속성 ablation)는 `python3 pilot/scripts/nlice_ablation.py --kg pilot/cache/prod/kg_v2.json --perm 200 --boot 2000`으로 산출된다. §6.4의 긴장 실험은 위 명령에 `--normexp 0.3` 또는 `--align_all 1`을 추가하여 재현된다(기본값은 각각 0.5, 0). 보강 KG는 `merge_ie_preds.py`(GeneReviews→StatPearls 순차 union)→`v155_normalize_umls.py`→`v157_build_kg.py`로 재생성된다. 본 연구에서 Table 2·3·4의 모든 정량 주장은 위 스크립트 재실행으로 재현을 확인하였다. 특히 속성 밀도 대조는 동일 스크립트에 `--kg pilot/cache/prod/kg.json`(간선 location 27%)을 주면 base 6.08→+location 7.52(+1.44pp, $p=0.025$), `--kg pilot/cache/prod/kg_v2.json`(38%)을 주면 base 8.20→11.84(+3.64pp, $p=0.005$)로 재현된다.

KG 구축·정규화·조립, 진단 스코어러(질환 계보 해석, 통합 근거, 분리 정규화, BioLORD 정렬, 잠재의미 융합, 속성 채널, 후보 동일성 병합), MRREL 기반 계층 구축, NLICE 속성 ablation의 구현 및 세부 근거는 부속 문서(`docs/method_ie.md`, `docs/method_attribute_diagnosis.md`, `docs/attribute_benchmark_analysis.md`, `docs/sota_reference.md`, `docs/eval_pipeline_forensic.md`)에 기술한다.

---

## References

[1] Bickley LS, Szilagyi PG. *Bates' Pocket Guide to Physical Examination and History Taking.* 7th ed. Philadelphia: Wolters Kluwer Health/Lippincott Williams & Wilkins; 2013. Ch. 3, "The Seven Attributes of a Symptom", p. 38.

[2] Jacobsen JOB, Baudis M, Baynam GS, et al. The GA4GH Phenopacket schema defines a computable representation of clinical data. *Nat Biotechnol.* 2022;40(6):817–820. doi:10.1038/s41587-022-01357-4.

[3] Elhadad N, Pradhan S, Gorman S, Manandhar S, Chapman W, Savova G. SemEval-2015 Task 14: Analysis of Clinical Text. In *Proc. 9th Int. Workshop on Semantic Evaluation (SemEval 2015)*, pp. 303–310.

[4] Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024: phenotypes around the world. *Nucleic Acids Res.* 2024;52(D1):D1333–D1346. doi:10.1093/nar/gkad1005.

[5] Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. *Nucleic Acids Res.* 2004;32(Database issue):D267–D270. doi:10.1093/nar/gkh061.

[6] SNOMED International. *SNOMED CT Editorial Guide — Body Structure* (Body structure 123037004; Finding site 363698007). https://docs.snomed.org/ (accessed 2026-06-30).

[7] Spärck Jones K. A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation.* 1972;28(1):11–21.

[8] Aronson AR, Lang FM. An overview of MetaMap: historical perspective and recent advances. *J Am Med Inform Assoc.* 2010;17(3):229–236. doi:10.1136/jamia.2009.002733.

[9] Altmann A, Toloşi L, Sander O, Lengauer T. Permutation importance: a corrected feature importance measure. *Bioinformatics.* 2010;26(10):1340–1347. doi:10.1093/bioinformatics/btq134. (Permutation-null p-value for a feature's contribution; cf. Breiman, Random Forests, *Machine Learning* 2001;45(1):5–32 for the origin of permutation importance.)

[10] Liu F, Shareghi E, Meng Z, Basaldella M, Collier N. Self-Alignment Pretraining for Biomedical Entity Representations. In *Proc. NAACL-HLT 2021*, pp. 4228–4238.

[11] Remy F, Demuynck K, Demeester T. BioLORD-2023: semantic textual representations fusing large language models and clinical knowledge graph insights. *J Am Med Inform Assoc.* 2024;31(9):1844–1855.

[12] Deerwester S, Dumais ST, Furnas GW, Landauer TK, Harshman R. Indexing by latent semantic analysis. *Journal of the American Society for Information Science.* 1990;41(6):391–407.

[13] Al-Ars Z, Agba OP, Guo Z, Boerkamp C, Jaber Z, Jaber T. NLICE: Synthetic Medical Record Generation for Effective Primary Healthcare Differential Diagnosis. *arXiv:2401.13756*; IEEE 2024.

[14] Zhao et al. An agentic system for rare disease diagnosis with traceable reasoning (DeepRare). *Nature* (2025); doi:10.1038/s41586-025-10097-9; *arXiv:2506.20430.* (Per-dataset Recall@1 from Extended Data Table 2, p.37, HPO-input only, DeepSeek-V3 backbone — 원표 대조로 전 cell 검증; 본문의 MME "78%"는 multimodal-input 설정으로 본 표의 HPO-only 70.0과 별개.)

[15] Lim et al. H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis. NeurIPS 2025 GenAI4Health workshop; *arXiv:2510.03700.*

[16] Ghosh S, et al. DDxT: Deep Generative Transformer Models for Differential Diagnosis. NeurIPS 2023 Deep Generative Models for Health (DGM4H) workshop; *arXiv:2312.01242.*

[17] He W, Chen J. Scalable Online Disease Diagnosis via Multi-Model-Fused Actor-Critic Reinforcement Learning. In *Proc. KDD 2022*; *arXiv:2206.03659.*

[18] Rose et al. MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis. In *Proc. ACL 2025*; *arXiv:2502.19175.*

[19] Chen et al. RareBench: Can LLMs Serve as Rare Diseases Specialists? In *Proc. KDD 2024*; *arXiv:2402.06341.*

[20] Fansi Tchango A, Goel R, Wen Z, Martel J, Ghosn J. DDXPlus: A New Dataset for Automatic Medical Diagnosis. In *Proc. NeurIPS 2022 (Datasets and Benchmarks)*; *arXiv:2205.09148.*

[21] Resnik P. Using information content to evaluate semantic similarity in a taxonomy. In *Proc. 14th Int. Joint Conf. on Artificial Intelligence (IJCAI 1995)*, pp. 448–453.

[22] McCray AT, Srinivasan S, Browne AC. Lexical methods for managing variation in biomedical terminologies. In *Proc. 18th Annu. Symp. on Computer Applications in Medical Care (SCAMC 1994)*, pp. 235–239. (SPECIALIST Lexicon and Lexical Tools; source of the normalized-string form used by MRXNS.)

*검증 완료:* [14] DeepRare per-dataset Recall@1(Ext. Data Table 2, p.37)·[17] MMF-AC SymCat-300 Top-3=78.24%는 원표와 대조하여 전 cell 정확 확인됨. Table 2·3의 우리 수치는 §7 명령으로 재실행 재현 확인됨. *투고 전 잔여:* [13][18][19] 저자 전체 표기 보완, [11] BioLORD-2023-C 체크포인트 서지 최종 확인.
