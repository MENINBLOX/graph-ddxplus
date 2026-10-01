# Method — Attribute Information Extraction and Normalization

> IE 파이프라인의 논문 Method 초안. 국문 서술 + 영문 용어·표기·표(작성 규약 준수).
> 대응 스크립트: S0 seed / S1 `v156_build_sources` / S2 `v153_production_ie` / S3 `v155_normalize_umls`(+`umls_link`,`attr_normalize`) / S4 `v157_build_kg`.
> **Part I = Method(개선 반영본). Part II = 자기비평 로그(발견한 학술적 헛점 + 반영한 수정).**

---

# Part I. Method

## 3.1 Overview and design constraints

우리는 질환별 임상 지식 텍스트에서 finding과 임상 표준 속성(clinical attributes)을 추출하여
표준 온톨로지에 정규화하고, 이를 disease–phenotype edge의 qualified statement로 부착한 단일
knowledge graph(KG)를 구축한다. 추출·정규화 파이프라인은 다음 제약 하에서 설계된다.

- **(C1) Strict zero-shot / benchmark-blind.** IE 입력은 *질환명(disease name)만*으로 조회한
  자유 텍스트이며, 평가 벤치마크의 question·symptom list·train label은 어느 단계에도 사용하지
  않는다. Disease–symptom 매핑을 직접 제공하는 구조화 자원(예: HPO annotation `phenotype.hpoa`,
  Orphanet HOOM)은 *KG source 로 사용하지 않는다* — 이는 disease–symptom 정답을 우회 주입하는
  것에 해당하기 때문이다. 이러한 자원은 §3.7 의 coverage 실측에서 *참조(reference)* 로만 쓴다.
- **(C2) LLM은 KG 구축 시점에만.** LLM은 IE 단계에서 텍스트로부터 추출에만 사용하고, 정규화·
  질의 시점에는 사용하지 않는다. 정규화는 전면 결정론적(deterministic) 사전 기반이다.
- **(C3) Few-shot 및 인위적 개수 제한 금지.** 프롬프트에 예시를 넣지 않으며("exhaustive list"
  directive 사용), finding 개수를 제한하지 않는다.
- **(C4) 재현성.** 디코딩은 greedy(temperature 0)로 sampling seed 가 존재하지 않으며, 잔여
  비결정성(GPU kernel·batch)은 §3.8 에 명시된 방법으로 고정한다.

Figure 1 은 seed → source → extraction → normalization → KG 의 5 단계를 요약한다.

## 3.2 Disease seed (S0)

Seed 는 UMLS Metathesaurus 의 Disorder 계열(semantic group DISO) 개념 중, rare disease 를
포함하도록 OMIM·ORPHANET·HPO·RareBench 를 union 한 **38,906 개 질환 CUI**(preferred name 포함)
이다. 이후 모든 조회는 이 CUI–name 쌍의 *name* 만을 입력으로 사용한다(C1).

## 3.3 Symptom-bearing source acquisition (S1)

각 질환에 대해 임상 발현(clinical presentation)을 기술하는 자유 텍스트 blob 을 조립한다.
자료원은 라이선스·재현성 기준으로 선정한다(상세: `methodology_ie_sources.md`).

- **PubMed abstracts** — NCBI E-utilities(`esearch`→`efetch`)로 질환명(및 UMLS 동의어)으로
  조회한 상위 abstract. 접근 재현 가능, 저작권상 텍스트 마이닝 허용.
- **Encyclopedic clinical sections** — MedlinePlus(U.S. public domain), Wikipedia "Signs and
  symptoms"(CC BY-SA)의 임상 발현 섹션. (StatPearls 는 ND 라이선스 제약으로 공개 KG 에서 제외.)

PubMed abstract 는 연구·유전 중심이라 임상 발현 밀도가 낮은 경향이 있으므로(선행 corpus-mismatch
관찰), passage 를 **symptom-cue 점수**(임상·소견 단어 빈도, 섹션 유형 가중)로 랭킹하여 상위
passage 를 단어 예산(word budget) 내에서 concatenate 한다. 선택은 결정론적이며, passage 별
provenance(PMID/문서 ID)를 보존한다(traceability).

산출 coverage(§3.7 정의): non-empty 자유텍스트 소스를 가진 질환 = **35,193 / 38,906 (90.5%)**.

## 3.4 Attribute schema

증상 특성화의 교과서 표준인 **Bates 의 attributes-of-a-symptom** 을 앵커로, KG 후보 6 속성을
둔다: **location, character, severity, timing(onset 포함), aggravating, relieving.** radiation 은
location 의 방향 qualifier 로 흡수하고, 개념형(location 등)은 온톨로지 CUI 로, 범주형
(severity·onset)은 통제어휘로 정규화한다(§3.6). *최종 KG 반영 속성은 downstream ablation
(leave-one-out)이 결정하며(§3.7), 본 절의 6 속성은 후보 집합이다.* 스키마의 임상적 정당성은
별도 문서(`docs/attribute_rationale_draft_ko.md`)에서 다룬다.

## 3.5 Single-pass LLM extraction (S2)

- **Model.** 로컬 open-weight LLM(gemma-4-E4B, bf16). vLLM backend, greedy decoding.
- **Prompt.** 질환명과 SOURCE 를 받아 *SOURCE 에 명시된 사실만으로* 원자적 finding 과 그 속성을
  출력하는 **단일 프롬프트**. 내부적으로 STEP 1(anatomical site 를 finding 명에서 분리해 원자화)
  → STEP 2(각 원자 finding 에 6 속성 배정, SOURCE 가 그 finding 에 대해 명시할 때만 채움)의
  2 단계 지시를 포함하나 **1 회 호출**로 수행한다(C2·비용). Benchmark-blind, few-shot 없음(C3),
  질환명·원인·치료·역학·정상소견·negation 은 제외하도록 지시한다.
- **Faithful raw extraction.** severity·onset 은 *원문 표현 그대로(raw)* 추출하고(정규화는 S3),
  개수 상한 없이 **SOURCE 가 명시한 범위 내에서 exhaustive**(source-relative)하게 추출하도록
  지시한다(C3). 따라서 실측 finding 밀도(8.6/disease)는 지시의 상한 부재가 아니라 자료원의 발현
  밀도에 의해 결정된다(§W11).

## 3.6 Deterministic normalization (S3)

정규화는 임계치(threshold)·학습 파라미터 없이 전면 결정론적으로 수행한다.

**(a) 개념형(finding / location / character) → UMLS CUI → SNOMED CT.**
UMLS 공식 정규화 문자열 인덱스 **MRXNS_ENG**(SPECIALIST Lexicon 의 *norm*: lowercase·구두점
제거·stop-word 제거·굴절 축약·어순 정렬)에 **exact match** 한다. 질의 mention 에는 동일 norm 에
더해 결정론적 형태 축약(단복수·동사원형 deverb·Latin/Greek 복수·수량 접두어 strip·head noun-
phrase)을 적용하되, **각 축약형은 반드시 MRXNS 에 존재하는 문자열과 exact match 될 때만 채택**
한다(축약이 허구 링크를 만들 수 없음). 이는 MetaMap·QuickUMLS·cTAKES 계열의 사전 기반 매핑과
동일 계열이며 유사도 임계치를 사용하지 않는다. 하나의 정규화 문자열이 복수 CUI 에 매핑되면
**semantic type 우선순위**(Sign/Symptom > Finding > … > Neoplasm)로 결정론적 disambiguation
한다. CUI→SNOMED CT 는 MRCONSO, CUI→semantic type 은 MRSTY 로 매핑한다. finding 은 임상 유효
semantic type(disease/symptom/finding 계열)만 인정하는 **clin-type filter** 로 해부·수식어 오링크
를 배제한다.

**(b) 범주형(severity / onset) → 통제어휘.**
severity → {mild, moderate, severe}, onset → {sudden, gradual} 로 결정론적 lexicon 매핑한다.
수치 척도(NRS 0–10)는 1–3 / 4–6 / 7–10 → mild / moderate / severe 로 대응하고, 정도를 나타내지
않는 변화어(increased·worsened 등)는 매핑을 *거부*(빈값)한다. LLM 에 정규화를 시키지 않으므로 raw
추출은 faithful 하게 유지된다.

## 3.7 Validation

새 gold 을 구축하지 않으며, IE 튜닝·선택·점수화에 벤치마크 gold 을 사용하지 않는다(C1). 검증은
두 축이다.

**(A) Intrinsic — gold-free.** 결정론적 측정(NLI·LLM-judge 아님):
- *Source-grounding (enforced).* 파이프라인은 추출 후 **source 에 token 이 없는 finding·속성값을
  제거**(filter)하므로, 최종 출력의 grounding 은 *설계상 1.00* 이다 — 즉 grounding=1.00 은 측정치가
  아니라 *강제된 불변식*이며, 그 자체로 faithfulness 의 독립 증거가 아니다(§W10).
- *Pre-filter hallucination rate (측정).* 진짜 faithfulness 신호는 **filter 이전** 단계에서 LLM 이
  source-부재 개념을 생성한 비율이다. 이 비율(및 filter 로 제거된 finding 수)을 보고한다. 단,
  token-level filter 는 표현 변형(동의어)로 인해 *유효 finding 도 탈락*시킬 수 있어 recall 을
  희생한다(§W1 연계).
- *Normalizability* = CUI-link · SNOMED-map · clin-type 적합 비율(정확도 아님, §W2).
- *Atomicity* = multi-site finding 분해율 등 구조 지표.

전 코퍼스(303,346 findings, filter 이후) 실측: Table 2. finding CUI-link 0.94, severity→vocab
0.76, onset→vocab 1.00. (grounding 은 강제값이라 표에 성능으로 싣지 않고, pre-filter drop 율로 보고.)

**(B) Extrinsic — downstream diagnosis.** IE 의 유효성은 완성 KG 를 단일 알고리즘으로 5 개
벤치마크(DDXPlus·SymCat·RareBench 등)에서 top-1 진단 정확도(GTPA@1)로 평가하고, **속성별
leave-one-out ablation** 으로 각 속성의 기여 delta 를 측정한다(KGrEaT 계열 extrinsic 평가). 이는
intrinsic 이 못 재는 recall·효용을 보완한다.

**Coverage(§3.3 참조).** benchmark 어휘(symptom/phenotype CUI) 대비 KG evidence coverage 를
사전 실측한다. rare disease phenotype 이 표준 자원에 존재함은 `phenotype.hpoa`(OMIM/ORPHA→HPO)로
확인하되, 이는 disease–symptom 매핑이므로 KG source 로는 쓰지 않고 coverage 참조로만 사용한다(C1).

## 3.8 Reproducibility

디코딩은 greedy(temperature 0)로 sampling seed 가 없다. 잔여 비결정성은 (i) deterministic CUDA
kernel(`torch.use_deterministic_algorithms`), (ii) 고정 batch size, (iii) 고정 library·model
revision·dtype 으로 제어한다. 모든 정규화 단계는 사전·규칙 기반으로 seed 무관하게 결정론적이다.
자료원은 조회 방법·라이선스와 함께 명시하고, 정규화 사전(MRXNS/MRCONSO/MRSTY)은 UMLS 릴리즈
버전으로 고정한다.

---

# Part II. 자기비평 로그 — 학술적 헛점과 반영한 수정

> 초안(Part I 이전 버전)을 리뷰어 관점에서 재독하며 찾은 약점과, 그에 대해 Part I 에 반영한
> 수정(또는 명시한 한계)이다. 심각도 순.

### 🔴 W10. source-grounding = 1.00 은 tautology(강제값, 측정 아님) — *재독 시 발견*
- **문제.** 최종 출력의 grounding 이 1.00 인 이유는 파이프라인의 `clean()` 이 **source 에 token 이
  없는 finding 을 애초에 제거**하기 때문이다(`v145:107`, `in_src` 필터). 즉 grounding=1.00 은
  측정 결과가 아니라 필터가 강제한 불변식이며, faithfulness 의 독립 증거로 제시하면
  **동어반복(과거 severity=1.00 tautology 와 동종 오류)**이다.
- **수정.** Part I §3.7(A)에서 grounding 을 *enforced invariant* 로 재정의하고 **성능 표에서 제외**.
  진짜 faithfulness 신호는 **filter 이전(pre-filter) 의 hallucination 생성율**로 재정의해 그것을
  보고하도록 변경. 아울러 token-level 강제 필터가 동의어 표현의 유효 finding 을 탈락시켜 **recall
  을 희생**함을 W1 과 연결해 명시. — 리뷰어가 가장 먼저 찌를 지점을 선제 노출.

### 🔴 W1. Intrinsic 검증이 recall(완전성)을 측정하지 못한다
- **문제.** source-grounding·normalizability 는 precision 축(추출한 것이 충실한가)만 잰다.
  *놓친 finding·속성*(recall)은 gold 없이 측정 불가하다. 리뷰어: "IE 가 얼마나 완전한지 어떻게
  아는가?" 답이 없으면 IE 품질 주장이 성립하지 않는다.
- **수정.** (i) Part I §3.7 에서 intrinsic 이 precision 축임을 명시하고, recall·효용은 **extrinsic
  ablation** 으로만 검증한다고 역할을 분리(과대주장 제거). (ii) 이를 방법의 *한계*로 §3.7·Limitations
  에 명문화. (iii) 초안의 "IE 품질을 검증한다"류 문장을 "faithfulness 와 normalizability 를 측정
  한다"로 하향 수정. **완전성 주장 자체를 하지 않는 것**이 정직한 처리다.

### 🔴 W2. "옳은 CUI"(link correctness)를 측정하지 않는다
- **문제.** normalizability 는 *어떤* CUI 로 링크되는지(link rate)만 재고, *옳은* CUI 인지(entity
  linking accuracy)는 재지 않는다. 공개 EL SOTA 도 MedMentions 에서 ~56% 수준이라, 링크율 0.94 를
  정확도로 오독하면 과대주장이다.
- **수정.** Part I 은 링크율을 "normalizability(정규화 가능성)"로만 명명하고 *정확도로 부르지
  않는다.* 방어 가능한 주장을 "threshold-free·deterministic(재현 가능)"으로 한정하고, link
  correctness 는 미측정 한계로 명시. exact-match + semantic-type disambiguation 은 정확도 보장이
  아니라 *재현성·비임의성* 근거임을 분명히 함.

### 🔴 W3. Negation / assertion status 미처리
- **문제.** grounding 이 "content token 이 SOURCE 에 존재" 인데, 부정·가정("does not cause
  jaundice", "unlike X")된 finding 도 token 은 존재해 grounded 로 집계될 수 있다. 임상 IE 리뷰어가
  가장 먼저 지적하는 지점(NegEx 등 assertion 표준 존재).
- **수정.** (i) §3.5 프롬프트 지시에 negation 제외를 명문화. (ii) grounding 이 assertion-aware 가
  아님을 **한계로 명시**하고, downstream 에서 부정 finding 오포함이 diagnosis 에 미치는 영향은
  extrinsic 으로 관측된다고 서술. (iii) 향후 rule-based negation(NegEx/ConText) 후처리를 명시적
  future work 로 남김. — 은폐하지 않고 노출.

### 🟠 W4. Source availability 가 cross-benchmark 비교의 교란변수
- **문제.** PubMed abstract 는 임상 발현 밀도가 낮고 encyclopedic 섹션은 common disease 에 편중
  → rare disease 는 속성 밀도가 구조적으로 낮다. 그러면 rare 벤치마크(RareBench) 열세가 *방법*이
  아니라 *자료원 가용성* 때문일 수 있어, cross-benchmark 주장이 교란된다.
- **수정.** §3.3·§3.7 에서 **source 유형별 finding/속성 밀도를 보고**(disease degree 분포)하도록
  명시하고, 자료원 가용성을 교란변수로 Limitations 에 기재. 진단 결과 해석 시 degree 를 통제/보고
  하는 조건을 둔다. (은폐 대신 정량 노출.)

### 🟠 W5. Model·prompt 선정 근거의 순환성
- **문제.** 추출 recipe(모델·프롬프트)를 과거 MACCROBAT 세팅으로 골랐으나 그 gold 는 폐기됨.
  그러면 "이 recipe 가 최적"이라는 근거가 사라져, 고정 recipe 제시가 임의적으로 보인다.
- **수정.** Part I 은 recipe 를 "고정된 구성"으로 *중립 기술*하고 최적성 주장을 하지 않는다. 선정
  정당화는 (i) gold-free 구조지표(fill·atomicity·clin-type) 상의 open-model 패널 비교와 (ii)
  extrinsic ablation 으로 미룬다고 명시. "benchmark 로 튜닝했다"는 오해를 차단.

### 🟠 W6. character 속성은 사실상 정규화되지 않음
- **문제.** character 는 통제 표준어휘가 빈약해 clin-type 적합률이 낮다(≈0.5). 6 속성을 "표준
  정규화"로 뭉뚱그리면 character 에서 과대주장.
- **수정.** §3.4·§3.6 에서 character 는 **free-text 보존** 속성으로 분리 표기하고, "정규화 interlingua"
  주장에서 제외. character 는 KG-only 가 아니라 KG+LLM 변형에서만 활용한다고 범위를 한정.

### 🟡 W7. Grounding 이 semantic entailment 이 아니라 token 포함
- **문제.** token 포함은 표현 변형·다의어·문맥을 못 잡는다(entailment 보다 약함). 단, 프로젝트는
  NLI/LLM-judge 를 의도적으로 배제(합의)했다.
- **수정.** §3.7 에서 grounding 을 "faithfulness *proxy*(token-level)"로 정확히 한정하고, NLI 를
  쓰지 않는 이유(judge 편향 회피)와 그 trade-off(약한 근거)를 함께 명시. 명명으로 과대주장 제거.

### 🟡 W8. 형태 축약 규칙은 수작업 heuristic (English-specific)
- **문제.** deverb·Latin plural 등은 curated 규칙이라 임의성·언어종속 지적 가능.
- **수정.** §3.6 에서 규칙의 안전성 근거를 명시 — **모든 축약형은 MRXNS exact-match 로 검증되어야
  채택**되므로 과다 축약이 허구 링크를 만들 수 없음(규칙은 recall 후보만 확장, precision 은
  exact-match 가 보장). 규칙 집합·언어 종속성은 재현 부록에 공개.

### 🟡 W9. 단일 패스 추출의 per-document 완전성 미보장
- **문제.** 비용 위해 iterative refine 을 포기 → 긴/복잡 소스에서 under-extraction 가능.
- **수정.** §3.5 에서 단일 패스를 *비용-완전성 trade-off* 로 명시하고, per-document 완전성은
  보장하지 않으며 대수적 규모(35k 질환·303k finding)와 extrinsic 으로 총량 효용을 검증한다고 서술.

### 🟡 W11. "exhaustive extraction" 주장 vs. 관측 밀도(8.6 finding/disease)
- **문제.** C3 는 개수 무제한·exhaustive 를 표방하나, 실측 밀도는 8.6 finding/disease, KG disease
  degree 중앙값 6 으로 낮다. "exhaustive" 를 절대적으로 읽으면 관측과 모순되어 과대주장이다.
- **수정.** §3.5 에서 exhaustive 를 *"source 가 명시한 범위 내에서 무제한"*(source-relative)으로
  한정하고, 낮은 밀도의 원인을 자료원의 발현 밀도(§3.3·W4)로 귀속. IE 지시의 상한 부재와 자료원의
  하한을 분리해 서술.

### 🟡 W12. extrinsic ablation 은 속성-기여를 격리하지 IE recall 을 검증하진 않는다
- **문제.** W1 에서 recall 을 extrinsic 으로 보완한다 했으나, leave-one-out ablation 은 KG 구조·
  정규화·scoring·source coverage 와 교락되어 *속성 유무의 delta* 만 격리한다. 이를 "IE recall
  검증"으로 확대하면 논리 비약이다.
- **수정.** §3.7(B)에서 extrinsic 의 검증 대상을 *속성-값의 진단 기여(delta)* 로 한정하고, 전역 IE
  recall 은 **여전히 미검증**임을 W1 과 함께 한계로 확정. 과잉 일반화 제거.

### 검토 결과 요약 (W1–W12)
- **가장 심각(재독 시 추가 발견) = W10**: source-grounding=1.00 이 필터가 강제한 tautology 였고,
  이를 faithfulness 증거로 제시하는 것은 과거 severity=1.00 오류의 반복. → grounding 을 표에서
  빼고 *pre-filter hallucination 율*로 신호를 재정의.
- **과대주장 제거**가 핵심 축: "IE 품질 검증"→"faithfulness·normalizability 측정", "링크 정확도"
  →"정규화 가능성", "표준 정규화 6 속성"→"5 정규화 + character free-text", "exhaustive"→
  "source-relative exhaustive", "extrinsic=IE 검증"→"extrinsic=속성 delta 격리".
- **은폐 금지 → Limitations 로 노출**: recall 미측정(W1)·link correctness 미측정(W2)·negation
  미처리(W3)·source 교란(W4)·grounding tautology(W10)·전역 recall 미검증(W12). 방어 가능한 축
  (재현성·결정론·threshold-free·traceability·extrinsic delta)으로 주장 범위 재조정.
- **남은 실행 항목(코드 변경 필요)**: pre-filter hallucination 율 계측 추가(W10), negation/assertion
  후처리(W3), source-type 별 finding·속성 밀도 보고(W4·W11), open-model 패널 구조지표(W5).
