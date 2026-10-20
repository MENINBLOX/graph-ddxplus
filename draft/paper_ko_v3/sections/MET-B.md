### 3.5 진단 질의

**순위화 질의.** 한 환자의 진단 순위는 Cypher 질의 두 개(소견 점수 질의와 통합 근거 질의)의 결과를 합쳐 낸다. 입력은 환자 소견 개념(UMLS CUI)의 집합 P다. 소견 점수 질의는 P의 각 개념을 개념 IS_A 관계로 한 단계 위까지 확장해 개념 집합 C(P)를 만든다. 확장된 개념 c의 가중 w_c는 c가 환자 개념 자체이면 1이고, 환자 개념의 직접 상위 개념으로만 들어오면 감쇠 δ다. 두 경로가 모두 있으면 큰 값을 쓴다.

질환 d의 소견 점수 s(d)는 d의 HAS_FINDING 간선 가운데 확장 개념에 닿는 간선의 기여를 더한 값이다.

$$s(d)=\sum_{c\in C(P),\ (d\to c)\in \mathrm{HAS\_FINDING}} w_c\cdot \mathrm{IDF}(c)\cdot n_{\mathrm{docs}}(d,c)^{\beta}\cdot g(d,c)$$

IDF(c) = ln(N / df(c))다. N은 소견 간선이 하나 이상 있는 질환 수이고, df(c)는 그중 c를 소견으로 가진 질환 수다. n_docs(d, c)는 간선을 지지하는 문서 수다. g(d, c)는 간선의 지지 문서 가운데 주제 문서가 하나라도 있으면 1이고, 없으면 γ(offtopic_w)다. 주제 문서는 제목이 그 질환의 이름 형태를 포함하는 문서다. topic-genes 빌드(S-mix, S-llm, S-dict, S-union)에서는 제목에 그 질환의 원인 유전자 기호가 있는 문서도 주제 문서로 본다(3.2절). S-D는 이 유전자 규칙 없이 빌드했다.

소견 점수는 질환 프로파일 크기로 길이를 정규화한다.

$$\mathrm{score}_F(d)=\frac{s(d)}{\left(n(d)+\mu\left(1-\dfrac{m(d)}{m(d)+\tilde m}\right)\bar L\right)^{\alpha}}$$

n(d)는 질환 d의 소견 개념 수이고, m(d)는 d에 대해 수집한 원문 문자 수다. m̃은 소견이 있는 질환들의 m(d) 중앙값이고, L̄은 그 질환들의 평균 n(d)다. μ 항(커버리지 스무딩)은 문헌이 적은 질환의 유효 길이를 늘려 그 질환의 과대평가를 줄이기 위한 항이다.

통합 근거는 환자 개념이 질환 노드이기도 한 경우를 다룬다. 환자 개념이 IS_DISEASE로 질환 d0을 가리키면, d0과 그 직접 하위 질환(질환 IS_A 한 단계)에 u · n_u(d)를 더한다. n_u(d)는 환자 개념 가운데 d 자신이거나 d의 직접 상위 질환인 개념의 수이고, u는 통합 근거 가중(unified_w)이다. IS_DISEASE 간선은 그 질환의 CUI가 그래프에 개념 노드로 있을 때만, 즉 어떤 질환의 소견 개념이거나 그 상위 개념일 때만 생긴다. 따라서 자기 간선 제거(3.4절)로 질환 자신의 CUI가 어느 질환의 소견 개념이나 그 상위 개념으로도 남지 않으면, 그 질환의 IS_DISEASE 간선도 사라진다. 이때 환자 소견에 그 질환의 CUI가 있어도, 그 개념으로는 소견 점수도 통합 근거도 생기지 않는다. 예를 들어 S-llm의 바탕인 v3 LLM 그래프에서 IS_DISEASE 간선은 자기 간선 제거 전 19,609개, 제거 후 16,468개다. 최종 점수는 score_F(d)와 u · n_u(d)의 합이다(score_F는 아래 후보 창 안의 질환에서만 읽는다). 동점은 질환 ID 순으로 정렬한다.

**동결 매개변수.** 채점 매개변수는 설계 D의 개발 선택에서 정하고 동결했다. 선택은 v2 문헌 gemma 그래프에서 좌표 탐색(최대 2회 통과, 한 통과에서 바뀐 값이 없으면 중단)으로 했다. 격자는 α {0.2, 0.25, 0.35, 0.5}, μ {0, 1, 2, 4}, δ {0, 0.3, 0.5, 0.7}, u {0, 0.5, 1, 2}, β {0, 0.5, 1, 2}, γ {1, 0.5, 0.25, 0}이다. 선택된 값은 α = 0.35, μ = 2, δ = 0.3, u = 2.0, β = 0.5다. 설계 D·E의 문헌 그래프 시스템(S-D, S-mix, S-llm, S-dict, S-union과 개발 변형, 2022 절단 그래프)은 이 값을 그대로 쓴다. 구 시스템인 Qwen3 v2.1c 그래프의 수치는 그 시스템에서 따로 고른 설정의 값이다. 그래프마다 γ만 {0, 0.25, 0.5, 1}에서 D3 규칙(3.6절)으로 다시 골랐다. 선택된 γ는 S-D 0, S-mix 0.5, S-llm 1.0, S-dict 0.5, S-union 0.5다. γ = 1이면 주제 문서 구분을 끈 것과 같다. 출처 가중(β, γ)은 간선을 지우지 않는다. γ = 0이어도 그 간선과 근거 문장은 추적 질의로 조회된다.

**후보 창.** 동결 순위화는 Top-k(k = 10)를 내기 위해 소견 점수 score_F를 상위 k × 4 = 40개 질환에서만 읽는다. 통합 근거 점수는 이 창 밖의 질환에도 더한다. 따라서 통합 근거를 받은 질환이 소견 점수로 상위 40개 밖이면 통합 근거 점수만 받는다. 이 창은 D 확증에 쓴 동결 코드(동결 커밋 cd130e7, 실행 커밋 2ecb064)에 들어 있었으나 설계 D 문서에는 명시되지 않았다. 확증 이후 개발 탐색에서 이 창이 자기 간선 제거와 상호작용함을 확인했다(4.4절). 통합 근거를 받은 창 밖 질환의 소견 점수까지 읽는 채점(이하 확장 채점)은 세 그래프 모두에서 개발 매크로 Top-10/Top-1을 낮췄다. 동결 그래프는 30.56/12.64%에서 30.30/12.37%로, v2 UMLS 사전 그래프는 34.60/10.21%에서 32.82/9.65%로, 같은 사전 그래프의 자기 간선 제거판은 34.50/14.24%에서 34.40/13.26%로 낮아졌다. 벤치마크별 방향이 일관된 것은 v2 UMLS 사전 그래프뿐이다(확장 채점에서 높아짐/낮아짐 Top-1 0/7, 부호검정 p = .016; Top-10 1/8, p = .039). 동결 그래프와 자기 간선 제거판은 벤치마크별로 방향이 갈렸다(동결 그래프 Top-10 3/5, p = .73, Top-1 2/4, p = .69; 자기 간선 제거판 Top-10 4/5, p = 1.0, Top-1 2/6, p = .29). 동결 그래프에서 확장 채점은 DDXPlus(+0.97%p), RAMEDIS(+3.21%p), SymCat(+1.19%p)의 Top-10을 오히려 높였다(모두 Holm p < .05). 그래프 단독 순위화의 창(40)은 동결 설정의 일부이므로 D 확증에서 그대로 썼고 E 확증에서도 바꾸지 않았다. 결합 안의 그래프 목록 창은 E에서 40으로 맞췄다(3.7절).

**정답 판정과 지표.** 후보는 그래프의 전체 질환 43,575개다. 벤치마크의 정답 질환 목록은 후보로 쓰지 않고 정답 대조에만 쓴다. 정답 식별자(UMLS CUI, OMIM, Orphanet, MONDO)는 교차참조로 질환 노드 키에 사상한다. 주 판정 lineage는 예측 질환이 정답 질환이거나 질환 IS_A 관계로 정답의 하위 3단계 이내에 있으면 정답으로 인정한다. 정답의 상위 질환은 정답으로 인정하지 않는다. 보조 판정 strict는 정답 키와 정확히 같을 때만 인정한다. 지표는 Top-1, Top-3, Top-10(상위 k 안에 정답이 있는 환자 비율)이다. 형식은 RareBench의 Recall@k와 같고, 정답 인정 기준은 위의 lineage/strict를 따른다. 판정은 Top-10을 중심으로 하고 Top-1을 함께 보고한다. 정답이 질환 노드로 사상되지 않거나 사상된 소견 개념이 없는 환자는 평가에서 뺀다. 예를 들어 D 확증의 Phenopacket-new는 369증례 중 261명을 평가했다.

### 3.6 벤치마크와 평가 프로토콜

**벤치마크.** 개발 세트는 9개다(Table 4). DDXPlus는 test 분할에서 시드 0으로 3,000명을 뽑았다. 이 표본에는 49개 질환 중 48개가 있다(세기관지염 환자가 없다). 희귀질환은 RareBench의 RAMEDIS(624명), LIRICAL(370명), HMS(88명), MME(40명), 북경협화병원 실제 증례 PUMCH_L(988명)과 PUMCH-ADM(75명), 출판 증례 모음 Phenopacket-store 0.1.27(10,310명)을 썼다. SymCat은 801개 질환마다 시드 0으로 10명씩 생성하고, 소견이 뽑히지 않은 표본과 정답 사상이 안 되는 질환을 빼 7,481명을 썼다. 입력 형식은 DDXPlus가 값이 있는 문진 코드, SymCat이 증상 목록, 나머지 7개가 HPO 표현형이다. PUMCH_L과 PUMCH-ADM은 PhenoBrain 시험 증례 묶음(Zenodo, doi:10.5281/zenodo.10774650)으로 CC BY 4.0 하에 배포된다.

벤치마크 입력은 결정론적 어휘 사상으로 개념 집합이 된다. DDXPlus는 기본값 응답을 근거에서 빼고, 통증 부위 응답은 부위별 통증 개념으로 합성한다. 이 규칙은 '여행 안 함' 응답(trav1_@_N)이 DDXPlus validate 환자의 93%에서 'Travel History'로 사상되던 결함을 고친 것이다(4.1절). 통증의 성격, 강도, 방사 부위, 위치가 분명한 정도, 생긴 속도를 묻는 문항은 기본값이 아닌 응답이면 값과 관계없이 통증 개념(Pain, C0030193)으로만 옮긴다. Phenopacket-store 증례는 관찰된 소견의 HPO 코드만 쓰고, 소견별 발병 시점은 쓰지 않는다. DDXPlus 정답은 영문 질환명의 UMLS 정확 일치를 우선한다. 이때 질환 의미유형만 보고, SNOMED > MeSH > MedlinePlus > NCI 순으로 고르며, 5자 이하 대문자 약어는 제외한다. 정확 일치가 없으면 ICD-10 경로를 쓴다. 49개 질환 중 31개가 이름 일치로, 18개가 ICD-10 경로로 사상되었다. 희귀질환 정답은 OMIM·Orphanet 식별자를 UMLS 또는 MONDO 교차참조로 사상한다.

DDXPlus 개발 환자 3,000명 중 15.3%(458명)는 입력 소견에 정답 질환의 CUI 자체가 있다. 정답의 상위 질환 개념까지 세면 22.7%다. 458명에서 그 소견이 나온 문항은 본인 과거력 202명, 증상 문항 106명, 가족력 95명, 노출 등 55명이다. 통합 근거(3.5절)는 이런 소견을 점수에 반영한다.

Table 4. Benchmarks and their roles

| Benchmark | Role | Encoding | Source type | Patients evaluated | Gold-disease clusters | Sampling |
|---|---|---|---|---|---|---|
| DDXPlus (test) | Development | Coded questionnaire answers with values | Synthetic (49-disease generator) | 3,000 | 48 | Seed 0 |
| RareBench RAMEDIS | Development | Phenotype terms (HPO) | RareBench subset | 624 | 74 | All |
| RareBench LIRICAL | Development | Phenotype terms (HPO) | RareBench subset | 370 | 252 | All |
| RareBench HMS | Development | Phenotype terms (HPO) | RareBench subset | 88 | 46 | All |
| RareBench MME | Development | Phenotype terms (HPO) | RareBench subset | 40 | 17 | All |
| PUMCH_L | Development | Phenotype terms (HPO) | Real hospital cases (PUMCH) | 988 | 100 | All |
| PUMCH-ADM | Development | Phenotype terms (HPO) | Real hospital cases (PUMCH) | 75 | 16 | All |
| Phenopacket-store 0.1.27 | Development | Phenotype terms (HPO) | Published case reports | 10,310 | 772 | All |
| SymCat | Development | Symptom list | Synthetic (801-disease probability table) | 7,481^a^ | 760 | 801 diseases × 10, seed 0 |
| DDXPlus (test) | Design D confirmation^b^ | As above | Synthetic | 3,000 | — | Seed 101, disjoint from development |
| Phenopacket-new | Design D confirmation^b^ | Phenotype terms (HPO) | Published cases added after 0.1.27 (main@85ffd78) | 261 of 369^c^ | — | All new cases minus 36 with a source PMID in the corpus |
| SymCat | Design D confirmation^b^ | Symptom list | Synthetic | 7,482 | — | 801 diseases × 10, seed 101 |
| DDXPlus-E (test) | Design E confirmation^d^ | As above | Synthetic | 3,000 | 49 | Seed 202 from 128,529 rows outside both earlier samples |
| SymCat-E | Design E confirmation^d^ | Symptom list | Synthetic | 7,463^a^ | 760 | 801 diseases × 10, seed 202 |

^a^ Samples with no drawn symptom and diseases without a gold-label mapping were excluded.
^b^ Sealed at commit 31122e2 and evaluated once on 2026-10-13. These sets have since been opened and are used only as development evidence in design E.
^c^ Patients without a mapped gold key or without mapped findings are not evaluated. Of the 369 cases, 264 have a mapped gold key (37 source disease IDs, 50 gold keys), and 261 of these also have mapped findings (at most 34 cases per disease).
^d^ Sealed at commit 3b10c36 and evaluated once at freeze commit 49901bc (2026-10-19 23:13 to 2026-10-20 01:34, one attempt). New patients drawn from the development diseases.

**설계 D 프로토콜.** 설계 D 이전의 탐색 단계(v1)에서 세 가지 문제가 있었다. 첫째, 그래프 초기화가 일부 실패해 이전 빌드의 노드와 간선이 남은 그래프로 시험 분할을 두 번 평가했고, 두 평가는 무효로 처리했다(4.1절). 둘째, DDXPlus validate로 고른 설정이 희귀질환 벤치마크로 전이되지 않았다. 셋째, 같은 문헌의 BM25가 9개 시험 세트 중 6개에서 그래프보다 높았다. 결과를 본 뒤 주장을 바꾸지 않도록, 설계 D는 목표와 판정을 먼저 고정하고 개발과 확증을 분리했다.

개발 데이터는 DDXPlus validate 3,000명과 v1에서 이미 열람한 위의 9개 세트다. 설정 선택 규칙 D3는 9개 개발 세트의 매크로 lineage Top-10이 가장 높은 단일 설정을 고른다. Top-10이 같으면 Top-1로 고른다. 벤치마크별 조정은 하지 않는다. 비교 시스템도 같은 규칙으로 각자의 설정을 다시 고른다.

확증 세트는 봉인 커밋 31122e2로 고정했다. 구성은 세 가지다. (1) DDXPlus test 중 개발 표본과 겹치지 않는 3,000명(시드 101). (2) Phenopacket-store main@85ffd78(2026-10-07)에서 0.1.27 이후 추가된 405증례 중 원출처 PMID가 코퍼스에 있는 36증례를 뺀 369증례. (3) SymCat 시드 101(7,482명). 확증 로더는 환경 변수 CONFIRM_UNSEAL=1이 없으면 실행을 거부한다.

목표는 네 가지이고 제약이 하나 있다.
- T1(주): 그래프 단독 > BM25. 개발 9개 중 8개 이상에서 우세(부호검정 p < .05)해야 한다. 그리고 확증 3개 각각에서 Top-10 차이의 대응 부트스트랩 95% CI 하한이 −1%p보다 커야 하고(비열등), 그중 2개 이상에서 Holm 보정 후 유의하게 우세해야 한다.
- T2(흔한 질환 정확도 목표): 확증 DDXPlus에서 Top-10 ≥ 40%, Top-1 ≥ 18%.
- T3: 개발 PUMCH_L·PUMCH-ADM의 Top-10이 같은 랭커에 HPO 주석을 넣은 값 이상. 기준값은 처음에 validate로 고른 HPO 주석 설정의 값(53.6%, 53.3%)으로 적었다. D3에 따라 HPO 주석 설정을 개발 세트에서 다시 고른 뒤에는 그 값(53.2%, 57.3%)으로 판정했다.
- T4: 그래프+BM25 결합이 확증 매크로 Top-10에서 그래프 단독과 BM25 단독보다 유의하게 높음.
- 제약(출처 불변식): 모든 간선이 근거 문장과 문헌에 연결됨. 적재 시 검사는 모든 소견 간선에 근거 언급(Mention)이 있는지 본다.

판정 결과는 충족과 미달을 모두 그대로 보고하고, T1이 미달이면 그 사실이 주 결과라고 미리 적었다. 동결 설정(frozen_config_D.json)은 커밋 cd130e7에 넣었고, 확증은 비교군 실행 스크립트만 더한 커밋 2ecb064에서 실행했다(설정 변경 없음). 확증은 예정일(2026-10-16)보다 앞당겨 2026-10-13에 한 번 실행했다. 사전에 지정한 PPR 부가 가설 H1(같은 그래프에서 개인화 PageRank가 1-홉 집계보다 높은가)과 H4(HPO 주석·PrimeKG 그래프에서도 같은가)를 같은 확증 세트에서 평가했다. 비교 지식원은 주 시스템 실행 뒤 별도 스크립트로 같은 세트에서 한 번 평가했다.

**설계 E 프로토콜.** 설계 E는 D 확증 이후의 개발 탐색에서 고른 시스템을 새 봉인 세트로 한 번 확증한다. D 확증에서 T1과 T2는 미달이었다(4.2절). 그 뒤 같은 9개 개발 세트에서 그래프 변형 32개를 비교했다(4.3–4.4절). 32개는 동결 그래프의 재현(base_nm)과, 수집 깊이, 추출·연결 방식, 빌더 규칙(등위 분해, 한정어 제외, 자기 간선 제거 등) 중 하나 이상을 바꾼 그래프 31개다. 문헌으로 나누면 v2 문헌 그래프가 base_nm을 포함해 18개, v2+v3 문헌 그래프가 14개다. v2+v3 문헌 그래프 14개에는 PMC 본문 전체를 읽는 판 2개(v3umlsfull_ns, v3dictfull_ns)가 들어 있다. 이 32개 가운데 S-mix를 주 시스템으로 골랐다. S-mix는 같은 세트에서 최댓값으로 고른 시스템이므로 개발값에는 선택 낙관이 들어 있다. D 확증 결과는 바꾸지 않고 주 결과의 하나로 보고한다. 설계 E는 D의 목표와 판정을 대체하는 새 확증 설계이며, 판정 기준과 개발값으로 본 예상을 실행 전에 저장소에 커밋했다(5ce9c3d 등).

확증 세트 E는 data/confirm_e/MANIFEST.json에 고정했다(2026-10-19 생성, 봉인 커밋 3b10c36). DDXPlus-E는 DDXPlus test 134,529행에서 개발 표본 3,000행과 D 확증 표본 3,000행을 뺀 128,529행 중 시드 202로 뽑은 3,000명이다. 매니페스트에는 뽑은 행의 번호와 각 행의 sha256을 기록했고, 개발 표본·D 확증 표본과 겹치는 행은 0개다. SymCat-E는 801개 질환마다 시드 202로 10명씩 생성한다(개발은 시드 0, D 확증은 시드 101). 두 세트는 개발 세트와 같은 질환 집합에서 새로 뽑은 환자다. DDXPlus-E 표본의 질환 중 개발 표본에 없던 것은 세기관지염(1명)뿐이다. 합성 환자를 개념으로 사상하면 일부 환자는 입력이 개발 환자와 같아진다. 열람된 D 확증 세트로 재면, (정답, 소견 개념 집합)이 개발 환자 한 명과 정확히 같은 환자가 DDXPlus 3,000명 중 791명(26.4%), SymCat 7,482명 중 1,151명(15.4%)이었다. E 세트에서는 이 비율을 재지 않았다. 확증 전에는 세트가 봉인되어 있었고, 확증 뒤에도 따로 재지 않았다. 따라서 E 확증은 같은 질환에서 새로 뽑은 환자 표본에서 결과가 재현되는지까지만 검사하며, 그 표본의 일부는 입력이 개발 환자와 같은 환자다.

희귀질환 확증 세트는 없다. Phenopacket-store main에는 2026-10-19 현재 D 확증 세트의 기준 커밋 85ffd78 이후 커밋이 없고, 스냅샷을 이날로 고정했으므로 새 증례가 0건이다. RareBench와 PUMCH 증례는 모두 열람되었다. D 확증의 Phenopacket-new 증례도 시간 분할 분석(3.9절)의 2026 스냅샷에 들어가 E의 그래프 시스템 5개(S-D, S-mix, S-llm, S-dict, S-union) 모두에서 평가되었다. 따라서 설계 E 시스템(S-mix 등)의 희귀질환 결과는 모두 개발 수준이다. 확증 수준의 희귀질환 결과는 D 확증의 Phenopacket-new(설계 D 시스템과 비교 지식원)뿐이다.

봉인 장치는 코드로 강제한다. E 확증 로더(load_confirm_e)는 CONFIRM_E_UNSEAL=1이 없으면 거부한다. 개발 로더는 DDXPlus-E의 행과 SymCat 시드 202를 거부한다.

동결 시스템은 7개다(Table 5). 그래프 시스템 5개는 3.5절의 동결 채점을 공유하고 γ만 다르다.
- S-mix(주): v2+v3 문헌, gemma 구절과 구절 안 UMLS 사전 재연결(3.3절), 자기 간선 제거(v3llmd_ns), γ 0.5.
- S-D: 설계 D의 동결 시스템(v2 문헌 gemma 그래프, base_nm 빌드), γ 0.
- S-llm: 같은 v2+v3 문헌의 LLM 추출만(v3llm_ns), γ 1.0.
- S-dict: 같은 문헌의 UMLS 사전만(v3umls_ns), γ 0.5.
- S-union: LLM 항목과 모든 문장의 UMLS 사전 항목의 합집합(v3unionu_ns), γ 0.5.
- S-bm25: 같은 v2+v3 문헌의 BM25(일반 문헌 규칙도 그래프와 같게).
- S-comb: S-mix 순위와 S-bm25 순위의 역순위 결합. 가중 w는 {0.25, 0.5, 0.75}에서 D3 규칙으로 고르고, 한쪽만 쓰는 0과 1은 후보에서 뺀다.

S-dict는 같은 문헌(v2·v3 모두 PMC 논문은 임상 절 12,000자, 도서 장은 24,000자)에서 가장 높은 사전 그래프다. v3 PMC 논문의 본문 전체를 읽는 v3umlsfull_ns(개발 매크로 Top-10 43.08%)는 문헌이 달라 비교군에서 뺐다. 그래프 시스템은 γ만 다시 고른 반면, S-bm25는 질의 방식·검색 깊이·집계 방식의 전체 격자를 다시 고른다. 조정 예산이 BM25 쪽에 더 많으므로 이 설계는 E-T3에 보수적이다.

목표는 다섯 가지다. 판정 세트는 DDXPlus-E와 SymCat-E이고, 판정 기준은 lineage, 후보는 43,575개 질환 전체다.
- E-T1(주): S-mix > S-D. 두 세트 각각에서 Top-10 차이의 환자 대응 부트스트랩 95% CI 하한 > 0이고, 정확 McNemar 검정의 Holm 보정 p < .05여야 한다(족 = 두 세트).
- E-T2(S-llm 비교는 주, S-dict·S-union 비교는 부): S-mix − S-llm, S-mix − S-dict, S-mix − S-union 각각의 매크로 Top-10 차이에서 95% CI 하한 > −1%p(비열등). 세 비교는 따로 판정한다.
- E-T3: S-mix 대 S-bm25. 두 세트 각각에서 Top-10 차이 CI 하한 > −1%p이고, 1개 이상에서 Holm 유의 우세.
- E-T4: S-mix 단독으로 DDXPlus-E Top-10 ≥ 40%, Top-1 ≥ 18%(D의 T2 기준 유지).
- E-T5: S-comb − S-mix와 S-comb − S-bm25 둘 다 매크로 Top-10 차이 CI 하한 > 0.

설계 E의 목표 달성(E0)은 E-T1 충족 그리고 E-T2의 S-llm 비교 충족으로 정의한다. 주 목표는 E-T1과 E-T2(S-llm 비교)이고, 나머지는 보고용 부 목표다. 목표 사이의 다중성은 보정하지 않는다.

개발값으로 본 예상도 실행 전에 커밋했다. E-T1은 충족을 예상했다(개발 S-mix − S-D Top-10: DDXPlus +7.6%p, SymCat +2.7%p). E-T2는 S-llm 비교의 충족(+1.9%p, +1.9%p)과 S-dict·S-union 비교의 미달(DDXPlus −1.6%p·−3.0%p, SymCat −1.1%p·−1.3%p)을 예상했다. E-T4는 Top-10이 기준 경계(개발 40.6%), Top-1이 미달(개발 11.8%)일 것으로 예상했다. E-T3은 충족을 예상했다(개발 S-mix − S-bm25 Top-10: DDXPlus +20.9%p [+19.2, +22.7], SymCat +3.2%p [+2.5, +4.0], 둘 다 Holm 유의). E-T5는 미달을 예상했다. 개발 DDXPlus·SymCat의 매크로 Top-10 차이에서 S-comb − S-mix는 −0.02%p [−0.54, +0.49](DDXPlus −0.8%p, SymCat +0.8%p)로 CI 하한이 0보다 낮았다. S-comb − S-bm25는 +12.04%p [+11.15, +12.94]로 CI 하한이 0보다 높았다. E-T3과 E-T5의 예상은 S-bm25와 S-comb를 개발 세트에서 처음 고른 뒤 커밋 61e3118에 기록했다. 그 뒤 첫 S-comb 선택이 비교 인스턴스에 S-mix가 아닌 그래프가 적재된 상태에서 실행된 것을 확인했다(3.7절). 그래서 S-comb를 다시 고르고, E-T5의 예상을 커밋 0cb1c59에서 위의 값으로 고쳤다. 고치기 전의 E-T5 예상도 미달이었다(S-comb − S-mix −1.67%p [−2.38, −1.01]). 두 커밋은 모두 동결 커밋보다 앞선다.

절차는 다섯 단계다. (1) 설계 E 커밋(5ce9c3d, 판정 기준과 E-T1·E-T2·E-T4 예상 포함). (2) 확증 세트 봉인, 로더 거부 장치, 매크로 통계 코드 커밋(3b10c36). (3) 확증 실행 스크립트 커밋, S-bm25 색인과 개발 선택, S-comb 가중 선택, 재연결 항목에 원 LLM 구절을 남기도록 고친 S-mix의 재빌드(지문 불변 확인), Mention 빌드 검사. (4) 동결 커밋(frozen_config_E.json: 시스템별 지문, 색인 sha256, 설정). (5) 확증 1회 실행. 확증 실행 스크립트와 동결 기록 스크립트는 a7d0680(2026-10-19)에 커밋했다. 개발 선택은 S-bm25가 질의 전체·깊이 1,000·역순위 결합 집계(개발 매크로 Top-10/Top-1 27.35/11.04%), S-comb가 w = 0.25(46.72/19.46%)였다. 원 LLM 구절(src_t)을 남기도록 고쳐 다시 빌드한 S-mix는 빌드 지문(15ba677…)이 바뀌지 않았고, 적재된 HAS_FINDING 간선은 5,221,285개였다. Mention 빌드 검사는 통과했다(아래 동결 검증). 동결 커밋은 49901bc(2026-10-19)다. 실행이 중단되면 같은 동결 입력으로 결과 파일이 없는 시스템만 다시 실행하고, 모든 시도를 기록하며, 시스템은 바꾸지 않기로 정했다. 확증은 동결 커밋에서 2026-10-19 23:13에 시작해 2026-10-20 01:34에 끝났다. 시도는 1회였고 중단은 없었다.

동결 검증도 실행 절차에 넣었다. 모든 그래프 시스템은 확증 실행 직전에 같은 비교 인스턴스에 다시 빌드한다. 빌드 요약의 지문(순위화에 쓰이는 모든 행의 해시)과 적재된 HAS_FINDING 수가 동결 기록과 같을 때만 평가한다. Disease 노드 수가 후보 질환 수(43,575)와 다르면 빌더가 멈춘다. S-bm25는 색인 파일들의 sha256을 동결 기록과 대조한다. 확증 실행에서 그래프 시스템 5개는 모두 지문과 HAS_FINDING 수가 동결 기록과 같았다. S-bm25 색인 파일의 sha256도 모두 동결 기록과 같았다. 출처 불변식 제약은 Mention을 넣어 별도 인스턴스에 빌드한 S-mix로 동결 전에 확인했다. 이 빌드는 그래프 검사(모든 HAS_FINDING 간선에 근거 언급 1개 이상, Disease 노드 43,575개)를 통과했다. 근거 언급은 11,582,169개였다. 빌드 지문(15ba677…)과 HAS_FINDING 수(5,221,285개)는 Mention 없는 빌드와 같았다. 개발 DDXPlus 3,000명 모두에서 정답 질환의 Top-10 내 위치(lineage)도 Mention 없는 빌드와 같았다.

확증 결과는 4.6절과 Table 16에 보고한다. 개발 세트와 같은 질환에서 새로 뽑은 환자에서 E-T1, E-T2의 S-llm 비교, E-T3이 충족되었다. 주 목표 두 개가 모두 충족되어 설계 E의 목표 달성(E0) 조건을 만족했다. E-T2의 S-dict·S-union 비교, E-T4, E-T5는 미달했다. E-T4에서 DDXPlus-E Top-10은 41.07%로 기준(40%) 이상이었으나 Top-1이 12.43%로 기준(18%)에 못 미쳤다. 7개 판정은 모두 실행 전에 커밋한 예상과 같았다. 판정에 쓰지 않은 보조 지표(3.8절)는 주장의 범위를 좁힌다. DDXPlus-E는 정답 질환이 49개뿐이어서, E-T1 Top-10 차이(+6.5%p, 환자 대응 CI [+4.6, +8.5])의 정답 질환 군집 CI는 [−6.1, +19.3]으로 0을 포함했다. 정답 질환이 760개인 SymCat-E에서는 군집 CI가 [+1.0, +3.9]였다. S-mix − S-D의 Top-1 차이는 DDXPlus-E +0.7%p [−0.6, +2.0], SymCat-E −0.8%p [−1.2, −0.4]였다. 따라서 E 확증으로는 질환 수준의 일반화와 Top-1 개선을 주장하지 않는다.

### 3.7 비교 시스템

**같은 문헌의 텍스트 검색(BM25, 설계 D).** v2 코퍼스에서 일반 문헌과 비인간 질환을 뺀 문서 122,661편을 약 1,000자 단위로 나눠 청크 655,965개를 색인했다. 이 문서 수는 D 동결 그래프에 근거가 적재된 문서 86,276편보다 많아 비교 시스템에 유리하다. 토큰화에는 영어 불용어 제거와 영어 어간 추출(PyStemmer)을 썼다. 환자 소견 개념은 UMLS 영어 선호명으로 바꿔 질의하므로 그래프와 같은 정보만 입력한다. 검색된 청크의 점수는 그 문헌에 연결된 질환들(그래프와 같은 질환–문헌 연결)로 모아 열린 집합 순위를 낸다. 질의 방식(소견 전체를 한 질의로 / 소견별 질의), 검색 깊이 {100, 1,000}, 집계 방식 {최댓값, 합, 역순위 결합}의 12개 조합에서 D3 규칙으로 골랐다. 선택된 설정은 소견별 질의, 깊이 1,000, 최댓값 집계였다(개발 매크로 Top-10/Top-1 24.46/8.75%).

**그래프+BM25 결합(설계 D의 T4).** 그래프 순위 r_G와 BM25 순위 r_B를 역순위 결합한다.

$$\mathrm{score}(d)=\frac{1-w}{60+r_G(d)}+\frac{w}{60+r_B(d)}$$

w ∈ {0, 0.25, 0.5, 0.75, 1}에서 D3 규칙으로 w = 0.25를 골랐다(개발 매크로 32.32/12.72%). D의 결합에서 두 순위는 각 시스템의 상위 200개 질환이었다. D의 결합 코드는 그래프 목록을 상위 200개로 냈으므로 그 안의 소견 점수 창이 200 × 4 = 800이었다. 따라서 결합 안의 그래프 순서는 그래프 단독 평가(창 40)와 달랐다. w = 0, 즉 그래프 목록만으로 낸 개발 매크로는 30.22/12.37%로, 그래프 단독 30.56/12.64%와 다르다. E의 S-comb는 그래프 목록을 그래프 단독 평가와 같은 창 40으로 낸다(커밋 a7d0680). 그래서 S-comb의 그래프 목록은 창 안의 40개 질환과 창 밖에서 통합 근거를 받은 질환으로 이루어지고, BM25 목록은 상위 200개 그대로다. 결합 결과에서 그래프가 순위를 낸 질환은 그래프의 근거 경로를 그대로 가진다.

**같은 랭커, 다른 지식원.** 정확도 차이가 순위 방법에서 오는지 지식에서 오는지 가르기 위해, 기존 지식원을 같은 스키마와 같은 후보 질환 집합(43,575개)으로 별도 Neo4j 인스턴스에 적재하고 같은 Cypher 질의로 평가했다. HPO 주석(phenotype.hpoa 2026-02-16판)은 간선 393,498개, 소견이 있는 질환 12,251개다. PrimeKG의 disease_phenotype_positive 관계는 간선 747,705개, 질환 10,898개다. MEDLINE MeSH 공출현(NLM 2024 요약, LLM 없음, 지지 수 = 공동 색인 인용 수)은 간선 3,798,931개, 질환 3,302개다. 각 지식원의 채점 설정은 D3 규칙으로 다시 골랐다(Table 5). 이 지식원들에는 근거 문장(Mention)이 없다. HPO 주석과 PrimeKG는 주 시스템 점수에 들어가지 않으며 지식 격차의 참조로만 제시한다. 희귀질환 증례와 HPO 주석은 같은 출판 증례에서 나온 경우가 많다. Phenopacket-store 0.1.27에서 정답 질환이 HPO 주석에 있는 증례의 74.6%가, 그 증례를 만든 논문을 근거로 인용하는 주석을 가진다 [ref: Jiang et al. 2026]. 따라서 희귀질환 비교는 HPO 주석 쪽에 유리하다.

**HippoRAG식 개인화 PageRank(PPR).** 같은 그래프를 무방향 가중 그래프로 내보내 환자 소견 개념을 시드로 PPR을 계산하고, 질환 노드의 확률을 점수로 쓴다. 간선은 HAS_FINDING(가중 n_docs^β, 주제 문서가 없으면 γ를 곱함), 개념 IS_A와 질환 IS_A(가중 h), IS_DISEASE(가중 1)다. 허브 보정은 질환 점수를 차수^η로 나눈다. 격자는 재시작 확률 {0.15, 0.3, 0.5, 0.7}, β {0, 0.5, 1}, γ {0, 1}, h {0, 0.5, 1}, η {0, 0.5, 1}, 시드 가중 {균등, IDF}, 시드 질환 제외 여부였고, 선택 규칙은 D3다. S-D 그래프에서는 재시작 0.3, β 1, γ 0, h 0.5, η 0, 균등 시드, 시드 질환 제외가 선택되었다(개발 매크로 23.57/6.33%). HPO 주석·PrimeKG 그래프에서는 재시작 확률만 탐색해 각각 0.5와 0.7을 골랐다. scipy 거듭제곱 계산과 Neo4j GDS pageRank의 질환 Top-10은 PUMCH_L 20명에서 100% 일치했다(간선 없는 고립 시드 제외 조건).

**설계 E의 비교군.** S-llm, S-dict, S-union은 S-mix와 같은 v2+v3 문헌, 같은 빌더 규칙, 같은 동결 채점을 쓰고 추출 방식과 γ(각자 D3로 선택)가 다르다. S-D는 문헌 깊이와 추출 방식이 모두 다른 설계 D의 동결 시스템이다. S-bm25는 위 BM25를 v2+v3 문헌으로 다시 색인하고 같은 12개 조합에서 다시 골랐다. 색인은 문서 1,001,916편, 청크 3,074,271개다. 선택된 설정은 질의 전체·깊이 1,000·역순위 결합 집계다(개발 매크로 27.35/11.04%). S-comb의 w는 개발 세트에서 0.25를 골랐다(개발 매크로 46.72/19.46%). 첫 가중 선택은 비교 인스턴스에 S-mix가 아닌 그래프가 적재된 상태에서 실행되어 폐기했고, 이 값은 적재된 간선 수로 S-mix를 확인한 뒤 다시 고른 것이다. MedCPT 밀집 검색은 v1 탐색 단계에서만 평가했다(DDXPlus validate로 설정 선택, 시험 분할 평가). 설계 D에서는 D3 재선택 대상으로 적었으나 재선택하지 않았고, D·E 확증의 비교군에 넣지 않았다.

Table 5. Systems and frozen configurations

| System | Literature | Extraction or knowledge source | Configuration | Edges | Diseases with findings | Concepts | Docs^a^ | Dev macro Top-10 / Top-1 (%) |
|---|---|---|---|---|---|---|---|---|
| **S-mix (main)** | v2 + v3 | gemma spans; spans the phrase table cannot link named by UMLS finding strings inside the span (v3llmd_ns) | Builder: split-coord, hpo-names, topic-genes, no-self; γ 0.5 | 5,221,285 | 33,537 | 85,761 | 711,350 | 46.11 / 19.66 |
| S-llm | v2 + v3 | gemma spans linked by the phrase table (v3llm_ns) | Same builder; γ 1.0 | 3,672,106 | 33,123 | 76,487 | 589,666 | 44.90 / 19.30 |
| S-dict | v2 + v3 | UMLS finding-string dictionary on every sentence (v3umls_ns) | Same builder; γ 0.5 | 14,506,788 | 34,356 | 95,608 | 984,061 | 42.88 / 18.11 |
| S-union | v2 + v3 | S-llm items + S-dict items (v3unionu_ns) | Same builder; γ 0.5 | 15,396,388 | 34,359 | 115,224 | 986,098 | 43.77 / 17.59 |
| S-D (design D frozen) | v2 | gemma spans linked by the v2 phrase table (base_nm) | No builder flags; γ 0 | 832,671 | 27,309 | 46,130 | 86,276 | 30.56 / 12.64 |
| S-bm25 | v2 + v3 | BM25 over the same documents | Joint query, depth 1,000, RRF | — | — | — | 1,001,916 | 27.35 / 11.04 |
| S-comb | v2 + v3 | Reciprocal-rank fusion of S-mix and S-bm25 | w 0.25 | — | — | — | — | 46.72 / 19.46 |
| BM25 (design D) | v2 | BM25 over 655,965 chunks of 122,661 documents | Per-finding query, depth 1,000, max | — | — | — | 122,661 | 24.46 / 8.75 |
| Graph + BM25 (design D) | v2 | Reciprocal-rank fusion of S-D and BM25 | w 0.25 | — | — | — | — | 32.32 / 12.72 |
| PPR (design D) | v2 | S-D graph | Restart 0.3, β 1, γ 0, h 0.5, η 0, uniform seeds | — | — | — | — | 23.57 / 6.33 |
| HPO annotations, 1-hop | — | phenotype.hpoa 2026-02-16 | α 0.2, μ 2, δ 0.5, u 2, β 0 | 393,498 | 12,251 | — | — | 52.45 / 30.37 |
| PrimeKG, 1-hop | — | disease_phenotype_positive | α 0.5, μ 2, δ 0.3, u 1, β 0 | 747,705 | 10,898 | — | — | 33.52 / 15.18 |
| MeSH co-occurrence, 1-hop | — | MEDLINE co-occurrence 2024 (no LLM) | α 0.35, μ 0, δ 0.3, u 1, β 0 | 3,798,931 | 3,302 | — | — | 9.13 / 2.61 |

All literature-graph systems of designs D and E share the frozen scoring of design D (α 0.35, μ 2, δ 0.3, u 2.0, β 0.5); only γ is re-selected per graph. Every graph has 43,575 Disease nodes. Development macro = mean over the nine development sets. S-mix had the highest development macro Top-10 of the 32 graph variants compared on the same sets (the base_nm reproduction of S-D plus 31 graphs that change collection depth, extraction or linking, or builder rules; 18 on v2 literature including base_nm, and 14 on v2 + v3 literature including two full-text builds), so its development value carries selection optimism. HPO annotations and PrimeKG are separate comparator systems and never enter the main system's scores.
^a^ Document units that support at least one edge; a table stored as its own document unit (TBL + PMCID) is counted apart from its article. For BM25, documents indexed.

### 3.8 통계

**세트별 대응 비교.** 같은 환자에 대한 두 시스템의 Top-k 차이는 환자 단위 대응 부트스트랩(2,000회, 시드 0, 백분위 95% CI)으로 구간을 낸다. 유의성은 두 시스템의 적중이 갈린 환자에 대한 양측 정확 McNemar 검정으로 보고, 판정 대상 세트를 족으로 삼아 Holm 보정한다(개발 9개, D 확증 3개, E 확증 2개). 판정 세트를 지정한 비교는 그 세트가 두 결과에 같은 환자 수로 들어 있지 않으면 멈춘다.

**정답 질환 군집 CI.** 여러 벤치마크에서 환자가 소수 질환에 몰려 있다. DDXPlus 개발 표본은 3,000명이 48개 질환, RAMEDIS는 624명이 74개 질환, PUMCH-ADM은 75명이 16개 질환이다. 환자 단위 CI는 같은 질환들에서의 차이만 말한다. 그래서 개발 비교에는 같은 정답 질환을 가진 환자를 통째로 다시 뽑는 군집 부트스트랩 CI(2,000회)를 함께 낸다.

**방향 일관성.** 개발 9개 세트에 걸친 방향 일관성은 차이 부호에 대한 양측 부호검정으로 본다. 차이가 0인 세트는 뺀다.

**매크로 비교.** 매크로 차이는 판정 세트 차이의 등가중 평균이다. CI는 세트별로 환자를 층화하고 두 시스템에 같은 재표본 인덱스를 쓰는 대응 부트스트랩 2,000회(시드 0, 백분위)로 낸다. p는 부트스트랩 분포에서 2 · min(P(Δ ≤ 0), P(Δ ≥ 0))이다. 이 계산(macro_compare)은 D 확증 T4의 세 수치를 그대로 재현한다: 결합 − 그래프 Top-10 +2.67%p [+1.81, +3.58], 결합 − BM25 Top-10 +5.10%p [+3.59, +6.67], 결합 − 그래프 Top-1 +0.96%p [+0.36, +1.54].

**단일 시스템 정확도.** 확증 결과에서 시스템 하나의 Top-k 95% CI는 환자 부트스트랩 1,000회(시드 0)로 낸다.

**개발 매크로.** 개발 매크로는 9개 세트 값(소수 둘째 자리까지 반올림한 값)의 평균이다. 이는 D3 선택에 쓴 계산과 같다. 세트별 표의 값은 환자별 순위에서 직접 반올림했다.

**비열등 한계와 보조 지표.** 비열등 한계는 D의 T1과 E의 E-T2·E-T3 모두 −1%p다. 설계 E는 대응 비교(E-T1·E-T2·E-T3·E-T5)의 Top-1·Top-3 차이, 정답 질환 군집 CI, 열람된 D 확증 세트와 개발 9개 세트에서의 같은 비교를 판정에 쓰지 않는 보조 지표로 정했다. 이 가운데 보고한 것은 Top-1 대응 차이, 정답 질환 군집 CI, 개발 세트 비교(표 12)다. Top-3는 시스템별 정확도(표 16a)로만 적고, 열람된 D 확증 세트의 비교는 계산하지 않았다. 단, E-T4는 S-mix 단독 Top-1(≥ 18%)을 판정한다.

### 3.9 누수 점검과 시간 분할

**원출처 제외.** v2 수집에서는 Phenopacket-store 0.1.27 증례의 원출처 PMID 1,733편을 코퍼스에서 뺐다. v3 수집에서는 그 뒤 추가된 증례의 원 논문까지 1,834편을 모두 뺐다. D 확증 세트에서는 원출처 PMID가 v2 코퍼스에 있던 신규 증례 36건을 뺐다(3.6절).

**데이터 기반 누수 점검.** RareBench 세트는 원출처 PMID를 제공하지 않는다. PMID를 뺀 Phenopacket-store에서도 같은 증례가 재보고나 리뷰의 표처럼 다른 문헌을 통해 남을 수 있다. 그래서 정확도를 보기 전에 고정한 기준으로 RareBench 4개 세트와 Phenopacket-store의 누수를 데이터로 점검한다. 환자마다 정답 질환의 근거로 그래프에 적재된 문서 가운데 환자 소견 개념을 가장 많이 포함한 문서 하나를 찾는다. 그 문서가 환자 개념의 50% 이상이면서 5개 이상을 포함하면, 그 환자를 원출처 증례일 수 있는 환자로 표시한다. 이 점검에는 근거 언급(Mention)이 있는 그래프가 필요하다. 점검은 구 시스템인 Qwen3 v2.1c 그래프(Qwen3-4B-Instruct-2507 추출, v2 문헌)에서만 했다. 그 결과 RAMEDIS·HMS·MME는 0명이었다. LIRICAL은 370명 중 8명이 표시되었고, 이들의 Top-10은 100%였으며, 이들을 뺀 Top-10은 11.6%(전체 13.5%)였다. Phenopacket-store는 10,310명 중 113명(1.1%)이 표시되었고, 이들의 Top-10은 84.1%였으며, 이들을 뺀 Top-10은 10.3%(전체 11.1%)였다. D 동결 gemma 그래프와 v2+v3 문헌 그래프(S-mix 포함)에서는 이 점검을 하지 않았다 [미실행: S-mix의 Mention 빌드에서 같은 점검 실행].

**시간 분할.** 회고 평가에서는 지식원이 평가 증례의 원 논문을 이미 근거로 쓰고 있을 수 있다. 그래서 지식원 시점 이후에 출판된 증례로 따로 평가한다. 이 분석은 사전 등록 확증이 아니라 개발 탐색이다.

- 2022 절단 그래프(_y2022): 출판연도가 2022년 이하인 문서의 근거만으로 간선을 다시 집계한다. 출판연도가 없는 문서도 뺀다. 간선마다 근거 문장과 문서가 저장되어 있으므로 재추출은 필요 없다. IDF, 질환별 원문 문자 수, 길이 정규화 상수는 절단 그래프에서 다시 계산된다. 원인 유전자 검색으로 찾은 문서와 원인 유전자 기호에 의한 주제 판정은 HPO 주석 2022-12-15판이 주석한 질환에만 쓴다. 2026년의 유전자–질환 연결이 2022 절단 그래프로 새지 않게 하기 위해서다.
- 증례: Phenopacket-store 0.1.27과 그 뒤 main에 추가된 증례 중 정답과 소견을 사상할 수 있는 10,603건이다. D 확증의 Phenopacket-new 증례도 여기에 들어간다. 원 논문(9편)이 v2 코퍼스에 들어 있는 32건은 모든 시간 분할 비교에서 빼고 10,571건을 쓴다. 증례 시점은 원 논문의 PubMed 출판일이며, 2023-01-01 이후이면 미래 증례다.
- 질환 상태: 2022-02 이후 HPO 주석 릴리스 36개에서 정답 OMIM을 처음 주석한 릴리스를 찾는다. 그 날짜가 2022-12-31보다 뒤이거나 2026-09 릴리스까지 주석되지 않았으면 '신규', 아니면 '기주석'이다. 정답 OMIM이 여럿이면 가장 이른 날짜를 쓴다.
- 네 묶음의 크기: 미래·신규 2,381건(263개 질환), 미래·기주석 1,106건(67개 질환), 과거·신규 1,091건(175개 질환), 과거·기주석 5,993건(405개 질환).
- 비교 기준: HPO 주석 2022-12-15판을 같은 1-홉 랭커(HPO 주석의 동결 설정)로 평가한 값과 2022 절단 S-D 그래프다. 문헌 그래프는 개발에서 고른 γ가 아니라 γ = 0과 γ = 1에서 모두 평가한다. 차이의 CI는 환자 대응 부트스트랩(2,000회, 시드 0)이다.
- 참고로 모든 문서를 쓰는 2026 스냅샷(누설 가능, 10,571건, γ = 0)도 함께 보고한다.

코퍼스 질의에는 2026년의 질환명과 동의어를 썼다. 따라서 2022 절단 그래프는 2022년 당시의 검색 가능성을 과대평가할 수 있다.

4.5.1절은 위 분석보다 앞서 수행한 시간 분할 분석(설계 T)의 결과도 보고한다. 이 분석은 설계 D의 동결 설정으로 v2 문헌 gemma 그래프의 연말 절단판 7개(2019–2025년)와 HPO 주석 스냅샷 6개를 위의 10,603증례(원 논문이 v2 코퍼스에 있는 32건 포함)에서 비교했다. 설계 T의 가설 H2는 '미래 증례에서 문헌 그래프가 HPO 주석보다 높다'였다.
