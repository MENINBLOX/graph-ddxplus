### 3.5 진단 질의

**순위화 질의.** 한 환자의 진단 순위는 Cypher 질의 두 개(소견 점수 질의와 통합 근거 질의)의 결과를 합쳐 낸다. 입력은 환자 소견 개념(UMLS CUI)의 집합 P다. 소견 점수 질의는 P의 각 개념을 개념 IS_A 관계로 한 단계 위까지 확장해 개념 집합 C(P)를 만든다. 확장된 개념 c의 가중 w_c는 c가 환자 개념 자체이면 1이고, 환자 개념의 직접 상위 개념으로만 들어오면 감쇠 δ다. 두 경로가 모두 있으면 큰 값을 쓴다.

질환 d의 소견 점수 s(d)는 d의 HAS_FINDING 간선 가운데 확장 개념에 닿는 간선의 기여를 더한 값이다.

$$s(d)=\sum_{c\in C(P),\ (d\to c)\in \mathrm{HAS\_FINDING}} w_c\cdot \mathrm{IDF}(c)\cdot n_{\mathrm{docs}}(d,c)^{\beta}\cdot g(d,c)$$

IDF(c) = ln(N / df(c))다. N은 소견 간선이 하나 이상 있는 질환 수이고, df(c)는 그중 c를 소견으로 가진 질환 수다. n_docs(d, c)는 간선을 지지하는 문서 수다. g(d, c)는 간선의 지지 문서 가운데 주제 문서가 하나라도 있으면 1이고, 없으면 γ(offtopic_w)다. 주제 문서는 제목이 그 질환의 이름 형태를 포함하는 문서다. topic-genes 빌드(S-mix, S-llm, S-dict, 순수 사전 그래프, S-union)에서는 제목에 그 질환의 원인 유전자 기호가 있는 문서도 주제 문서로 본다(3.2절). S-D는 이 유전자 규칙 없이 빌드했다.

소견 점수는 질환 프로필 크기로 길이를 정규화한다.

$$\mathrm{score}_F(d)=\frac{s(d)}{\left(n(d)+\mu\left(1-\dfrac{m(d)}{m(d)+\tilde m}\right)\bar L\right)^{\alpha}}$$

n(d)는 질환 d의 소견 개념 수이고, m(d)는 d에 대해 수집한 원문 문자 수다. m̃은 소견이 있는 질환들의 m(d) 중앙값이고, L̄은 그 질환들의 평균 n(d)다. μ 항(커버리지 스무딩)은 문헌이 적은 질환의 유효 길이를 늘려 그 질환의 과대평가를 줄이기 위한 항이다.

통합 근거는 환자 개념이 질환 노드이기도 한 경우를 다룬다. 환자 개념이 IS_DISEASE로 질환 d0을 가리키면, d0과 그 직접 하위 질환(질환 IS_A 한 단계)에 u · n_u(d)를 더한다. n_u(d)는 환자 개념 가운데 d 자신이거나 d의 직접 상위 질환인 개념의 수이고, u는 통합 근거 가중(unified_w)이다. IS_DISEASE 간선은 그 질환의 CUI가 그래프에 개념 노드로 있을 때만 생긴다. 그래서 자기 간선 제거(3.4절) 뒤 질환 자신의 CUI가 어느 질환의 소견 개념이나 그 상위 개념으로도 남지 않으면, 환자 소견에 그 CUI가 있어도 그 CUI에서 오는 소견 점수 기여와 통합 근거가 모두 생기지 않는다. 환자의 다른 소견에서 오는 점수는 그대로 남는다. 최종 점수는 score_F(d)와 u · n_u(d)의 합이다. 동점은 질환 ID 순으로 정렬한다.

**동결 매개변수.** 채점 매개변수는 설계 D의 개발 선택에서 정하고 동결했다. 선택은 v2 문헌 gemma 그래프(S-D)에서 좌표 탐색(최대 2회 통과, 한 통과에서 바뀐 값이 없으면 중단)으로 했다. 격자는 α {0.2, 0.25, 0.35, 0.5}, μ {0, 1, 2, 4}, δ {0, 0.3, 0.5, 0.7}, u {0, 0.5, 1, 2}, β {0, 0.5, 1, 2}, γ {1, 0.5, 0.25, 0}이다. 선택된 값은 α = 0.35, μ = 2, δ = 0.3, u = 2.0, β = 0.5다. 설계 D·E의 문헌 그래프 시스템(S-D, S-mix, S-llm, S-dict, S-union, 순수 사전 그래프, 개발 변형, 2022 절단 그래프)은 이 값을 그대로 쓴다. S-mix, S-llm, S-dict, S-union과 개발 변형은 γ만 {0, 0.25, 0.5, 1}에서 D3 규칙(3.6절)으로 다시 골랐다. 선택된 γ는 S-mix 0.5, S-llm 1.0, S-dict 0.5, S-union 0.5다. S-D의 γ 0은 위 좌표 탐색에서 고른 값이다. 순수 사전 그래프는 S-dict와 견주기 위해 S-dict의 γ 0.5를 그대로 썼다(3.10절). 2022 절단 그래프는 γ를 고르지 않고 γ 0과 1에서 모두 평가했다(3.9절). γ = 1이면 주제 문서 구분을 끈 것과 같다. 출처 가중(β, γ)은 간선을 지우지 않는다. γ = 0이어도 그 간선과 근거 문장은 추적 질의로 조회된다.

**후보 창.** 동결 순위화는 Top-k(k = 10)를 내기 위해 소견 점수 score_F를 상위 k × 4 = 40개 질환에서만 읽는다. 통합 근거 점수는 이 창 밖의 질환에도 더한다. 따라서 창 밖에서 통합 근거를 받은 질환은 통합 근거 점수만 받는다. 이 창은 D 확증에 쓴 동결 코드(동결 커밋 cd130e7, 실행 커밋 2ecb064)에 들어 있었으나 설계 D 문서에는 적혀 있지 않았다. 창은 동결 설정의 일부이므로 D와 E 확증, 그리고 같은 랭커로 평가한 모든 비교 지식원에서 그대로 썼다. 결합 시스템 안의 그래프 목록에 쓴 창은 3.7절에 적는다. 창과 자기 간선 제거의 상호작용, 창 밖 질환의 소견 점수까지 읽는 채점과의 비교는 보충자료에 둔다.

**정답 판정과 지표.** 후보는 그래프의 전체 질환 43,575개다. 벤치마크의 정답 질환 목록은 후보로 쓰지 않고 정답 대조에만 쓴다. 정답 식별자(UMLS CUI, OMIM, Orphanet, MONDO)는 교차참조로 질환 노드 키에 사상한다. 주 판정 lineage는 예측 질환이 정답 질환이거나 질환 IS_A 관계로 정답의 하위 3단계 이내에 있으면 정답으로 인정한다. 정답의 상위 질환은 정답으로 인정하지 않는다. 보조 판정 strict는 정답 키와 정확히 같을 때만 인정한다. 지표는 Top-1, Top-3, Top-10(상위 k 안에 정답이 있는 환자 비율)이다. 형식은 RareBench의 Recall@k와 같고, 정답 인정 기준은 위의 lineage/strict를 따른다. 판정은 Top-10을 중심으로 하고 Top-1을 함께 보고한다. 정답이 질환 노드로 사상되지 않거나 사상된 소견 개념이 없는 환자는 평가에서 뺀다. 예를 들어 D 확증의 Phenopacket-new는 369증례 중 261명을 평가했다.

### 3.6 벤치마크와 평가 프로토콜

**벤치마크.** 개발 세트는 9개다(표 3). DDXPlus는 test 분할에서 시드 0으로 3,000명을 뽑았다. 이 표본에는 49개 질환 중 48개가 있다(세기관지염 환자가 없다). 희귀질환은 RareBench의 RAMEDIS(624명), LIRICAL(370명), HMS(88명), MME(40명), 북경협화병원 실제 증례 PUMCH_L(988명)과 PUMCH-ADM(75명), 출판 증례 모음 Phenopacket-store 0.1.27(10,310명)을 썼다. SymCat은 801개 질환마다 시드 0으로 10명씩 생성하고, 소견이 뽑히지 않은 표본과 정답 사상이 안 되는 질환을 빼 7,481명을 썼다. 9개 세트의 환자는 모두 22,976명이다. 입력 형식은 DDXPlus가 값이 있는 문진 코드, SymCat이 증상 목록, 나머지 7개가 HPO 표현형이다. PUMCH_L과 PUMCH-ADM은 PhenoBrain 시험 증례 묶음(Zenodo, doi:10.5281/zenodo.10774650)으로 CC BY 4.0 하에 배포된다.

벤치마크 입력은 결정론적 어휘 사상으로 개념 집합이 된다. DDXPlus는 기본값 응답을 근거에서 빼고, 통증 부위 응답은 부위별 통증 개념으로 합성한다. 이 규칙은 '여행 안 함' 응답(trav1_@_N)이 DDXPlus validate 환자의 93%에서 'Travel History'로 사상되던 결함을 고친 것이다(보충자료). 통증의 성격, 강도, 방사 부위, 위치가 분명한 정도, 생긴 속도를 묻는 문항은 기본값이 아닌 응답이면 값과 관계없이 통증 개념(Pain, C0030193)으로만 옮긴다. Phenopacket-store 증례는 관찰된 소견의 HPO 코드만 쓰고, 소견별 발병 시점은 쓰지 않는다. DDXPlus 정답은 영문 질환명의 UMLS 정확 일치를 우선한다. 이때 질환 의미유형만 보고, SNOMED > MeSH > MedlinePlus > NCI 순으로 고르며, 5자 이하 대문자 약어는 제외한다. 정확 일치가 없으면 ICD-10 경로를 쓴다. 49개 질환 중 31개가 이름 일치로, 18개가 ICD-10 경로로 사상되었다. 희귀질환 정답은 OMIM·Orphanet 식별자를 UMLS 또는 MONDO 교차참조로 사상한다.

**입력 내 정답 명시 층.** DDXPlus 개발 환자 3,000명 중 458명(15.3%)은 입력 소견에 정답 질환의 키(CUI)가 그대로 있다. 정답 키의 직속 상위 질환 키까지 세면 681명(22.7%)이다. 458명에서 그 소견이 나온 문항은 본인 과거력 202명, 증상 문항 106명, 가족력 95명, 노출 등 55명이다. 입력에 정답 질환 키나 그 직속 상위 질환 키가 있는 환자를 입력 내 정답 명시 층이라 부른다. 통합 근거(3.5절)는 이런 소견을 점수에 반영한다. 확증 세트 E에서 이 층은 DDXPlus-E 717명(23.9%), SymCat-E 275명(3.7%)이다. 개발 SymCat에서는 285명(3.8%)이다. 이 층에 따른 결과는 4.6절에서 다룬다.

Table 3. Benchmarks and their roles

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
^b^ Sealed at commit 31122e2 and evaluated once on 2026-10-13. The comparator sources (HPOA, PrimeKG, MeSH co-occurrence) were run once on these sets after the main systems, without a pre-committed criterion. The two secondary PPR hypotheses were committed before the run and included in it, but no criterion for these sets was written; their values on these sets are sealed descriptive. These sets have since been opened and are used only as development evidence in design E.
^c^ Patients without a mapped gold key or without mapped findings are not evaluated. Of the 369 cases, 264 have a mapped gold key (37 source disease IDs, 50 gold keys), and 261 of these also have mapped findings (at most 34 cases per disease).
^d^ Sealed at commit 3b10c36 and evaluated once at freeze commit 49901bc (2026-10-19 23:13 to 2026-10-20 01:34, one attempt). New patients drawn from the development diseases. After confirmation these sets were re-analysed in the review analyses of 2026-10-20 (no decision rule) and in the post hoc measurements of Section 3.10 (decision rules committed before measurement for the rule rows only; strata and auxiliary values have no rule). Commit hashes are those in the run records; Section 3.6 maps them to the public history.

**설계 D 프로토콜.** 설계 D 이전의 탐색 단계(v1)에서는 무효 평가로 시험 분할을 두 번 열었고, DDXPlus validate로 고른 설정이 희귀질환 세트로 전이되지 않았으며, 같은 문헌의 BM25가 9개 시험 세트 중 6개에서 그래프보다 높았다. 그래서 설계 D는 목표와 판정을 먼저 고정하고 개발과 확증을 나눴다.

개발 데이터는 DDXPlus validate 3,000명과 v1에서 이미 열람한 위의 9개 세트다. 설정 선택 규칙 D3는 9개 개발 세트의 매크로 lineage Top-10이 가장 높은 단일 설정을 고른다. Top-10이 같으면 Top-1로 고른다. 벤치마크별 조정은 하지 않는다. 비교 시스템도 같은 규칙으로 각자의 설정을 다시 고른다.

확증 세트는 봉인 커밋 31122e2로 고정했다. 구성은 세 가지다. (1) DDXPlus test 중 개발 표본과 겹치지 않는 3,000명(시드 101). (2) Phenopacket-store main@85ffd78(2026-10-07)에서 0.1.27 이후 추가된 405증례 중 원출처 PMID가 코퍼스에 있는 36증례를 뺀 369증례. (3) SymCat 시드 101(7,482명). 확증 로더는 환경 변수 CONFIRM_UNSEAL=1이 없으면 실행을 거부한다.

목표는 네 가지이고 제약이 하나 있다.
- T1(주): 그래프 단독 > BM25. 개발 9개 중 8개 이상에서 우세(부호검정 p < .05)해야 한다. 그리고 확증 3개 각각에서 Top-10 차이의 대응 부트스트랩 95% CI 하한이 −1%p보다 커야 하고(비열등), 그중 2개 이상에서 Holm 보정 후 유의하게 우세해야 한다.
- T2(흔한 질환 정확도 목표): 확증 DDXPlus에서 Top-10 ≥ 40%, Top-1 ≥ 18%.
- T3: 개발 PUMCH_L·PUMCH-ADM의 Top-10이 같은 랭커에 HPO 주석을 넣은 값 이상. 기준값은 처음에 validate로 고른 HPO 주석 설정의 값(53.6%, 53.3%)으로 적었다. D3에 따라 HPO 주석 설정을 개발 세트에서 다시 고른 뒤에는 그 값(53.2%, 57.3%)으로 판정했다.
- T4: 그래프+BM25 결합이 확증 매크로 평균에서 그래프 단독과 BM25 단독보다 유의하게 높음. 판정할 k는 적혀 있지 않아 D3의 주 지표인 Top-10으로 판정했다. Top-1로 판정해도 결과는 같다.
- 제약(출처 불변식): 모든 간선이 근거 문장과 문헌에 연결됨. 적재 시 검사는 모든 소견 간선에 근거 언급(Mention)이 있는지 본다.

판정 기준은 실행 전(2026-10-12, 공개 이력 커밋 3fbd29a)에 커밋했다. 이 증빙도 로컬 커밋 시각이며, 공개 등록부와 실행 전 원격 기록은 없다. 충족과 미달을 모두 그대로 보고하고, T1이 미달이면 그 사실이 주 결과라고 함께 적었다. 동결 설정(frozen_config_D.json)은 커밋 cd130e7에 넣었고, 확증은 비교군 실행 스크립트만 더한 커밋 2ecb064에서 실행했다(설정 변경 없음). 확증은 예정일(2026-10-16)보다 앞당겨 2026-10-13에 한 번 실행했다. PPR 부가 가설 두 개(PPR-H1, PPR-H4)는 실행 전에 커밋했고(공개 이력 커밋 50a64db), 확증 실행에 PPR을 넣기로 미리 정했다. PPR-H1은 같은 그래프에서 개인화 PageRank가 1-홉 집계보다 높은가이고, PPR-H4는 HPO 주석·PrimeKG 그래프에서도 같은가다. PPR-H1의 판정은 개발 매크로 Top-10의 부호검정과 Holm 보정으로 적었고, PPR-H4에는 판정 방법을 적지 않았다. 봉인 세트에서의 판정 기준은 두 가설 모두 적지 않았다. 그래서 봉인 세트의 PPR 값은 [봉인 기술]로 표시한다(3.8절). 비교 지식원(HPO 주석, PrimeKG, MeSH 공출현)은 주 시스템 실행 뒤 별도 스크립트로 같은 세트에서 한 번 돌렸다. HPO 주석·PrimeKG의 PPR도 이 스크립트에서 돌렸다. 이 비교에도 판정 기준을 미리 정하지 않았으므로 [봉인 기술]로 표시한다.

**설계 E 프로토콜.** 설계 E는 D 확증 이후의 개발 탐색에서 고른 시스템을 새 봉인 세트로 한 번 확증한다. D 확증에서 T1과 T2는 미달이었다(4.1절). 그 뒤 같은 9개 개발 세트에서 그래프 변형 32개를 비교했다(4.6절, 보충자료). 32개는 동결 그래프의 재현(base_nm)과, 수집 깊이, 추출·연결 방식, 빌더 규칙(등위 분해, 한정어 제외, 자기 간선 제거 등) 중 하나 이상을 바꾼 그래프 31개다. 문헌으로 나누면 v2 문헌 그래프가 base_nm을 포함해 18개, v2+v3 문헌 그래프가 14개다. v2+v3 문헌 그래프 14개에는 PMC 본문 전체를 읽는 판 2개(v3umlsfull_ns, v3dictfull_ns)가 들어 있다. 이 32개 가운데 개발 매크로 Top-10이 가장 높은 S-mix를 주 시스템으로 골랐다. 따라서 S-mix의 개발값에는 선택 낙관이 들어 있다. D 확증 결과는 바꾸지 않고 주 결과의 하나로 보고한다. 설계 E는 D의 목표와 판정을 대체하는 새 확증 설계다. 판정 기준과 개발값으로 본 예상은 실행 전에 커밋했다(5ce9c3d 등). 이 증빙은 로컬 커밋 시각이며, 공개 등록부와 실행 전 원격 기록은 없다.

확증 세트 E는 data/confirm_e/MANIFEST.json에 고정했다(2026-10-19 생성, 봉인 커밋 3b10c36). DDXPlus-E는 DDXPlus test 134,529행에서 개발 표본 3,000행과 D 확증 표본 3,000행을 뺀 128,529행 중 시드 202로 뽑은 3,000명이다. 매니페스트에는 뽑은 행의 번호와 각 행의 sha256을 기록했고, 개발 표본·D 확증 표본과 겹치는 행은 0개다. SymCat-E는 801개 질환마다 시드 202로 10명씩 생성한다(개발은 시드 0, D 확증은 시드 101). 두 세트는 개발 세트와 같은 질환 집합에서 새로 뽑은 환자다. DDXPlus-E 표본의 질환 중 개발 표본에 없던 것은 세기관지염(1명)뿐이다. 합성 환자를 개념으로 사상하면 일부 환자는 입력이 개발 환자와 같아진다. 열람된 D 확증 세트로 재면, (정답, 소견 개념 집합)이 개발 환자 한 명과 정확히 같은 환자가 DDXPlus 3,000명 중 791명(26.4%), SymCat 7,482명 중 1,151명(15.4%)이었다. E 세트에서는 이 비율을 재지 않았다. 따라서 E 확증은 같은 질환에서 새로 뽑은 환자 표본에서 결과가 재현되는지까지만 검사하며, 그 표본의 일부는 입력이 개발 환자와 같은 환자다.

희귀질환 확증 세트는 없다. Phenopacket-store main에는 2026-10-19 현재 D 확증 세트의 기준 커밋 85ffd78 이후 커밋이 없었고, 스냅샷을 이날로 고정했으므로 새 증례가 0건이다. RareBench와 PUMCH 증례는 모두 열람되었다. D 확증의 Phenopacket-new 증례도 시간 분할 분석(3.9절)의 2026 스냅샷에 들어가 E의 그래프 시스템 다섯 개 모두에서 평가되었다. 따라서 설계 E 시스템의 희귀질환 결과는 모두 개발 수준이다. 봉인 세트에서 나온 희귀질환 값은 D 확증의 Phenopacket-new 값(설계 D 시스템과 비교 지식원)뿐이다.

봉인 장치는 코드로 강제한다. E 확증 로더(load_confirm_e)는 CONFIRM_E_UNSEAL=1이 없으면 거부한다. 개발 로더는 DDXPlus-E의 행과 SymCat 시드 202를 거부한다. E를 다시 읽는 확증 이후 분석(2026-10-20 리뷰 분석과 3.10절의 측정)의 스크립트도 이 로더를 거치므로, 같은 해제 변수가 있어야 실행된다.

동결 시스템은 7개다(표 4). 그래프 시스템 5개는 3.5절의 동결 채점을 공유하고 γ만 다르다.
- S-mix(주): v2+v3 문헌, gemma 구절과 구절 안 UMLS 사전 재연결(3.3절), 자기 간선 제거(v3llmd_ns), γ 0.5.
- S-D: 설계 D의 동결 시스템(v2 문헌 gemma 그래프, base_nm 빌드), γ 0.
- S-llm: 같은 v2+v3 문헌의 LLM 추출만(v3llm_ns), γ 1.0.
- S-dict: 같은 문헌의 UMLS 사전만(v3umls_ns), γ 0.5.
- S-union: LLM 항목과 모든 문장의 UMLS 사전 항목의 합집합(v3unionu_ns), γ 0.5.
- S-bm25: 같은 v2+v3 문헌의 BM25(일반 문헌 규칙도 그래프와 같게).
- S-comb: S-mix 순위와 S-bm25 순위의 역순위 결합. 가중 w는 {0.25, 0.5, 0.75}에서 D3 규칙으로 고르고, 한쪽만 쓰는 0과 1은 후보에서 뺀다.

S-dict는 같은 텍스트(PMC 논문은 임상 절 12,000자, 도서 장은 24,000자)로 만든 UMLS 사전 추출 변형 가운데 개발 매크로 Top-10이 가장 높은 것이다. v3 PMC 논문의 본문 전체를 읽는 v3umlsfull_ns(개발 매크로 Top-10 43.08%)는 텍스트가 달라 비교군에서 뺐다. 순수 사전 그래프(3.3절)는 확증 뒤에 만들었으므로 동결 시스템이 아니며, E에서의 값은 [사후]다(3.10절).

목표는 다섯 가지이고, E-T2의 세 비교를 따로 세면 판정은 7개다. 판정 세트는 DDXPlus-E와 SymCat-E이고, 판정 기준은 lineage, 후보는 43,575개 질환 전체다.
- E-T1(주): S-mix > S-D. 두 세트 각각에서 Top-10 차이의 환자 대응 부트스트랩 95% CI 하한 > 0이고, 정확 McNemar 검정의 Holm 보정 p < .05여야 한다(족 = 두 세트).
- E-T2(S-llm 비교는 주, S-dict·S-union 비교는 부): S-mix − S-llm, S-mix − S-dict, S-mix − S-union 각각의 매크로 Top-10 차이에서 95% CI 하한 > −1%p(비열등). 세 비교는 따로 판정한다.
- E-T3(부): S-mix 대 S-bm25. 두 세트 각각에서 Top-10 차이 CI 하한 > −1%p이고, 1개 이상에서 Holm 유의 우세.
- E-T4(부): S-mix 단독으로 DDXPlus-E Top-10 ≥ 40%, Top-1 ≥ 18%(D의 T2 기준 유지).
- E-T5(부): S-comb − S-mix와 S-comb − S-bm25 둘 다 매크로 Top-10 차이 CI 하한 > 0.

설계 E의 주 목표(E0)는 S-mix가 설계 D 그래프보다 높고(E-T1), 같은 문헌의 LLM 단독 그래프보다 낮지 않은 것(E-T2의 S-llm 비교)이다. E0 달성은 이 두 판정이 모두 충족되는 것으로 정의했다. 나머지 다섯 판정(E-T2의 S-dict·S-union 비교, E-T3, E-T4, E-T5)은 보고용 부 목표다. 결과 절의 첫 비교(4.2절)인 E-T3도 실행 전에 커밋한 7개 판정 가운데 부 목표다. 목표 사이의 다중성은 보정하지 않는다.

개발값으로 본 예상도 실행 전에 커밋했다. E-T1은 충족을 예상했다(개발 S-mix − S-D Top-10: DDXPlus +7.6%p, SymCat +2.7%p). E-T2는 S-llm 비교의 충족(+1.9%p, +1.9%p)과 S-dict·S-union 비교의 미달(DDXPlus −1.6%p·−3.0%p, SymCat −1.1%p·−1.3%p)을 예상했다. E-T4는 Top-10이 기준 경계(개발 40.6%), Top-1이 미달(개발 11.8%)일 것으로 예상했다. E-T3은 충족을 예상했다(개발 S-mix − S-bm25 Top-10: DDXPlus +20.9%p [+19.2, +22.7], SymCat +3.2%p [+2.5, +4.0], 둘 다 Holm 유의). E-T5는 미달을 예상했다. 개발 DDXPlus·SymCat의 매크로 Top-10 차이에서 S-comb − S-mix는 −0.02%p [−0.54, +0.49](DDXPlus −0.8%p, SymCat +0.8%p)로 CI 하한이 0보다 낮았다. S-comb − S-bm25는 +12.04%p [+11.15, +12.94]로 CI 하한이 0보다 높았다. E-T3과 E-T5의 예상은 S-bm25와 S-comb를 개발 세트에서 처음 고른 뒤 커밋 61e3118에 기록했다. 그 뒤 첫 S-comb 선택이 비교 인스턴스에 S-mix가 아닌 그래프가 적재된 상태에서 실행된 것을 확인했다(3.7절). 그래서 S-comb를 다시 고르고, E-T5의 예상을 커밋 0cb1c59에서 위의 값으로 고쳤다. 고치기 전의 E-T5 예상도 미달이었다(S-comb − S-mix −1.67%p [−2.38, −1.01]). 두 커밋은 모두 동결 커밋보다 앞선다.

절차는 다섯 단계다. (1) 설계 E 커밋(5ce9c3d, 판정 기준과 E-T1·E-T2·E-T4 예상 포함). (2) 확증 세트 봉인, 로더 거부 장치, 매크로 통계 코드 커밋(3b10c36). (3) 확증 실행 스크립트 커밋(a7d0680), S-bm25 색인과 개발 선택, S-comb 가중 선택, 재연결 항목에 원 LLM 구절을 남기도록 고친 S-mix의 재빌드(지문 불변 확인), Mention 빌드 검사. (4) 동결 커밋(frozen_config_E.json: 시스템별 지문, 색인 sha256, 설정). (5) 확증 1회 실행. 개발 선택은 S-bm25가 질의 전체·깊이 1,000·역순위 결합 집계(개발 매크로 Top-10/Top-1 27.35/11.04%), S-comb가 w = 0.25(46.72/19.46%)였다. 동결 커밋은 49901bc(2026-10-19)다. 실행이 중단되면 같은 동결 입력으로 결과 파일이 없는 시스템만 다시 실행하고, 모든 시도를 기록하며, 시스템은 바꾸지 않기로 정했다. 확증은 동결 커밋에서 2026-10-19 23:13에 시작해 2026-10-20 01:34에 끝났다. 시도는 1회였고 중단은 없었다. 확증 직전에 다섯 그래프 시스템을 같은 비교 인스턴스에 다시 빌드했고, 빌드 지문과 적재된 HAS_FINDING 수가 동결 기록과 같을 때만 평가했다. S-bm25는 색인 파일의 sha256을 동결 기록과 대조했다. 동결 검증과 출처 불변식 검사의 결과는 3.4절에 적었다.

**커밋 기록.** 이 절의 커밋 해시는 실행 기록(동결 설정 파일과 확증 시도 로그)에 남은 값이다. 공개 저장소의 이력은 확증 뒤에 파일 하나를 이력에서 지우며 다시 썼다. 그래서 같은 커밋이 공개 이력에서는 다른 해시를 가진다. 두 해시의 커밋 시각은 같고, 커밋 내용은 지운 파일 하나만 다르다. 대응은 다음과 같다: 31122e2 → 5108e8d, cd130e7 → dce8e46, 2ecb064 → c8b33e6, 5ce9c3d → 9e87ae7, 3b10c36 → 9910f61, a7d0680 → f4da0c6, 61e3118 → da73c65, 0cb1c59 → 1e54e60, 49901bc → 780f684. 실행 기록에 없는 설계 D 판정 기준 커밋과 PPR 설계 커밋은 공개 이력의 해시(3fbd29a, 50a64db)로 적는다. 재작성 전 해시는 각각 a167a9a, 19fc7df다. 확증 이후 분석의 커밋(5800581, 400d4cf, f676539, 3ea76b7, dfa9541)은 공개 이력의 해시 그대로다.

### 3.7 비교 시스템과 같은 랭커 지식원 교체 평가

**같은 랭커 지식원 교체 평가.** 정확도 차이가 순위 방법에서 오는지 지식에서 오는지 가르기 위해, 기존 지식원을 문헌 그래프와 같은 스키마(Disease, Concept, HAS_FINDING, IS_A, IS_DISEASE)와 같은 후보 질환 집합(43,575개)으로 별도 Neo4j 인스턴스에 적재했다. 그리고 문헌 그래프와 같은 Cypher 1-홉 랭커 코드로 평가했다. 이 통제 비교를 같은 랭커 지식원 교체 평가라 부른다. 랭커 코드는 같지만 채점 설정은 지식원마다 D3 규칙으로 다시 골랐다. 그래서 이 원고는 이 조건을 "같은 랭커 코드, 지식원별로 조정한 설정"이라고 쓴다. 고정한 랭커에 지식원을 바꿔 넣는 평가는 선례가 있다 [ref: Groza et al. 2026]. 비교 지식원은 주 시스템의 점수에 들어가지 않는다(3.1절). 비교 지식원에는 근거 문장(Mention)이 없다.

**비교 지식원.** 다섯 가지를 같은 랭커에 넣었다(표 4).
- HPO 주석(HPOA): phenotype.hpoa 2026-02-16판이다. NOT 주석이 아닌 표현형(aspect P) 행 264,245개를 읽었다. 질환 ID는 교차참조로 후보 질환 키에, HPO 용어는 UMLS CUI에 사상했다. 간선은 393,498개, 소견이 있는 질환은 12,251개다.
- HPOA-LOPO: HPOA에서 평가 증례의 원논문을 인용한 주석을 지운 판이다(정의는 3.9절). 간선은 363,488개, 소견이 있는 질환은 11,841개다.
- PrimeKG: disease_phenotype_positive 관계(MONDO 질환–HPO 표현형)다. 간선은 747,705개, 질환은 10,898개다.
- MeSH 공출현: NLM MEDLINE 공출현 2024 요약이다. LLM을 쓰지 않으며, 지지 수는 공동 색인된 인용 수다. 간선은 3,798,931개, 질환은 3,302개다.
- SemMedDB: VER43 2024 PREDICATION 130,480,195행 가운데 세 술어(MANIFESTATION_OF, COEXISTS_WITH, ASSOCIATED_WITH; NEG_ 술어 제외)의 행은 8,389,567개다. 이 가운데 질환–소견 쌍이 될 수 있는 2,524,982행을 썼다. MANIFESTATION_OF는 목적어 쪽만 질환으로 받았다. 방향이 없는 연관인 나머지 두 술어는 어느 쪽 끝이든 질환으로 받았다. SemMedDB가 일반 개념(novelty 0)으로 표시한 개념이 낀 333,537행은 뺐다. 지지 수는 서로 다른 PMID 수다. 간선은 889,315개, 질환은 11,522개다. 방향과 일반 개념의 처리는 평가 전에 실행 스크립트 머리말에 적었으나, 이 결정의 커밋 기록은 없다. 질환 주어만 받는 판본과 일반 개념을 남긴 판본은 민감도 분석으로 보충자료에 둔다.

MeSH 공출현과 SemMedDB는 질환을 UMLS CUI로 식별한다. 그래서 두 지식원은 CUI를 키로 가진 후보 질환 35,281개에만 닿고, MONDO 키 질환 7,985개와 GeneReviews 장 309개에는 닿지 않는다. 두 지식원의 소견 쪽은 LLM 경로의 허용 의미유형에서 검사·시술 유형을 뺀 26개로 걸렀다.

**같은 랭커가 비교 지식원에 주는 영향.** 비교 지식원의 간선은 모두 주제 간선으로 적재했으므로 γ는 점수에 영향이 없다. 질환별 원문 문자 수 m(d)는 모든 질환에 같은 상수로 두었으므로, 커버리지 스무딩은 모든 질환에 같은 길이를 더한다. HPOA와 PrimeKG 간선의 지지 수는 1이므로 β도 영향이 없다. HPOA의 빈도, 발병 시기, 수식어 열은 쓰지 않는다. 따라서 "같은 1-홉 랭커의 HPOA"는 큐레이션 지식으로 얻을 수 있는 정확도의 상한이 아니다. HPO 의미 유사도를 쓰는 표현형 매칭 도구와는 이 조건으로 직접 비교할 수 없다(2.3절).

**같은 문헌의 BM25.** 설계 D의 BM25는 v2 코퍼스에서 일반 문헌과 비인간 질환을 뺀 문서 122,661편을 약 1,000자 단위로 나눈 청크 655,965개를 색인한다. 이 문서 수는 D 동결 그래프에 근거가 적재된 문서 86,276편보다 많다. 설계 E의 S-bm25는 같은 방식으로 v2+v3 문헌을 색인한다(문서 1,001,916편, 청크 3,074,271개). 토큰화에는 영어 불용어 제거와 영어 어간 추출(PyStemmer)을 썼고, 점수 함수는 bm25s 0.3.13의 Lucene 변형(k1 1.5, b 0.75)이다. 환자 소견 개념은 UMLS 영어 선호명으로 바꿔 질의하므로 그래프와 같은 정보만 입력한다. 검색된 청크의 점수는 그 문헌에 연결된 질환들(그래프와 같은 질환–문헌 연결)로 모아 열린 집합 순위를 낸다. 설정은 질의 방식(소견 전체를 한 질의로 / 소견별 질의) 2개 × 검색 깊이 {100, 1,000} 2개 × 집계 방식 {최댓값, 합, 역순위 결합} 3개의 12개 조합이다. k1과 b는 모든 조합에서 고정했다. 설계 D는 소견별 질의·깊이 1,000·최댓값 집계를 골랐다(개발 매크로 Top-10/Top-1 24.46/8.75%). 설계 E의 S-bm25는 질의 전체·깊이 1,000·역순위 결합 집계를 골랐다(27.35/11.04%). 확증 뒤에는 12개 조합 전부와 설계 D의 v2 BM25를 E에서 다시 돌렸다(V1, 3.10절).

**결합.** 그래프 순위 r_G와 BM25 순위 r_B는 역순위 결합한다.

$$\mathrm{score}(d)=\frac{1-w}{60+r_G(d)}+\frac{w}{60+r_B(d)}$$

설계 D의 결합(T4)은 w ∈ {0, 0.25, 0.5, 0.75, 1}에서 D3 규칙으로 w = 0.25를 골랐다(개발 매크로 32.32/12.72%). D의 결합에서 두 순위는 각 시스템의 상위 200개 질환이었다. 그래서 그래프 목록 안의 소견 점수 창은 200 × 4 = 800이었고, 결합 안의 그래프 순서는 그래프 단독 평가(창 40)와 달랐다. w = 0, 즉 그래프 목록만으로 낸 개발 매크로는 30.22/12.37%로 그래프 단독 30.56/12.64%와 다르다. 설계 E의 S-comb(E-T5)는 그래프 목록을 그래프 단독 평가와 같은 창 40으로 낸다(커밋 a7d0680). 그래서 그래프 목록은 창 안의 40개 질환과 통합 근거를 받은 질환으로 이루어진다. BM25 목록은 상위 200개 그대로다. 첫 S-comb 가중 선택은 비교 인스턴스에 S-mix가 아닌 그래프가 적재된 상태에서 실행되어 폐기했다. 다시 고른 값은 w = 0.25(개발 매크로 46.72/19.46%)다. 확증 이후의 M1(3.10절)은 같은 식으로 S-mix와 HPOA를 결합했다. 두 목록은 각자의 그래프와 설정으로 낸 순위다. 소견 점수는 상위 40개 질환(창 40)에서만 읽으므로, 목록은 그 40개와 통합 근거를 받은 질환으로 이루어진다(요청 길이 200). 가중은 w ∈ {0.25, 0.5, 0.75}에서 D3 규칙으로 w = 0.75를 골랐다(개발 9개 세트 매크로 Top-10 51.43, 48.84, 57.03%). M1은 간선 합집합도 만들었다. 두 지식원에 모두 있는 간선은 HPOA 주석을 주제 지지 문서 1편으로 세어 지지 수와 주제 지지 수에 1씩 더했다. HPOA에만 있는 간선은 지지 수와 주제 지지 수를 1로 두었다. 소견 수, IDF, 개념 계층, IS_DISEASE는 합집합에서 다시 계산했고, 질환별 원문 문자 수는 S-mix 값을 썼다. 합집합은 간선 5,506,536개(공유 108,247개, HPOA 전용 285,251개)이고, 소견이 있는 질환은 33,537개에서 34,516개로 늘었다. 합집합은 S-mix의 동결 설정(γ 0.5)으로 순위화했다.

**튜닝 예산.** 조정 예산은 시스템마다 다르다(표 4). S-D의 채점 설정은 6개 매개변수의 좌표 탐색(평가 38회)으로 골랐다. S-mix, S-llm, S-dict, S-union은 이 설정을 물려받고 γ 4값만 다시 골랐다. S-mix는 같은 9개 개발 세트에서 그래프 32개 × γ 4값 가운데 개발 매크로 Top-10이 가장 높은 것으로 골랐다. 그래서 S-mix의 탐색 범위가 가장 넓고, 선택 낙관도 가장 클 수 있다. 선택 낙관의 크기는 시스템별로 재지 않았다. 비교 지식원은 6개 채점 매개변수의 같은 격자에서 같은 시작점의 좌표 탐색으로 다시 골랐다. 평가 횟수는 HPOA 38회(서로 다른 설정 28개), PrimeKG 38회(25개), MeSH 공출현 39회(25개), SemMedDB 39회(28개)다. SemMedDB는 설계 D의 동결 채점으로 낸 값도 함께 보고한다. HPOA-LOPO는 HPOA 설정을 다시 고르지 않았다. BM25는 12개 조합에서 골랐다. 채점 매개변수만 보면 BM25(12개 조합)의 조정 예산이 그래프(γ 4값)보다 많다. 다만 시스템 선택까지 세면 S-mix 쪽 탐색(그래프 32개 × γ 4값)이 더 크다.

**개인화 PageRank(PPR).** 같은 그래프를 무방향 가중 그래프로 내보내 환자 소견 개념을 시드로 PPR을 계산하고, 질환 노드의 확률을 점수로 쓴다. 간선은 HAS_FINDING, 개념 IS_A, 질환 IS_A, IS_DISEASE이고, 허브 보정은 질환 점수를 차수의 거듭제곱으로 나눈다. 매개변수는 D3 규칙으로 골랐고, 격자와 선택값은 보충자료에 둔다. scipy 거듭제곱 계산과 Neo4j GDS pageRank의 질환 Top-10은 PUMCH_L 20명 모두에서 같았다(간선 없는 고립 시드 제외 조건).

**설계 E의 비교군.** S-llm, S-dict, S-union은 S-mix와 같은 v2+v3 문헌, 같은 빌더 규칙, 같은 동결 채점을 쓰고 추출 방식과 γ가 다르다. S-D는 문헌 깊이와 추출 방식이 모두 다른 설계 D의 동결 시스템이다. MedCPT 밀집 검색은 v1 탐색 단계에서만 평가했다(DDXPlus validate로 설정 선택, 시험 분할 평가). 설계 D에서는 D3 재선택 대상으로 적었으나 재선택하지 않았고, D·E 확증과 확증 이후 측정의 비교군에 넣지 않았다. 그래서 이 원고의 검색 비교는 어휘 검색(BM25)에 한정된다.

Table 4. Systems, knowledge sources and tuning budget

| System | Literature | Extraction or knowledge source | Configuration | Edges | Diseases with findings | Tuning budget | Dev macro Top-10 / Top-1 (%) |
|---|---|---|---|---|---|---|---|
| **S-mix (main)** | v2 + v3 | gemma spans; spans the phrase table cannot link are named by UMLS finding strings inside the span (v3llmd_ns) | Builder: split-coord, hpo-names, topic-genes, no-self; γ 0.5 | 5,221,285 | 33,537 | 1 of 32 graph variants × 4 γ values; scoring taken from S-D | 46.11 / 19.66 |
| S-llm | v2 + v3 | gemma spans linked by the phrase table (v3llm_ns) | Same builder; γ 1.0 | 3,672,106 | 33,123 | 4 γ values | 44.90 / 19.30 |
| S-dict | v2 + v3 | UMLS finding-string dictionary on every sentence; CUI-less items linked by HPO names and the LLM phrase table (v3umls_ns) | Same builder; γ 0.5 | 14,506,788 | 34,356 | 4 γ values^a^ | 42.88 / 18.11 |
| Pure dictionary | v2 + v3 | As S-dict with an empty phrase table: CUI-less items linked by HPO names only (v3umlsnt_ns) | Same builder; γ 0.5 (taken from S-dict) | 13,883,910 | 34,354 | None^b^ | 42.38 / 17.75 |
| S-union | v2 + v3 | S-llm items + S-dict items (v3unionu_ns) | Same builder; γ 0.5 | 15,396,388 | 34,359 | 4 γ values | 43.77 / 17.59 |
| S-D (design D frozen) | v2 | gemma spans linked by the v2 phrase table (base_nm) | No builder flags; γ 0 | 832,671 | 27,309 | Coordinate search over six scoring parameters, 38 evaluations (37 distinct); 1 of 2 IE models | 30.56 / 12.64 |
| S-bm25 | v2 + v3 | BM25 over 3,074,271 chunks of 1,001,916 documents | Joint query, depth 1,000, RRF | — | — | 12 configurations | 27.35 / 11.04 |
| BM25 (design D) | v2 | BM25 over 655,965 chunks of 122,661 documents | Per-finding query, depth 1,000, max | — | — | 12 configurations | 24.46 / 8.75 |
| S-comb | v2 + v3 | Reciprocal-rank fusion of S-mix and S-bm25 | w 0.25 | — | — | 3 w values | 46.72 / 19.46 |
| Graph + BM25 (design D) | v2 | Reciprocal-rank fusion of S-D and BM25 (design D) | w 0.25 | — | — | 5 w values | 32.32 / 12.72 |
| PPR (design D) | v2 | S-D graph | Restart 0.3, β 1, γ 0, h 0.5, η 0, uniform seeds | — | — | 28 evaluations (25 distinct) | 23.57 / 6.33 |
| HPOA, 1-hop | — | phenotype.hpoa 2026-02-16 (aspect P, NOT rows excluded) | α 0.2, μ 2, δ 0.5, u 2, β 0 | 393,498 | 12,251 | Coordinate search, 38 evaluations (28 distinct) | 52.45 / 30.37 |
| HPOA-LOPO, 1-hop | — | HPOA without the 22,261 rows citing any of 1,745 Phenopacket-store case PMIDs^c^ | HPOA setting | 363,488 | 11,841 | None | (not an LOPO estimate) |
| PrimeKG, 1-hop | — | disease_phenotype_positive | α 0.5, μ 2, δ 0.3, u 1, β 0 | 747,705 | 10,898 | Coordinate search, 38 evaluations (25 distinct) | 33.52 / 15.18 |
| MeSH co-occurrence, 1-hop | — | MEDLINE co-occurrence 2024 (no LLM) | α 0.35, μ 0, δ 0.3, u 1, β 0 | 3,798,931 | 3,302 | Coordinate search, 39 evaluations (25 distinct) | 9.13 / 2.61 |
| SemMedDB, 1-hop | — | SemMedDB VER43 2024: MANIFESTATION_OF, COEXISTS_WITH, ASSOCIATED_WITH; generic concepts dropped | α 0.35, μ 1, δ 0.5, u 0.5, β 0 | 889,315 | 11,522 | Coordinate search, 39 evaluations (28 distinct)^d^ | 23.20 / 7.15 |

Status: development, for every accuracy value. Dev macro is the mean over the nine development sets of the per-set lineage Top-10 and Top-1. All literature-graph systems share the frozen scoring of design D (α 0.35, μ 2, δ 0.3, u 2.0, β 0.5); only γ is re-selected per graph by rule D3, except that S-D keeps the γ of its original coordinate search and the pure dictionary graph takes S-dict's γ. Every graph has 43,575 Disease nodes. The comparator sources are scored by the same ranker code with settings re-selected per source by rule D3 on the same grid; for them every edge counts as on-topic and the source length is a constant, and HPOA and PrimeKG edges carry one supporting unit, so γ (and, for HPOA and PrimeKG, β) has no effect. S-mix had the highest development macro Top-10 of the 32 graph variants compared on the same sets, so its development value carries selection optimism. The pure dictionary graph, HPOA-LOPO and SemMedDB were built after the design E confirmation (Section 3.10). Comparator sources never enter the main system's scores. "Evaluations" counts tuning trials, repeats included.
^a^ The highest development macro Top-10 among the dictionary graphs built from the same 12,000-character clinical-section text; the full-text build v3umlsfull_ns (43.08) was excluded because its text differs.
^b^ Scored at S-dict's γ 0.5 for a like-for-like comparison; rule D3 over γ would select 0.25 (42.43 / 17.83).
^c^ Deletion is complete only for Phenopacket-store (Section 3.9), so the nine-set macro is not an LOPO estimate and is not reported; the Phenopacket-store values are in Section 4.5. Three same-size random deletions leave 362,888–362,940 edges.
^d^ With the frozen design D scoring, 17.53 / 7.60. SemMedDB and MeSH co-occurrence reach only the 35,281 CUI-keyed candidate diseases.

### 3.8 통계와 근거 지위

**세트별 대응 비교.** 같은 환자에 대한 두 시스템의 Top-k 차이는 환자 단위 대응 부트스트랩(2,000회, 시드 0, 백분위 95% CI)으로 구간을 낸다. 유의성은 두 시스템의 적중이 갈린 환자에 대한 양측 정확 McNemar 검정으로 보고, 판정 대상 세트를 족으로 삼아 Holm 보정한다(개발 9개, D 확증 3개, E 확증 2개). 판정 세트를 지정한 비교는 그 세트가 두 결과에 같은 환자 수로 들어 있지 않으면 멈춘다.

**정답 질환 군집 CI.** 여러 벤치마크에서 환자가 소수 질환에 몰려 있다. DDXPlus 개발 표본은 3,000명이 48개 질환, RAMEDIS는 624명이 74개 질환, PUMCH-ADM은 75명이 16개 질환이다. 그래서 같은 정답 질환 키를 가진 환자를 통째로 다시 뽑는 군집 부트스트랩 CI(2,000회)를 환자 단위 CI와 함께 낸다. 이 CI의 점추정은 환자 가중 차이 그대로다. 군집 CI는 같은 질환 집합 안에서 질환 단위의 변동을 반영할 뿐이며, 새 질환으로의 일반화를 보이지 않는다. 확증 세트 E도 개발과 같은 질환 집합이므로 E의 군집 CI도 같은 뜻이다.

**질환 등가중.** 환자 가중 차이는 환자마다 1표를 준다. 질환 등가중 차이는 정답 질환 키마다 환자 차이의 평균을 내고, 그 평균들을 같은 무게로 다시 평균한다. 한 질환이 세트의 큰 몫을 차지하면 두 값이 크게 갈린다. 질환 등가중 CI는 질환을 다시 뽑는 부트스트랩(2,000회, 시드 0, 백분위)으로 낸다. 정답 질환 키는 환자의 정답을 사상한 질환 노드 키의 정렬 튜플이며, 군집 CI와 같은 키다. 확증 이후 측정 가운데 M1과 M2는 환자 가중 값, 군집 CI, 질환 등가중 값을 함께 보고한다. P0 측정, V1, P1-3은 환자 CI와 군집 CI를 보고한다. V1과 P1-3은 "모든 비교는 환자 가중과 질환 등가중을 함께 보고한다"는 측정 전 규칙(커밋 f676539)에 속하지만, 질환 등가중 값을 내지 않았다. P0-4의 Phenopacket-store 질환 등가중 값은 M1의 참조 비교에서 나온다(3.10절).

**방향 일관성.** 개발 9개 세트에 걸친 방향 일관성은 차이 부호에 대한 양측 부호검정으로 본다. 차이가 0인 세트는 뺀다.

**매크로 비교.** 매크로 차이는 판정 세트 차이의 등가중 평균이다. CI는 세트별로 환자를 층화하고 두 시스템에 같은 재표본 인덱스를 쓰는 대응 부트스트랩 2,000회(시드 0, 백분위)로 낸다. p는 부트스트랩 분포에서 2 · min(P(Δ ≤ 0), P(Δ ≥ 0))이다. 이 계산(macro_compare)은 D 확증 T4의 세 수치를 그대로 재현한다: 결합 − 그래프 Top-10 +2.67%p [+1.81, +3.58], 결합 − BM25 Top-10 +5.10%p [+3.59, +6.67], 결합 − 그래프 Top-1 +0.96%p [+0.36, +1.54].

**단일 시스템 정확도.** 확증 결과에서 시스템 하나의 Top-k 95% CI는 환자 부트스트랩 1,000회(시드 0)로 낸다.

**개발 매크로.** 개발 매크로는 9개 세트 값(소수 둘째 자리까지 반올림한 값)의 평균이다. 이는 D3 선택에 쓴 계산과 같다. 세트별 표의 값은 환자별 순위에서 직접 반올림했다.

**비열등 한계와 보조 지표.** 비열등 한계는 D의 T1과 E의 E-T2·E-T3 모두 −1%p다. 설계 E는 대응 비교(E-T1·E-T2·E-T3·E-T5)의 Top-1·Top-3 차이, 정답 질환 군집 CI, 열람된 D 확증 세트와 개발 9개 세트에서의 같은 비교를 판정에 쓰지 않는 보조 지표로 정했다. 이 가운데 보고한 것은 Top-1 대응 차이, 정답 질환 군집 CI, 개발 세트 비교다. Top-3는 시스템별 정확도로만 보충자료에 적고, 열람된 D 확증 세트의 비교는 계산하지 않았다. 단, E-T4는 S-mix 단독 Top-1(≥ 18%)을 판정한다.

**커버리지와 프로필 두께.** 계보 커버리지는 정답 질환이나 그 IS_A 하위 질환(깊이 3 이내)에 소견 간선이 하나라도 있는 환자의 비율이다. 이 규칙은 정확도의 lineage 판정과 같다. 정확 일치 커버리지는 정답 키 자신에 간선이 있는지만 본다. 본문은 계보 커버리지를 기본으로 쓰고 정확 일치 값은 괄호에 둔다. 프로필 두께는 한 지식원의 같은 랭커 그래프에서 한 질환 키가 가진 소견 간선(HAS_FINDING)의 수, 곧 CUI로 사상한 서로 다른 소견 개념의 수다. 원 지식원의 주석 행 수와는 다르다. 정답 키가 여럿이면 그 최댓값을 쓴다. 두께는 "간선 1개 이상"인 커버리지보다 강한 조건이다. 두께 층은 0, 1–9, 10–29, 30 이상의 네 층이다.

**부호 규약.** 문헌 그래프와 비교 대상의 차이는 "S-mix(또는 해당 문헌 그래프) − 비교 대상"으로 쓴다. 양수는 문헌 그래프가 높다는 뜻이다. 설계 기록에 "비교 대상 − S-mix"로 적힌 값은 부호를 바꿔 옮긴다. 문헌 그래프 대 비교 대상이 아닌 비교(결합, 랭커, 그래프 판본, 비교 지식원끼리)는 문장에 뺄셈 방향을 적는다.

**근거 지위 표기.** 본문과 표의 수치에는 다음 표기 중 하나를 붙인다.
- [확증]: 봉인 세트에서 한 번 평가하고, 실행 전에 커밋한 기준으로 판정한 결과다. 설계 D의 T1, T2, T4와 설계 E의 E-T1–E-T5가 여기에 든다. 설계 D의 T3는 실행 전에 커밋한 목표이지만 개발 세트에서 판정했으므로 [개발]로 표시한다. PPR 부가 가설(PPR-H1, PPR-H4)은 봉인 세트의 판정 기준이 없으므로 [봉인 기술]이다(3.6절).
- [확증-보조]: 설계 E에 미리 적어 둔 보조 지표(질환 군집 CI, Top-1·Top-3)다. 판정에는 쓰지 않는다.
- [봉인 기술]: D 봉인 세트에서 판정 기준 없이 한 번 돌린 값이다. 동결 비교 지식원(HPOA, PrimeKG, MeSH 공출현)과 PPR의 값이 여기에 든다.
- [사후]: 이미 쓴 확증 세트 E를 확증 뒤에 다시 분석한 값이다. 3.10절 측정의 결정 규칙 행은 규칙을 측정 전에 커밋했다. 그 밖의 값(2026-10-20 리뷰 분석, 측정의 층별 값과 보조 행)은 결정 규칙 없이 낸 값이다.
- [개발]: 9개 개발 세트의 값이다. 선택 낙관이 들어 있다.
- [탐색]: 재현율 분해, 시간 분할, 시뮬레이션, 큐레이션 자원(HPOA)과의 간선 일치 비교의 값이다.
- [측정]: 비용과 규모를 직접 잰 값이다.
- [구조 검사]: 적재 시 불변식, 순위 동일성 같은 구조 검사의 결과다.

영어로 쓴 표에서는 같은 표기를 confirmed, confirmed-auxiliary, sealed descriptive, post hoc, development, exploratory, measured, structural check로 쓴다.

### 3.9 누수와 순환

평가 증례가 이미 지식원에 들어가 있으면 정확도가 부풀 수 있다. 이 상태를 순환(circularity)이라 부르고, 평가 증례의 정보가 지식원에 들어가는 경로를 누수(leakage)라 부른다. 출판 증례 벤치마크에서는 문헌 그래프와 큐레이션 주석 모두 증례의 원논문을 근거로 쓸 수 있다. 그래서 문헌 그래프 쪽(누수 점검)과 큐레이션 쪽(LOPO)을 함께 점검한다.

**그래프 쪽: 원출처 제외.** 문헌 수집은 Phenopacket-store 증례의 원논문을 모든 검색 경로에서 뺐다(v2 1,733편, v3 1,834편; 3.1절). D 확증 세트에서는 원출처 PMID가 v2 코퍼스에 있던 신규 증례 36건을 뺐다(3.6절). 확증 이후 LOPO 측정에서 쓴 증례 PMID 가운데 이 제외 목록에 없던 22개도 S-mix의 문헌에 하나도 없음을 확인했다.

**그래프 쪽: 데이터 기반 누수 점검.** RareBench 세트는 원출처 PMID를 제공하지 않는다. PMID를 뺀 Phenopacket-store에서도 같은 증례가 재보고나 리뷰의 표처럼 다른 문헌을 통해 남을 수 있다. 그래서 정확도를 보기 전에 고정한 기준으로 누수를 데이터로 점검한다. 환자마다 정답 질환의 근거로 그래프에 적재된 문서 가운데 환자 소견 개념을 가장 많이 포함한 문서 하나를 찾는다. 그 문서가 환자 개념의 50% 이상이면서 5개 이상을 포함하면, 그 환자를 원출처 증례일 수 있는 환자로 표시한다. 문서 단위는 근거 언급 그래프의 문서 ID이며, 표 문서는 원 논문과 따로 센다. 이 점검은 처음에 구 Qwen3 그래프에서만 했다. 확증 뒤(P0-1, 3.10절)에는 같은 점검을 개발 9개 세트에서 그래프 15개로 넓혔다. 15개는 S-mix와, 같은 추출을 v2 문헌과 v2+v3 문헌에 적용한 7쌍(S-dict 포함)이다. S-mix는 근거 언급 그래프에서 지지 문서를 직접 읽었다. 나머지 14개는 빌더의 간선 생성을 추출 파일에서 그대로 재현하는 스크립트로 지지 문서를 구했다. 이 재현은 15개 그래프 모두에서 빌드 요약 수치 6개(질환, 소견 있는 질환, 개념, 간선, 언급, 문서)가 같았다. S-mix에서는 개발 환자에게 필요한 (정답 질환, 환자 개념) 21,054쌍의 지지 문서 집합이 근거 언급 그래프와 모두 같았다. 두 그래프를 비교할 때는 어느 한쪽이라도 표시한 환자를 양쪽에서 뺀다. S-D(base_nm)에서는 이 점검을 하지 않았다.

**큐레이션 쪽: 출판 증례 순환.** 자기 인용률은 평가 증례의 원논문 PMID를 근거로 단 HPOA 주석을 정답 질환에 가진 증례의 비율이다. Jiang 등은 Phenopacket-store 0.1.27에서 정답 질환이 HPOA에 있는 증례의 74.6%가 그 증례의 논문을 인용한 주석을 가진다고 보고했다 [ref: Jiang et al. 2026]. 같은 연구는 증례 논문의 주석을 뺐을 때(leave-one-publication-out, LOPO) Phenomizer의 R@1이 0.4481에서 0.1217로 내려간다고 보고했다.

LOPO는 평가 증례가 실린 논문에서 나온 지식을 지식원에서 빼고 채점하는 누수 통제다. 증례마다 그래프를 다시 만들 수는 없으므로, HPOA에서는 전역 삭제판을 주 LOPO 그래프로 썼다(P0-4, 3.10절). 전역 삭제판은 평가하는 Phenopacket-store 증례 10,310건의 파일명과 externalReferences에 있는 PMID 1,745개 가운데 하나라도 인용한 HPOA 행을 모두 지운다. 지운 행은 22,261개다. 다른 출처를 함께 인용한 행도 지웠다. 한 증례로 보면 자기 논문뿐 아니라 다른 증례의 논문도 함께 지워진다. 이 삭제는 경쟁 질환의 주석도 얇게 만들어 정답의 순위를 올릴 수 있다. 그래서 증례별 LOPO와 견준 치우침의 방향은 정해지지 않는다. 전역 삭제판의 HPOA 값은 증례별 LOPO 값의 하한이나 상한으로 읽지 않는다. 대조로는 같은 수의 행을 읽은 행 전체에서 무작위로 지운 그래프 세 개(시드 0–2)를 만들었다. 같은 질환 대조(시드 0)는 질환마다 LOPO가 지운 수만큼의 행을 그 질환에서 LOPO가 남긴 행 가운데 무작위로 지운다. 남은 행이 모자란 질환에서는 남은 행만 지운다. 증례별 LOPO에 가까운 10겹 판도 냈다. 이 판은 PMID를 공유하는 증례를 한 겹에 묶어 증례를 10겹으로 나누고(시드 0), 각 겹의 증례를 그 겹 논문의 행만 지운 그래프로 채점한다. Phenopacket-store에서는 두 판을 함께 보고한다. LOPO 뒤에 HPOA의 채점 설정은 다시 고르지 않았고, S-mix도 동결 설정 그대로 두었다. LOPO가 완전한 세트는 Phenopacket-store뿐이다. 저장소에는 LIRICAL, MME, HMS, RAMEDIS 증례의 PMID가 없다. 이 네 세트와 PUMCH_L, PUMCH-ADM도 같은 LOPO 그래프로 채점되므로, Phenopacket-store 논문의 행을 지운 영향만 받는다. LIRICAL은 출판 증례 모음이며, 환자의 68.1%에서 정답 질환이 행을 잃었다. 그러나 지운 행이 LIRICAL 증례 자신의 원논문인지는 확인할 수 없으므로 LIRICAL의 LOPO는 불완전하다. MME, HMS, RAMEDIS에서는 원논문 주석이 지워졌는지 검사하지 못했다. PUMCH_L과 PUMCH-ADM은 비출판 병원 증례다.

**시간 분할.** 회고 평가에서는 지식원이 평가 증례의 원 논문을 이미 근거로 쓰고 있을 수 있다. 그래서 지식원 시점 이후에 출판된 증례로 따로 평가한다. 이 분석은 [탐색]이다.
- 2022 절단 그래프: 출판연도가 2022년 이하인 문서의 근거만으로 간선을 다시 집계한다. 출판연도가 없는 문서도 뺀다. IDF, 질환별 원문 문자 수, 길이 정규화 상수는 절단 그래프에서 다시 계산된다. 원인 유전자 검색으로 찾은 문서와 원인 유전자 기호에 의한 주제 판정은 HPO 주석 2022-12-15판이 주석한 질환에만 쓴다.
- 증례: Phenopacket-store 0.1.27과 그 뒤 main에 추가된 증례 중 정답과 소견을 사상할 수 있는 10,603건이다. 원 논문(9편)이 v2 코퍼스에 들어 있는 32건은 모든 시간 분할 비교에서 빼고 10,571건을 쓴다. 증례 시점은 원 논문의 PubMed 출판일이며, 2023-01-01 이후이면 미래 증례다.
- 질환 상태: 2022-02 이후 HPO 주석 릴리스 36개에서 정답 OMIM을 처음 주석한 릴리스를 찾는다. 그 날짜가 2022-12-31보다 뒤이거나 2026-09 릴리스까지 주석되지 않았으면 '신규', 아니면 '기주석'이다.
- 네 묶음의 크기: 미래·신규 2,381건(263개 질환), 미래·기주석 1,106건(67개 질환), 과거·신규 1,091건(175개 질환), 과거·기주석 5,993건(405개 질환).
- 비교 대상은 HPO 주석 2022-12-15판을 같은 1-홉 랭커(HPOA의 동결 설정)로 평가한 값이다. 문헌 그래프는 2022 절단판 11개(S-D, S-mix, S-llm, S-dict, S-union의 절단판 포함)를 개발에서 고른 γ가 아니라 γ = 0과 γ = 1에서 모두 평가한다. 본문은 γ = 0의 값을 보고하고, γ = 1의 값은 보충자료에 둔다. 차이의 CI는 환자 대응 부트스트랩(2,000회, 시드 0)이다.

코퍼스 질의에는 2026년의 질환명과 동의어를 썼다. 따라서 2022 절단 그래프는 2022년 당시의 검색 가능성을 과대평가할 수 있다. 이보다 앞서 설계 D 그래프로 한 시간 분할 분석은 보충자료에 둔다.

### 3.10 확증 이후 측정과 결정 규칙

설계 E 확증 뒤에 주장의 범위를 정하려고 측정 아홉 가지를 더 했다. 각 측정의 결정 규칙은 측정 전에 저장소에 커밋했다. 규칙이 판정하지 않는 층별 값과 보조 값은 결정 규칙 없이 보고한다. P0-1–P0-5의 규칙은 커밋 400d4cf(2026-10-20 19:42)에 넣었다. M1, M2, V1, P1-3의 규칙과 P0-5의 수정, P0-4의 해석 규칙은 커밋 f676539(2026-10-20 21:44)에 넣었다. 증빙은 로컬 커밋 시각이다. 결과와 측정 스크립트는 커밋 3ea76b7(P0, 2026-10-21)과 dfa9541(M1, M2, V1, P1-3, 2026-10-21)에 있다. 확증 세트 E를 쓴 측정은 [사후], 개발 세트만 쓴 측정은 [개발]이다. 이 측정들은 확증 판정을 바꾸지 않는다. 판정 결과는 4.1절의 판정 일람에 모았다. 이 측정보다 앞서 2026-10-20의 리뷰 분석(커밋 5800581)도 E의 저장된 순위를 결정 규칙 없이 다시 분석했다. 그 분석은 S-mix − S-dict, S-mix − S-D, S-union − S-dict를 군집 CI와 함께 계산했고, 그 값도 [사후]다.

기존 시스템을 다시 빌드하거나 다시 순위화한 측정은 저장된 순위를 모든 환자에서 재현할 때만 진행했다. HPOA와 PrimeKG의 재빌드는 개발 9개 세트 22,976명의 저장 순위를 모두 재현했다. strict 재순위화는 lineage 순위가 확증 실행과 모든 환자에서 같았다. BM25의 동결 설정 재실행도 확증의 S-bm25 순위를 모든 환자에서 재현했다.

- **P0-1 누수 제외 재계산** [개발]: 3.9절의 표시 환자를 빼고 두 결과를 다시 계산한다. 규칙은 다음과 같다. 같은 추출의 v2 → v2+v3 7쌍에서 수집 깊이 효과(개발 매크로 Top-10)가 모두 원래의 절반 이상 남으면 개발 수준 주장으로 유지하고, 아니면 누수에 민감한 결과로 적는다. 희귀 7개 세트의 S-mix − S-dict는 제외 뒤 매크로 CI 하한이 0보다 크면 유지하고, 아니면 뺀다.
- **P0-2 순수 사전 그래프** [개발; E 평가는 사후]: S-dict는 CUI 없는 사전 항목을 HPO 이름 색인과 LLM 구절 표로 연결했다(3.3절). 순수 사전 그래프(v3umlsnt_ns)는 문헌, 사전 항목, 빌더 규칙이 S-dict와 같고 구절 표만 비웠다. 그래서 CUI 없는 항목은 HPO 이름 색인으로만 연결된다. 규칙은 개발 매크로 Top-10이 S-dict와 1%p 안이면 구축 비용 비교를 "LLM 없는 사전 경로"로 쓰고, 아니면 "구축한 그대로의 S-dict"로 쓰는 것이다. 이 규칙은 개발 값에만 적용되며, E 값은 결정 규칙 없이 보고한다.
- **P0-3 strict 판정** [사후]: E-T2의 네 시스템을 E에서 다시 순위화해 strict 순위를 얻고, E-T2 세 비교의 strict 대응 검정을 낸다. 규칙은 strict에서 부호가 바뀌고 CI가 0을 배제하는 비교가 있으면 그 결론을 판정 규칙에 달린 것으로 쓰는 것이다. 규칙의 판정 단위는 E-T2와 같은 두 세트 매크로 Top-10이며, 세트별 strict 값은 규칙 없이 보고한다. lineage와 strict는 함께 보고한다. Top-10 자리 가운데 정답이 아닌 하위 질환이 차지한 비율(하위 질환 칸 비율)도 그래프별로 보고한다.
- **P0-4 HPOA LOPO** [개발; 해석 문장은 탐색]: 3.9절의 LOPO 그래프와 무작위 삭제 대조로 같은 랭커 채점을 다시 한다. 이 측정은 통과 기준이 없는 보고 규칙이며, 희귀 세트의 차이를 LOPO 전후로 모두 보고한다. 해석 규칙은 Phenopacket-store의 질환 등가중 값을 기준으로 한다. HPOA의 우위가 10%p 이하로 줄고 무작위 삭제보다 크게 줄면 "순환을 걷어내면 격차가 작아졌다"를 [탐색]으로 쓴다. HPOA의 우위가 30%p 이상 남으면 "순환을 걷어내도 격차가 컸다"를 쓴다. 그 사이면 수치만 쓴다. P0-4 산출물(커밋 3ea76b7)에는 환자 가중 값과 군집 CI만 있다. Phenopacket-store의 질환 등가중 S-mix − HPOA는 LOPO 전과 후 모두 M1의 참조 비교에서 계산되었다(커밋 dfa9541). 이 비교는 같은 HPOA·LOPO 그래프와 같은 정답 질환 키를 쓴다. 무작위 삭제 대조의 질환 등가중 값은 계산하지 않았다. 그래서 "무작위 삭제보다 크게 줄었는가"는 환자 가중 값으로 판단하며, 이 부분만 규칙의 기준과 다르다.
- **P0-5 SemMedDB 같은 랭커** [개발]: 원래 규칙은 SemMedDB를 동결 1-홉 채점으로 평가하는 것이었다. 이 채점은 설정을 다시 고른 HPOA·PrimeKG보다 SemMedDB에 불리하다. 그래서 측정 전에 같은 격자와 D3 규칙으로 다시 고른 값을 주 결과로 바꾸고, 동결 채점 값을 함께 보고하기로 고쳤다. 첫째 규칙은 SemMedDB의 개발 매크로 Top-10이 S-dict 이상이면 "공개 규칙 기반 자원이 같거나 높다"를 본문에 쓰는 것이다. 둘째 규칙은 흔한 질환 두 세트의 S-dict − SemMedDB를 질환 군집 CI로 읽는다. 두 세트 모두 하한이 0보다 크면 "시험한 공개 지식원 가운데 이 공간에서 가장 높았다"를 더한다. 한 세트라도 상한이 0보다 작으면 문헌 그래프의 우세 문장을 빼고 "공개 규칙 기반 자원이 같은 공간을 이미 다루며 더 높았다"를 쓴다. 그 밖이면 "SemMedDB와 구별되지 않았다"고 쓰고, 우세 문장은 HPOA, PrimeKG, MeSH 공출현 대비로만 쓴다.
- **M2 E 세트의 큐레이션 비교군** [사후]: 설계 D의 동결 HPOA·PrimeKG 1-홉 설정을 E에 한 번 돌린다. 결과는 HPOA 계보 커버리지 여부와 HPOA 프로필 두께로 층화한다. 두께 층의 경계는 개발 세트 결과를 본 뒤 정했고, E 측정 전에 커밋했다. 규칙은 두 세트 모두 S-mix − HPOA의 군집 CI 하한이 0보다 크면 큐레이션 주석이 얇은 공간에 관한 문장에 E 값을 더하고, 아니면 그 세트의 문장을 빼는 것이다. 두께 30 이상 층은 판정 없이 보고한다.
- **M1 HPOA와 S-mix의 결합** [개발]: 순위 결합과 간선 합집합(3.7절)을 세 층으로 평가한다. 층은 흔한 질환(DDXPlus, SymCat), 비출판 병원 증례(PUMCH_L, PUMCH-ADM), 공개 희귀질환 증례(RAMEDIS, LIRICAL, HMS, MME, Phenopacket-store; HPOA-LOPO판과 함께)다. 규칙은 비출판 병원층 매크로에서 결합 − HPOA와 결합 − S-mix의 CI 하한이 모두 0보다 클 때만 "보완"이라고 쓰는 것이다. 측정 스크립트는 환자 층화 CI, 군집 CI, 질환 등가중 CI가 모두 이 조건을 만족해야 통과로 했다.
- **V1 BM25 민감도** [사후]: v2+v3 BM25 12개 조합 전부와 설계 D의 v2 BM25를 E에서 돌린다. 측정 전에 커밋한 규칙은 12개 조합 모두에 대해 두 세트에서 S-mix − BM25 Top-10의 CI 하한이 0보다 크면 "BM25 설정과 무관하게"로 쓰고, 아니면 "개발에서 고른 BM25 설정 대비"로 한정하는 것이다. 원고는 이 문구를 "시험한 12개 BM25 설정 모두에 대해"로 좁혀 쓴다. 규칙을 환자 CI와 군집 CI에서 따로 읽는 방식은 측정 스크립트에 적었다. 이 스크립트는 결과와 같은 커밋(dfa9541)에 처음 들어갔으므로, 이 읽기 방식은 측정 전 커밋으로 확인되지 않는다. v2 BM25는 희석의 크기를 봉인 환자에서 재는 데 쓴다. 이 원고는 흔한 질환 세트에서 같은 설정의 BM25가 더 큰 코퍼스에서 낮아진 현상을 희석이라 부른다.
- **P1-3 입력 내 정답 명시 층화** [사후]: E-T2의 비교를 입력 내 정답 명시 층(3.6절)과 나머지 층(없음 층)으로 나눈다. 저장된 순위만 쓴다. 규칙은 DDXPlus-E 없음 층의 lineage Top-10에서 S-mix − S-dict의 CI가 0을 포함하거나 양수이면, S-dict의 우세가 입력 구조에서 온다는 설명을 봉인 세트 수준([사후])으로 쓰고, 아니면 개발 수준으로만 두는 것이다.

M1과 M2의 규칙에는 판정할 k가 적혀 있지 않았다. 그래서 D3의 주 지표인 Top-10으로 판정했다. M1은 이 선택과, 환자 층화 CI·군집 CI·질환 등가중 CI가 모두 조건을 만족해야 한다는 요구를 측정 스크립트에 적었다. 이 스크립트도 결과와 같은 커밋(dfa9541)에 처음 들어갔으므로, 이 둘은 측정 전 커밋으로 확인되지 않는다. 같은 기록에 적은 보조 측정 가운데 MedCPT 재선택, HPO 고유 의미 유사도 랭커 평가, 수집 비용 기록(v3 수집 로그의 벽시계·CPU 시간)은 하지 않았다. 4.4절의 v3 수집 시간은 2026-10-20 리뷰 분석의 비용 표에 적은 벽시계 구간이다.

<!-- 작성 메모(본문 아님): (1) 커밋 해시. 설계 문서, frozen_config_D.json, confirmE/attempts.log는 이력 재작성 전 해시(31122e2, cd130e7, 2ecb064, 5ce9c3d, 3b10c36, a7d0680, 61e3118, 0cb1c59, 49901bc)를 쓴다. 이 해시들은 origin/main에 없다. 각 쌍은 `git diff --name-only`로 docs/audit/provenance_audit_v21_user.csv 하나만 다르고 작성·커밋 시각이 같다. 3.6절 "커밋 기록"에 대응을 적었다. MET-A 3.4와 RES 절도 같은 규약(실행 기록 해시 + 3.6절 대응표)을 따라야 한다. (2) 설계 E 브랜치가 원격에 처음 생긴 시각은 2026-10-20 01:52 KST(GitHub activity API)로, 확증 종료(01:34) 뒤다. "실행 전 원격 기록은 없다"의 근거다. (3) [커밋 필요] 표기는 3.8절에 정의하지 않았다. 커밋하지 않은 값은 원고에 인용하지 않기 때문이다. 다른 절에 이 표기가 남아 있으면 원고를 마치기 전에 모두 해소해야 한다. (4) 질환 등가중의 질환 키는 커밋된 스크립트(m2_curated_e.py, m1_combine.py)가 쓰는 variants_analysis.gold_clusters(사상된 정답 키의 정렬 튜플)로 적었다. PUMCH_L의 질환 등가중 +1.5는 이 키의 값이지만 아직 스크래치패드에만 있다. (5) P0-4의 Phenopacket-store 질환 등가중 값은 pilot/cache/v3/framing/m1/stats.json(커밋 dfa9541)의 참조 행 "HPOA - S-mix", "HPOA-LOPO - S-mix"에 있다(Top-10, 772개 질환, 규칙과 같은 방향인 HPOA − S-mix: LOPO 전 +55.73 [+52.32, +58.89], LOPO 후 −3.28 [−5.79, −0.76]). 같은 행의 환자 가중 값은 P0-4와 같다(+42.66, −5.58). 무작위 삭제 대조의 질환 등가중 값은 계산·커밋되지 않았다. 이 값을 커밋하면 3.10절 P0-4의 편차 문장을 지운다. §9.34와 P0 요약은 질환 등가중 값을 적지 않았다. (6) 3.9절은 hpoa_lopo.py 머리말과 P0-4 요약을 따라 전역 삭제의 치우침 방향을 "정해지지 않는다"로 적었다(Phenopacket-store HPOA Top-10: 전역 25.45, 10겹 25.60). §9.34, §9.35 M1 항목, 프레이밍 검토 5.1 행 3의 "HPOA에 불리한 전역 삭제"·"순환 효과의 상한"은 이 기록과 어긋난다. RES-B 4.5도 3.9절을 따라야 한다. (7) M1 보고서에 따르면 빌드 요약 지문의 IDF 항은 저장된 IDF가 아니라 log(Disease 노드 수/df)로 계산된다. df 변화는 여전히 지문을 바꾸지만, MET-A 3.4의 "순위화가 읽는 모든 행의 지문"은 이에 맞게 고쳐야 한다. (8) 개요의 HPOA 튜닝 "38설정"은 좌표 탐색 평가 38회이며 서로 다른 설정은 28개다(ref_hpoa/tune.json trail). 다른 절도 "평가 38회(서로 다른 설정 28개)"로 쓴다. (9) PPR-H1·PPR-H4를 [확증]에서 [봉인 기술]로 옮겼다. 근거는 설계 D10(공개 커밋 50a64db)이다. D10은 E-H1을 개발 매크로 Top-10의 부호검정과 Holm으로 적었고, 확증은 "PPR 시스템을 함께 넣는다"고만 적었다. RES-A 4.1의 표 6a(행 5–6, "판정 13개")와 본문의 [확증] 표기는 이에 맞게 고쳐야 한다. 표 6a에 남기고 [봉인 기술]로 표시할지, 표 밖으로 옮길지는 저자가 정한다. (10) 설계 D 판정 기준 커밋 3fbd29a와 PPR 설계 커밋 50a64db는 공개 이력 해시다(재작성 전 a167a9a, 19fc7df, `git diff --name-only`로 provenance_audit_v21_user.csv 하나만 다름). GitHub activity API에서 2026-07-13 뒤 첫 원격 활동은 2026-10-19 16:22 KST(feat/ie-noise-sensitivity 브랜치 생성)다. 그래서 설계 D의 기준과 확증(10-12–10-13)에도 실행 전 원격 기록이 없다. RES-A 표 6a의 "Criteria at 3fbd29a and sets sealed at 31122e2"는 공개 해시와 실행 기록 해시가 섞여 있다. (11) V1·P1-3의 질환 등가중 값은 계산되지 않았다. 계산해 커밋할지, 3.8절의 편차 문장으로 둘지는 저자가 정한다. -->
