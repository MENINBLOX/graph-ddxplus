## 4. 결과

### 4.1 코퍼스와 그래프

표 2는 두 코퍼스의 규모를 비교한다. v2 코퍼스는 질의한 51,894개 질환 중 12,886개(24.8%)에서 문헌을 얻지 못했다. 비인간 질환 8,319개를 포함한 이 51,894개 질환 기준으로 질환–문서 연결은 304,428건, 고유 문서는 136,373개였다. 원출처 PMID 1,733편 가운데 검색 결과에 나타난 269편을 수집에서 제외했다. 표 2의 v2 열은 v3 열과 같이 사람 질환 43,575개 기준이다. 사람 질환 중 문헌이 있는 질환은 32,545개(74.7%)였다. 사람 질환의 질환–문서 연결은 246,203건, 고유 문서는 123,772개였다. 텍스트는 고유 문서 기준 6.0억 자였다. 질환–문서 연결마다 본문 길이를 더하면(같은 문서가 여러 질환에 연결되면 여러 번 셈) 12.5억 자였다. 질환 파일(비인간 질환 포함)에 모두 20회 이상 나열된 논문 1,119편을 일반 문헌 규칙으로 빼면 2,663개 질환은 문헌이 하나도 남지 않았다.

v3 재수집은 사람 질환 43,575개 전부에 같은 상한을 적용했다(3.2절). v3의 새 문서는 해당 질환의 v2 목록에 없던 문서다. 새 문서를 하나 이상 얻은 질환은 35,006개였다. 새 질환–문서 연결은 1,619,168건이었다. 그 구성은 PubMed 초록 920,514건, PMC 표 문서 494,462건, 새 PMC 문헌의 임상 절 204,192건이었다. 고유 문서 ID는 906,347개, 고유 PMID는 797,366개였다. 그중 문서 ID 24,236개는 v2에서 다른 질환의 문헌으로 이미 수집된 것이었다. PMID로는 92,221개가 v2와 겹쳤다. 표 문서는 원 논문의 PMID를 가지므로, v2 논문의 표 문서도 이 겹침에 들어간다. 새 텍스트는 고유 문서 기준 21.4억 자로 v2 사람 질환 텍스트(6.0억 자)의 약 3.6배였다. 그중 표가 10.0억 자, 초록이 7.6억 자, PMC 임상 절이 3.8억 자였다. 질환–문서 연결마다 더하면 42.6억 자(표 22.5억 자, 초록 12.7억 자, PMC 임상 절 7.5억 자)였다. 이는 같은 방식으로 센 v2 사람 질환 텍스트(12.5억 자)의 약 3.4배다.

v2와 v3를 합치면 논문 단위(표 문서를 원 논문에 묶은 단위)로 882,014편이다. 같은 일반 문헌 규칙으로 4,122편이 제외되었다. 문헌이 있는 사람 질환은 35,308개(81.0%)로 늘었다. 일반 문헌 규칙 때문에 문헌이 모두 빠지는 질환은 2,663개에서 938개로 줄었다. 사람 질환 8,267개(19.0%)는 v3에서도 문헌이 없다.

LLM 추출 항목(원문에서 위치가 확인된 구절)은 v2 문헌에서 533만 개였다(모델 출력 560만 개). v3 문헌의 추출까지 합치면 17,654,170개다. 이 중 구절 표로 개념에 연결된 항목은 36.9%(6,518,369개)였다. 연결되지 않은 사유 중 가장 큰 것은 근거 부족(3.3절 규칙 5)으로, 전체 항목의 43.0%였다. 근거 부족은 연결된 개념의 이름·별칭이 구절 안에 연속으로 나타나지 않거나, 나타나도 구절 내용어의 절반 미만만 덮는 경우다.

Table 2. Corpus statistics, v2 vs v3. Human diseases = 43,575 candidates (8,319 non-human animal diseases removed). In the v2 column, links, unique documents and text count human diseases only (footnote b gives all 51,894 queried diseases). v3 counts cover documents not in the same disease's v2 list; 24,236 v3 document IDs (92,221 PMIDs, which include the table documents of v2 articles) also occur in v2 under other diseases.

| Item | v2 | v3 (new documents) | v2 + v3 |
|---|---|---|---|
| Collection caps per disease | PMC OA reviews 15, case reports 10 (clinical sections, 12,000 characters per article); StatPearls and GeneReviews chapters by title match (24,000 characters per chapter) | PMC OA reviews 30 + case reports 30 (clinical sections, 12k characters, plus tables); PubMed abstracts 50; per causal gene (≤ 2 genes per disease): abstracts 50 + PMC 20 | — |
| Diseases queried | 51,894 (including 8,319 non-human) | 43,575 | — |
| Human diseases with ≥ 1 document | 32,545 (74.7%) | 35,006 (with new documents) | 35,308 (81.0%) |
| Human diseases without any document | 11,030 (25.3%) | — | 8,267 (19.0%) |
| Disease–document links | 246,203^b^ | 1,619,168 (abstracts 920,514; table documents 494,462; PMC clinical sections 204,192) | — |
| Unique documents | 123,772^b^ | 906,347 document IDs (797,366 PMIDs) | 882,014 articles^a^ |
| Text over unique documents (characters) | 0.60 B^b^ | 2.14 B (tables 1.00 B; abstracts 0.76 B; PMC clinical sections 0.38 B) | — |
| Text summed over disease–document links (characters)^c^ | 1.25 B^b^ | 4.26 B (tables 2.25 B; abstracts 1.27 B; PMC clinical sections 0.75 B) | — |
| Source-paper PMIDs excluded (leakage control) | 1,733 (269 appeared in search results)^d^ | 1,834 | — |
| Generic articles excluded (listed ≥ 20 times across disease files)^e^ | 1,119 | — | 4,122 |
| Human diseases left with no document by the generic rule | 2,663 | — | 938 |
| LLM items (location-verified spans) | 5.33 M (5.60 M model outputs) | — | 17.65 M |
| LLM items linked by the phrase table | — | — | 36.9% |

^a^ A table document is counted with its source article.
^b^ All 51,894 queried diseases, non-human included: 304,428 links; 136,373 unique documents; 0.66 B characters over unique documents; 1.56 B characters summed over links.
^c^ A document linked to several diseases is counted once per disease.
^d^ Phenopacket-store release 0.1.27 only. The source papers (9) of 32 cases added after 0.1.27 remain in the v2 portion; the v3 collection excluded all 1,834.
^e^ Counted over all disease files, non-human diseases included; a file that lists an article twice counts twice.

표 5b는 이 논문에서 비교하는 문헌 그래프의 규모를 보여 준다. 모든 그래프는 Disease 노드 43,575개를 후보로 가지며, 소견 간선이 없는 질환도 후보에 남는다. 주 시스템 S-mix는 HAS_FINDING 간선 5,221,285개, 소견 있는 질환 33,537개, 개념 85,761개, 근거 문서 711,350개로 이루어진다. S-mix의 간선 수는 설계 D 동결 그래프 S-D(832,671개)의 6.3배이고, 같은 v2+v3 문헌의 LLM 단독 그래프 S-llm(3,672,106개)의 1.4배다. UMLS 사전 그래프 S-dict의 간선(14,506,788개)은 S-mix의 2.8배다. 소견 있는 질환은 S-D 27,309개에서 S-mix 33,537개로 늘었다. 자기 간선 제거로 빠지는 간선은 v3 LLM 그래프에서 24,562개로, 제거 전 간선의 0.7%였다.

Table 5b. Literature graphs compared in this paper (build detail for the graph systems of Table 5). Scoring keys other than offtopic_w are the design D frozen values (α 0.35, μ 2, decay 0.3, unified_w 2.0, β 0.5). Every graph has 43,575 Disease nodes.

| System | Graph | Literature | Extraction | Builder rules | offtopic_w^a^ | Diseases with findings | Concepts | HAS_FINDING edges | Mentions^b^ | Documents^c^ | Dev macro Top-10 / Top-1 (%)^d^ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-D | base_nm | v2 | LLM (gemma-4-E4B-it) | none; self-edges kept | 0 (frozen) | 27,309 | 46,130 | 832,671 | 1,201,064 | 86,276 | 30.56 / 12.64 |
| S-llm | v3llm_ns | v2 + v3 | LLM | coordination split, HPO names, causal-gene titles; self-edges removed | 1.0 | 33,123 | 76,487 | 3,672,106 | 7,010,155 | 589,666 | 44.90 / 19.30 |
| S-dict | v3umls_ns | v2 + v3 | UMLS finding-string dictionary (no LLM extraction)^e^ | same as S-llm | 0.5 | 34,356 | 95,608 | 14,506,788 | 59,796,310 | 984,061 | 42.88 / 18.11 |
| S-union | v3unionu_ns | v2 + v3 | LLM + UMLS dictionary | same as S-llm | 0.5 | 34,359 | 115,224 | 15,396,388 | 66,898,838 | 986,098 | 43.77 / 17.59 |
| S-mix | v3llmd_ns | v2 + v3 | LLM spans; spans the phrase table cannot link (except negated, over-8-word and table-absent spans) are named by UMLS strings inside the span | same as S-llm | 0.5 | 33,537 | 85,761 | 5,221,285 | 11,582,169 | 711,350 | 46.11 / 19.66 |

^a^ Re-selected per graph on the nine development sets (design D rule D3) from {0, 0.25, 0.5, 1}.
^b^ Aggregated evidence-mention count. Development builds have no Mention nodes; the frozen S-D graph and the S-mix rebuild with Mentions (Section 4.1) do.
^c^ Distinct document IDs supporting at least one edge; a PMC table document is counted separately from its article.
^d^ Development values (nine sets). At each graph's offtopic_w selected on these sets, S-mix had the highest macro Top-10 and Top-1 of the 32 graphs evaluated (base_nm included; 18 on v2 literature, 14 on v2 + v3 literature, two of which read the full PMC text instead of the 12k clinical sections), so its value includes selection optimism (Section 4.4).
^e^ CUI-less dictionary items are linked through HPO names and the v2 phrase table, which was built from LLM phrases (Section 3.3). In a 1/20 sample of literature files, 4.6% of dictionary items received their concept only through this table (Section 4.4.7).

구축과 분석 중 발견한 무증상 실패와 그 수정을 정리한다. 수집에서는 질환명 변형 함수가 6자 미만 이름을 버려 Gout, Croup, Mumps 등 25개 질환의 질의가 한 번도 나가지 않았다. 원래 이름을 항상 질의하도록 고친 뒤 이 질환들을 재수집했다. 벤치마크 전처리에서는 DDXPlus 근거 사상이 응답 값을 무시해, 여행하지 않았다는 응답(trav1_@_N)이 환자 93%에서 'Travel History'로 사상되었다. 기본값 응답은 근거에서 빼도록 고쳤다. DDXPlus 정답 사상은 ICD-10 경로 때문에 Pneumonia가 'Pneumonia in diseases classified elsewhere'로 연결되는 식으로 어긋나 있었다. 영문 질환명의 UMLS 정확 일치를 우선하도록 고쳤다. 정규화에서는 의미유형 표 누락으로 언급 20.6만 개가 잘못 탈락했다. 의미유형을 UMLS MRSTY 원본에서 읽도록 고쳤다.

그래프 적재에서는 기존 그래프를 지우는 일괄 삭제가 일부 배치에서 실패했으나, 실패가 서버 경고로만 남고 빌드는 정상 종료했다. 그 결과 제외 대상 비인간 질환 노드 2,682개와 근거 문장이 없는 이전 빌드 간선 62,497개가 그래프에 남았다. 이 실패는 Disease 노드 수(46,257)가 후보 질환 수(43,575)와 다르다는 불변식 위반으로 드러났다. 이 그래프로 낸 설계 D 이전 탐색 단계의 v2.0 수치와 첫 v2.1 수치는 무효 처리했다. 이후 빌드는 삭제 실패나 Disease 수 불일치가 있으면 중단한다. Mention을 넣은 빌드는 언급 없는 간선이 있어도 중단한다. 표 5b의 v3 그래프(S-llm, S-dict, S-union, S-mix)의 개발값은 Mention 없는 빌드에서 나왔다. 이 빌드들은 언급 검사를 거치지 않았다. S-mix는 동결 전에 Mention을 넣어 다시 빌드했고(3.6절 절차의 (3)단계), 이 빌드는 출처 불변식 검사를 통과했다. 언급 없는 HAS_FINDING 간선은 0개, Disease 노드는 43,575개, Mention 노드는 11,582,169개였다. 순위화가 읽는 행의 지문은 Mention 없는 빌드와 같았다(15ba677…). 개발 DDXPlus 3,000명 모두에서 정답 질환의 Top-10 내 위치(lineage, Top-10 밖이면 없음)도 Mention 없는 빌드와 같았다.

설계 D 확증 이후의 분석 도구에는 에이전트 기반 코드 리뷰를 두 차례 했고, 확인된 결함 8건과 11건을 모두 수정했다. 2026년 유전자–질환 매핑으로 찾은 문서가 2022년 절단 그래프에 남아 있었다. 같은 논문의 초록과 PMC 본문이 중복 적재되었다. 표 문서가 원 논문과 별개로 일반 문헌 규칙을 통과했다. 분석 스크립트가 평가와 다른 정답 집합을 읽었다(Phenopacket 정답 집합을 평가는 10,310명, 분석 스크립트는 8,823명으로 읽음). 결과 재사용 판정이 요약 수치 6개만 비교해, 순위화에 쓰이는 행이 바뀌어도 이전 결과가 남을 수 있었다. 마지막 결함은 빌드가 순위화에 쓰이는 모든 행의 지문(fingerprint)과 채점 설정을 기록해 비교하도록 고쳤다.

### 4.2 설계 D 확증

설계 D의 동결 시스템을 2026-10-13에 봉인 확증 세트에서 한 번 평가했다(동결 커밋 cd130e7, 실행 커밋 2ecb064). 주 시스템은 v2 문헌·gemma-4-E4B-it 추출 그래프 S-D의 1-홉 질의였다. 확증 세트는 DDXPlus 3,000명, Phenopacket-new 369명(정답 키와 소견 개념이 있는 261명 평가), SymCat 7,482명이다. 후보는 43,575개 질환 전체이고 판정은 lineage다. 표 6은 모든 시스템의 Top-1/3/10을, 표 7은 사전에 고정한 판정 기준과 결과를 나란히 보여 준다.

주 목표 T1(그래프 단독이 같은 문헌의 BM25보다 높음)은 미달이었다. 개발 9개 세트에서는 S-D의 Top-10이 9개 모두 BM25보다 높았다(부호검정 p = .0039). 확증에서 S-D와 BM25의 Top-10 차이는 DDXPlus +7.1%p [+5.0, +9.1](Holm 보정 p = 3.5 × 10⁻¹²), Phenopacket-new +0.0%p [−4.2, +3.8], SymCat +0.2%p [−0.6, +1.0]이었다. Phenopacket-new는 CI 하한(−4.2%p)이 비열등 한계(−1%p)보다 낮아 비열등 조건을 충족하지 못했다. Holm 유의 우세는 DDXPlus 하나였고 기준은 두 세트 이상이었다. 개발에서의 9/9 우세는 확증에서 흔한 질환 세트 하나에서만 재현되었다. Top-1에서는 SymCat(+1.3%p [+0.9, +1.7])만 유의하게 높았다.

T2(DDXPlus Top-10 ≥ 40%, Top-1 ≥ 18%)도 미달이었다. S-D는 35.2/12.4%, 그래프+BM25 결합은 37.8/14.9%로 둘 다 기준보다 낮았다. 개발 세트에서 판정하는 T3(PUMCH_L·PUMCH-ADM Top-10이 같은 랭커의 HPO 주석 이상)도 미달이었다. PUMCH_L은 S-D 50.7% 대 HPO 주석 53.2%, PUMCH-ADM은 56.0% 대 57.3%였다. 두 차이(−2.5%p [−5.6, +0.5]; −1.3%p [−13.3, +10.7])는 유의하지 않았다(보정 전 정확 McNemar p = .11, 1; 개발 9개 세트 Holm 보정 시 PUMCH_L p = .22). HPO 주석 기준값은 D3 규칙에 따라 개발 세트에서 다시 고른 설정의 값이다. 사전 기준표의 validate 선택값(53.6/53.3%)으로 판정해도 PUMCH_L에서 미달이다.

T4(결합이 확증 매크로에서 두 단독보다 유의하게 높음)는 충족했다. 세 세트 등가중 매크로 Top-10에서 결합은 S-D보다 +2.7%p [+1.8, +3.6], BM25보다 +5.1%p [+3.6, +6.7] 높았다. Top-1에서도 결합은 S-D보다 +1.0%p [+0.4, +1.5], BM25보다 +1.8%p [+1.1, +2.6] 높았다(세트별 층화 대응 부트스트랩 2,000회, 모두 p ≤ .001). 세트별로도 결합의 Top-10은 세 세트 모두 S-D보다 Holm 유의하게 높았다(DDXPlus +2.7%p, Phenopacket-new +3.1%p, SymCat +2.3%p). BM25와 비교하면 DDXPlus(+9.8%p)와 SymCat(+2.5%p)에서 유의했고, Phenopacket-new(+3.1%p [−1.1, +7.3])에서는 유의하지 않았다.

Table 6. Design D confirmation: lineage Top-1 / Top-3 / Top-10 (%, patient-bootstrap 95% CI, B = 1,000). Open set of 43,575 diseases. Evaluated once on 2026-10-13 (freeze commit cd130e7; run at commit 2ecb064, no configuration change). Phenopacket-new: 261 of 369 cases evaluable.

| System | Dataset | Top-1 | Top-3 | Top-10 |
|---|---|---|---|---|
| Literature graph S-D, 1-hop | DDXPlus (3,000) | 12.4 [11.3, 13.6] | 20.8 [19.3, 22.2] | 35.2 [33.6, 36.9] |
| | Phenopacket-new (261) | 1.9 [0.4, 3.8] | 5.4 [3.1, 8.0] | 9.2 [5.7, 12.3] |
| | SymCat (7,482) | 3.3 [2.9, 3.7] | 6.0 [5.5, 6.5] | 11.1 [10.4, 11.8] |
| BM25 (same v2 literature) | DDXPlus | 12.0 [10.9, 13.2] | 19.0 [17.7, 20.5] | 28.0 [26.5, 29.8] |
| | Phenopacket-new | 1.1 [0.0, 2.7] | 4.2 [1.9, 6.9] | 9.2 [5.7, 12.6] |
| | SymCat | 2.0 [1.7, 2.3] | 4.6 [4.2, 5.1] | 10.9 [10.2, 11.6] |
| Graph + BM25 fusion (w = 0.25) | DDXPlus | 14.9 [13.7, 16.2] | 21.0 [19.6, 22.3] | 37.8 [36.1, 39.5] |
| | Phenopacket-new | 1.9 [0.4, 3.4] | 5.4 [2.7, 8.0] | 12.3 [8.4, 16.1] |
| | SymCat | 3.7 [3.3, 4.1] | 7.3 [6.7, 7.8] | 13.4 [12.6, 14.1] |
| Literature graph S-D, PPR^c^ | DDXPlus | 4.5 [3.7, 5.2] | 12.2 [11.1, 13.5] | 19.4 [18.0, 20.8] |
| | Phenopacket-new | 3.1 [1.1, 5.0] | 5.0 [2.7, 8.0] | 7.7 [4.6, 10.7] |
| | SymCat | 2.6 [2.3, 3.0] | 5.1 [4.7, 5.6] | 9.5 [8.9, 10.2] |
| HPO annotations, 1-hop^a,b^ | DDXPlus | 2.3 [1.8, 2.8] | 3.9 [3.2, 4.6] | 10.3 [9.2, 11.3] |
| | Phenopacket-new | 14.9 [10.7, 19.5] | 24.1 [19.2, 29.5] | 36.0 [30.3, 41.8] |
| | SymCat | 0.8 [0.6, 0.9] | 1.5 [1.2, 1.8] | 3.8 [3.3, 4.2] |
| HPO annotations, PPR^a,c^ | DDXPlus | 4.1 [3.4, 4.8] | 6.8 [6.0, 7.7] | 11.2 [10.1, 12.3] |
| | Phenopacket-new | 10.7 [7.3, 14.6] | 16.5 [11.9, 20.7] | 23.8 [18.8, 29.1] |
| | SymCat | 2.2 [1.8, 2.5] | 3.6 [3.3, 4.1] | 6.6 [6.1, 7.2] |
| PrimeKG, 1-hop^a,b^ | DDXPlus | 4.2 [3.5, 4.9] | 7.5 [6.6, 8.5] | 12.0 [10.9, 13.2] |
| | Phenopacket-new | 8.0 [5.0, 11.5] | 14.2 [10.3, 18.0] | 31.8 [26.1, 36.8] |
| | SymCat | 1.9 [1.5, 2.2] | 3.5 [3.0, 3.8] | 5.6 [5.1, 6.0] |
| PrimeKG, PPR^a,c^ | DDXPlus | 2.8 [2.2, 3.4] | 3.7 [3.1, 4.4] | 7.2 [6.3, 8.2] |
| | Phenopacket-new | 6.1 [3.4, 9.2] | 7.3 [4.2, 10.3] | 9.2 [5.7, 12.6] |
| | SymCat | 2.2 [1.9, 2.5] | 3.6 [3.2, 4.1] | 5.7 [5.2, 6.3] |
| MeSH co-occurrence, 1-hop^b^ | DDXPlus | 10.2 [9.1, 11.3] | 14.9 [13.7, 16.3] | 25.6 [24.1, 27.2] |
| | Phenopacket-new | 0.0 [0.0, 0.0] | 1.1 [0.0, 2.3] | 1.5 [0.4, 3.1] |
| | SymCat | 2.8 [2.5, 3.2] | 4.5 [4.1, 5.0] | 7.5 [7.0, 8.2] |

^a^ Curated from published cases that overlap with rare-disease cases; shown as a reference for the knowledge gap, not as a competing system.
^b^ Separate Neo4j instance with the same Cypher ranker.
^c^ PPR computed by sparse power iteration on the graph exported from Neo4j, with settings selected per graph on the development sets (restart probability: S-D 0.3 with uniform seeds; HPO annotations 0.5 and PrimeKG 0.7 with IDF seeds).

Table 7. Design D judgments (criteria fixed before the confirmation run). Differences are the second system minus the first in percentage points, with paired patient-bootstrap 95% CI (2,000 replicates) and exact McNemar test with Holm adjustment across the three confirmation sets; macro = equal-weight mean of the three sets with set-stratified paired bootstrap.

| Target | Criterion | Result | Verdict |
|---|---|---|---|
| T1 (primary) | Graph > BM25: dev ≥ 8/9 sets (sign test p < .05) and, in each confirmation set, Top-10 difference CI lower bound > −1 pp, with ≥ 2 sets Holm-significant | Dev 9/9 (p = .0039). Confirmation Top-10: DDXPlus +7.1 [+5.0, +9.1] (Holm p = 3.5 × 10⁻¹²); Phenopacket-new +0.0 [−4.2, +3.8]; SymCat +0.2 [−0.6, +1.0] | Not met |
| T2 | DDXPlus Top-10 ≥ 40% and Top-1 ≥ 18% | Graph 35.2 / 12.4; fusion 37.8 / 14.9 | Not met |
| T3 (dev)^a^ | PUMCH_L and PUMCH-ADM Top-10 ≥ HPO annotations with the same ranker | PUMCH_L 50.7 vs 53.2 (−2.5 [−5.6, +0.5], p = .11); PUMCH-ADM 56.0 vs 57.3 (−1.3 [−13.3, +10.7], p = 1) | Not met |
| T4 | Fusion significantly above both single systems on the confirmation macro | Fusion − graph: Top-10 +2.7 [+1.8, +3.6], Top-1 +1.0 [+0.4, +1.5]; fusion − BM25: Top-10 +5.1 [+3.6, +6.7], Top-1 +1.8 [+1.1, +2.6]; all p ≤ .001 | Met |
| H1 (secondary)^b^ | PPR vs 1-hop on the same graph (two-sided) | Top-10: DDXPlus −15.8 [−17.3, −14.3], Phenopacket-new −1.5 [−4.6, +1.5], SymCat −1.6 [−2.2, −1.0] | PPR lower (2 of 3 sets Holm-significant) |
| H4 (secondary)^b^ | PPR vs 1-hop on curated graphs | HPO annotations: SymCat Top-10 +2.8 [+2.3, +3.3], DDXPlus Top-1 +1.8 [+1.2, +2.4], Phenopacket-new Top-10 −12.3 [−16.5, −7.7]; PrimeKG: DDXPlus Top-10 −4.8 [−5.8, −3.7], Phenopacket-new Top-10 −22.6 [−27.6, −17.2] | PPR lower on the rare-disease set for both sources |

^a^ Development sets. p values are unadjusted exact McNemar p; with Holm adjustment across the nine development sets, PUMCH_L p = .22 and PUMCH-ADM p = 1.
^b^ H1 and H4 are the secondary PPR hypotheses of design D, specified before the confirmation run (Section 3.6).

사전 지정 부가 가설의 결과는 개발 결과와 같은 방향이었다. 같은 그래프에서 PPR − 1-홉 Top-10 차이(H1)는 DDXPlus −15.8%p [−17.3, −14.3], SymCat −1.6%p [−2.2, −1.0]로 Holm 유의했다. Phenopacket-new의 차이(−1.5%p [−4.6, +1.5])는 유의하지 않았다. DDXPlus Top-1 차이는 −8.0%p [−9.1, −6.9]였다. 큐레이션 그래프에 같은 방식의 PPR을 적용하면(H4; 재시작 확률은 그래프별로 개발 세트에서 다시 골라 HPO 주석 0.5, PrimeKG 0.7, S-D 0.3), HPO 주석에서는 흔한 질환 세트의 정확도가 소폭 올랐다(SymCat Top-10 +2.8%p, DDXPlus Top-1 +1.8%p). 같은 HPO 주석에서 Phenopacket-new Top-10은 12.3%p 낮아졌다. PrimeKG에서는 PPR이 DDXPlus Top-10을 4.8%p, Phenopacket-new Top-10을 22.6%p 낮췄다(모두 Holm 유의). 개발 세트에서도 PPR의 Top-1은 9개 세트 모두 S-D 1-홉보다 낮았다(부호검정 p = .0039). HPO 주석과 PrimeKG 그래프에서도 PPR의 Top-10은 9개 세트 중 8개에서 1-홉보다 낮았다(각 p = .039). 개발에서 PPR이 1위로 낸 질환의 차수 중앙값은 75로 1-홉(85)보다 낮았다. 정답 질환을 차수 기준 삼분위로 나누면, PPR의 손실은 상위 1/3에서 컸다(Top-10 38.3% → 29.7%).

같은 1-홉 랭커에 다른 지식원을 넣은 비교에서, S-D의 Top-10은 MEDLINE MeSH 공출현 그래프보다 세 세트 모두 유의하게 높았다(DDXPlus +9.6%p, Phenopacket-new +7.7%p, SymCat +3.6%p). HPO 주석·PrimeKG와 비교하면 S-D는 DDXPlus에서 높았다(+24.8%p, +23.2%p). Phenopacket-new에서는 S-D가 낮았다(−26.8%p, −22.6%p). SymCat에서는 S-D가 높았다(+7.3%p, +5.5%p). Phenopacket-new의 정답 질환 중 S-D 그래프에 소견 간선이 하나라도 있는 질환의 비율은 59.1%였다. 같은 비율은 DDXPlus 92.0%, SymCat 94.8%였다(정답 정확 일치 기준).

### 4.3 재현율 손실 분해와 수집 깊이

이 절의 결과는 모두 설계 D 확증 이후의 탐색이며 확증 판정을 바꾸지 않는다. 이 절의 시뮬레이션과 단계 분해는 Phenopacket-store 0.1.27과 그 이후 main에 추가된 증례 중 정답과 소견을 사상할 수 있는 10,603건을 쓴다. 이 증례에는 설계 D 확증의 Phenopacket-new 증례 파일이 포함된다. v2 수집은 0.1.27의 원출처 PMID 1,733편만 뺐다. 그래서 0.1.27 이후 증례 중 32건은 원 논문(9편)이 v2 코퍼스에 들어 있다. 단계 분해(표 9A)와 S-D 그래프의 참고값은 이 32건을 포함한 값이다. 이 32건을 빼면(10,571건) S-D 그래프의 Top-10은 16.7%에서 16.5%가 된다.

**랭커의 오류·재현율 민감도.** 문헌 그래프에는 간선별 정답이 없다. 그래서 깨끗한 HPO 주석 그래프(HPOA v2024-12-12)에서 참 간선을 빼거나 틀린 간선을 넣고 순위를 냈다. 순위화에는 HPO 주석 그래프용으로 개발 세트에서 고른 동결 1-홉 설정(α 0.2, μ 2, 감쇠 0.5, unified_w 2)을 썼다(시드 0, 점마다 단일 실행). 참 간선은 정한 비율(재현율)만큼 무작위로 남겼다. 틀린 간선은 남긴 참 간선 1개당 정한 수(후보 비율)의 후보를 뽑아 넣었다. 격자의 후보 비율 2.33과 1은 명목 정밀도 1/(1 + 후보 비율) 30%와 50%에 해당한다. 틀린 간선의 구성은 세 가지로 돌렸다. 형제 구성은 같은 부모를 둔 형제 질환의 소견(형제가 없으면 무작위 질환의 소견)에서 후보를 뽑는다. 관련 질환의 실제 소견이 잘못 붙는 오류를 본뜬 것이다. 어휘 구성은 v2 구절 표(3.3절)의 개념 가운데 빌더가 간선으로 받는 의미유형의 개념에서 무작위로 뽑는다. 추출 어휘의 임의 개념이 붙는 오류를 본뜬 것이다. 혼합 구성은 후보의 70%를 형제 구성의 방식으로, 30%를 어휘 구성의 방식으로 뽑는다. 후보 비율과 세 구성은 모두 가정이며, 이 값들을 뒷받침하는 측정값은 없다. 이미 있는 간선과 겹치는 후보는 건너뛰므로, 실현 정밀도(참 간선 수/전체 간선 수)는 명목 정밀도 이상이고 같은 후보 비율에서도 구성마다 다르다. 재현율 100%에서 후보 비율 2.33의 실현 정밀도는 형제 구성 40.0%, 혼합 구성 35.9%, 어휘 구성 30.5%였다. 그래서 두 순수 구성은 같은 후보 비율로도, 같은 재현율에서 혼합 구성과 실현 정밀도가 같아지는 비율로도 돌렸다. 재현율 30%와 15%에서는 세 구성 모두 실현 정밀도가 20%가 되는 비율로도 돌렸다.

표 8a·8b는 혼합 구성의 결과다. 재현율 100%에서 혼합 구성의 실현 정밀도와 Top-10 손실은 두 순수 구성의 값 사이에 있었다(표 8c). 모든 참 간선을 남기고 간선 정밀도를 100%에서 55.3%, 35.9%로 낮추면 Top-10은 56.0%에서 53.5%, 49.5%로 낮아졌다(35.9%에서 차이 −6.5%p [−7.1, −5.9]). 손실은 HPOA가 증례 논문을 이미 인용한 과거 증례에 몰렸다(84.0% → 75.2%). 미래 증례는 7.7%에서 7.4%로 거의 변하지 않았다. 참 간선을 남기는 비율(재현율)을 낮추면 Top-10은 더 크게 낮아졌다(정밀도 100%·재현율 15%에서 17.3%). 재현율 15%에서 같은 후보 비율(참 간선 1개당 2.33개)로 넣으면 정밀도는 32.6%, Top-10은 15.5%였다. 이 지점에서 재현율을 30%로 올리면(정밀도 33.3%) Top-10은 26.4%로 10.9%p [10.2, 11.7] 올랐다. 같은 지점에서 틀린 간선을 모두 빼면(정밀도 100%) Top-10은 17.3%로 1.8%p [1.3, 2.4] 올랐다. 같은 시드에서 질환별 소견의 순회 순서만 바꿔 틀린 간선을 다시 뽑으면, 재현율 100% 점의 Top-1은 최대 1.5%p, Top-10은 최대 0.8%p 달라졌다. 표 8의 CI는 증례 표집만 반영하고 이 실현 변동은 반영하지 않는다.

틀린 간선의 구성과 후보 비율을 바꿔도 재현율 15% 지점(아래에서 보듯 S-D 그래프의 재현율과 같은 범위, 이하 운용점)의 비교 방향은 바뀌지 않았다(표 8c). 비교한 조건은 세 구성에 세 정밀도 수준(혼합 구성의 후보 비율 2.33과 1에 해당하는 수준, 20%)과 같은 후보 비율·정밀도 맞춤 비율을 조합한 13개다. 이 조건들의 실현 정밀도는 재현율 15·30% 점에서 20.0–54.2%였다(재현율 15% 점만 보면 20.0–53.2%). 13개 조건에서 재현율을 15%에서 30%로 늘린 효과는 Top-10 +8.0~+11.6%p, Top-1 +2.3~+6.6%p였다. 같은 재현율 15%에서 정밀도를 100%로 올린 효과는 Top-10 +0.1~+2.7%p, Top-1 −0.1~+1.2%p였다. 13개 조건 모두에서 재현율 효과의 CI 하한이 정밀도 효과의 CI 상한보다 컸다. 두 구간이 가장 가까운 조건은 형제 구성·정밀도 20%의 Top-1이었다([1.8, 2.8] 대 [0.7, 1.7]). Top-10에서 재현율 효과는 모든 조건에서 정밀도 효과의 3.7배 이상이었다. 가장 작은 비는 어휘 구성·정밀도 20%의 3.7배(10.0 대 2.7)였다. 재현율 100%에서는 구성에 따라 손실의 크기가 달랐다. 실현 정밀도를 35.9%로 맞추면 Top-10 손실은 형제 구성 10.1%p, 혼합 구성 6.5%p, 어휘 구성 1.7%p였다(오류가 없는 56.0% 대비 상대 18%, 12%, 3%). 따라서 재현율 100%에서 오류가 Top-10을 얼마나 낮추는지는 구성 가정에 따라 달라진다. 운용점에서는 어느 구성에서도 정밀도로 인한 Top-10 손실이 2.7%p 이하였다.

시뮬레이션의 두 축 가운데 재현율 쪽 운용점은 HPOA를 기준으로 잰 S-D 그래프의 재현율과 같은 범위에 있다. S-D 그래프는 HPOA(2026-02-16판) 주석 쌍 가운데 10.9%를 같은 개념의 간선으로 가졌다(비교 가능한 간선이 있는 8,132개 질환의 주석 쌍 281,825개 기준). 아래 단계 분해에서 정답 질환의 큐레이션 소견 가운데 간선으로 남은 비율은 14.1%였다(표 9A). 정밀도 쪽에는 측정값이 없다. S-D 그래프의 간선 정밀도에 대해 알 수 있는 것은 HPOA 일치율이라는 하한뿐이다. S-D 그래프에서 HPOA 주석 질환의 간선 가운데 개념이 HPO 용어에 대응되는 168,587개를 HPOA 주석과 대조하면 정확 일치율은 18.2%(HPO 부모 포함 22.1%)였다. 질환을 무작위로 바꾼 대응에서는 2.8%(3.7%)였다(4.7절). HPOA에 없는 간선도 맞을 수 있으므로 이 값은 그 간선들의 정밀도 하한이다. 또 이 값은 HPOA 주석이 있는 질환, 주로 희귀질환의 간선에만 해당한다. 표 8c의 정밀도 수준은 이 하한 근처의 20%부터 100%까지를 포함한다.

운용점의 Top-10은 세 구성과 정밀도 20–100% 전체에서 14.7–17.3%였다. 같은 증례에서 S-D 그래프의 Top-10(16.7%)은 이 범위 안에 있었다. 이 일치는 일관성 확인이며 S-D 그래프의 정밀도를 재는 근거가 아니다. 운용점의 Top-10은 정밀도에 거의 반응하지 않아, 오류가 없는 점(17.3%)도 16.7%에 가깝기 때문이다. Top-1은 맞지 않았다(시뮬레이션 5.0–6.3% 대 S-D 그래프 7.4%). 두 순위화는 채점 설정(HPOA α 0.2·감쇠 0.5·β 0, S-D α 0.35·감쇠 0.3·β 0.5·offtopic_w 0)과 개념 어휘가 다르다. 지식 시점도 다르다(HPOA 2024-12-12판 대 시간 절단 없는 문헌). 2024년까지 출판된 문헌만으로 다시 집계한 S-D 그래프의 Top-1/Top-10은 6.6/15.2%였다. 따라서 이 비교로 말할 수 있는 것은 S-D 그래프의 Top-10이 재현율 15% 수준 HPOA 그래프와 같은 범위에 있다는 데까지다.

Table 8. Ranker sensitivity on the HPOA graph (v2024-12-12) with injected wrong edges and removed true edges. HPOA 1-hop configuration re-selected on the development sets and frozen (α 0.2, μ 2, decay 0.5, unified_w 2); 10,603 phenopacket cases; seed 0, one run per point. Top-k in %. The wrong-edge compositions and the candidate ratios are assumptions; no measurement supports them. The mixed composition (panels a and b) draws 70% of the candidates from the findings of a sibling disease (a random disease if there is none) and 30% from v2 phrase-table concepts of the semantic types the builder accepts; panel c adds the two pure compositions. The grid ratios 2.33 and 1 give nominal precisions 1 / (1 + ratio) of 30% and 50%. A candidate that is already an edge is skipped, so the realized precision (true edges / all edges) is at least the nominal value and depends on the composition. Differences are computed from unrounded values. CIs are paired patient-bootstrap 95% intervals; they do not include the variation between wrong-edge realizations (at full recall, up to 1.5 pp in Top-1 and 0.8 pp in Top-10).

(a) Edge precision, all true edges kept (mixed composition).

| Wrong-edge candidates per true edge | Edge precision | Edges | All cases Top-1 / Top-10 | Past, cited^a^ Top-10 (n = 5,008) | Past, uncited^a^ Top-10 (n = 4,748) | Future^a^ Top-10 (n = 847) | Top-10 difference vs 100% [95% CI] |
|---|---|---|---|---|---|---|---|
| 0 | 100% | 379,291 | 36.3 / 56.0 | 84.0 | 35.2 | 7.7 | — |
| 1 | 55.3% | 685,801 | 32.6 / 53.5 | 81.1 | 32.6 | 7.3 | −2.6 [−3.0, −2.1] |
| 2.33 | 35.9% | 1,056,700 | 26.8 / 49.5 | 75.2 | 29.9 | 7.4 | −6.5 [−7.1, −5.9] |

(b) Recall (share of true edges kept), all cases, mixed composition. Wrong edges are added at 2.33 candidates per kept true edge, so the realized precision falls as fewer true edges are kept.

| True edges kept | No wrong edges (precision 100%): Top-1 / Top-10 | With wrong edges: precision | With wrong edges: Top-1 / Top-10 |
|---|---|---|---|
| 100% | 36.3 / 56.0 | 35.9% | 26.8 / 49.5 |
| 50%^b^ | 18.4 / 36.0 | — | — |
| 30% | 13.3 / 27.9 | 33.3% | 10.8 / 26.4 |
| 15% | 6.2 / 17.3 | 32.6% | 5.1 / 15.5 |

(c) Composition of wrong edges: recall effect and precision effect (pp [95% CI]), all cases. Recall effect = recall 30% minus recall 15% at the same precision level. Precision effect = no wrong edges minus wrong edges at the same recall. Sibling only = all candidates from sibling-disease findings; vocabulary only = all candidates from phrase-table concepts.

| Composition | Candidates per true edge^c^ | Realized precision at recall 100 / 30 / 15% | Recall effect, Top-1 / Top-10 | Precision effect at recall 15%, Top-1 / Top-10 | Precision effect at recall 100%, Top-1 / Top-10 |
|---|---|---|---|---|---|
| Mixed (70 / 30) | 2.33 | 35.9 / 33.3 / 32.6 | +5.7 [5.1, 6.3] / +10.9 [10.2, 11.7] | +1.1 [0.7, 1.6] / +1.8 [1.3, 2.4] | +9.5 [8.9, 10.2] / +6.5 [5.9, 7.1] |
| Sibling only | 2.33 | 40.0 / 35.7 / 34.6 | +4.9 [4.3, 5.5] / +10.8 [10.1, 11.6] | +1.0 [0.5, 1.4] / +0.4 [−0.1, 1.0] | +10.1 [9.4, 10.8] / +8.1 [7.5, 8.7] |
| Sibling only | matched | 35.9 / 33.3 / 32.6 | +5.3 [4.7, 5.9] / +10.0 [9.2, 10.7] | +1.0 [0.6, 1.5] / +1.2 [0.6, 1.8] | +12.7 [12.0, 13.4] / +10.1 [9.5, 10.8] |
| Vocabulary only | 2.33 | 30.5 / 30.2 / 30.1 | +5.9 [5.3, 6.4] / +9.3 [8.6, 10.0] | +0.5 [0.2, 0.7] / +0.9 [0.5, 1.3] | +3.1 [2.7, 3.5] / +2.6 [2.2, 3.0] |
| Vocabulary only | matched | 35.9 / 33.2 / 32.5 | +6.3 [5.7, 6.9] / +10.0 [9.3, 10.7] | +0.5 [0.2, 0.7] / +1.6 [1.2, 1.9] | +2.1 [1.7, 2.5] / +1.7 [1.3, 2.1] |
| Mixed (70 / 30) | 1 | 55.3 / 52.5 / 51.8 | +6.0 [5.4, 6.6] / +10.6 [9.8, 11.3] | +0.5 [0.2, 0.9] / +0.5 [−0.0, 0.9] | +3.7 [3.2, 4.3] / +2.6 [2.1, 3.0] |
| Sibling only | 1 | 58.5 / 54.2 / 53.2 | +6.3 [5.7, 6.9] / +11.6 [10.9, 12.4] | +0.7 [0.4, 1.1] / +0.4 [−0.1, 0.8] | +5.5 [5.0, 6.1] / +4.6 [4.1, 5.1] |
| Sibling only | matched | 55.4 / 52.4 / 51.8 | +5.4 [4.8, 6.0] / +9.7 [8.9, 10.4] | −0.1 [−0.5, 0.3] / +0.1 [−0.5, 0.5] | +6.1 [5.5, 6.6] / +4.6 [4.1, 5.1] |
| Vocabulary only | 1 | 50.3 / 50.1 / 50.0 | +6.4 [5.8, 7.0] / +10.5 [9.8, 11.2] | +0.0 [−0.2, 0.2] / +0.6 [0.3, 0.9] | +1.7 [1.3, 2.1] / +1.6 [1.2, 1.9] |
| Vocabulary only | matched | 55.3 / 52.5 / 51.9 | +6.6 [6.0, 7.2] / +10.4 [9.8, 11.2] | +0.2 [−0.0, 0.4] / +0.6 [0.3, 0.8] | +1.4 [1.1, 1.7] / +1.4 [1.1, 1.7] |
| Mixed (70 / 30) | precision 20% | — / 20.0 / 20.0 | +3.8 [3.3, 4.3] / +8.9 [8.2, 9.6] | +1.2 [0.8, 1.7] / +1.5 [0.8, 2.1] | — |
| Sibling only | precision 20% | — / 20.0 / 20.0 | +2.3 [1.8, 2.8] / +8.0 [7.2, 8.7] | +1.2 [0.7, 1.7] / +1.0 [0.3, 1.6] | — |
| Vocabulary only | precision 20% | — / 20.0 / 20.0 | +6.3 [5.8, 6.9] / +10.0 [9.3, 10.7] | +1.1 [0.8, 1.4] / +2.7 [2.2, 3.1] | — |

^a^ Past, cited: published before the HPOA release and cited by it as evidence; past, uncited: published before but not cited; future: published after the release.
^b^ The recall-50% point was not rerun with wrong edges.
^c^ 2.33 and 1: the grid ratios (nominal precision 30% and 50%), run for all three compositions. Matched: the ratio that gives the mixed composition's realized precision at each recall (sibling only 2.86, 2.63, 2.58 and 1.15, 1.08, 1.06; vocabulary only 1.82, 2.02, 2.08 and 0.82, 0.91, 0.93; at recall 100, 30, 15%). Precision 20%: the ratio that gives a realized precision of 20% (mixed 4.95, 4.77; sibling only 6.0, 5.59; vocabulary only 4.05, 4.03; at recall 30, 15%). Over the 13 conditions, the realized precision at recall 30 and 15% spans 20.0–54.2% (20.0–53.2% at recall 15% alone).
For reference, the S-D literature graph reaches Top-1 / Top-10 7.4 / 16.7% on the same cases (Top-10 16.5% on the 10,571 cases left after removing the 32 whose source papers are in the v2 corpus) under its own scoring configuration (α 0.35, decay 0.3, β 0.5, offtopic_w 0) and concept vocabulary. Its Top-10 lies within the range of the recall-15% points over all compositions and precisions 20–100% (14.7–17.3%); this is a consistency check, not an estimate of its edge precision.

**재현율 손실의 단계 분해.** Phenopacket 증례에서 관찰된 소견 중 HPO 주석이 정답 질환에 붙인 소견을 분해 대상으로 삼았다(증례×정답×소견 82,751건, 질환 878개). 각 소견은 정답 질환의 문헌 근거가 처음 사라진 단계로 분류했다(S-D 그래프의 문헌·추출·구절 표 기준). 소견의 22.4%는 쓸 만한 문헌이 없는 질환에 속했다(표 9A). 44.7%는 문헌은 있으나 소견 이름이 텍스트에 없었다. 따라서 최대 55,465건(67.0%)이 수집 단계에서 빠졌다(반올림한 두 비율의 합은 67.1%). 이는 전체 손실(85.9%)의 78%다. 간선으로 남은 소견은 14.1%였다. 텍스트에 이름이 있는 소견(33.0%) 중 약 74%는 추출과 개념 연결을 통과했다. 추출되지 않은 소견은 2.3%, 추출되었으나 연결되지 않거나 다른 CUI로 연결된 소견은 6.2%였다. 연결 실패의 주된 유형은 등위 구절 미분해('arachnodactyly, kyphosis or scoliosis')와 같은 개념의 다른 CUI(failure to thrive C0015544 대 C0231246)였다. 소견 이름을 UMLS 영어 문자열로 넓혀도 '텍스트에 없음'의 0.7%만 추가로 발견되었다. 문자열 일치는 소견 존재의 하한이다. 따라서 '텍스트에 없음'은 수집 손실의 상한이고 '추출 안 됨'은 추출 손실의 하한이다. 질환–소견 쌍(22,626쌍) 단위로 세면 간선이 있는 쌍은 5.8%였다.

**v2+v3 그래프의 단계 분해.** 같은 82,751건을 v2+v3 문헌의 세 그래프(S-mix, S-llm, S-dict)에서 다시 분해했다(표 9A). 이 재측정은 설계 E 확증 실행 뒤에 했다. 각 그래프의 분해는 그 그래프의 문헌, 추출, 구절 표와 빌더 규칙(등위 분해, HPO 이름 연결, 원인 유전자 제목 주제 인정, 자기 간선 제거)을 따랐다. 분해 스크립트를 이에 맞게 고친 뒤에도 기본 설정은 S-D 분해를 질환–소견 쌍별 판정까지 그대로 재현했다. 수집 손실(문헌 없음과 텍스트에 이름 없음의 합)은 S-D 67.0%에서 S-mix 37.3%로 줄었다(S-llm 37.7%, S-dict 36.4%). 쓸 만한 문헌이 없는 소견은 22.4%에서 0.3%로 줄었다. 문헌은 있으나 소견 이름이 텍스트에 없는 소견은 44.7%에서 37.0%로 줄었다. S-mix에서도 이 단계가 가장 큰 손실이었다. 이 값도 문자열 일치 기준이므로 수집 손실의 상한이다. 다른 말로 적힌 소견은 이 단계에 들어간다. 추출·연결 손실(추출 안 됨과 연결 안 됨의 합)은 늘지 않았다(8.5% → 5.1%). S-D와 같은 v2 문헌·추출에 v2+v3 그래프의 빌더 규칙만 적용해도 추출·연결 손실은 5.0%(4,153건)였다. 따라서 S-D 대비 감소는 문헌 확장보다 빌더 규칙의 차이에 해당한다. 정답 개념의 주제 간선이 된 소견은 14.1%에서 41.3%로 늘었다. 단계 분류와 별개로, 동결 순위화가 그 소견으로 정답 질환에 0보다 큰 점수를 주는 비율은 14.9%에서 55.5%로 늘었다(S-llm 53.4%, S-dict 59.3%). 질환–소견 쌍 단위로 세면 간선이 있는 쌍은 5.8%에서 22.5%로 늘었다.

분해의 정의에는 두 가지 단서가 있다. 첫째, 분해는 HPO 소견을 HPO 온톨로지(hp.json)의 UMLS 교차참조 CUI로 찾는다. 순위화는 소견마다 환자 쪽 평가에 쓰는 HPO–UMLS 대응의 CUI 하나를 쓴다. 소견을 순위화의 CUI로 찾으면 수집 손실은 S-D 66.6%, S-mix 36.0%였다. 두 CUI 중 하나로 찾는 경우까지 포함해, 네 그래프 모두에서 수집 손실은 교차참조 기준 값에서 3.2%p 이내로 움직였다. 둘째, v2+v3 그래프에서 '주제 밖 문서에만 있음' 단계(4.4–4.9%)는 손실이 아니다. 이 그래프들의 동결 offtopic_w는 0보다 크다(S-mix·S-dict 0.5, S-llm 1.0). 그래서 이 단계 소견의 97.8–98.0%가 점수를 받는다. S-D(offtopic_w 0)에서는 이 단계가 손실이다.

Table 9A. Stage at which a curated finding of the gold disease is first lost: the design D graph (S-D; v2 literature, gemma-4-E4B-it extraction, v2 phrase table, offtopic_w 0) and three v2 + v3 graphs of Table 5b (each with its own extraction, phrase table and builder rules). Cases: the 10,603 phenopacket cases, including 32 whose source papers are in the v2 corpus. Every column uses the same 82,751 case × gold × finding triples (22,626 disease–finding pairs, 878 gold diseases). Findings are matched by the UMLS cross-references of their HPO terms (hp.json)^d^. Percentages are rounded; subtotals are computed from counts.

(a) Case × gold × finding triples, n (%).

| Stage | S-D | S-mix | S-llm | S-dict |
|---|---|---|---|---|
| No usable document (none, or all generic) | 18,502 (22.4) | 208 (0.3) | 208 (0.3) | 206 (0.2) |
| Documents exist, finding name not in text | 36,963 (44.7) | 30,656 (37.0) | 30,958 (37.4) | 29,901 (36.1) |
| In text, not extracted | 1,906 (2.3) | 2,022 (2.4) | 2,053 (2.5) | 277 (0.3) |
| Extracted, not linked or linked to a different CUI | 5,113 (6.2) | 2,219 (2.7) | 2,857 (3.5) | 2,439 (2.9) |
| Linked, removed by a filter^a^ | 2,881 (3.5) | 3,113 (3.8) | 3,160 (3.8) | 3,087 (3.7) |
| Only in off-topic documents^b^ | 3,569 (4.3) | 3,627 (4.4) | 3,775 (4.6) | 4,042 (4.9) |
| Only a parent or child concept has an edge | 2,132 (2.6) | 6,755 (8.2) | 6,622 (8.0) | 7,104 (8.6) |
| Edge present | 11,685 (14.1) | 34,151 (41.3) | 33,118 (40.0) | 35,695 (43.1) |
| Collection loss (first two stages) | 55,465 (67.0) | 30,864 (37.3) | 31,166 (37.7) | 30,107 (36.4) |
| Extraction and linking loss (third and fourth stages) | 7,019 (8.5) | 4,241 (5.1) | 4,910 (5.9) | 2,716 (3.3) |
| Scored by the frozen ranker for the gold disease^c^ | 12,303 (14.9) | 45,891 (55.5) | 44,222 (53.4) | 49,090 (59.3) |

(b) Disease–finding pairs (%; n = 22,626).

| Stage | S-D | S-mix | S-llm | S-dict |
|---|---|---|---|---|
| No usable document (none, or all generic) | 31.9 | 0.5 | 0.5 | 0.5 |
| Documents exist, finding name not in text | 49.6 | 56.4 | 56.9 | 55.1 |
| In text, not extracted | 1.7 | 3.0 | 3.0 | 0.3 |
| Extracted, not linked or linked to a different CUI | 3.7 | 2.2 | 2.9 | 2.5 |
| Linked, removed by a filter^a^ | 3.0 | 4.9 | 4.8 | 5.2 |
| Only in off-topic documents^b^ | 2.6 | 4.1 | 4.0 | 4.9 |
| Only a parent or child concept has an edge | 1.8 | 6.6 | 6.5 | 7.2 |
| Edge present | 5.8 | 22.5 | 21.4 | 24.3 |

^a^ Generic document or excluded semantic type; in the v2 + v3 columns also a concept naming the disease or a parent disease (self-edge removal).
^b^ Scored at the graph's offtopic_w. A loss for S-D (offtopic_w 0); not a loss for the v2 + v3 graphs (frozen design E values: S-mix 0.5, S-llm 1.0, S-dict 0.5), where 97.8–98.0% of these triples receive a score.
^c^ Independent of the stage: the frozen ranker gives the gold disease a score above 0 through the ranker's CUI for the finding (footnote d) or a graph IS_A parent of that CUI, over HAS_FINDING edges only (off-topic edges weighted by offtopic_w).
^d^ The ranker queries one CUI per finding (the HPO–UMLS mapping used for patients). Matching findings by that CUI, or by either CUI, moves the collection loss by at most 3.2 pp in any column (ranker CUI: S-D 66.6%, S-mix 36.0%, S-llm 37.1%, S-dict 34.1%).

**수집 확장 사전테스트.** 어떤 문헌 원천이 큐레이션 소견을 더 담는지 텍스트 출현율로 비교했다. HPO 주석이 있는 OMIM 질환 300개를 무작위로 뽑았고(시드 7, 벤치마크 정보 미사용), 그중 99개는 v2 코퍼스에 쓸 문헌이 없었다. 지표는 큐레이션 (질환, HPO) 쌍의 소견 이름이 그 질환 텍스트에 나오는 비율이다. 대조는 다른 무작위 질환의 큐레이션 소견이 나오는 비율로, 텍스트 양의 효과를 잰다. v2 코퍼스의 재현율은 18.4%, 대조는 2.9%였다(표 9B). PMC OA 리뷰·증례를 30+30편(임상 절, 12k 절단)으로 늘리면 재현율은 28.5%였다. 같은 문헌을 절단 없이 전문으로 읽으면 47.8%, PubMed 초록 50편을 더하면 49.4%, 원인 유전자 기호 검색을 더하면 58.1%였다. 재현율과 대조의 차이(질환 특이 신호)는 15.5%p에서 40.3%p로 2.6배가 되었다. 대조도 2.9%에서 17.8%로 6배가 되었다. 차이 기준 가장 큰 단일 상승은 12k 절단과 절 필터를 없앤 단계였다(+10.2%p). 무작위 40질환에서 임상 절 12k 텍스트에 없던 큐레이션 소견 181개 중 58.6%는 표에 있었다(중복 허용; 기타 절 38.7%, 고찰 30.4%, 서론 26.0%, 결과 19.9%). v2 수집은 본문의 `<p>` 요소만 저장해 표를 버렸다. 표를 더하면 PMC 30+30 임상 절의 재현율은 28.5%에서 40.9%로 올랐다. 재현율/대조 비는 임상 절+표 3.8, 전문+표 3.1이었다.

Table 9B. Collection pretest: share of curated (disease, HPO) pairs whose finding name occurs in the disease's text (300 random HPOA-annotated OMIM diseases, seed 7). Control = occurrence of another random disease's curated findings.

| Collection variant | Recall (%) | Control (%) | Difference (pp) |
|---|---|---|---|
| v2 corpus | 18.4 | 2.9 | 15.5 |
| + PMC OA reviews and case reports 30 + 30 (clinical sections, 12k cut) | 28.5 | 6.1 | 22.4 |
| Same articles, full text, no cut | 47.8 | 15.2 | 32.6 |
| + PubMed abstracts 50 | 49.4 | 15.4 | 34.0 |
| + causal-gene symbol search (per gene, ≤ 2 genes per disease: abstracts 50 + PMC 20) | 58.1 | 17.8 | 40.3 |
| v2 corpus + tables | 23.5 | 4.4 | 19.1 |
| PMC 30 + 30 clinical sections + tables | 40.9 | 10.8 | 30.1 |
| Full text + tables | 51.1 | 16.5 | 34.6 |

**수집 깊이의 효과(개발 세트 탐색).** v2 문헌과 v2+v3 문헌을 같은 추출 방법과 같은 빌더 규칙(등위 분해, HPO 이름 연결, 원인 유전자 제목 주제 인정, 자기 간선 제거)으로 그래프화했다. 동결 1-홉 설정에서 offtopic_w만 D3 규칙으로 다시 골라 개발 9개 세트에서 비교했다. UMLS 사전 추출에서 매크로 Top-10/Top-1은 34.50/14.24%에서 42.88/18.11%로 올랐다(표 9C). 선택된 offtopic_w는 두 그래프 모두 0.5였다. Top-10은 9개 세트 중 8개에서 오르고 내린 세트는 없었다(부호검정 p = .008). Holm 유의한 상승은 DDXPlus(+4.43%p), RAMEDIS(+8.65%p), LIRICAL(+6.76%p), Phenopacket(+7.71%p)이었다. 정답 질환 군집 CI가 0을 넘은 세트는 RAMEDIS, LIRICAL, HMS(+10.23%p), PUMCH-ADM(+14.67%p), Phenopacket이었다. Top-1은 7개 세트에서 올랐으나 부호검정은 유의하지 않았다(7/2, p = .18). LLM 추출에서는 매크로 Top-10/Top-1이 31.94/13.06%에서 44.90/19.30%로 올랐다. LLM 쪽 v2 그래프는 v2 구절 표를, v3 그래프는 그 표를 같은 연결기로 v3 구절까지 확장한 표를 썼다. 선택된 offtopic_w도 0에서 1.0으로 바뀌었다. 따라서 LLM 쪽 차이에는 구절 표 확장과 offtopic_w 변화가 함께 들어 있다. offtopic_w를 두 그래프 모두 0으로 고정하면 v3 그래프의 매크로 Top-10은 43.64%였고, 상승 폭은 11.70%p였다. Top-10은 9개 세트 모두 올랐고(p = .004), 9개 모두 Holm 유의했다. 상승 폭은 MME(+27.50%p), RAMEDIS(+23.40%p), HMS(+14.77%p), LIRICAL(+13.51%p), PUMCH-ADM(+13.33%p), Phenopacket(+13.28%p)에서 컸다. DDXPlus(+3.63%p)와 SymCat(+1.06%p)에서는 작았다. 매크로 Top-10 상승은 LLM +13.0%p, UMLS 사전 +8.4%p였다. 세트별로는 LLM의 상승이 7개 세트에서 컸고 DDXPlus와 PUMCH-ADM에서 작았다(7/2, 부호검정 p = .18). 두 상승 폭의 차이를 환자 수준에서 검정하지는 않았다. 같은 추출 방식과 같은 빌더 규칙(자기 간선 유무 포함)으로 v2 그래프와 v2+v3 그래프를 짝지어 검정한 비교는 이 두 쌍을 포함해 7쌍이다. 자기 간선을 둔 LLM 쌍(base_split_tg → v3llm)은 같은 조건의 여덟째 쌍이지만 짝 검정에 넣지 않았다. 이 쌍의 매크로 Top-10은 32.59%에서 44.09%로 11.50%p 올랐다. 7쌍 모두에서 매크로 Top-10이 올랐다(+6.95~+12.96%p). 자기 간선을 둔 쌍은 UMLS 사전 +6.95%p, HPO 사전 +7.56%p였다. 자기 간선을 뺀 쌍은 HPO 사전 +7.86%p, UMLS 사전 +8.38%p, LLM + UMLS 사전 +9.11%p, LLM + HPO 사전 +10.64%p, LLM +12.96%p였다. 합집합 두 쌍의 v3 그래프도 LLM 항목을 확장한 구절 표로 연결했다. HPO 사전 두 쌍에서는 Top-10이 오른 세트가 7개, 내린 세트가 2개였다(부호검정 p = .18). 나머지 다섯 쌍에서는 오른 세트가 8개 이상, 내린 세트가 1개 이하였다(p ≤ .039). Top-10에서 수집 깊이의 효과는 같은 v2 문헌 위에서 추출·연결 방식이나 빌더 규칙을 바꾼 어떤 변형보다 컸다(동결 그래프 대비 최대 +4.1%p, v2unionu_ns, 4.4절). Top-1에서는 추출 방법에 따라 달랐다. UMLS 사전 그래프에서는 자기 간선 제거의 효과(v2 +4.03%p, v3 +7.56%p)가 수집 깊이의 효과(+3.87%p)보다 컸다(4.4절). LLM 그래프에서는 수집 깊이의 효과(+6.24%p)가 자기 간선 제거의 효과(v2 +0.90%p, v3 +2.86%p)보다 컸다.

PMC 본문을 12k 임상 절로 자르지 않고 전부 읽은 사전 그래프(*full)를 12k 판과 비교했다. 매크로 Top-10은 UMLS 사전 +0.20%p(43.08% 대 42.88%), HPO 사전 +0.87%p(41.38% 대 40.51%) 높았다. 매크로 Top-1은 UMLS 사전 +1.30%p(19.41% 대 18.11%), HPO 사전 +2.33%p(18.27% 대 15.94%) 높았다. 세트별로는 갈렸다(UMLS Top-1 6/3, p = .51, Top-10 4/5, p = 1; HPO Top-1 6/1, p = .13, Top-10 6/2, p = .29). 간선은 UMLS 사전 1.9배(28,096,533개), HPO 사전 1.7배였다. UMLS 전문 그래프의 빌드는 프로세스 RSS 약 24 GB와 swap 약 32 GB를 썼다. 주 그래프는 12k 임상 절 판을 쓴다. 참고로 v3 사전 그래프의 개발 매크로 Top-10은 모두 같은 랭커의 PrimeKG(33.52%)보다 높고 HPO 주석(52.45%)보다 낮았다(대응 검정 없음).

Table 9C. Collection depth: v2 vs v2 + v3 literature with the same extraction (development sets, lineage Top-10, %). Δ = v3 − v2 with patient-bootstrap 95% CI and gold-disease cluster-bootstrap 95% CI (B = 2,000). * Holm-adjusted exact McNemar p < .05 within each comparison. UMLS dictionary: v2umls_ns (offtopic_w 0.5) → v3umls_ns (0.5). LLM: base_split_tg_ns (0) → v3llm_ns (1.0); the v3 LLM graph also uses the phrase table extended to v3 spans.

| Benchmark | n (gold diseases) | UMLS dict v2 → v3 | Δ [patient CI] [cluster CI] | LLM v2 → v3 | Δ [patient CI] [cluster CI] |
|---|---|---|---|---|---|
| DDXPlus | 3,000 (48) | 37.83 → 42.27 | +4.43* [3.07, 5.77] [−1.35, 10.43] | 35.13 → 38.77 | +3.63* [1.97, 5.23] [−5.71, 13.13] |
| RAMEDIS | 624 (74) | 39.26 → 47.92 | +8.65* [5.61, 11.54] [0.70, 14.29] | 32.53 → 55.93 | +23.40* [19.23, 27.24] [12.18, 32.18] |
| LIRICAL | 370 (252) | 25.95 → 32.70 | +6.76* [2.70, 11.08] [1.45, 12.57] | 25.41 → 38.92 | +13.51* [9.46, 17.84] [7.10, 19.94] |
| HMS | 88 (46) | 32.95 → 43.18 | +10.23 [2.27, 18.18] [2.70, 18.82] | 29.55 → 44.32 | +14.77* [6.82, 23.86] [4.82, 26.14] |
| MME | 40 (17) | 30.00 → 52.50 | +22.50 [7.50, 37.50] [0.00, 53.85] | 30.00 → 57.50 | +27.50* [15.00, 42.50] [9.23, 59.09] |
| PUMCH_L | 988 (100) | 64.78 → 64.78 | 0.00 [−2.13, 2.23] [−4.08, 4.23] | 57.19 → 63.36 | +6.17* [3.95, 8.40] [0.13, 12.62] |
| PUMCH-ADM | 75 (16) | 45.33 → 60.00 | +14.67 [5.33, 24.00] [1.33, 30.68] | 49.33 → 62.67 | +13.33* [2.67, 24.00] [−2.90, 30.00] |
| Phenopacket-store | 10,310 (772) | 20.37 → 28.08 | +7.71* [7.08, 8.31] [4.78, 10.80] | 17.87 → 31.14 | +13.28* [12.63, 14.00] [9.52, 17.01] |
| SymCat | 7,481 (760) | 14.05 → 14.46 | +0.41 [−0.27, 1.03] [−0.85, 1.71] | 10.43 → 11.48 | +1.06* [0.43, 1.71] [−0.03, 2.15] |
| Macro Top-10 | 9 sets | 34.50 → 42.88 | +8.38 (sets up/down 8/0, sign p = .008) | 31.94 → 44.90 | +12.96 (9/0, p = .004) |
| Macro Top-1 | 9 sets | 14.24 → 18.11 | +3.87 (7/2, p = .18) | 13.06 → 19.30 | +6.24 (7/2, p = .18) |
