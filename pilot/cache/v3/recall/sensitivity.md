# Recall decomposition: feature CUI 민감도 (xref / ranker / both)

생성: `recall_decomp.py --feature-cui {xref,ranker,both}`. 단위는 (case, gold, curated feature) triple 82,751건, 괄호는 그 비율이다. 입력 인자는 각 JSON의 `args`에 있다.

| 모드 | feature 매칭 CUI | near | 결과 파일 |
|---|---|---|---|
| xref (공개 결과) | hp.json UMLS xref (`CA.hpo_to_cui`) | HPO is_a 부모 **또는 자식**의 xref CUI에 topic edge | D `pilot/cache/v2/recall/decomp.json`, v3 `decomp_{smix,sllm,sdict}.json` |
| ranker | ranker가 조회하는 CUI 하나 (`hpo_umls_mapping.json`, `temporal_eval.load_cases`와 동일) | 그래프 Concept IS_A(`phenotype_parents.json`) **부모**에 topic edge. 출발 CUI가 edge를 가진 concept일 때만 IS_A가 있다(build_graph_v2) | `decomp_<graph>_rankercui.json` |
| both | ranker CUI 또는 xref CUI | 둘 모두의 그래프 IS_A 부모 | `decomp_<graph>_bothcui.json` |

ranker/both 모드에서 추가된 stage: `no_cui`(ranker CUI가 없어 ranker 입력에서 빠지는 feature), `near_child`(자식 concept에만 topic edge, ranker는 점수 없음), `near_offtopic`(부모의 off-topic edge만, decay x offtopic_w로 점수). 판정 순서는 edge > near > offtopic_only > near_offtopic > near_child > filtered > no_docs > not_in_text > not_linked > not_extracted 이다.

`scored`는 stage 정의와 무관한 기준이다. graph_dx.RANK가 이 feature(ranker CUI와 그 그래프 IS_A 부모)로 gold 질병에 0보다 큰 점수를 주면 1이다. topic edge는 1, off-topic edge는 frozen design E `offtopic_w`(D 0, S-mix 0.5, S-llm 1.0, S-dict 0.5)를 곱한다. idf > 0도 확인한다. ranker 모드 실행에서 계산했고 both 실행에서도 pair 단위로 같다.

whole-graph pass 검증: 각 ranker/both 실행이 모든 질병의 item을 다시 링크해서 얻은 diseases_with_findings / edges / concepts / Concept IS_A가 빌드 로그·summary와 같다:

| graph | diseases_with_findings | edges | concepts | Concept IS_A |
|---|---|---|---|---|
| D (base_nm) ranker | 27,309 | 832,671 | 46,130 | 34,068 |
| D (base_nm) both | 27,309 | 832,671 | 46,130 | 34,068 |
| S-mix ranker | 33,537 | 5,221,285 | 85,761 | 56,340 |
| S-mix both | 33,537 | 5,221,285 | 85,761 | 56,340 |
| S-llm ranker | 33,123 | 3,672,106 | 76,487 | 49,757 |
| S-llm both | 33,123 | 3,672,106 | 76,487 | 49,757 |
| S-dict ranker | 34,356 | 14,506,788 | 95,608 | 76,044 |
| S-dict both | 34,356 | 14,506,788 | 95,608 | 76,044 |

## 요약: 헤드라인 수치의 이동

- **collection loss**(no_docs+not_in_text): D 대 S-mix가 xref 67.0% 대 37.3%(차 29.7pp)에서 ranker 66.6% 대 36.0%(차 30.5pp), both 65.8% 대 35.4%(차 30.4pp)가 된다. 어느 graph에서도 xref 대비 3.2pp 이내로 움직인다. '수집 손실이 67%에서 37%로 줄었다'는 결론은 매핑 기준에 민감하지 않다.
- **extraction+linking**(not_extracted+not_linked), xref→ranker: D (base_nm) 8.5→9.0%, S-mix 5.1→3.3%, S-llm 5.9→4.2%, S-dict 3.3→1.2%. 움직임은 거의 not_linked다(D (base_nm) 5,113→5,658, S-mix 2,219→799, S-llm 2,857→1,495, S-dict 2,439→773). v3 graph는 --hpo-names가 바로 ranker CUI를 graph에 넣기 때문에 xref 기준 not_linked가 부풀어 있었고, D는 반대로 약간 과소였다. 그래서 D와 v3의 차이는 ranker 기준에서 더 커진다(D−S-mix 3.4pp → 5.8pp).
- **edge**, xref→ranker: D (base_nm) 14.1→13.1%, S-mix 41.3→43.1%, S-llm 40.0→41.8%, S-dict 43.1→45.3%. D−S-mix 차이는 27.1pp → 30.0pp. near는 ranker 정의(그래프 IS_A 부모만)에서 줄고(S-mix 8.2→5.9%), 자식만 닿는 near_child(1.3/0.9/0.8/1.0%)는 따로 센다.
- **offtopic_only**: v3 graph에서는 손실이 아니다. frozen E offtopic_w(0.5/1.0/0.5)에서 ranker 기준 offtopic_only와 near_offtopic triple은 모두 점수를 받는다(S-mix 4,036+1,298, S-llm 4,147+1,143, S-dict 4,429+1,940, 3절). D(offtopic_w 0)에서는 0%다.
- **ranker가 실제로 점수를 주는 triple**(stage 정의와 무관): D (base_nm) 14.9%, S-mix 55.5%, S-llm 53.4%, S-dict 59.3%.

## 1. 헤드라인 수치가 얼마나 움직이나

셀은 triple 비율(%)이고 xref → ranker (both) 순이다. 세 값의 분모는 모두 같은 82,751 triple이다.

| 지표 | D (base_nm) | S-mix | S-llm | S-dict |
|---|---|---|---|---|
| collection loss (no_docs+not_in_text) | 67.0 → **66.6** (65.8) | 37.3 → **36.0** (35.4) | 37.7 → **37.1** (36.5) | 36.4 → **34.1** (33.2) |
| extraction+linking (not_extracted+not_linked) | 8.5 → **9.0** (7.1) | 5.1 → **3.3** (2.9) | 5.9 → **4.2** (3.8) | 3.3 → **1.2** (1.0) |
| filtered | 3.5 → **3.1** (3.6) | 3.8 → **3.9** (3.8) | 3.8 → **3.9** (3.9) | 3.7 → **3.8** (3.7) |
| offtopic_only | 4.3 → **3.8** (4.5) | 4.4 → **4.9** (4.8) | 4.6 → **5.0** (5.0) | 4.9 → **5.4** (5.3) |
| near_offtopic | - → **0.9** (1.0) | - → **1.6** (1.7) | - → **1.4** (1.5) | - → **2.3** (2.6) |
| near_child | - → **1.3** (1.0) | - → **0.9** (0.8) | - → **0.8** (0.8) | - → **1.0** (0.9) |
| near | 2.6 → **1.8** (1.8) | 8.2 → **5.9** (6.4) | 8.0 → **5.3** (5.6) | 8.6 → **6.3** (7.1) |
| edge | 14.1 → **13.1** (14.8) | 41.3 → **43.1** (43.7) | 40.0 → **41.8** (42.4) | 43.1 → **45.3** (45.7) |
| near+edge | 16.7 → **14.9** (16.6) | 49.4 → **49.0** (50.2) | 48.0 → **47.0** (48.0) | 51.7 → **51.6** (52.8) |
| no_cui | - → **0.5** (0.5) | - → **0.5** (0.5) | - → **0.5** (0.5) | - → **0.5** (0.5) |
| scored by ranker | 12,303 (14.9%) | 45,891 (55.5%) | 44,222 (53.4%) | 49,090 (59.3%) |
| extraction+linking / (collection 손실과 no_cui를 뺀 triple) | 25.7 → **27.4** (21.2) | 8.2 → **5.1** (4.5) | 9.5 → **6.8** (6.0) | 5.2 → **1.9** (1.5) |

건수(triple)로 본 핵심 지표:

| 지표 | 모드 | D (base_nm) | S-mix | S-llm | S-dict |
|---|---|---|---|---|---|
| collection loss (no_docs+not_in_text) | xref | 55,465 (67.0%) | 30,864 (37.3%) | 31,166 (37.7%) | 30,107 (36.4%) |
| collection loss (no_docs+not_in_text) | ranker | 55,079 (66.6%) | 29,827 (36.0%) | 30,678 (37.1%) | 28,255 (34.1%) |
| collection loss (no_docs+not_in_text) | both | 54,461 (65.8%) | 29,288 (35.4%) | 30,192 (36.5%) | 27,462 (33.2%) |
| extraction+linking (not_extracted+not_linked) | xref | 7,019 (8.5%) | 4,241 (5.1%) | 4,910 (5.9%) | 2,716 (3.3%) |
| extraction+linking (not_extracted+not_linked) | ranker | 7,470 (9.0%) | 2,696 (3.3%) | 3,510 (4.2%) | 1,020 (1.2%) |
| extraction+linking (not_extracted+not_linked) | both | 5,903 (7.1%) | 2,368 (2.9%) | 3,151 (3.8%) | 835 (1.0%) |
| edge | xref | 11,685 (14.1%) | 34,151 (41.3%) | 33,118 (40.0%) | 35,695 (43.1%) |
| edge | ranker | 10,824 (13.1%) | 35,661 (43.1%) | 34,551 (41.8%) | 37,492 (45.3%) |
| edge | both | 12,231 (14.8%) | 36,196 (43.7%) | 35,064 (42.4%) | 37,835 (45.7%) |
| near+edge | xref | 13,817 (16.7%) | 40,906 (49.4%) | 39,740 (48.0%) | 42,799 (51.7%) |
| near+edge | ranker | 12,303 (14.9%) | 40,557 (49.0%) | 38,932 (47.0%) | 42,721 (51.6%) |
| near+edge | both | 13,705 (16.6%) | 41,511 (50.2%) | 39,737 (48.0%) | 43,699 (52.8%) |

## 2. Stage별 전체 분포 (triple)

### D (base_nm) (frozen offtopic_w 0.0)

| stage | xref | ranker | both |
|---|---|---|---|
| no_cui | - | 393 (0.5%) | 393 (0.5%) |
| no_docs | 18,502 (22.4%) | 18,566 (22.4%) | 18,295 (22.1%) |
| not_in_text | 36,963 (44.7%) | 36,513 (44.1%) | 36,166 (43.7%) |
| not_extracted | 1,906 (2.3%) | 1,812 (2.2%) | 1,739 (2.1%) |
| not_linked | 5,113 (6.2%) | 5,658 (6.8%) | 4,164 (5.0%) |
| filtered | 2,881 (3.5%) | 2,554 (3.1%) | 2,995 (3.6%) |
| near_child | - | 1,061 (1.3%) | 815 (1.0%) |
| near_offtopic | - | 744 (0.9%) | 789 (1.0%) |
| offtopic_only | 3,569 (4.3%) | 3,147 (3.8%) | 3,690 (4.5%) |
| near | 2,132 (2.6%) | 1,479 (1.8%) | 1,474 (1.8%) |
| edge | 11,685 (14.1%) | 10,824 (13.1%) | 12,231 (14.8%) |

### S-mix (frozen offtopic_w 0.5)

| stage | xref | ranker | both |
|---|---|---|---|
| no_cui | - | 393 (0.5%) | 393 (0.5%) |
| no_docs | 208 (0.3%) | 204 (0.2%) | 204 (0.2%) |
| not_in_text | 30,656 (37.0%) | 29,623 (35.8%) | 29,084 (35.1%) |
| not_extracted | 2,022 (2.4%) | 1,897 (2.3%) | 1,742 (2.1%) |
| not_linked | 2,219 (2.7%) | 799 (1.0%) | 626 (0.8%) |
| filtered | 3,113 (3.8%) | 3,228 (3.9%) | 3,143 (3.8%) |
| near_child | - | 716 (0.9%) | 649 (0.8%) |
| near_offtopic | - | 1,298 (1.6%) | 1,409 (1.7%) |
| offtopic_only | 3,627 (4.4%) | 4,036 (4.9%) | 3,990 (4.8%) |
| near | 6,755 (8.2%) | 4,896 (5.9%) | 5,315 (6.4%) |
| edge | 34,151 (41.3%) | 35,661 (43.1%) | 36,196 (43.7%) |

### S-llm (frozen offtopic_w 1.0)

| stage | xref | ranker | both |
|---|---|---|---|
| no_cui | - | 393 (0.5%) | 393 (0.5%) |
| no_docs | 208 (0.3%) | 204 (0.2%) | 204 (0.2%) |
| not_in_text | 30,958 (37.4%) | 30,474 (36.8%) | 29,988 (36.2%) |
| not_extracted | 2,053 (2.5%) | 2,015 (2.4%) | 1,867 (2.3%) |
| not_linked | 2,857 (3.5%) | 1,495 (1.8%) | 1,284 (1.6%) |
| filtered | 3,160 (3.8%) | 3,267 (3.9%) | 3,230 (3.9%) |
| near_child | - | 681 (0.8%) | 646 (0.8%) |
| near_offtopic | - | 1,143 (1.4%) | 1,256 (1.5%) |
| offtopic_only | 3,775 (4.6%) | 4,147 (5.0%) | 4,146 (5.0%) |
| near | 6,622 (8.0%) | 4,381 (5.3%) | 4,673 (5.6%) |
| edge | 33,118 (40.0%) | 34,551 (41.8%) | 35,064 (42.4%) |

### S-dict (frozen offtopic_w 0.5)

| stage | xref | ranker | both |
|---|---|---|---|
| no_cui | - | 393 (0.5%) | 393 (0.5%) |
| no_docs | 206 (0.2%) | 204 (0.2%) | 202 (0.2%) |
| not_in_text | 29,901 (36.1%) | 28,051 (33.9%) | 27,260 (32.9%) |
| not_extracted | 277 (0.3%) | 247 (0.3%) | 163 (0.2%) |
| not_linked | 2,439 (2.9%) | 773 (0.9%) | 672 (0.8%) |
| filtered | 3,087 (3.7%) | 3,153 (3.8%) | 3,046 (3.7%) |
| near_child | - | 840 (1.0%) | 783 (0.9%) |
| near_offtopic | - | 1,940 (2.3%) | 2,147 (2.6%) |
| offtopic_only | 4,042 (4.9%) | 4,429 (5.4%) | 4,386 (5.3%) |
| near | 7,104 (8.6%) | 5,229 (6.3%) | 5,864 (7.1%) |
| edge | 35,695 (43.1%) | 37,492 (45.3%) | 37,835 (45.7%) |

## 3. offtopic_only를 손실로 볼 수 있나: graph weight > 0 비율

offtopic_only triple 중 ranker가 실제로 0보다 큰 점수를 주는 비율(`scored`)이다. xref 행은 공개 결과의 offtopic_only triple에 ranker 실행의 pair별 `scored`를 붙인 것이다.

| graph | offtopic_w | xref offtopic_only | 그중 scored | ranker offtopic_only | 그중 scored | ranker near_offtopic | 그중 scored | both offtopic_only | 그중 scored |
|---|---|---|---|---|---|---|---|---|---|
| D (base_nm) | 0.0 | 3,569 | 0 (0.0%) | 3,147 | 0 (0.0%) | 744 | 0 (0.0%) | 3,690 | 0 (0.0%) |
| S-mix | 0.5 | 3,627 | 3,555 (98.0%) | 4,036 | 4,036 (100.0%) | 1,298 | 1,298 (100.0%) | 3,990 | 3,900 (97.7%) |
| S-llm | 1.0 | 3,775 | 3,694 (97.9%) | 4,147 | 4,147 (100.0%) | 1,143 | 1,143 (100.0%) | 4,146 | 4,042 (97.5%) |
| S-dict | 0.5 | 4,042 | 3,954 (97.8%) | 4,429 | 4,429 (100.0%) | 1,940 | 1,940 (100.0%) | 4,386 | 4,309 (98.2%) |

## 4. 공개(xref) stage별로 ranker가 실제 점수를 주는 triple

셀은 scored / 해당 stage triple. xref 정의에서 손실로 분류된 triple 중 ranker가 쓰는 것, edge·near로 분류됐지만 ranker가 쓰지 못하는 것을 보여 준다.

| xref stage | D (base_nm) | S-mix | S-llm | S-dict |
|---|---|---|---|---|
| no_docs | 0 / 18,502 | 0 / 208 | 0 / 208 | 0 / 206 |
| not_in_text | 319 / 36,963 | 2,025 / 30,656 | 1,571 / 30,958 | 3,045 / 29,901 |
| not_extracted | 65 / 1,906 | 348 / 2,022 | 271 / 2,053 | 39 / 277 |
| not_linked | 574 / 5,113 | 1,537 / 2,219 | 1,523 / 2,857 | 1,610 / 2,439 |
| filtered | 3 / 2,881 | 315 / 3,113 | 249 / 3,160 | 440 / 3,087 |
| offtopic_only | 0 / 3,569 | 3,555 / 3,627 | 3,694 / 3,775 | 3,954 / 4,042 |
| near | 855 / 2,132 | 4,256 / 6,755 | 4,078 / 6,622 | 4,544 / 7,104 |
| edge | 10,487 / 11,685 | 33,855 / 34,151 | 32,836 / 33,118 | 35,458 / 35,695 |

## 5. 주요 이동 xref → ranker (triple, 상위 10)

- **D (base_nm)** (바뀐 triple 6,442, 7.8%): edge→not_linked 862, near→not_in_text 527, not_in_text→near_offtopic 515, offtopic_only→not_linked 487, near→near_child 435, not_linked→edge 423, not_in_text→near 296, filtered→no_docs 271, edge→near_child 243, edge→near 209
- **S-mix** (바뀐 triple 9,515, 11.5%): near→not_in_text 1,500, not_linked→edge 1,254, not_in_text→near_offtopic 1,031, not_in_text→near 924, near→edge 727, near→near_child 456, near→offtopic_only 401, not_in_text→filtered 249, not_in_text→no_cui 243, filtered→near 237
- **S-llm** (바뀐 triple 8,814, 10.7%): near→not_in_text 1,584, not_linked→edge 1,199, not_in_text→near_offtopic 866, near→edge 669, not_in_text→near 643, near→near_child 419, near→offtopic_only 335, not_in_text→filtered 265, not_in_text→no_cui 243, near→not_extracted 224
- **S-dict** (바뀐 triple 10,853, 13.1%): near→not_in_text 1,688, not_in_text→near_offtopic 1,600, not_in_text→near 1,380, not_linked→edge 1,237, near→edge 848, near→offtopic_only 527, near→near_child 499, offtopic_only→near 310, filtered→near 275, not_in_text→no_cui 243

## 6. HPO is_a와 그래프 IS_A의 차이 (ranker 모드)

ranker 모드에서 점수가 없는 stage(near_child, filtered, no_docs, not_in_text, not_extracted, not_linked, no_cui)의 triple 중, 같은 CUI 매핑으로 HPO is_a 부모(또는 자식만)의 CUI에 topic edge가 있는 triple 수다. ranker는 HPO 계층을 쓰지 않으므로 점수가 없다. 공개 정의의 near는 이 경우를 포함했다. 반대로 그래프 IS_A(UMLS PAR/RB)로만 닿는 경우는 공개 정의에서 빠졌다(5절의 not_in_text→near, not_in_text→near_offtopic).

| graph | HPO 부모 topic edge | HPO 자식만 topic edge | 그중 판정이 near_child |
|---|---|---|---|
| D (base_nm) | 887 | 714 | 736 |
| S-mix | 1,905 | 803 | 556 |
| S-llm | 1,985 | 737 | 519 |
| S-dict | 2,020 | 816 | 611 |

## 7. 보충: v2 corpus에 v3 flag만 적용 (base_split_tg_ns와 같은 graph)

`recall_decomp.py --link full --topic-genes --no-self [--feature-cui ranker --offtopic-w 0]` (v2 sources, ie_gemma_full, phrase2cui_gemma). ranker 실행의 whole-graph 수치는 build_base_split_tg_ns.log와 같다(27,473 / 1,144,472 / 46,529 / 33,298).

| | xref | ranker |
|---|---|---|
| not_linked | 2,276 | 1,546 |
| extraction+linking | 4,153 | 3,236 |
| edge | 14,514 | 15,001 |

flag만으로 D의 not_linked가 줄어드는 폭은 ranker 기준에서 더 크다(xref 5,113→2,276, ranker 5,658→1,546). flag를 고정하고 corpus만 v3로 바꾸면 extraction+linking은 xref 기준 4,153→4,241(S-mix)로 거의 그대로지만, ranker 기준으로는 3,236→2,696로 줄어든다.

## 주의

- 문자열 단계(no_docs, not_in_text, not_extracted, not_linked)의 판정은 모드와 무관하다. 모드 간 차이는 그 앞의 CUI 단계(edge, near, offtopic_only, near_offtopic, near_child, filtered)가 먼저 가져가는 triple이 달라서 생긴다.
- no_cui 393 triple(155 pair)은 hpo_umls_mapping CUI도 hp.json xref도 없다. 그래서 both에서도 같다. load_cases는 이 feature를 환자 증거에서 뺀다.
- `scored`는 HAS_FINDING 경로만 본다. unified evidence(환자 CUI가 gold 질병이나 그 부모 질병 자체인 경우)는 넣지 않았다. --no-self 때문에 이런 CUI는 filtered로 분류되지만 ranker는 unified evidence로 점수를 준다(앞선 검증에서 xref 기준 S-mix 149, S-llm 146, S-dict 79 triple).
- 문자열 존재 검사는 부정문 속 언급도 'text에 있음'으로 센다. 그래서 not_extracted는 엄밀한 추출 손실 하한이 아니다(앞선 검증 지적, 이번 변경과 무관).
