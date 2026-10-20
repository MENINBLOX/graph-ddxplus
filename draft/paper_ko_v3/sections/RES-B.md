### 4.4 같은 문헌 위 추출 방식 비교

이 절의 결과는 모두 설계 D 확증(2026-10-13) 이후 같은 9개 개발 세트에서 수행한 탐색이다(4.4.7절은 설계 E 확증 실행 뒤의 분석이며, 확증 세트 E의 사후 분석을 일부 포함한다). 4.2의 확증 결과는 바뀌지 않는다. 각 그래프는 근거 언급(Mention) 노드 없이 비교용 인스턴스에 빌드했다. 채점은 동결 1-홉 설정을 그대로 쓰고, 그래프마다 주제 밖 가중치(offtopic_w, 이하 ot)만 {0, 0.25, 0.5, 1}에서 D3 규칙(9개 세트 매크로 Top-10, 동률이면 Top-1)으로 다시 골랐다. 표 12의 Top-10 비교에는 환자 대응 부트스트랩 95% CI와 정답 질환 군집 부트스트랩 95% CI를 함께 적는다. 그 밖의 세트별 차이는 Holm 보정 정확 McNemar 검정으로 유의 여부를 표시한다(3.8절). 9개 세트에서 방향이 일관되는지는 양측 부호검정으로 본다. 모든 비교는 개념 연결 단계까지 포함한 파이프라인 비교다.

#### 4.4.1 v2 문헌: LLM 추출과 UMLS 사전 매칭의 비교

표 10은 v2 문헌에서 추출·연결 방식이나 빌더 규칙을 바꾼 그래프 18개(기준 1개와 변형 17개)를 보여 준다. 자기 간선을 둔 채로는 UMLS 사전 매칭(v2umls)의 매크로 Top-10이 34.60%로 LLM 추출(base_split_tg, 32.59%)보다 높았다. 같은 조건에서 매크로 Top-1은 LLM 추출이 12.16%로 UMLS 사전(10.21%)보다 높았다. 세트별 Top-1은 LLM 추출이 7개 세트에서 높았다(2/7, p = .18). Holm 유의한 차이는 RAMEDIS −3.7%p, PUMCH_L −3.6%p, Phenopacket −1.0%p, SymCat +0.6%p였다(UMLS 사전 − LLM). 자기 간선(4.4.4)을 빼면 매크로 Top-10/Top-1은 UMLS 사전(v2umls_ns) 34.50/14.24%, LLM 추출(base_split_tg_ns) 31.94/13.06%로 두 지표 모두 UMLS 사전이 높았다. 세트별 부호는 Top-1 7/2(p = .18), Top-10 7/1(p = .070)로 유의하지 않았다. Holm 보정 후 유의한 세트별 Top-1 차이는 DDXPlus +1.8%p, SymCat +1.2%p, Phenopacket −0.8%p였다. HPO 사전(v2dict_ns)의 Top-1은 9개 세트 모두에서 UMLS 사전보다 낮았다(9/0, p = .004).

UMLS 사전에 LLM 항목을 더한 몫은 작았다. 합집합 v2unionu_ns의 매크로 34.66/14.77%는 v2 그래프 가운데 두 지표 모두 가장 높았다. 그러나 UMLS 사전 단독과의 차이는 Top-1 +0.53%p(세트 6/2, p = .29), Top-10 +0.16%p(5/3, p = .73)였다. 자기 간선을 둔 그래프에서 HPO 사전에 LLM 항목을 더하면 매크로 Top-1이 9.99%에서 11.93%로 올랐다(세트 8/0, p = .008). 자기 간선을 뺀 그래프에서는 12.28%에서 14.23%로 올랐다(7/0, p = .016). 자기 간선을 둔 그래프의 세트별 차이 가운데 DDXPlus +4.9%p는 Holm 유의였고, 정답 질환 군집 CI [1.5, 9.0]도 0을 포함하지 않았다.

평가에 쓰는 CUI로 세면 흔한 질환 벤치의 소견 언급도 대부분 HPO 개념이었다(SymCat 65%, DDXPlus 70%). 따라서 표 10의 비교는 사전 어휘와 환자 어휘가 겹치는 데서 오는 편향을 배제하지 않는다.

Table 10. Extraction and linking variants on the same v2 literature (development, nine sets). Macro = mean of the nine set values (%, lineage). ot = offtopic weight re-selected per graph by rule D3. Higher/lower = number of sets above/below the frozen graph (ties excluded), two-sided sign test p in parentheses. No LLM extraction = dictionary matching without an LLM; CUI-less dictionary items (all HPO dictionary items, and UMLS strings not resolved to one concept) are linked by HPO name and through the v2 phrase table, which was built from LLM phrases (3.3, 4.4.7). In a 1/20 sample of the v2 literature, 4.6% of UMLS dictionary items received their concept only through that table.

| Graph | Extraction / linking | Edges | ot | Macro Top-10 / Top-1 | vs frozen: Top-10 higher/lower (p) · Top-1 |
|---|---|---|---|---|---|
| base_nm (= S-D, frozen) | LLM | 832,671 | 0 | 30.56 / 12.64 | — |
| base_split | LLM + coordination split, HPO-name relink | 1,153,875 | .5 | 32.21 / 12.17 | 7/1 (.070) · 5/2 |
| base_split_tg | above + causal-gene title as topic | 1,153,875 | .5 | 32.59 / 12.16 | 7/1 (.070) · 5/2 |
| v2dict | HPO dictionary (no LLM extraction) | 1,880,407 | 0 | 31.64 / 9.99 | 7/2 (.18) · 3/6 |
| v2umls | UMLS finding-type dictionary (no LLM extraction) | 5,016,970 | .5 | 34.60 / 10.21 | 8/1 (.039) · 4/5 |
| v2union | LLM + HPO dictionary | 2,440,332 | .5 | 33.39 / 11.93 | 8/1 (.039) · 5/4 |
| v2unionu | LLM + UMLS dictionary | 5,257,772 | .25 | 34.28 / 11.25 | 8/1 (.039) · 5/3 |
| *Qualifiers removed (_nq)* | | | | | |
| base_split_tg_nq | | 1,141,998 | .5 | 32.61 / 12.15 | 7/1 (.070) · 5/2 |
| v2dict_nq | | 1,527,478 | .25 | 31.75 / 9.97 | 7/2 (.18) · 2/6 |
| v2union_nq | | 2,088,086 | .5 | 33.78 / 11.92 | 8/1 (.039) · 6/3 |
| v2unionu_nq | | 5,189,584 | 0 | 34.38 / 11.24 | 8/1 (.039) · 5/3 |
| *Self-edges removed (_ns)* | | | | | |
| base_nm_ns | | 824,624 | .5 | 30.14 / 12.87 | 3/3 (1) · 3/3 |
| base_split_tg_ns | | 1,144,472 | 0 | 31.94 / 13.06 | 7/2 (.18) · 5/4 |
| v2dict_ns | | 1,872,312 | .25 | 32.65 / 12.28 | 7/2 (.18) · 4/5 |
| v2umls_ns | | 4,995,414 | .5 | 34.50 / 14.24 | 8/1 (.039) · 6/1 |
| v2union_ns | | 2,427,194 | .25 | 33.36 / 14.23 | 8/1 (.039) · 6/3 |
| v2union_nq_ns | | 2,074,961 | .5 | 33.30 / 14.31 | 8/1 (.039) · 6/2 |
| v2unionu_ns | | 5,235,933 | .5 | 34.66 / 14.77 | 8/1 (.039) · 7/2 |

#### 4.4.2 v2+v3 문헌: LLM 단독, 사전, 합집합

v2+v3 문헌에서 매크로 Top-10/Top-1은 LLM 단독 추출(S-llm) 44.90/19.30%, UMLS 사전(S-dict) 42.88/18.11%였다(표 11). 매크로 차이는 +2.0/+1.2%p였지만 세트별 부호는 Top-10 6/3(p = .51), Top-1 5/3(p = .73)으로 유의하지 않았다. 희귀질환 세트 7개 중 6개에서 S-llm의 Top-10이 높았다(S-llm − S-dict, Holm 유의: RAMEDIS +8.0%p, LIRICAL +6.2%p, Phenopacket +3.1%p). 나머지 PUMCH_L에서는 S-llm이 1.4%p 낮았다(Holm 비유의). 흔한 질환 합성 세트 두 개(DDXPlus 48질환, SymCat 760질환)에서는 S-dict가 높았다(Holm 유의: DDXPlus Top-10 −3.5%p·Top-1 −5.6%p, SymCat Top-10 −3.0%p·Top-1 −1.1%p). PUMCH_L의 Top-1도 S-dict가 4.3%p 높았다(Holm 유의). HPO 사전(v3dict_ns)과 비교하면 S-llm이 Top-10과 Top-1 모두 8개 세트에서 높았다(각각 8/1, p = .039).

v2+v3 문헌에서는 LLM 항목에 사전 항목을 더한 합집합의 매크로가 LLM 단독보다 낮았다. LLM + HPO 사전(v3union_ns)은 44.00/18.16%, LLM + UMLS 사전(S-union)은 43.77/17.59%였다. 이 매크로 차이에는 대응 검정을 하지 않았다. LLM 단독보다 Top-10이 높은 세트와 낮은 세트의 수는 HPO 합집합 4/5, UMLS 합집합 3/6이었다. 두 합집합 모두 RAMEDIS·LIRICAL·HMS·MME·Phenopacket에서 낮았고 DDXPlus·PUMCH_L·SymCat에서 높았다. 반대로 UMLS 사전에 LLM 항목을 더하면 Top-10은 7개 세트에서 오르고 내린 세트가 없었으나(7/0, p = .016), Top-1은 4/5로 갈렸다. v2에서 동결 그래프와 LLM + HPO 사전 합집합(v2union)을 비교했을 때, Top-10을 잃은 PUMCH-ADM·PUMCH_L·DDXPlus 환자에서는 정답 질환 프로필이 약 2배가 되었다. 그러나 환자와 맞는 소견은 중앙값 1~2개만 늘었고, 소견 점수 합은 0.91~1.11배였다. 길이 정규화의 분모가 점수 합보다 크게 늘어 정답 점수가 희석된 것이다. 이 기전이 v3 합집합에서도 작동하는지는 확인하지 않았다.

S-llm의 ot는 격자의 끝값인 1.0(주제 구분을 끈 값)에서 선택되었다. S-llm의 매크로 Top-10은 ot가 0에서 1.0으로 갈수록 43.64%에서 44.90%로 올랐다. 같은 격자에서 매크로 Top-1은 19.76%(ot 0), 19.83%(0.25), 19.78%(0.5), 19.30%(1.0)로, 선택된 1.0에서 가장 낮았다. 1.0을 뺀 세 ot(0, 0.25, 0.5)에서 S-llm의 Top-1은 모두 S-mix(19.66%)보다 높았다. 따라서 S-mix의 Top-1 우세(+0.4%p, 4.4.3)는 S-llm이 ot 1.0을 고른 데서만 나온다.

#### 4.4.3 혼합 추출(S-mix)

S-mix(v3llmd_ns)의 개발 매크로 Top-10/Top-1은 46.11/19.66%로, 같은 순위화로 평가한 그래프 32개 가운데 두 지표 모두 가장 높았다(그래프마다 고른 ot 기준). 32개는 v2 문헌 그래프 18개(base_nm 포함, 표 10)와 v2+v3 문헌 그래프 14개다. v2+v3 문헌 그래프에는 본문 전체를 읽는 판 2개(v3umlsfull_ns, v3dictfull_ns)와 일반 문헌을 되살린 판 2개(4.4.6)가 들어 있다. 표 11에 없는 v2+v3 문헌 그래프 4개의 매크로는 v3dict 39.20/10.83%(표 13a), v3dictfull_ns 41.38/18.27%, v3llm_ns_kn 44.98/19.26%, v3umls_ns_kn 42.71/18.05%다. ot 격자의 매크로는 0에서 45.78/19.34%, 0.25에서 46.11/19.51%, 0.5에서 46.11/19.66%, 1.0에서 45.68/18.75%였다. Top-10 동률을 Top-1로 깨서 0.5를 골랐다. DDXPlus의 Top-10/Top-1은 40.6/11.8%, SymCat은 13.3/2.5%였다.

설계 D 동결 시스템(S-D) 대비 S-mix의 Top-10은 9개 세트 모두에서 높았다(9/0, p = .004; 표 12). 차이는 DDXPlus +7.6%p [5.7, 9.4], SymCat +2.7%p [1.9, 3.4], Phenopacket +14.3%p [13.6, 15.0]였다. 가장 큰 차이는 MME +30.0%p와 RAMEDIS +26.0%p였다. 정답 질환 군집 CI는 DDXPlus [−6.1, 22.1]와 PUMCH-ADM [−1.5, 26.8]에서 0을 포함했다. Top-1은 7개 세트에서 높고 2개 세트에서 낮았다(7/2, p = .18). 낮아진 세트는 RAMEDIS(−7.7%p)와 SymCat(−0.7%p)이며 둘 다 Holm 유의였다. S-D 대비 S-mix의 RAMEDIS Top-1 손실 102건 중 95건은 정답이 PKU인 환자였다. S-D는 PKU 환자 148명 중 95명을 Top-1로 맞혔고, S-mix는 0명을 맞혔다. S-mix가 새로 맞힌 54건에는 PKU 환자가 없었다. PKU 환자를 빼면 RAMEDIS Top-1은 S-D 4.0%, S-mix 13.9%였다(4.4.6).

S-mix와 S-llm의 매크로 차이는 Top-10 +1.2%p, Top-1 +0.4%p였다. 세트별 부호는 두 지표 모두 5/3(p = .73)이었다. Holm 유의 Top-10 상승은 DDXPlus +1.9%p [0.9, 2.8], PUMCH_L +3.7%p [2.0, 5.6], SymCat +1.9%p [1.4, 2.4]였다. Holm 유의 Top-1 상승은 PUMCH_L +2.7%p, Phenopacket +1.2%p였다. Holm 유의 하락은 없었다. DDXPlus Top-10 차이의 정답 질환 군집 CI는 [−2.7, 5.8]로 0을 포함했다.

S-mix와 S-dict의 매크로 차이는 Top-10 +3.2%p, Top-1 +1.6%p였다(Top-10 세트 7/2, p = .18; Top-1 5/3, p = .73). Top-10은 RAMEDIS +9.3%p, LIRICAL +5.7%p, Phenopacket +3.0%p에서 높고 SymCat −1.1%p에서 낮았다(모두 Holm 유의). DDXPlus Top-10 차이는 −1.6%p [−2.9, −0.4]였고 Holm 보정 p는 .06이었다. Top-1은 DDXPlus −4.9%p, SymCat −0.8%p에서 낮고 Phenopacket +1.5%p에서 높았다(모두 Holm 유의).

S-mix와 S-union의 매크로 차이는 Top-10 +2.3%p, Top-1 +2.1%p였다(Top-10 세트 6/2, p = .29; Top-1 7/2, p = .18). Top-10은 RAMEDIS +9.3%p, LIRICAL +3.8%p, Phenopacket +2.7%p에서 높고 DDXPlus −3.0%p, SymCat −1.3%p에서 낮았다(모두 Holm 유의). Top-1은 Phenopacket +1.8%p에서 높고 DDXPlus −4.6%p, SymCat −1.0%p에서 낮았다(모두 Holm 유의). 이 대비에는 사전을 적용한 범위(LLM 구절 안 대 모든 문장)의 차이와, S-mix가 재연결 대상 LLM 항목을 지운 효과가 함께 들어 있다.

재연결 대상 항목을 지우지 않고 사전 항목만 덧붙인 판(v3llmda_ns, 간선 5,433,244개)의 매크로는 45.88/19.41%로 교체판(S-mix)보다 조금 낮았다(Top-10 −0.2%p, 세트 1/5, p = .22). Holm 유의한 세트별 차이는 없었다.

S-mix는 같은 9개 개발 세트에서 이 그래프 32개 가운데 매크로 Top-10이 가장 높은 것으로 골랐고, ot도 그래프마다 다시 골랐다. 따라서 S-mix의 개발값에는 선택 낙관이 들어 있다. 변형 가운데 2위인 v3llmda_ns(구절 안 사전 항목을 덧붙인 판, 45.88%)와의 매크로 Top-10 차이는 0.2%p이고, LLM 단독(S-llm, 44.90%)과의 차이는 1.2%p다. 같은 개발 세트에서 1-홉 랭커의 매개변수를 각각 다시 고른 HPO 주석은 52.45/30.37%, PrimeKG는 33.52/15.18%였다. v2 문헌의 BM25는 24.46/8.75%였다. 이 세 값과는 대응 검정을 하지 않았다.

Table 11. Graphs on the v2 + v3 literature (development). (a) Build size and macro accuracy. Edges and diseases with findings are from the build summaries; all graphs have 43,575 Disease nodes. vs S-D = number of sets above/below the frozen design-D graph (sign test p). v3umlsfull_ns reads full PMC text instead of 12k clinical sections, so its literature differs and it is listed for reference only.

| System | Graph | Extraction | Self-edges | Edges | Diseases with findings | Concepts | ot | Macro Top-10 / Top-1 | vs S-D: Top-10 (p) · Top-1 (p) |
|---|---|---|---|---|---|---|---|---|---|
| S-D | base_nm (v2 literature) | LLM | kept | 832,671 | 27,309 | 46,130 | 0 | 30.56 / 12.64 | — |
| | v3dict_ns | HPO dictionary | removed | 5,159,708 | 34,202 | 14,421 | 0 | 40.51 / 15.94 | 9/0 (.004) · 6/3 (.51) |
| | v3umls | UMLS dictionary | kept | 14,544,635 | 34,359 | 96,653 | 0 | 41.55 / 10.55 | 9/0 (.004) · 5/4 (1) |
| S-dict | v3umls_ns | UMLS dictionary | removed | 14,506,788 | 34,356 | 95,608 | .5 | 42.88 / 18.11 | 9/0 (.004) · 8/1 (.039) |
| | v3llm | LLM | kept | 3,696,668 | 33,487 | 79,890 | 1.0 | 44.09 / 16.44 | 8/0 (.008) · 8/1 (.039) |
| S-llm | v3llm_ns | LLM | removed | 3,672,106 | 33,123 | 76,487 | 1.0 | 44.90 / 19.30 | 9/0 (.004) · 6/3 (.51) |
| | v3union_ns | LLM + HPO dictionary | removed | 7,119,107 | 34,246 | 77,249 | .25 | 44.00 / 18.16 | 9/0 (.004) · 6/3 (.51) |
| S-union | v3unionu_ns | LLM + UMLS dictionary (all sentences) | removed | 15,396,388 | 34,359 | 115,224 | .5 | 43.77 / 17.59 | 9/0 (.004) · 8/1 (.039) |
| | v3llmda_ns | LLM + in-span UMLS strings added | removed | 5,433,244 | 33,558 | 86,045 | .25 | 45.88 / 19.41 | 9/0 (.004) · 7/2 (.18) |
| **S-mix** | **v3llmd_ns** | **LLM span; unlinked spans named by in-span UMLS strings** | removed | **5,221,285** | **33,537** | **85,761** | **.5** | **46.11 / 19.66** | 9/0 (.004) · 7/2 (.18) |
| (ref.) | v3umlsfull_ns | UMLS dictionary, full text | removed | 28,096,533 | 34,562 | 108,809 | .25 | 43.08 / 19.41 | 9/0 (.004) · 8/1 (.039) |

(b) Per-set lineage Top-10 / Top-1 (%), development.

| Set (n) | S-D | S-dict | S-llm | v3union_ns | S-union | S-mix |
|---|---|---|---|---|---|---|
| DDXPlus (3,000) | 33.0 / 11.2 | 42.3 / 16.6 | 38.8 / 11.0 | 38.8 / 10.8 | 43.6 / 16.4 | 40.6 / 11.8 |
| RAMEDIS (624) | 31.3 / 18.3 | 47.9 / 8.2 | 55.9 / 11.4 | 54.0 / 10.1 | 47.9 / 8.7 | 57.2 / 10.6 |
| LIRICAL (370) | 23.2 / 9.2 | 32.7 / 15.7 | 38.9 / 17.8 | 36.5 / 16.5 | 34.6 / 16.2 | 38.4 / 17.8 |
| HMS (88) | 28.4 / 9.1 | 43.2 / 15.9 | 44.3 / 17.1 | 38.6 / 15.9 | 43.2 / 14.8 | 44.3 / 15.9 |
| MME (40) | 25.0 / 12.5 | 52.5 / 22.5 | 57.5 / 37.5 | 55.0 / 25.0 | 55.0 / 25.0 | 55.0 / 35.0 |
| PUMCH_L (988) | 50.7 / 20.1 | 64.8 / 36.1 | 63.4 / 31.9 | 65.9 / 34.2 | 65.4 / 33.5 | 67.1 / 34.6 |
| PUMCH-ADM (75) | 56.0 / 22.7 | 60.0 / 30.7 | 62.7 / 30.7 | 64.0 / 33.3 | 61.3 / 26.7 | 68.0 / 33.3 |
| Phenopacket (10,310) | 16.7 / 7.5 | 28.1 / 13.9 | 31.1 / 14.2 | 29.7 / 14.6 | 28.3 / 13.6 | 31.0 / 15.4 |
| SymCat (7,481) | 10.7 / 3.2 | 14.5 / 3.3 | 11.5 / 2.2 | 13.4 / 3.0 | 14.6 / 3.5 | 13.3 / 2.5 |

Table 12. Paired development comparisons for the design-E contrasts (S-mix minus comparator, percentage points). Top-10 cells: difference [patient-paired bootstrap 95% CI] / [gold-disease cluster bootstrap 95% CI]; Top-1 cells: difference [patient CI]. * Holm-adjusted exact McNemar p < .05 within the nine sets. Last row: sets where S-mix is higher/lower (two-sided sign test p).

(a) Top-10

| Set | S-mix − S-D | S-mix − S-llm | S-mix − S-dict | S-mix − S-union |
|---|---|---|---|---|
| DDXPlus | +7.6 [5.7, 9.4] / [−6.1, 22.1]* | +1.9 [0.9, 2.8] / [−2.7, 5.8]* | −1.6 [−2.9, −0.4] / [−8.3, 4.5] | −3.0 [−4.1, −1.7] / [−9.1, 2.6]* |
| RAMEDIS | +26.0 [22.0, 29.7] / [15.3, 39.2]* | +1.3 [−0.8, 3.4] / [−1.3, 4.4] | +9.3 [6.9, 12.0] / [2.5, 20.2]* | +9.3 [6.7, 12.2] / [2.7, 20.3]* |
| LIRICAL | +15.1 [10.8, 19.5] / [9.2, 22.0]* | −0.5 [−2.2, 1.1] / [−2.2, 1.2] | +5.7 [2.7, 8.7] / [2.6, 9.0]* | +3.8 [1.1, 6.5] / [1.2, 6.8]* |
| HMS | +15.9 [8.0, 25.0] / [7.7, 26.0]* | 0.0 [−6.8, 6.8] / [−6.4, 6.9] | +1.1 [−5.7, 8.0] / [−4.2, 7.8] | +1.1 [−4.6, 6.8] / [−3.3, 7.0] |
| MME | +30.0 [15.0, 45.0] / [11.1, 63.3]* | −2.5 [−7.5, 0.0] / [−11.1, 0.0] | +2.5 [−7.5, 12.5] / [−8.0, 18.2] | 0.0 [−10.0, 10.0] / [−10.2, 13.0] |
| PUMCH_L | +16.4 [13.8, 19.1] / [8.4, 25.6]* | +3.7 [2.0, 5.6] / [0.6, 7.7]* | +2.3 [0.4, 4.3] / [−2.1, 6.5] | +1.7 [−0.2, 3.6] / [−2.6, 5.1] |
| PUMCH-ADM | +12.0 [2.7, 22.7] / [−1.5, 26.8]* | +5.3 [−1.3, 13.3] / [−1.3, 11.4] | +8.0 [1.3, 16.0] / [1.3, 17.3] | +6.7 [0.0, 13.3] / [0.0, 15.8] |
| Phenopacket | +14.3 [13.6, 15.0] / [10.9, 18.0]* | −0.1 [−0.5, 0.3] / [−1.7, 1.2] | +3.0 [2.4, 3.5] / [0.5, 5.5]* | +2.7 [2.2, 3.2] / [0.3, 5.3]* |
| SymCat | +2.7 [1.9, 3.4] / [1.4, 4.0]* | +1.9 [1.4, 2.4] / [1.0, 2.7]* | −1.1 [−1.7, −0.5] / [−2.2, −0.2]* | −1.3 [−1.8, −0.7] / [−2.2, −0.4]* |
| Higher/lower (p) | 9/0 (.004) | 5/3 (.73) | 7/2 (.18) | 6/2 (.29) |
| Macro difference | +15.6 | +1.2 | +3.2 | +2.3 |

(b) Top-1

| Set | S-mix − S-D | S-mix − S-llm | S-mix − S-dict | S-mix − S-union |
|---|---|---|---|---|
| DDXPlus | +0.6 [−0.7, 1.8] | +0.7 [−0.1, 1.5] | −4.9 [−5.8, −3.9]* | −4.6 [−5.6, −3.6]* |
| RAMEDIS | −7.7 [−11.4, −3.9]* | −0.8 [−2.7, 1.1] | +2.4 [0.3, 4.5] | +1.9 [−0.2, 4.0] |
| LIRICAL | +8.7 [4.9, 12.4]* | 0.0 [−1.9, 1.9] | +2.2 [−0.3, 4.6] | +1.6 [−0.3, 3.8] |
| HMS | +6.8 [0.0, 14.8] | −1.1 [−4.6, 2.3] | 0.0 [−6.8, 6.8] | +1.1 [−4.6, 6.8] |
| MME | +22.5 [10.0, 37.5]* | −2.5 [−10.0, 5.0] | +12.5 [0.0, 25.0] | +10.0 [0.0, 22.5] |
| PUMCH_L | +14.5 [11.7, 17.1]* | +2.7 [1.0, 4.6]* | −1.5 [−3.7, 0.8] | +1.1 [−0.7, 3.1] |
| PUMCH-ADM | +10.7 [1.3, 21.3] | +2.7 [0.0, 6.7] | +2.7 [−6.7, 12.0] | +6.7 [−1.3, 14.7] |
| Phenopacket | +7.9 [7.4, 8.4]* | +1.2 [0.9, 1.6]* | +1.5 [1.0, 1.9]* | +1.8 [1.4, 2.2]* |
| SymCat | −0.7 [−1.1, −0.2]* | +0.3 [0.0, 0.6] | −0.8 [−1.2, −0.5]* | −1.0 [−1.3, −0.7]* |
| Higher/lower (p) | 7/2 (.18) | 5/3 (.73) | 5/3 (.73) | 7/2 (.18) |
| Macro difference | +7.0 | +0.4 | +1.6 | +2.1 |

#### 4.4.4 자기 간선

질환에서 자기 개념과 직접 상위 질환의 개념으로 가는 간선(자기 간선)을 빼면 매크로 Top-1이 올랐다(표 13a). v2 문헌에서 매크로 Top-1은 동결 그래프 12.64→12.87%, LLM(분해·재연결) 12.16→13.06%, HPO 사전 9.99→12.28%, UMLS 사전 10.21→14.24%, LLM + HPO 사전 11.93→14.23%, LLM + UMLS 사전 11.25→14.77%로 바뀌었다. 빠지는 간선은 그래프마다 8,047~21,839개로 전체 간선의 0.4~1%였다. 같은 제거로 DDXPlus Top-10은 2.6~5.5%p 낮아졌다(모두 Holm 유의). 제거의 매크로 Top-1 효과는 v2 LLM 그래프에서 +0.2~0.9%p, v2 사전 그래프에서 +2.3~4.0%p였다.

v2+v3 문헌의 UMLS 사전 그래프에서는 매크로 Top-1이 10.55%에서 18.11%로 올랐다(세트 8/1, p = .039; PUMCH_L +21.8%p, PUMCH-ADM +17.3%p). v3 LLM 그래프에서는 16.44%에서 19.30%로 올랐다(세트 7/2, p = .18; Holm 유의 PUMCH_L +9.8%p, Phenopacket +2.9%p). 같은 v3 LLM 그래프에서 자기 간선 제거는 DDXPlus의 Top-10을 6.1%p, Top-1을 4.4%p 낮췄다. SymCat은 Top-10 1.8%p, Top-1 1.3%p, RAMEDIS는 Top-10 2.2%p 낮아졌다(모두 Holm 유의). D3 규칙은 v3의 사전·LLM 그래프 모두에서 제거판을 골랐다.

기전을 보기 위해 UMLS 사전 그래프의 두 판(v2umls_ns, v2umls; 둘 다 ot 0.5)에서 환자별 순위를 비교했다(표 13b). 이 두 판 사이에서 자기 간선 제거로 Top-1이 오른 세트는 7개, 낮아진 세트는 DDXPlus와 SymCat이었다. Top-1 분석은 오른 7개 세트 가운데 MME를 뺀 6개 세트(PUMCH_L, Phenopacket, RAMEDIS, LIRICAL, HMS, PUMCH-ADM)에서만 했다. 대상은 자기 간선을 뺀 판에서는 Top-1이 맞고 자기 간선이 있는 판에서는 틀린 환자다. 이런 환자는 PUMCH_L 130명(반대 방향 18명), Phenopacket 168명(57명), RAMEDIS 36명(10명), LIRICAL 12명(0명), HMS 5명(0명), PUMCH-ADM 7명(0명)이었다. 이들에게서 정답을 밀어낸 질환의 자기 개념(또는 상위 질환 개념)은 거의 모두 환자 소견에 있었다(PUMCH_L 124/130명, Phenopacket 165/168명, RAMEDIS 36/36명). 그 자기 간선은 밀어낸 질환의 점수 항 가운데 가장 컸다(PUMCH_L 115명, Phenopacket 158명, RAMEDIS 35명). 이 6개 세트의 358명에서 밀어낸 질환의 최대 점수 항은 자기 개념 262명, 상위 질환 개념 65명, 그 밖 31명이었다. 밀어낸 질환에는 이름이 곧 징후인 질환 노드가 많았다(Chylopericardium, Nocturia, Hypokalemia, Thumb aplasia, Syndactyly, Obesity). UMLS 사전 그래프의 자기 CUI 간선은 13,036개이고 평균 지지 문서 수(n_docs)는 3.15편으로 다른 간선(1.42편)보다 많았다. LLM 동결 그래프에서는 자기 CUI 간선이 5,856개, 평균 지지 문서 수가 1.65편(다른 간선 1.13편)이었다.

반대 방향인 DDXPlus Top-10에서는 같은 두 판 사이에 자기 간선을 빼서 Top-10을 잃은 환자가 122명, 얻은 환자가 44명이었다. 잃은 122명 중 117명은 정답 질환 개념이나 그 상위 질환 개념을 입력 소견으로 가졌다. 그중 106명은 그 간선이 정답 질환 점수의 최대 항이었다. 이 106명의 정답에는 급성 폐부종 43명(상위 개념 Pulmonary edema), GERD 36명, 공황 발작 13명(상위 개념 Anxiety)이 포함되었다. DDXPlus 개발 환자의 15.3%(458/3,000)는 입력 소견에 정답 질환 CUI 자체가 있고, 상위 질환 개념까지 넣으면 22.7%다(3.6절).

#### 4.4.5 후보 창과 자기 간선의 상호작용

동결 순위화는 소견 점수를 상위 40개 질환(k × 4, k = 10)에서만 읽고, 통합 근거 점수는 그 밖의 질환에도 더한다(3.5절). 앞의 DDXPlus 117명에서 정답 질환의 통합 근거 항은 두 판에서 똑같이 +2.0이었다. 이 117명 가운데 적어도 44명은, 자기 간선을 뺀 판에서 정답의 소견 점수를 읽었다면 Top-10 안의 대체 질환보다 총점이 높았다. 이 44명은 정답이 소견 점수 창(상위 40) 밖으로 밀려 통합 근거 점수만 받은 경우다. 나머지 73명은 이 분석의 기록으로는 창의 영향인지 판단할 수 없다.

창 밖 질환 가운데 통합 근거를 받은 질환의 소견 점수도 읽는 채점(이하 확장 채점, 탐색)에서는 자기 간선 제거가 DDXPlus Top-10도 올렸다(+1.0%p [0.3, 1.8], Holm 유의). 확장 채점에서 v2umls_ns는 v2umls보다 Top-1이 9개 세트 모두에서(9/0, p = .004), Top-10이 7개 세트에서(7/0, p = .016) 높았다. 그러나 확장 채점의 매크로는 창 있는 채점보다 낮았다(표 13c). 동결 그래프는 30.56/12.64%에서 30.30/12.37%로, v2umls는 34.60/10.21%에서 32.82/9.65%로(세트 Top-1 0/7, p = .016; Top-10 1/8, p = .039), v2umls_ns는 34.50/14.24%에서 34.40/13.26%로 낮아졌다. 확장 채점에서는 통합 근거를 받은 질환(환자 소견 개념이 가리키는 질환과 그 직접 하위 질환)이 모두 소견 점수까지 받는다. 창은 D 확증에 쓰인 동결 설정의 일부이므로 바꾸지 않았다.

Table 13. Self-edge removal and the candidate window (development).

(a) Effect of removing disease→own-concept and disease→direct-parent-disease-concept edges. Each graph keeps its own D3-selected ot. Top-1 sign = sets where the self-edge-free graph is higher/lower. DDXPlus Top-10 difference = removed − kept [patient CI]; all nine rows are Holm-significant.

| Graph | Edges removed | Macro Top-1 kept → removed | Top-1 sign (p) | Macro Top-10 kept → removed | DDXPlus Top-10 diff |
|---|---|---|---|---|---|
| base_nm (frozen LLM, v2) | 8,047 | 12.64 → 12.87 | 3/3 (1) | 30.56 → 30.14 | −3.1 [−3.9, −2.3] |
| base_split_tg (LLM, v2) | 9,403 | 12.16 → 13.06 | 6/2 (.29) | 32.59 → 31.94 | −4.4 [−5.4, −3.5] |
| v2dict (HPO dictionary) | 8,095 | 9.99 → 12.28 | 5/3 (.73) | 31.64 → 32.65 | −5.5 [−6.6, −4.5] |
| v2umls (UMLS dictionary) | 21,556 | 10.21 → 14.24 | 7/2 (.18) | 34.60 → 34.50 | −2.6 [−3.4, −1.8] |
| v2union (LLM + HPO dict.) | 13,138 | 11.93 → 14.23 | 6/2 (.29) | 33.39 → 33.36 | −5.3 [−6.2, −4.4] |
| v2unionu (LLM + UMLS dict.) | 21,839 | 11.25 → 14.77 | 6/2 (.29) | 34.28 → 34.66 | −3.2 [−4.1, −2.3] |
| v3dict (HPO dictionary, v3) | 11,531 | 10.83 → 15.94 | 7/2 (.18) | 39.20 → 40.51 | −2.0 [−3.0, −1.1] |
| v3umls (UMLS dictionary, v3) | 37,847 | 10.55 → 18.11 | 8/1 (.039) | 41.55 → 42.88 | −1.8 [−2.8, −0.7] |
| v3llm (LLM, v3) | 24,562 | 16.44 → 19.30 | 7/2 (.18) | 44.09 → 44.90 | −6.1 [−7.2, −5.1] |

(b) Patient-level mechanism, v2umls (self-edges kept) vs v2umls_ns (removed), ot 0.5. Top-1 rows cover six of the seven sets where removal raised Top-1 (MME was not analysed); DDXPlus and SymCat, where removal lowered Top-1, were analysed only for Top-10 (DDXPlus).

| Set | Top-1 lost with self-edges / gained | Displacing disease's own (or parent) concept in patient findings | That self-edge is the displacing disease's largest score term |
|---|---|---|---|
| PUMCH_L | 130 / 18 | 124 | 115 |
| Phenopacket | 168 / 57 | 165 | 158 |
| RAMEDIS | 36 / 10 | 36 | 35 |
| LIRICAL, HMS, PUMCH-ADM | 12 / 0, 5 / 0, 7 / 0 | – | – |
| All 358 lost in the six sets | – | – | own concept 262, parent disease concept 65, other 31 |
| DDXPlus Top-10 lost / gained after removal | 122 / 44 | 117 (gold or parent disease concept in findings) | 106 (gold's largest term) |

(c) Candidate window (finding score read for the top k × 4 = 40 diseases) vs extended scoring (finding score also read for diseases outside the window that receive unified evidence; exploratory). Macro Top-10 / Top-1 (%); sign = sets where extended scoring is higher/lower.

| Graph | Window (frozen) | Extended scoring | Top-1 sign (p) | Top-10 sign (p) |
|---|---|---|---|---|
| base_nm | 30.56 / 12.64 | 30.30 / 12.37 | 2/4 (.69) | 3/5 (.73) |
| v2umls | 34.60 / 10.21 | 32.82 / 9.65 | 0/7 (.016) | 1/8 (.039) |
| v2umls_ns | 34.50 / 14.24 | 34.40 / 13.26 | 2/6 (.29) | 4/5 (1) |

#### 4.4.6 그 밖의 빌더 규칙

HPO 사전 매칭의 27.9%는 소견이 아닌 HPO 한정어('right', 'left', 'severe', 'mild', 'bilateral', 'moderate' 등)였다. 이 한정어는 질환 프로필 HPO 개념의 19.4%를 차지했다(LLM 경로의 HPO 재연결에서는 2.5%). HPOA aspect 규약에 따라 임상 수식어(M)·유전 양식(I)·빈도 항(542개 이름)을, 연결된 개념이 다른 HPO 항의 CUI가 아닐 때만 빼는 한정어 제외 규칙을 시험했다. LLM + HPO 사전 그래프에서 이 규칙은 간선을 2,440,332개에서 2,088,086개로 14.4% 줄였고 매크로 Top-10을 33.39%에서 33.78%로 바꿨다. 다른 그래프에서는 매크로 효과가 더 작고 방향이 일정하지 않아 기본값으로 쓰지 않았다.

질환 파일에 20회 이상 나열된 문헌(한 질환 파일에 두 번 나오면 2회로 센다)을 모든 질환에서 빼는 일반 문헌 규칙도 점검했다. 제목에 정답 질환의 정확한 이름이 있는 일반 문헌을 잃는 개발 환자는 v2 문헌에서 909명(21,396명 중), v2+v3 문헌에서 1,886명(22,858명 중)이었다. 해당 고유 정답 질환은 74개에서 165개로 늘었다. 이름이 제목에 있는 질환에만 일반 문헌 2,628편을 되살리는 빌드는 매크로를 ±0.2%p 안에서만 바꿨다(S-llm 44.90/19.30→44.98/19.26%, S-dict 42.88/18.11→42.71/18.05%). 이 규칙은 채택하지 않았다.

RAMEDIS 624명 중 148명(24%)의 정답은 PKU(C0031485)였고, 동결 그래프는 그중 95명을 Top-1로 맞혔다. 이 95명 중 92명의 1위는 고전적 PKU 노드(C0751434)였다. 이 노드는 소견이 2개(문헌 3,220자)뿐이어서 고유 소견 하나로 길이 정규화 점수 1위가 된다. 간선이 늘면 같은 환자의 다른 소견(Hypertyrosinemia)을 가진 질환이 앞선다. RAMEDIS Top-1 손실 가운데 PKU 환자의 몫은 분해·재연결 그래프 74건 중 68건, LLM + HPO 사전 75건 중 70건, UMLS 사전 101건 중 95건이었다. S-mix에서도 S-D 대비 RAMEDIS Top-1 손실 102건 중 95건이 PKU 환자였다(4.4.3). 이 하락은 단일 질환의 입도 효과이므로 변형 판단에 쓰지 않았다.

#### 4.4.7 사전 그래프 대비 혼합 추출: 비용과 기전

이 절은 S-mix와 S-dict를 구축 비용, 정확도, 간선 구성, HPO 주석 일치로 비교한다(표 14). 분석은 모두 설계 E 확증 실행 뒤에 했다. 순위 분석은 개발 세트의 저장된 순위로 했으므로 개발 결과다. 확증 세트 E에서 계산한 S-union − S-dict 비교는 사전에 정하지 않은 사후 분석이다.

**구축 비용.** 여기서 비용은 그래프를 한 번 만드는 데 든 시간이다. 문헌 수집은 네 시스템에 같으므로 넣지 않았다. LLM 경로(S-llm·S-mix·S-union)의 추출은 RTX 4090 1장에서 순수 생성 시간으로 127.4 GPU-시간이 걸렸다(v2 문헌 66.9시간, v3 새 문헌 60.6시간). 출력 토큰은 약 3.5억 개였다. 입력 토큰은 기록이 없어, 표본 창을 토크나이저로 센 평균으로 약 24억 개로 추정했다. 구절 정규화에는 CPU 단일 프로세스로 6.7시간 이하가 더 들었다. S-dict의 UMLS 사전 매칭은 GPU 없이 CPU 6~8 프로세스로 약 6분이 걸렸다(v2 2.2분 실측, v3 3.9분 추정). 단계별 시간을 더하면 추출부터 그래프 빌드까지 S-mix는 약 136시간, S-dict는 약 0.45시간이었다. 약 300배 차이이며, 이 차이는 거의 모두 LLM 추출의 GPU 시간이다. v2 문헌의 추출은 v3보다 느린 설정(eager 모드, 배치 64)으로 돌았다. v3 설정이었다면 LLM 경로의 생성 시간은 약 86.5시간이었을 것으로 추정된다(v2 출력 토큰 수 ÷ v3 평균 생성 속도). 추론에서는 네 시스템이 같은 Cypher 질의를 쓰고 LLM을 쓰지 않는다. 간선이 S-mix의 2.8배인 S-dict는 환자당 순위화 시간이 더 길었다(확증 실행의 DDXPlus-E에서 S-dict 155 ms, S-mix 87 ms). S-dict 구축도 LLM 산출물과 완전히 독립적이지는 않다. CUI 없이 저장된 사전 항목은 LLM 구절로 만든 v2 구절 표로 연결된다(3.3절). 문헌 파일의 1/20 표본에서 사전 항목의 4.6%가 이 표로만 개념을 얻었다. 최종 S-dict 간선 가운데 이 표에 기대는 간선의 비율은 재지 않았다.

**정확도.** 어느 그래프가 높은지는 세트 종류에 따라 갈렸다. 흔한 질환 합성 세트 두 개에서는 S-dict가 높았다. 두 세트 매크로 Top-10의 S-mix − S-dict는 개발(DDXPlus 48질환, SymCat 760질환)에서 −1.38%p [−2.06, −0.69], 확증 E(DDXPlus-E 49질환, SymCat-E 760질환)에서 −1.80%p [−2.44, −1.12]였다. 세트별로 보면 SymCat의 차이는 정답 질환 군집 CI도 0을 포함하지 않았다(개발 −1.1%p, 군집 CI [−2.2, −0.2]; E −1.2%p, 군집 CI [−2.1, −0.2]). DDXPlus Top-10의 차이는 환자 대응 CI만 0을 포함하지 않았다(E −2.4%p [−3.7, −1.2], 군집 CI [−8.8, +3.5]). DDXPlus Top-1의 차이는 군집 CI도 0을 포함하지 않았다(개발 −4.9%p, 군집 CI [−9.4, −1.1]; E −5.2%p, 군집 CI [−9.8, −1.4]). 희귀질환 개발 세트 7개에서는 S-mix가 높았다. 7개 세트 매크로 Top-10의 S-mix − S-dict는 +4.55%p [2.44, 6.73]였다. 세트별로는 RAMEDIS +9.3%p, LIRICAL +5.7%p, Phenopacket +3.0%p가 Holm 유의했고, 세 세트 모두 정답 질환 군집 CI도 0을 포함하지 않았다(표 12a). 이 세 값은 S-mix − S-dict다. 4.4.2절의 RAMEDIS +8.0%p, LIRICAL +6.2%p, Phenopacket +3.1%p는 S-llm − S-dict다.

사전 그래프에 LLM 항목을 더한 몫(S-union − S-dict)은 작았다. 흔한 질환 두 세트의 매크로 Top-10에서 이 차이는 개발 +0.74%p [0.22, 1.26], E +0.86%p [0.36, 1.36](사후 분석)였다. 두 CI는 환자 단위다. 두 세트의 세트별 정답 질환 군집 CI는 개발과 E 모두 0을 포함했다. 희귀질환 7개 세트 매크로에서는 +0.94%p [−0.15, 2.22]로 CI가 0을 포함했다.

**간선 구성.** 두 그래프의 간선을 (질환, 개념) 쌍으로 비교했다(표 14b). S-mix 간선 5,221,285개 가운데 86.9%(4,535,292개)는 S-dict에도 있었다. S-dict 간선은 14,506,788개로 S-mix의 2.8배였고, 그 가운데 68.7%는 S-mix에 없었다. S-mix에만 있는 간선(이하 S-mix 전용 간선)은 685,993개였다. 이 간선의 97.3%는 근거 언급이 모두 재연결되지 않은 LLM 항목(구절 표 경로)에서 나왔다. 근거 언급이 모두 구절 안 사전 재연결에서 나온 간선은 2.5%였다. S-mix 전용 간선의 88.7%는 지지 문서가 1편이었다. 공유 간선에서 이 비율은 S-mix 기준 71.0%였다. S-mix 전용 간선의 48.5%(332,817개)는 사전이 이름 붙일 수 없는 개념 위에 있었다. 곧 UMLS 사전 문자열도, 그 문자열의 HPO 이름도 그 개념을 가리키지 않았다. 이 간선의 개념에 많은 의미유형은 의약품(T121, 97,787개), 유기화합물(T109, 78,127개), 인구집단(T098, 46,455개)이었다(한 개념이 여러 유형을 가질 수 있다). 사전의 소견 의미유형에 드는 개념의 간선은 55,503개(16.7%)였다.

**표본 판독.** S-mix 전용 간선 30개를 무작위로 뽑아(시드 0) 근거 문장과 대조했다. 이 판독은 저자의 판단이며 임상 판정이 아니다. 18개는 오류로 보였다. 근거 문장이 그 질환의 소견을 지지하지 않는 경우가 6개, 다른 개념으로 연결된 경우가 6개였다(예: 약어 FCD가 fleck corneal dystrophy로 연결됨). 소견이 아닌 항목(검사 표의 행, 임상시험 결과 항목, 인구집단)이 4개, 표 형식의 잔재('Yes', 'Unknown (PCR)' 같은 칸 값)가 2개였다. 9개는 개념의 사전 문자열이 텍스트에 있어 사전도 잡았어야 하는 경우였다. S-dict가 같은 문자열을 다른 CUI로 연결한 경우가 5개, 복수형이라 정확 일치에 실패한 경우가 3개, 앞 표 칸의 'normal'이 부정 단서로 걸린 경우가 1개였다. 사전이 잡을 수 없는 새 표현은 3개였다. 그중 2개는 사전이 쓰는 UMLS 문자열에 그 개념의 문자열이 없는 경우였고, 1개는 형용사 'embolic'을 Embolism으로 읽은 경우였다.

**순위 반사실.** 개발 세트에서 한 그래프만 정답을 lineage Top-10에 넣은 환자를 그 그래프의 승리로 셌다(표 14c). 희귀질환 세트에서 S-mix의 승리는 RAMEDIS 68명, LIRICAL 26명, Phenopacket 505명이었다. 이 가운데 정답 질환에서 환자 소견 개념으로 가는 S-mix 전용 간선이 있는 환자는 16.2%, 61.5%, 17.8%였다. 같은 세트의 전체 환자에서 이 비율은 14.7%, 30.8%, 13.1%였다. 부분 반사실로, 정답 계보 질환의 S-mix 전용 간선 점수만 빼고 다른 질환의 점수와 길이 정규화는 그대로 두어 순위를 다시 계산했다. 이 계산에서도 승리는 대부분 유지되었다(RAMEDIS 63/68명, 92.6%; LIRICAL 20/26명, 76.9%; Phenopacket 448/505명, 88.7%). 같은 환자에서 S-dict가 정답 위에 올린 10개 질환 가운데 S-dict 전용 간선을 가진 질환은 중앙값으로 9개, 8개, 5개였다. 이 10개 질환의 소견 점수 합에서 S-dict 전용 간선의 몫은 중앙값으로 18.7%, 5.8%, 4.9%였다. 경쟁 질환에서 S-dict 전용 간선을 빼는 반사실은 하지 않았다. 반대로 DDXPlus에서 S-dict의 승리는 S-dict 전용 간선에 기댔다. DDXPlus의 S-dict 승리 208명(정답 15질환)에서 정답의 S-dict 전용 간선을 빼면 94명(45.2%)만 Top-10에 남았다. SymCat에서는 S-dict 승리 291명 중 211명(72.5%)이 남았다.

**HPO 주석 일치.** 4.7절과 같은 방법으로 두 그래프의 간선을 HPO 주석(phenotype.hpoa, 2026-02-16)과 대조했다(표 14d). HPO 주석은 희귀질환 주석이므로 이 결과는 흔한 질환 세트에 대해 말하지 않는다. 정확 일치율은 S-mix 12.6%(비교 가능 간선 856,023개, 무작위 순열 2.5%), S-dict 7.8%(1,572,570개, 무작위 1.9%)였다. 지지 문서 수의 네 구간 모두에서 S-mix의 일치율이 높았다(1편 7.4% 대 4.4%, 2편 15.2% 대 8.5%, 3–4편 24.9% 대 14.6%, 5편 이상 41.7% 대 24.6%). 두 그래프에 함께 있는 간선의 일치율은 13.0%였다. S-dict 전용 간선은 2.31%(무작위 1.31%), S-mix 전용 간선은 7.25%(무작위 2.02%)였다. 반대 방향으로, 이 질환들의 큐레이터 주석 쌍 가운데 그래프에 있는 쌍의 비율은 S-dict 32.8%, S-mix 29.4%였다.

**해석.** 다음은 측정이 아니라 해석이다. 이 데이터에서 LLM 구절이 한 일은 사전이 이름 붙이지 못하는 소견을 찾는 일보다, 어떤 텍스트가 그 질환의 소견인지 고르는 선택에 가깝다. 이 해석은 S-mix 간선의 86.9%가 S-dict에도 있고, S-dict 전용 간선의 HPO 일치율(2.31%)이 공유 간선(13.0%)보다 낮다는 측정에 기댄다. 따라서 'LLM은 사전에 없는 새 형태의 소견을 찾는 데만 쓴다'는 설계는 이 데이터에서 희귀질환 우세의 출처를 겨냥하지 않는다. 표본에서 새 표현은 30개 중 3개였고, 희귀질환 세트의 S-mix 승리는 S-mix 전용 간선을 빼도 대부분 유지되었다. 흔한 질환 합성 세트 두 개가 목표라면 사전 그래프가 약 300분의 1의 구축 시간으로 더 높았다. LLM 선택의 이점은 희귀질환 개발 세트에서만 관찰되었고 확증되지 않았다. 주제 문서로 사전 일치를 제한하는 방법이나 작은 문장 분류기 같은 더 싼 선택기는 시험하지 않았다(5.2절).

Table 14. S-mix versus the UMLS dictionary graph S-dict, analysed after the design-E confirmation run. The rank analysis (c) uses the stored development runs and is a development result.

(a) One-time build cost and ranking time. Literature collection, identical for all systems, is excluded. Hardware: one RTX 4090 24 GB GPU and an Intel i7-13700K CPU; other jobs shared the machine. Totals add the stage times.

| Item | S-mix | S-dict |
|---|---|---|
| LLM extraction, GPU generation time | 127.4 h (v2 literature 66.9 h; v3 new literature 60.6 h)^a^ | – |
| LLM tokens, output / input | 3.5 × 10⁸ / ≈ 2.4 × 10⁹ (input estimated) | – |
| CPU extraction steps | Phrase normalization ≤ 6.7 h^b^; in-span dictionary relink 3.5 min | UMLS dictionary matching ≈ 6.1 min^c^ |
| Graph build without Mentions | 9.2 min | 20.6 min |
| Extraction to graph, total | ≈ 136 h | ≈ 0.45 h |
| Edges | 5,221,285 | 14,506,788 |
| Ranking time per patient, DDXPlus-E / SymCat-E (confirmation run) | 87.0 / 35.8 ms | 155.0 / 60.4 ms |

^a^ v2 ran in eager mode with batch 64 and v3 with CUDA graphs and batch 512; at the v3 setting the generation time is estimated at ≈ 86.5 h. The total uses GPU process time including model loading (129.3 h). ^b^ Excludes ≈ 1.0 h estimated for phrases inherited from the Qwen3 phrase table. ^c^ v2 2.2 min measured; v3 3.9 min estimated from a 1/20 benchmark. In a 1/20 sample, 4.6% of dictionary items received their concept only through the v2 phrase table, which was built from LLM phrases.

(b) Edge overlap at the (disease, concept) level.

| Edge group | Edges | Share | Supported by 1 document (%) | ≥ 1 topic document (%) |
|---|---|---|---|---|
| Shared | 4,535,292 | 86.9% of S-mix; 31.3% of S-dict | 71.0 (S-mix counts); 47.1 (S-dict counts) | 67.0; 70.1 |
| S-mix only | 685,993 | 13.1% of S-mix | 88.7 | 68.2 |
| – all Mentions from non-relinked LLM items | 667,686 | 97.3% of S-mix only | 88.5 | 68.2 |
| – concept not nameable by the dictionary^d^ | 332,817 | 48.5% of S-mix only | 87.4 | 71.3 |
| S-dict only | 9,971,496 | 68.7% of S-dict | 72.2 | 47.0 |

^d^ Neither a UMLS dictionary string nor the HPO name of such a string names the concept; strings without a CUI that reach a concept through the phrase table are not counted. Of these edges, 55,503 (16.7%) have a concept in one of the dictionary's finding semantic types. The most frequent types are Pharmacologic Substance (T121, 97,787 edges), Organic Chemical (T109, 78,127) and Population Group (T098, 46,455); a concept may have several types. Reading of 30 random S-mix-only edges (seed 0; the author's reading, not a clinical judgment): likely wrong 18; a dictionary string of the concept is in the text but S-dict did not keep the edge 9; wording that no dictionary string matches 3.

(c) Development rank wins: patients whose gold is in the winner's lineage Top-10 and not in the other graph's. Gold via winner-only edge = patients whose gold disease reaches a patient concept through an edge only the winner has (wins / all evaluated patients of the set). Kept = wins still in Top-10 after a partial counterfactual that removes those edges from the gold lineage diseases and leaves all other diseases and the length normalization unchanged. Other graph's top 10 = the ten diseases the other graph ranked above the gold: median number of them with an edge only that graph has, and median share of such edges in their summed 1-hop finding score.

| Set | Winner | Wins (gold diseases) | Gold via winner-only edge, wins / all (%) | Kept | Other graph's top 10: diseases with own-only edge (score share) |
|---|---|---|---|---|---|
| RAMEDIS | S-mix | 68 (18) | 16.2 / 14.7 | 63 (92.6%) | 9 (18.7%) |
| LIRICAL | S-mix | 26 (23) | 61.5 / 30.8 | 20 (76.9%) | 8 (5.8%) |
| Phenopacket | S-mix | 505 (111) | 17.8 / 13.1 | 448 (88.7%) | 5 (4.9%) |
| DDXPlus | S-dict | 208 (15) | 67.3 / 65.7 | 94 (45.2%) | 4 (1.7%) |
| SymCat | S-dict | 291 (123) | 55.0 / 42.9 | 211 (72.5%) | 1 (0.6%) |

(d) Exact-concept agreement with HPO curated annotations (phenotype.hpoa 2026-02-16), computed as in Table 17; a lower bound on edge correctness, for rare-disease annotations only. Random = the same edges with diseases permuted (mean of 20). Curated pairs recovered = share of the curated (disease, phenotype) pairs of these diseases that the graph contains. The graphs differ in literature, extraction, builder rules and edge count; the causes of their differences were not separated.

| Graph or edge group | HPO-comparable edges | All (random) | 1 doc | 2 docs | 3–4 docs | ≥ 5 docs | Curated pairs recovered |
|---|---|---|---|---|---|---|---|
| S-mix | 856,023 | 12.6 (2.5) | 7.4 | 15.2 | 24.9 | 41.7 | 29.4 |
| S-dict | 1,572,570 | 7.8 (1.9) | 4.4 | 8.5 | 14.6 | 24.6 | 32.8 |
| Edges in both graphs | 801,933 | 13.0 (2.5) | – | – | – | – | – |
| S-mix-only edges | 54,090 | 7.25 (2.02) | – | – | – | – | – |
| S-dict-only edges | 770,637 | 2.31 (1.31) | – | – | – | – | – |
| Old Qwen3 v2.1 graph (Table 17) | 194,318 | 16.4 (2.6) | 13.3 | 31.7 | 46.3 | 63.2 | 11.3 |
| S-D graph (gemma v2.1) | 168,587 | 18.2 (2.8) | 15.0 | 34.0 | 48.7 | 62.8 | 10.9 |

### 4.5 시간 분할

이 절도 설계 D 확증 이후의 탐색이다. 대상 증례에는 D 확증에 쓰인 Phenopacket-store 신규 증례가 들어 있으므로, 이 결과는 확증이 아니라 개발 근거다. 2022년 말까지 출판된 문헌만으로 다시 집계한 그래프(2022 절단)를 HPOA 2022-12-15판(같은 1-홉 랭커)과 비교했다. 대상은 Phenopacket-store 증례 10,603건 중 원 논문(9편)이 v2 코퍼스에 들어 있는 32건을 뺀 10,571건이다. 증례는 출판일(2023-01-01 이후 = 미래)과 정답 질환의 HPOA 첫 등재 시점(2022-12-31 이후 = 신규)으로 네 묶음으로 나뉜다(3.9절). 표 15는 모든 그래프를 ot 0으로 평가한 값이다. 이 값은 S-mix(.5)·S-llm(1.0)·S-dict(.5)·S-union(.5)이 개발 세트에서 고른 ot의 값이 아니므로, 이 절에서는 각 시스템의 그래프를 ot 0으로 쓴 결과로 읽어야 한다. S-mix 그래프는 시간 분할에서 ot 0과 1.0으로만 평가했다.

2022년 말 HPOA에 아직 등재되지 않은 질환의 미래 증례(2,381건, 263질환)에서 2022 절단 문헌 그래프 11개의 Top-10은 모두 2.2% 이하였다. 11개 가운데 표 15에 없는 v2 그래프 4개(v2dict, v2umls, v2union, v2unionu_ns)는 0.5~1.3%였다. HPOA 2022-12의 Top-10은 0.1%였다. v3 그래프의 Top-10은 1.7~2.2%로, HPOA보다 1.6~2.1%p, 2022 절단 동결 그래프(S-D)보다 0.8~1.3%p 높았다.

이미 등재된 질환에서는 v3 문헌을 더한 그래프가 HPOA와의 Top-10 격차를 줄였다. 미래·기주석 증례(1,106건, 67질환)에서 같은 추출끼리 비교하면 HPOA 대비 격차가 UMLS 사전 −15.6%p에서 −8.1%p로, LLM −16.1%p에서 −7.3%p [−10.4, −4.1]로 줄었다. 과거·기주석 증례(5,993건, 405질환)에서는 UMLS 사전 −28.2%p에서 −17.3%p로, LLM −33.2%p에서 −17.9%p로 줄었다. LLM 쪽 v2 기준(base_split_tg)은 자기 간선을 두고 v2 구절 표를 쓴 그래프다. S-mix 그래프의 HPOA 대비 격차는 미래·기주석 −7.1%p [−10.1, −3.9], 과거·기주석 −15.5%p [−16.9, −14.0]로, ot 0에서 비교한 문헌 그래프 가운데 점추정이 가장 작았다. 다른 v3 그래프와의 차이는 0.2~2.4%p였고, v3 그래프끼리는 대응 검정을 하지 않았다.

Top-1에서는 미래·기주석 증례에서 S-mix 그래프가 35.5%로 HPOA 2022-12(28.0%)보다 7.5%p [4.2, 10.9] 높았다. S-llm(+7.3%p), S-union(+7.2%p), S-dict(+6.3%p) 그래프도 같은 방향이었다. 과거·기주석 증례에서 S-mix 그래프의 Top-1은 18.2%, HPOA는 27.0%였다(차이 −8.9%p [−10.1, −7.7]; 반올림 전 값으로 계산). 이 Top-1 비교는 사전에 정한 분석이 아니다. 미래·기주석 증례는 67개 질환에 몰려 있으며, CI는 환자 단위로만 계산했다.

ot를 1.0으로 바꾸면 과거·신규 증례(1,091건, 175질환)의 Top-10이 S-llm 그래프 11.0%, S-mix 그래프 10.5%로 ot 0(4.6%, 4.9%)보다 높았다. 나머지 묶음에서 ot 1.0과 ot 0의 차이는 ±2%p 이내였다. 시간 절단 없이 모든 문헌을 쓴 2026 스냅샷(출처 누설 가능, 10,571건, ot 0)의 Top-10은 S-mix 30.5%, S-llm 29.8%, S-union 28.0%, S-dict 27.9%, S-D 16.5%였다.

Table 15. Time split: graphs built from literature published up to 2022 vs HPOA 2022-12-15, same 1-hop ranker, ot 0, Phenopacket-store cases (n = 10,571 after removing 32 cases whose source paper is in the v2 corpus). "New" = disease first annotated in HPOA after 2022-12-31; "future" = case published from 2023-01-01. Differences vs HPOA in percentage points with patient-paired bootstrap 95% CI. Snapshot = all literature without time cut (leakage possible). Rows labelled S-dict, S-llm, S-union and S-mix use those graphs at ot 0, not each system's selected ot (S-mix .5, S-llm 1.0, S-dict .5, S-union .5); S-mix was not evaluated at ot .5 in the time split. Four further 2022-cut v2 graphs not shown (v2dict, v2umls, v2union, v2unionu_ns) give future · new Top-10 of 0.5–1.3%. The cases include the Phenopacket-store cases used in the design-D confirmation, so these are development results.

| Knowledge source (2022 cut) | Future · new, 2,381 (263) Top-10 | Future · curated, 1,106 (67) Top-10 (Δ vs HPOA) | Past · new, 1,091 (175) Top-10 | Past · curated, 5,993 (405) Top-10 (Δ vs HPOA) | Future · curated Top-1 | 2026 snapshot Top-10 |
|---|---|---|---|---|---|---|
| HPOA 2022-12 | 0.1 | 49.4 | 0.6 | 52.9 | 28.0 | – |
| S-D (base_nm) | 0.9 | 32.0 (−17.4 [−20.3, −14.2]) | 2.2 | 19.7 (−33.2 [−34.6, −31.7]) | 23.8 | 16.5 |
| base_split_tg (v2, LLM) | 1.1 | 33.3 (−16.1 [−19.2, −13.0]) | 2.7 | 19.7 (−33.2 [−34.6, −31.8]) | 28.3 | 17.6 |
| v2umls_ns (v2, UMLS dict.) | 1.4 | 33.7 (−15.6 [−18.6, −12.5]) | 2.6 | 24.7 (−28.2 [−29.7, −26.6]) | 28.3 | 20.1 |
| S-dict graph (v3umls_ns) | 1.9 | 41.2 (−8.1 [−11.1, −5.0]) | 4.4 | 35.6 (−17.3 [−18.8, −15.8]) | 34.4 | 27.9 |
| S-llm graph (v3llm_ns) | 1.7 | 42.0 (−7.3 [−10.4, −4.1]) | 4.6 | 35.0 (−17.9 [−19.3, −16.4]) | 35.4 | 29.8 |
| S-union graph (v3unionu_ns) | 2.2 | 41.6 (−7.8 [−10.8, −4.7]) | 5.8 | 36.0 (−16.8 [−18.3, −15.4]) | 35.3 | 28.0 |
| S-mix graph (v3llmd_ns) | 1.7 | 42.2 (−7.1 [−10.1, −3.9]) | 4.9 | 37.4 (−15.5 [−16.9, −14.0]) | 35.5 | 30.5 |

#### 4.5.1 파일럿의 비재현과 설계 D 동결 설정의 시간 분할

수집 확장 파일럿은 사전테스트 질환 300개 중 50개(Phenopacket 정답 질환 22개 포함)에만 추가 문헌 3,367편과 표 763문서를 넣은 그래프였다. 정답이 파일럿 질환인 154증례(22질환)에서 같은 1-홉 랭커로 평가한 HPO 주석의 Top-10은 2024-12판 10.4%, 2026-09판 84.4%였다. 2026-09판은 Phenopacket-store 증례를 주석에 흡수한 뒤의 판이므로 이 값에는 출처 누설이 들어 있다. 이 파일럿에서 파일럿 질환의 미래 증례 90건(13질환)에 대한 2022 절단 확장 문헌 그래프의 Top-10은 15.6%였다. 같은 증례에서 HPOA 2022-12는 0.0%, 현행 문헌은 4.4%였다(현행 대비 +11.1%p [5.6, 17.8]). 모든 질환을 같은 상한으로 다시 모은 v3에서는 같은 90건의 Top-10이 4.4~6.7%였다(S-llm·S-dict 6.7%, S-union 5.6%, S-mix 4.4%; 모두 ot 0). S-mix 그래프의 Top-10은 2022 절단 현행 문헌 그래프(4.4%)와 같았다. v3 그래프는 모두 HPOA(0.0%)보다 높았지만 파일럿 이득의 크기는 재현되지 않았다.

파일럿만 맞힌 증례는 S-mix 대비 10건, S-union 대비 9건, S-llm·S-dict 대비 8건이었다. 어느 비교에서나 그중 6건은 지적발달장애 두 질환(OMIM:620988, OMIM:620401)의 증례였다. v3 그래프만 맞힌 증례는 없었다. 두 질환의 근거는 제목에 "intellectual developmental disorder"가 들어간 이름 검색 문헌이었다. 50질환 파일럿에서는 이 문헌을 쓰는 질환이 5~9개라 일반 문헌 규칙의 기준(질환 파일에 20회 이상 나열)에 미치지 않았다. 전 질환 v3에서는 약 200개 질환이 같은 문헌을 써서 이 문헌이 제외되었다. 또 파일럿의 시간 분할 빌드는 유전자 경로로 찾은 문서를 시간으로 자르지 않아, 2026년 유전자–질환 연결로 찾은 문서가 2022 절단 그래프에 남아 있었다. v3의 2022 절단 빌드는 이런 문서를 뺐다. 이 두 차이가 비재현의 원인으로 보이지만 각각의 몫은 따로 측정하지 않았다.

설계 D의 동결 설정과 v2 문헌 그래프로 수행한 시간 분할 분석(10,603증례)의 결과는 다음과 같다. 미래 증례의 Top-10은 HPOA 16.5%, 문헌 그래프 10.8%로, "미래 증례에서 문헌이 큐레이션을 앞선다"는 가설은 기각되었다. 과거 증례–HPOA 스냅샷 쌍에서 HPOA가 그 증례 논문을 이미 인용한 경우 Top-10은 81.7%(22,426쌍), 인용하지 않은 경우 41.1%(30,036쌍)였다. HPOA 1-홉의 Top-10은 관행 평가(Phenopacket-store 0.1.27, HPOA 2026-09)에서 77.4%였고, 시간 분할의 미래 증례에서는 16.5%였다. 스냅샷 시점에 HPOA에 아직 등재되지 않은 질환의 미래 증례에서 Top-10은 HPOA 0.1%, 문헌 그래프 0.7%였다.

### 4.6 설계 E 확증

설계 E의 확증 세트는 DDXPlus-E와 SymCat-E 두 개다. DDXPlus-E는 DDXPlus test 134,529행에서 개발 표본(시드 0)과 D 확증 표본(시드 101)을 뺀 128,529행 가운데 시드 202로 뽑은 3,000명이다. 매니페스트에는 각 행의 번호와 sha256이 기록되어 있고, 개발 표본·D 확증 표본과 겹치는 행은 0개다. SymCat-E는 질환당 10명을 시드 202로 생성한 환자다. Phenopacket-store main에는 2026-10-19 기준 D 확증 스냅샷(85ffd78) 이후 커밋이 없으므로 희귀질환 확증 세트는 없다. 두 세트는 개발 세트와 같은 질환 생성기·확률표(DDXPlus 49질환, SymCat 801질환)에서 새로 뽑은 환자다. DDXPlus-E의 질환 가운데 개발 표본에 없던 것은 세기관지염 1명뿐이다.

매니페스트와 로더 거부 장치는 커밋 3b10c36(2026-10-19)으로 봉인했다. 판정 기준과 E-T1·E-T2·E-T4의 개발값 예상은 확증 실행 전에 설계 E 커밋(5ce9c3d)에 기록했다. S-bm25는 개발 세트의 12개 조합 가운데 질의 전체·깊이 1,000·역순위 결합 집계를 골랐다(개발 9개 세트 매크로 Top-10/Top-1 27.35/11.04%). S-comb는 w = 0.25를 골랐다(개발 매크로 46.72/19.46%; w = 0.5는 43.77/18.66%, 0.75는 35.61/17.11%). 첫 S-comb 선택은 비교용 인스턴스에 S-mix가 아닌 그래프가 올라간 상태에서 실행되었으므로 폐기했다. 위 값은 적재 간선 수를 확인한 S-mix로 다시 고른 것이다. E-T3·E-T5의 예상은 첫 S-comb 선택 뒤에 기록했다(커밋 61e3118). 이때의 E-T5 예상은 폐기한 첫 선택의 값이었다. 그래서 S-comb를 다시 고른 뒤 E-T5의 예상을 그 값으로 고쳤다(커밋 0cb1c59). 두 커밋 모두 동결 전이다. E-T3은 충족으로, E-T5는 S-comb − S-mix 비교의 미달로 예상했다(표 16b). 설계 D 확증에서는 결합 목표 T4가 충족되었다(4.2절). 시스템은 커밋 49901bc(2026-10-19)로 동결했다. 시스템별 지문, S-bm25 색인 sha256, 설정은 이 커밋의 frozen_config_E.json에 기록했다. 확증 세트 E는 시스템 동결 뒤 한 번만 쓰도록 사전에 정했다. 확증은 동결 커밋에서 2026-10-19 23:13에 시작해 2026-10-20 01:34에 끝났다. 시도는 1회였고 중단은 없었다.

통계 코드는 실행 전에 회귀 확인을 거쳤다. 매크로 비교 함수(macro_compare)는 D 확증의 T4 매크로 수치를 같은 값으로 재현했다: 결합 − 그래프 Top-10 +2.67%p [1.81, 3.58], 결합 − BM25 Top-10 +5.10%p [3.59, 6.67], 결합 − 그래프 Top-1 +0.96%p [0.36, 1.54].

확증 직전에 그래프 시스템 5개를 비교용 인스턴스에 다시 빌드했다. 다섯 빌드 모두 빌드 요약의 지문과 적재된 HAS_FINDING 간선 수가 동결 기록(frozen_config_E.json)과 같았다. 빌더는 Disease 노드 수가 후보 질환 수(43,575개)와 다르면 멈추는데, 다섯 빌드 모두 멈추지 않았다. 평가 로그에 기록된 소견 있는 질환 수도 표 11a의 빌드 요약과 같았다(S-D 27,309개, S-llm 33,123개, S-dict 34,356개, S-union 34,359개, S-mix 33,537개). S-bm25 색인 파일 8개의 sha256도 모두 동결 기록과 같았다.

표 16a는 일곱 시스템의 확증 정확도다. DDXPlus-E는 3,000명(49질환)을 모두 평가했다. SymCat-E는 개발 SymCat과 같은 규칙으로 소견이 뽑히지 않은 표본과 정답 사상이 안 되는 질환을 빼고 7,463명(760질환)을 평가했다. S-mix의 Top-10/Top-1은 DDXPlus-E 41.07/12.43%, SymCat-E 12.92/2.45%였다. 점추정으로 두 세트의 Top-10과 Top-1이 모두 가장 높은 시스템은 S-union이었다(DDXPlus-E 45.07/17.93%, SymCat-E 14.24/3.26%).

주 목표 E-T1은 충족되었다. S-mix의 Top-10은 S-D보다 DDXPlus-E에서 +6.5%p [+4.6, +8.5](Holm 보정 p = 2.1 × 10⁻¹¹), SymCat-E에서 +2.4%p [+1.7, +3.1](Holm 보정 p = 3.4 × 10⁻¹¹) 높았다. 두 세트 모두 CI 하한이 0보다 크고 Holm 보정 p가 .05보다 작았다. 차이의 방향은 개발 결과(DDXPlus +7.6%p, SymCat +2.7%p)와 같았고, 크기는 개발보다 작았다.

E-T2의 S-llm 비교(주 목표)도 충족되었다. 두 세트 매크로 Top-10에서 S-mix는 S-llm보다 +1.64%p [+1.17, +2.18] 높았다. CI 하한은 비열등 한계 −1%p뿐 아니라 0보다도 높았다. 세트별로도 S-mix가 DDXPlus-E에서 +1.6%p [+0.8, +2.4], SymCat-E에서 +1.7%p [+1.2, +2.2] 높았다(둘 다 Holm 유의).

E-T2의 S-dict 비교와 S-union 비교는 미달이었다. 매크로 Top-10 차이(S-mix − 비교 시스템)는 S-dict 대비 −1.80%p [−2.44, −1.12], S-union 대비 −2.66%p [−3.27, −2.07]로, 두 CI의 하한이 모두 −1%p보다 낮았다. 두 CI는 상한도 0보다 낮았다. 곧 두 확증 세트에서 S-mix의 Top-10은 S-dict와 S-union보다 낮았다. 세트별 차이는 S-dict 대비 DDXPlus-E −2.4%p [−3.7, −1.2], SymCat-E −1.2%p [−1.7, −0.6], S-union 대비 DDXPlus-E −4.0%p [−5.1, −2.9], SymCat-E −1.3%p [−1.8, −0.8]로 네 경우 모두 Holm 유의했다. 이 두 미달은 실행 전에 예상한 결과다(개발 DDXPlus −1.6%p·−3.0%p, SymCat −1.1%p·−1.3%p).

E-T3은 충족되었다. S-mix의 Top-10은 S-bm25보다 DDXPlus-E에서 +19.3%p [+17.6, +21.0], SymCat-E에서 +3.0%p [+2.2, +3.8] 높았다(두 세트 모두 Holm 보정 p < 10⁻¹³). 두 세트 모두 CI 하한이 −1%p보다 높고, 두 세트 모두 Holm 유의한 우세였다.

E-T4는 미달이었다. S-mix의 DDXPlus-E Top-10은 41.07%로 기준 40% 이상이었다. Top-1은 12.43%로 기준 18%보다 5.57%p 낮았다. Top-1의 CI [11.3, 13.6]은 전부 기준보다 낮았다. Top-10의 CI 하한(39.3%)도 기준보다 낮았다. 이 결과는 실행 전 예상(Top-10 경계, 개발 40.6%; Top-1 미달, 개발 11.8%)과 같았다.

E-T5는 미달이었다. S-comb − S-mix의 매크로 Top-10 차이는 +0.11%p [−0.39, +0.59](p = .71)로, CI 하한이 0보다 낮았다. 세트별로는 DDXPlus-E −0.8%p [−1.6, +0.1](Holm 비유의), SymCat-E +1.0%p [+0.5, +1.5](Holm 유의)로 방향이 갈렸다. S-comb − S-bm25의 매크로 차이는 +11.25%p [+10.36, +12.14]로 기준을 넘었다. E-T5는 두 비교가 모두 기준을 넘어야 충족이므로 미달이다. 이 결과는 실행 전 예상(S-comb − S-mix 미달, S-comb − S-bm25 충족)과 같았다. 설계 D 확증에서는 결합이 두 단독 시스템보다 높았다(T4, 4.2절). 설계 E 확증에서는 S-mix에 S-bm25 순위를 결합한 Top-10 이득이 확인되지 않았다.

따라서 E0은 달성되었다. 주 목표인 E-T1과 E-T2의 S-llm 비교가 모두 충족되었다. 일곱 판정은 모두 실행 전에 커밋한 예상과 같았다(표 16b). 이 확증이 보여 주는 것은 개발 세트와 같은 질환 집합에서 새로 생성한 합성 환자에서의 재현이다. E 세트에서 입력이 개발 환자와 똑같은 환자의 비율은 재지 않았다(3.6절). 새 질환으로의 일반화는 이 확증의 범위 밖이다. 희귀질환 확증 세트는 없었으므로, 4.4–4.5절의 희귀질환·시간 분할 결과는 개발 결과로 남는다.

판정에 쓰지 않은 보조 지표는 다음과 같다. Top-1에서는 S-D 대비 개선이 확증되지 않았다. S-mix − S-D의 Top-1 차이는 DDXPlus-E +0.7%p [−0.6, +2.0](Holm 보정 p = .30), SymCat-E −0.8%p [−1.2, −0.4](Holm 보정 p = .001)로, SymCat-E에서는 S-mix가 낮았다. 두 방향은 개발 결과(DDXPlus +0.6%p, SymCat −0.7%p; 표 12b)와 같았다. S-mix − S-llm의 Top-1 차이는 DDXPlus-E +0.53%p [−0.27, +1.33], SymCat-E +0.24%p [−0.03, +0.52]로 둘 다 Holm 비유의였다. DDXPlus-E의 Top-1은 S-dict 17.63%, S-union 17.93%로 S-mix보다 5.2%p, 5.5%p 높았다. SymCat-E에서는 그 차이가 0.7%p, 0.8%p였다(네 차이 모두 Holm 유의). S-dict와 S-union도 E-T4의 Top-1 기준(18%)에는 미치지 못했다. Top-3는 표 16a에만 적고 대응 검정을 하지 않았다. S-mix의 Top-3는 S-D보다 DDXPlus-E에서 4.83%p 높고(24.30% 대 19.47%), SymCat-E에서 0.15%p 낮았다(5.61% 대 5.76%).

정답 질환 군집 부트스트랩 CI(2,000회)는 같은 정답 질환의 환자를 통째로 다시 뽑는다. DDXPlus-E는 정답 질환이 49개뿐이어서 이 CI가 넓다. E-T1의 DDXPlus-E 차이(+6.5%p)의 군집 CI는 [−6.1, +19.3]으로 0을 포함했다. SymCat-E(760질환)의 군집 CI는 [+1.0, +3.9]로 0보다 컸다. S-mix − S-llm의 군집 CI는 DDXPlus-E [−1.3, +4.3], SymCat-E [+0.8, +2.6]이었다. S-mix − S-dict와 S-mix − S-union의 DDXPlus-E 군집 CI도 각각 [−8.8, +3.5], [−10.3, +1.3]으로 0을 포함했다. 반면 SymCat-E의 S-mix − S-dict 차이(−1.2%p)는 군집 CI [−2.1, −0.2]가 0을 포함하지 않았다. DDXPlus-E Top-1의 S-mix − S-dict 차이(−5.2%p)도 군집 CI [−9.8, −1.4]가 0을 포함하지 않았다. 곧 흔한 질환 합성 세트 두 개(DDXPlus-E 49질환, SymCat-E 760질환)에서 S-dict의 Top-10 우세는 SymCat-E에서는 질환 수준에서도, DDXPlus-E에서는 환자 수준에서만 확인되었다. S-mix − S-bm25의 군집 CI는 두 세트 모두 0보다 컸다(DDXPlus-E [+8.8, +30.7], SymCat-E [+1.4, +4.5]). 따라서 DDXPlus-E에서 S-D·S-llm 대비 우세는 같은 49개 질환의 새 환자에서 확인된 것이다. 다른 질환 집합에서도 같은 차이가 나는지는 이 확증으로 말할 수 없다.

개발 9개 세트에서의 같은 비교는 4.4.3절과 표 12에 있다. 열람된 D 확증 세트(DDXPlus·SymCat 시드 101)에서는 이 비교를 계산하지 않았다. S-mix의 Top-10/Top-1은 DDXPlus에서 개발 40.6/11.8%, 확증 41.1/12.4%였고, SymCat에서 개발 13.3/2.5%, 확증 12.9/2.5%였다. 그래프 시스템 5개의 Top-10에서 확증값 − 개발값(표 11b)은 DDXPlus +0.4~+1.5%p, SymCat −0.4~−0.2%p였다. 같은 질환의 새 환자 표본에서 개발값의 낙관은 이 범위 안에 있었다. 질환 수준의 낙관은 이 확증으로 검사되지 않는다.

표 16b는 실행 전에 기록한 개발값을 확증값과 나란히 둔다. E-T1과 E-T3의 개발값은 확증 판정과 같은 세트별 Top-10 차이다. E-T5의 개발값도 확증 판정과 같은 두 세트 매크로 Top-10 차이다. E-T4의 개발값은 S-mix 단독의 개발 DDXPlus 정확도다. E-T2만 정의가 다르다. 개발값은 세트별 차이이고, 확증 판정은 두 세트 매크로 차이다.

Table 16. Design E confirmation (lineage, open set of 43,575 diseases; one run at freeze commit 49901bc, 2026-10-19 to 10-20). New synthetic patients of the development diseases only; no rare-disease confirmation set.

(a) Accuracy per system: lineage Top-1 / Top-3 / Top-10 (%) with patient-bootstrap 95% CI (1,000 resamples). DDXPlus-E: 3,000 sampled, 3,000 evaluated (49 diseases). SymCat-E: 7,463 evaluated (760 diseases).

| System | DDXPlus-E Top-1 | DDXPlus-E Top-3 | DDXPlus-E Top-10 | SymCat-E Top-1 | SymCat-E Top-3 | SymCat-E Top-10 |
|---|---|---|---|---|---|---|
| **S-mix** | 12.43 [11.3, 13.6] | 24.30 [22.8, 26.0] | 41.07 [39.3, 43.0] | 2.45 [2.1, 2.8] | 5.61 [5.1, 6.2] | 12.92 [12.2, 13.6] |
| S-D | 11.73 [10.6, 12.9] | 19.47 [18.1, 20.9] | 34.53 [32.8, 36.3] | 3.23 [2.8, 3.6] | 5.76 [5.2, 6.3] | 10.49 [9.8, 11.2] |
| S-llm | 11.90 [10.8, 13.2] | 23.23 [21.7, 24.9] | 39.47 [37.7, 41.3] | 2.21 [1.9, 2.6] | 4.64 [4.2, 5.1] | 11.24 [10.5, 12.0] |
| S-dict | 17.63 [16.2, 19.0] | 29.33 [27.7, 31.0] | 43.50 [41.7, 45.4] | 3.12 [2.7, 3.5] | 6.31 [5.7, 6.8] | 14.08 [13.3, 14.9] |
| S-union | 17.93 [16.6, 19.3] | 28.50 [27.0, 30.2] | 45.07 [43.3, 47.0] | 3.26 [2.9, 3.6] | 6.15 [5.6, 6.7] | 14.24 [13.4, 15.0] |
| S-bm25 | 8.57 [7.6, 9.5] | 13.57 [12.4, 14.8] | 21.77 [20.3, 23.2] | 2.09 [1.8, 2.4] | 4.68 [4.2, 5.2] | 9.93 [9.2, 10.6] |
| S-comb | 12.10 [11.0, 13.3] | 24.80 [23.3, 26.4] | 40.30 [38.5, 42.1] | 2.64 [2.3, 3.0] | 6.24 [5.7, 6.8] | 13.90 [13.1, 14.6] |

(b) Judgments. Every "Dev expectation" was committed before the run: E-T1, E-T2 and E-T4 in the design E commit (5ce9c3d); E-T3 and E-T5 after S-bm25 and the first S-comb were selected on the development sets (commit 61e3118). That first S-comb selection was discarded, and the E-T5 expectation was corrected in commit 0cb1c59 after S-comb was reselected with the S-mix load verified. Both commits precede the freeze (commit 49901bc). Dev expectation values are development per-set Top-10 differences (DDXPlus; SymCat), with patient-paired bootstrap 95% CI where given, unless stated. Confirmation results are Top-10 differences in percentage points (DDXPlus-E; SymCat-E, or the equal-weight macro over the two sets) with patient-paired bootstrap 95% CI (2,000 resamples; set-stratified for the macro) and Holm-adjusted exact McNemar p (family = two sets). All seven judgments matched their committed expectations.

| Target | Comparison | Criterion | Dev expectation | Confirmation result | Met |
|---|---|---|---|---|---|
| E-T1 (primary) | S-mix − S-D, Top-10, each set | patient CI lower bound > 0 and exact McNemar Holm p < .05 (family = two sets) | +7.6; +2.7 (expected met) | +6.5 [+4.6, +8.5], Holm p = 2.1 × 10⁻¹¹; +2.4 [+1.7, +3.1], Holm p = 3.4 × 10⁻¹¹ | Met |
| E-T2 (primary) | S-mix − S-llm, macro Top-10 | CI lower bound > −1 pp | +1.9; +1.9 (expected met) | Macro +1.64 [+1.17, +2.18] (DDXPlus-E +1.6; SymCat-E +1.7) | Met |
| E-T2 | S-mix − S-dict, macro Top-10 | CI lower bound > −1 pp | −1.6; −1.1 (expected not met) | Macro −1.80 [−2.44, −1.12] (DDXPlus-E −2.4; SymCat-E −1.2) | Not met |
| E-T2 | S-mix − S-union, macro Top-10 | CI lower bound > −1 pp | −3.0; −1.3 (expected not met) | Macro −2.66 [−3.27, −2.07] (DDXPlus-E −4.0; SymCat-E −1.3) | Not met |
| E-T3 | S-mix − S-bm25, Top-10, each set | CI lower bound > −1 pp in both and ≥ 1 Holm-significant gain | +20.9 [19.2, 22.7]; +3.2 [2.5, 4.0], both Holm-significant (expected met) | +19.3 [+17.6, +21.0]; +3.0 [+2.2, +3.8], both Holm-significant (Holm p < 10⁻¹³) | Met |
| E-T4 | S-mix alone on DDXPlus-E | Top-10 ≥ 40%, Top-1 ≥ 18% | Top-10 40.6 (borderline); Top-1 11.8 (expected not met) | Top-10 41.07 [39.3, 43.0]; Top-1 12.43 [11.3, 13.6] | Not met (Top-1) |
| E-T5 | S-comb − S-mix and S-comb − S-bm25, macro Top-10 | both CI lower bounds > 0 | S-comb − S-mix, macro over DDXPlus and SymCat: −0.02 [−0.54, 0.49] (DDXPlus −0.8; SymCat +0.8; expected not met). S-comb − S-bm25, same macro: +12.04 [11.15, 12.94] (expected met) | S-comb − S-mix macro +0.11 [−0.39, +0.59] (DDXPlus-E −0.8; SymCat-E +1.0). S-comb − S-bm25 macro +11.25 [+10.36, +12.14] | Not met (S-comb − S-mix) |
| E0 | E-T1 and E-T2 (S-llm) | both met | – | E-T1 met; E-T2 (S-llm) met | Achieved |

### 4.7 출처 추적과 근거 품질

이 절의 간선 감사는 구 시스템에서만 했다. 큐레이션 대조는 구 시스템과 S-D 그래프에서 측정했고, 설계 E 확증 실행 뒤에 S-mix와 S-dict 그래프에서도 측정했다(4.4.7절). 구 시스템은 Qwen3-4B-Instruct-2507로 추출한 v2 문헌 그래프다. 감사 표본은 v2.1 그래프에서 뽑았고, 주제 표시를 더한 v2.1c 그래프는 간선 1,184,344개, 근거 언급 1,648,509개, 문서 88,266건이다. S-mix 그래프의 간선 감사는 하지 않았다.

구 Qwen3 그래프의 근거 언급 가운데 93.3%는 PMID로, 1.1%는 PMCID로, 5.6%는 StatPearls·GeneReviews 장 ID로 원 문헌이 식별되었다.

같은 그래프에서 시드 1로 뽑은 간선 200개(간선마다 근거 언급 1개)를 연구팀과 협력 병원의 의료진이 LLM 판정을 보지 않고 판정했다. 판정은 supports 59개(29.5%, Wilson 95% CI 23.6–36.2), not_supported 99개, wrong_concept 42개였다. 같은 200행에 대한 LLM(Claude) 판정의 supports는 75개(37.5%)였다. 두 판정의 일치율은 supports 여부(이진) 기준 89%(Cohen's κ 0.75), 세 범주 기준 70%(κ 0.48)였다. 세 범주 불일치 60행 중 38행은 의료진 wrong_concept 대 LLM not_supported였다. LLM이 supports로, 의료진이 not_supported·wrong_concept로 판정한 행은 19행(16행·3행)이었고, 반대 방향은 3행이었다(순차이 16행).

의료진 판정 기준으로 주제 문서가 1편 이상인 간선의 supports는 38.3%(51/133), 주제 밖 문서뿐인 간선은 11.9%(8/67)였다. 지지 문서가 2편 이상인 간선은 52.9%(9/17), 1편인 간선은 27.3%(50/183)였다. 의료진이 wrong_concept로 판정한 42행에는 약물·치료, 검사·방법·경과·결과 표현, 인구통계 개념, 질환 자체나 다른 질환명, 구절 절단이 포함되었다. 같은 200행의 LLM 판정에서 가장 많은 실패 유형은 근거 문서가 인접 아형이나 증후군 등 다른 질환을 다루는 경우(약 45행)였다. 이 200행은 이후 보류(held-out) 평가 세트로 지정했다.

큐레이션 대조에는 HPO 주석(phenotype.hpoa, 2026-02-16)을 썼다. 이 판에서 주석이 있는 질환은 12,296개이고, 그중 UMLS 개념으로 사상되는 표현형이 하나 이상인 질환은 12,251개다(3.7절 표 5의 HPO 주석 그래프). 이 질환들의 비교 가능한 간선 194,318개를 큐레이터 표현형과 대조했다(표 17). 정확 일치율은 16.4%(HPO 부모 포함 19.9%)로, 질환을 무작위로 순열한 대응(20회 평균) 2.6%보다 높았다. 일치율은 지지 문서 1편 13.3%(168,512간선), 2편 31.7%(18,262간선), 3–4편 46.3%(6,269간선), 5편 이상 63.2%(1,275간선)로 지지 문서 수에 따라 단조 증가했다. 주제 문서가 1편 이상인 간선은 19.3%, 주제 밖 문서뿐인 간선은 11.5%였다. 큐레이션은 불완전하므로 큐레이션에 없는 간선도 맞을 수 있다. 따라서 일치율은 정밀도의 하한이다. 반대 방향으로, 큐레이터가 주석한 쌍 가운데 그래프가 재현한 비율은 11.3%였다. IE 모델만 gemma-4-E4B로 바꾼 v2.1 그래프, 곧 S-D 그래프(간선 832,671개, 근거 언급 1,201,064개, 문서 86,276건)에서는 일치율이 18.2%(무작위 2.8%)였고, 지지 문서 1편 15.0%에서 5편 이상 62.8%로 올랐다.

Table 17. Evidence quality by provenance signal on the old Qwen3-4B-Instruct-2507 graph (audit sample drawn from v2.1; topic marks added in v2.1c): clinical audit (n = 200) and agreement with HPO curated annotations (exact concept match; a lower bound on edge correctness). The S-mix graph has not been audited; its HPO agreement, computed the same way, is in Table 14d.

| Edge group | Audit: supported (%) | HPO agreement (%) | HPO-comparable edges |
|---|---|---|---|
| All | 29.5 (59/200) | 16.4 | 194,318 |
| Random disease permutation | – | 2.6 | 194,318 |
| ≥1 topic document | 38.3 (51/133) | 19.3 | 121,862 |
| Off-topic documents only | 11.9 (8/67) | 11.5 | 72,456 |
| Supported by 1 document | 27.3 (50/183) | 13.3 | 168,512 |
| Supported by ≥2 documents | 52.9 (9/17) | 31.7–63.2^a^ | 25,806 |

^a^ 2 documents 31.7% (18,262 edges), 3–4 documents 46.3% (6,269), ≥5 documents 63.2% (1,275). The gemma-4-E4B v2.1 graph, which is the S-D graph, gives 18.2% overall agreement (random 2.8%), rising from 15.0% (1 document) to 62.8% (≥5 documents).

S-mix 그래프의 출처 상태는 다음과 같다. S-mix 간선의 의료진·LLM 판정은 하지 않았다. S-mix 그래프와 HPO 주석의 대조는 위와 같은 방식으로 설계 E 확증 실행 뒤에 계산했다(표 14d). 정확 일치율은 12.6%(비교 가능 간선 856,023개, 무작위 2.5%)였다. 지지 문서 1편, 2편, 3–4편, 5편 이상 구간의 일치율은 7.4%, 15.2%, 24.9%, 41.7%였다. 이 값은 같은 구간의 구 Qwen3 그래프(13.3%, 31.7%, 46.3%, 63.2%)와 S-D 그래프(15.0%, 34.0%, 48.7%, 62.8%)보다 모든 구간에서 낮았다. 주제 문서가 1편 이상인 S-mix 간선의 일치율은 14.9%, 주제 밖 문서뿐인 간선은 7.1%였다. S-mix와 두 구 그래프는 문헌, 추출 모델과 방식, 빌더 규칙, 간선 수가 다르다. 따라서 이 차이의 원인은 분리하지 않았다. 같은 문헌의 S-dict 그래프는 일치율이 7.8%(무작위 1.9%)로 S-mix보다 낮았다. 빌더는 재연결 항목의 원 LLM 구절을 근거 언급의 span 속성으로 저장하고, 근거 문장을 그 구절을 중심으로 자르도록 바뀌었다(커밋 3b10c36). 이 빌더로 S-mix를 비교용 인스턴스에 다시 빌드한 그래프의 지문은 바뀌지 않았다(15ba677…). 적재된 HAS_FINDING 간선은 5,221,285개로 빌드 요약과 같았다. 근거 언급을 넣어 다시 빌드한 S-mix는 동결 전에 출처 불변식 검사를 통과했다. 근거 언급이 없는 HAS_FINDING 간선은 0개였고, Disease 노드는 43,575개였다. 근거 언급은 11,582,169개였고, 지문은 바뀌지 않았다(15ba677…). 개발 DDXPlus 3,000명 모두에서 정답 질환의 Top-10 내 위치(lineage)가 근거 언급이 없는 빌드와 같았다. span 속성은 근거 언급의 LLM 구절과 사전 이름을 따로 감사할 수 있게 하지만, 그 감사는 하지 않았다. 4.3의 시뮬레이션에서 간선 정밀도를 36%에서 55%로 올렸을 때의 Top-10 기대 상승은 상대 약 6%였다. 근거 경로를 추적할 수 있다는 것이 간선이 정확하다는 뜻은 아니다.
