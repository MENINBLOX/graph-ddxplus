## 4. 결과

이 장의 정확도는 따로 적지 않으면 lineage 판정의 Top-k(%)이고, 후보는 사람 질환 43,575개 전체다. 차이는 따로 적지 않으면 "S-mix(또는 해당 문헌 그래프) − 비교 대상"의 %p다. 대괄호 안의 구간은 환자 대응 부트스트랩 95% CI다. 정답 질환 군집 CI(이하 군집 CI)는 같은 질환 집합 안의 질환 단위 변동만 반영하며, 새 질환으로의 일반화를 보이지 않는다(3.8절). 수치 뒤의 근거 지위 표기는 3.8절의 정의를 따른다. 4.1절의 표 6은 설계 D와 E에서 실행 전에 커밋한 판정을 충족 여부와 관계없이 모두 시간순으로 보인다. 4.2–4.4절은 그 판정과 확증 이후 측정을 H1–H3 순서로 다시 묶고, 표 6의 행 번호를 가리킨다. H4와 확증된 음성 결과는 4.5절, 제한 요인과 레버는 4.6절, 출처는 4.7절에서 다룬다.

### 4.1 코퍼스, 그래프, 커버리지와 판정 일람

**코퍼스.** 일반 문헌 규칙(3.2절)을 적용하기 전에 세면, v2 코퍼스에서 문헌이 있는 사람 질환은 43,575개 중 32,545개(74.7%)였다(표 5a) [측정]. v3 재수집은 사람 질환 전부에 같은 상한을 적용했다(3.2절). v3는 새 질환–문서 연결 1,619,168건을 더했다 [측정]. 그 구성은 PubMed 초록 920,514건, PMC 표 문서 494,462건, 새 PMC 문헌의 임상 절 204,192건이었다 [측정]. v2와 v3를 합치면 문헌이 하나라도 있는 사람 질환은 35,308개(81.0%)였다 [측정]. 8,267개(19.0%)는 v3 수집 뒤에도 문헌이 없었다 [측정]. 논문 단위(표 문서를 원 논문에 묶은 단위)로 세면 v2와 v3의 문헌은 882,014편이다 [측정]. 일반 문헌 규칙은 그중 4,122편을 뺐다 [측정]. 이 두 수는 그래프 빌더와 같이 비인간 질환의 문헌 파일까지 넣어 센 값이다. 일반 문헌 규칙 때문에 938개 질환은 쓸 수 있는 문헌을 모두 잃었다 [측정]. 그래서 규칙을 적용한 뒤 쓸 수 있는 문헌이 있는 사람 질환은 34,370개(78.9%)다 [측정]. v2만으로는 같은 방식으로 2,663개가 비어 29,882개(68.6%)가 남았다 [측정]. 문자 수와 문서 겹침을 포함한 전체 통계는 보충 표에 둔다.

**그래프와 커버리지.** 모든 그래프는 Disease 노드 43,575개를 후보로 가진다 [구조 검사]. 소견 간선이 없는 질환도 후보에 남는다. 소견 간선이 하나 이상 있는 질환은 S-mix 33,537개(77.0%), S-dict 34,356개(78.8%), 순수 사전 그래프 34,354개(78.8%), S-D 27,309개(62.7%)였다(표 5b) [측정]. 같은 랭커에 넣은 비교 지식원에서는 HPOA 12,251개(28.1%), PrimeKG 10,898개(25.0%), SemMedDB 11,522개(26.4%), MeSH 공출현 3,302개(7.6%)였다 [측정]. S-mix에서 소견 간선이 없는 질환은 10,038개(23.0%)였다 [측정]. 확증 세트 E에서 HPOA 계보 커버 환자는 DDXPlus-E 3,000명 중 1,212명(40.4%, 정답 질환 21개)이었다 [사후]. SymCat-E에서는 7,463명 중 2,989명(40.1%, 정답 질환 321개)이었다 [사후]. 정답 키가 정확히 일치하는 기준으로 세면 각각 232명(7.7%, 6개)과 1,142명(15.3%, 123개)이었다 [사후]. PrimeKG의 계보 커버 환자는 1,470명(49.0%, 25개)과 3,384명(45.3%, 361개)이었다 [사후].
<!-- [커밋 필요] 프로필 두께 중앙값 문장(SymCat에서 HPOA가 정확 일치로 다루는 질환 123개의 HPOA 프로필 중앙값 대 S-mix 프로필 중앙값)은 0.3 #1 처리(adv/profile_size.py 커밋) 뒤에 여기에 넣는다. -->

**LLM 경로의 연결.** v2와 v3 문헌의 LLM 추출 항목(원문에서 위치가 확인된 구절)은 모두 17,654,170개였다 [측정]. 이 중 구절 표로 개념에 연결된 항목은 36.9%(6,518,369개)였다 [측정]. 연결되지 않은 사유 가운데 가장 큰 것은 근거 부족(3.3절 규칙 5)으로, 전체 항목의 43.0%였다 [측정]. 구축 중 발견해 고친 무증상 실패와 코드 리뷰 결과는 보충자료에 둔다(3.4절).

**판정 일람.** 표 6a는 두 설계에서 실행 전에 커밋한 판정 13개를 시간순으로 보인다. 설계 D에서는 주 목표 T1(S-D가 같은 v2 문헌의 BM25보다 높음)과 흔한 질환 정확도 목표 T2가 미달했다(행 1–2) [확증]. 결합 목표 T4는 충족했다(행 4) [확증]. 개발 세트에서 판정하는 T3도 미달했다(행 3) [개발]. T3의 기준값은 커밋 뒤 D3 규칙에 따라 개발 세트에서 다시 고른 HPOA 값으로 바꿔 판정했다. 커밋한 원래 기준값으로 판정해도 미달이다(표 6 각주 a). 두 PPR 부가 가설(PPR-H1, PPR-H4)에서 PPR은 S-D의 DDXPlus와 SymCat에서 1-홉 질의보다 Top-10이 낮았고, Phenopacket-new에서는 차이가 유의하지 않았다(행 5) [확증]. 두 큐레이션 지식원에서는 희귀질환 세트 Phenopacket-new의 Top-10이 낮았다(행 6) [확증]. 흔한 질환에서는 방향이 갈렸다. HPOA의 SymCat에서는 PPR이 높았고(+2.8%p, Holm 유의), PrimeKG의 DDXPlus에서는 낮았다(−4.8%p, Holm 유의) [확증]. 설계 E의 주 목표 E0는 E-T1(S-mix가 S-D보다 높음)과 E-T2의 S-llm 비교(S-mix가 같은 문헌의 LLM 단독보다 낮지 않음)다. 두 판정은 모두 충족했다(행 7–8) [확증]. 부 목표 다섯 가운데 E-T3(같은 문헌 BM25 대비)는 충족했다(행 11) [확증]. E-T2의 S-dict·S-union 비교, E-T4(DDXPlus-E Top-1 ≥ 18%), E-T5(BM25 결합)는 미달했다(행 9–10, 12–13) [확증]. 설계 E의 7개 판정은 판정 기준과 예상을 실행 전에 커밋했고, 결과는 예상과 같았다. 설계 D는 예상을 따로 적지 않았으므로, 표에는 실행 전에 기록한 개발값을 적었다. 실행 전 커밋의 증빙은 로컬 커밋 시각이며, 공개 등록부와 실행 전 원격 기록은 없다. 표 6b는 확증 이후 측정 9건의 결정 규칙, 결과, 지위를 보인다. 이 규칙들은 측정 전에 커밋했다(3.10절). 다만 M1과 M2의 규칙에는 판정할 k가 없었고, D3의 주 지표인 Top-10으로 판정했다. M1 스크립트는 Top-10을 적었지만, 그 선택이 측정 전에 커밋되었는지는 확인되지 않는다. M2는 Top-1에서도 판정이 같았다. M1은 Top-1에서 간선 합집합이 규칙을 충족했다(표 6 각주 c). 이 Top-1 결과는 보완의 근거로 쓰지 않는다.

Table 5. Corpus and coverage. (a) Corpus over the 43,575 human candidate diseases; character counts and document overlaps are in the supplement. (b) Diseases with at least one finding edge in each source, and patients of the sealed design E sets whose gold disease or an IS_A descendant (depth 3) has at least one finding edge in the source (lineage coverage, the rule of the lineage judgement). Status: counts of diseases and edges are measurements; design E coverage is a post hoc analysis of the already used sets.

(a) Corpus.

| Item | v2 | v3 (new documents) | v2 + v3 |
|---|---|---|---|
| Human diseases with ≥ 1 document, before the generic rule | 32,545 (74.7%) | 35,006 with new documents | 35,308 (81.0%) |
| Human diseases without any document | 11,030 (25.3%) | — | 8,267 (19.0%) |
| Disease–document links | 246,203 | 1,619,168 (abstracts 920,514; table documents 494,462; PMC clinical sections 204,192) | — |
| Articles (table documents counted with their article) | 136,373^a^ | — | 882,014^a^ |
| Generic articles excluded (listed in ≥ 20 disease files) | 1,119^a^ | — | 4,122^a^ |
| Human diseases left with no document by the generic rule | 2,663 | — | 938 |
| Human diseases with ≥ 1 usable document, after the generic rule | 29,882 (68.6%) | — | 34,370 (78.9%) |
| Source-paper PMIDs excluded from collection | 1,733 | 1,834 | — |
| LLM items (location-verified spans) | 5,329,390 | 12,324,780 | 17,654,170 (36.9% linked by the phrase table) |

^a^ Counted over all queried disease files, the 8,319 non-human diseases of v2 included, as the graph builder counts them (v2: 123,772 unique documents over human diseases only).

(b) Coverage.

| Source | Diseases with ≥ 1 finding edge (of 43,575) | HAS_FINDING edges | DDXPlus-E patients covered by lineage (n = 3,000) | SymCat-E patients covered by lineage (n = 7,463) |
|---|---|---|---|---|
| S-mix | 33,537 (77.0%) | 5,221,285 | n.c. | n.c. |
| S-dict | 34,356 (78.8%) | 14,506,788 | n.c. | n.c. |
| Pure dictionary graph | 34,354 (78.8%) | 13,883,910 | n.c. | n.c. |
| S-D | 27,309 (62.7%) | 832,671 | n.c. | n.c. |
| HPOA (phenotype.hpoa 2026-02-16) | 12,251 (28.1%) | 393,498 | 1,212 (40.4%; 21 diseases); exact key 232 (7.7%; 6) | 2,989 (40.1%; 321 diseases); exact key 1,142 (15.3%; 123) |
| PrimeKG | 10,898 (25.0%) | 747,705 | 1,470 (49.0%; 25 diseases) | 3,384 (45.3%; 361 diseases) |
| SemMedDB VER43 2024 | 11,522 (26.4%)^b^ | 889,315 | n.e. | n.e. |
| MeSH co-occurrence | 3,302 (7.6%) | 3,798,931 | n.e. | n.e. |

n.c. = not computed. n.e. = source not evaluated on design E.
^b^ Only the 35,281 candidate diseases with a UMLS CUI key can be reached.

Table 6. All judgments committed before the runs, designs D and E, in time order (a), and the decision rules of the post-confirmation measurements (b). Lineage, open set of 43,575 diseases. Differences in percentage points with paired patient-bootstrap 95% CI (2,000 replicates, seed 0); Holm-adjusted exact McNemar p across the confirmation sets of one target; macro = equal-weight mean over the sets with a set-stratified paired bootstrap. The evidence for "committed before the run" is the local commit time; there is no public registry and no pre-run remote record.

(a) Pre-committed judgments.

| # | Target (role) | Criterion | Committed before the run | Result [95% CI] | Verdict |
|---|---|---|---|---|---|
| | **Design D.** Criteria at 3fbd29a and sets sealed at 31122e2 (2026-10-12); systems frozen at cd130e7 and run once at 2ecb064 (2026-10-13). DDXPlus 3,000; Phenopacket-new 261 of 369 evaluable; SymCat 7,482 | | | | |
| 1 | D-T1 (primary): S-D > BM25 on the same v2 corpus | Dev ≥ 8 of 9 sets (sign test p < .05); in each confirmation set Top-10 CI lower bound > −1 pp; ≥ 2 sets Holm-significant | Dev 9 of 9 sets (p = .0039) | DDXPlus +7.1 [+5.0, +9.1] (Holm p = 3.5 × 10⁻¹²); Phenopacket-new +0.0 [−4.2, +3.8]; SymCat +0.2 [−0.6, +1.0] | Not met |
| 2 | D-T2: DDXPlus accuracy | Top-10 ≥ 40% and Top-1 ≥ 18% | Dev S-D 33.0 / 11.2 | S-D 35.2 / 12.4; S-D + BM25 fusion 37.8 / 14.9 | Not met |
| 3 | D-T3^a^: unpublished hospital cases | PUMCH_L and PUMCH-ADM Top-10 ≥ same-ranker HPOA (53.6 / 53.3 at commit; re-judged against the HPOA re-selected on the development sets by rule D3, 53.2 / 57.3) | Not met on the development sets | S-D − HPOA: PUMCH_L −2.5 [−5.6, +0.5]; PUMCH-ADM −1.3 [−13.3, +10.7] | Not met |
| 4 | D-T4: S-D + BM25 fusion | Fusion significantly above both single systems, three-set macro Top-10 | Dev macro: fusion 32.32, S-D 30.56 | Fusion − S-D +2.7 [+1.8, +3.6]; fusion − BM25 +5.1 [+3.6, +6.7]; both p ≤ .001 | Met |
| 5 | PPR-H1^b^ (secondary, two-sided) | PPR vs 1-hop on S-D | Dev: PPR lower (Top-1 in 9 of 9 sets; macro Top-10 23.57 vs 30.56) | PPR − 1-hop: DDXPlus −15.8 [−17.3, −14.3]; Phenopacket-new −1.5 [−4.6, +1.5]; SymCat −1.6 [−2.2, −1.0] | PPR lower (2 of 3 sets Holm-significant) |
| 6 | PPR-H4^b^ (secondary) | PPR vs 1-hop on HPOA and on PrimeKG | Dev: PPR lower on both (Top-10 in 8 of 9 sets each) | PPR − 1-hop, Phenopacket-new: HPOA −12.3, PrimeKG −22.6; HPOA SymCat +2.8; PrimeKG DDXPlus −4.8 (all Holm-significant); HPOA DDXPlus Top-1 +1.8 (Holm-significant) | PPR lower on the rare-disease set for both sources; higher for HPOA on SymCat |
| | **Design E.** Criteria at 5ce9c3d (2026-10-19), expectations completed by the freeze commit; sets sealed at 3b10c36; systems frozen at 49901bc; run once from 2026-10-19 23:13 to 10-20 01:34. DDXPlus-E 3,000; SymCat-E 7,463 | | | | |
| 7 | E-T1 (primary, E0): S-mix > S-D | Both sets: Top-10 CI lower bound > 0 and Holm p < .05 | Met (dev DDXPlus +7.6, SymCat +2.7) | DDXPlus-E +6.5 [+4.6, +8.5] (Holm p = 2.1 × 10⁻¹¹); SymCat-E +2.4 [+1.7, +3.1] (Holm p = 3.4 × 10⁻¹¹) | Met |
| 8 | E-T2, S-llm (primary, E0): S-mix ≥ LLM only | S-mix − S-llm macro Top-10 CI lower bound > −1 pp | Met (dev +1.9, +1.9) | +1.64 [+1.17, +2.18] | Met |
| 9 | E-T2, S-dict (secondary) | S-mix − S-dict macro Top-10 CI lower bound > −1 pp | Not met (dev −1.6, −1.1) | −1.80 [−2.44, −1.12] | Not met |
| 10 | E-T2, S-union (secondary) | S-mix − S-union macro Top-10 CI lower bound > −1 pp | Not met (dev −3.0, −1.3) | −2.66 [−3.27, −2.07] | Not met |
| 11 | E-T3 (secondary): S-mix vs S-bm25 | Both sets Top-10 CI lower bound > −1 pp; ≥ 1 set Holm-significant | Met (dev +20.9 [+19.2, +22.7], +3.2 [+2.5, +4.0]) | DDXPlus-E +19.3 [+17.6, +21.0]; SymCat-E +3.0 [+2.2, +3.8]; both Holm p < 10⁻¹³ | Met |
| 12 | E-T4 (secondary): DDXPlus-E accuracy | S-mix Top-10 ≥ 40% and Top-1 ≥ 18% | Top-10 borderline (dev 40.6); Top-1 not met (dev 11.8) | Top-10 41.07 [39.3, 43.0]; Top-1 12.43 [11.3, 13.6] | Not met (Top-1) |
| 13 | E-T5 (secondary): S-mix + S-bm25 fusion (S-comb) | S-comb − S-mix and S-comb − S-bm25 macro Top-10 CI lower bounds > 0 | Not met (dev S-comb − S-mix −0.02 [−0.54, +0.49]) | S-comb − S-mix +0.11 [−0.39, +0.59]; S-comb − S-bm25 +11.25 [+10.36, +12.14] | Not met |

^a^ Judged on the development sets. Exact McNemar p = .11 and 1; with Holm adjustment across the nine development sets, .22 and 1. The committed criterion held fixed HPOA values chosen on the validation split (53.6 / 53.3); rule D3 later re-selected HPOA on the development sets, and T3 was re-judged against those values. Against the committed values PUMCH-ADM met the criterion (56.0 ≥ 53.3) and PUMCH_L did not (50.7 < 53.6), so the verdict is the same.
^b^ The secondary PPR hypotheses (called E-H1 and E-H4 in the design D record; Section 3.6), renamed here to avoid confusion with claims H1–H4. Row 6 gives point estimates.

(b) Post-confirmation measurements. Every rule was committed before its measurement (Section 3.10); the M1 and M2 rules did not name k (footnote c). Post hoc = analysis of the already used design E sets; development = the nine development sets.

| Measurement [status] | Rule committed before the measurement | Result | Outcome |
|---|---|---|---|
| | **Rules of §9.32 at 400d4cf (2026-10-20); results at 3ea76b7 (2026-10-21)** | | |
| P0-1 leakage re-check [development] | Remove patients flagged by the single-document rule (one document supports ≥ 50% and ≥ 5 of the patient's concepts). Keep the collection-depth claim if all 7 same-extraction pairs keep ≥ half of their dev macro Top-10 effect; keep the rare-set S-mix − S-dict claim if its CI lower bound stays > 0 | Remaining share 0.69–0.93 in the 7 pairs; rare 7-set S-mix − S-dict +4.55 → +4.79 [+2.21, +7.57] | Both kept (Section 4.6) |
| P0-2 pure dictionary graph [development; design E post hoc] | If pure dictionary − S-dict dev macro Top-10 is within 1 pp, describe the cost as that of an LLM-free dictionary path | −0.50 [−0.96, −0.07] | LLM-free dictionary path (Section 4.4) |
| P0-3 strict judgement [post hoc] | If an E-T2 comparison reverses its sign under strict judgement with a CI excluding 0, its conclusion depends on the judgement rule | S-mix − S-dict: lineage −1.80, strict +1.04 [+0.39, +1.72]; S-mix − S-union: strict +0.15 [−0.46, +0.76]; S-mix − S-llm: strict +4.04 | S-dict comparison depends on the rule; the other two do not (Section 4.6) |
| P0-4 HPOA LOPO [development] | Report S-mix − HPOA before and after deleting the HPOA annotations that cite the benchmark case papers, with a random-deletion control of the same size. Interpretation rule (f676539), on the Phenopacket-store disease-equal HPOA − S-mix: ≤ 10 pp and shrinking far more than under random deletion → "the gap shrank once circularity was removed" [exploratory]; ≥ 30 pp → "the gap stayed large"; otherwise numbers only | Phenopacket-store −42.66 → +5.58 [+4.75, +6.42]; random deletion −41.19 (seeds 1–2: −40.97, −41.44). Patient-weighted: the committed outputs hold no disease-equal value, so the rule's basis was not used | Reported (Section 4.5) |
| P0-5 SemMedDB, same ranker [development] | (1) If SemMedDB ≥ S-dict in dev macro Top-10, state that a public rule-based resource is equal or higher. (2) S-dict − SemMedDB on the two common sets by cluster CI: both lower bounds > 0 → "highest of the tested public sources"; any upper bound < 0 → withdraw the lead sentence; otherwise "not distinguished from SemMedDB". Revised before the measurement (f676539) to give SemMedDB the tuning budget of HPOA and PrimeKG | SemMedDB 23.20 (frozen scoring 17.53) vs S-dict 42.88. S-dict − SemMedDB cluster CI: DDXPlus [−14.53, +9.16], SymCat [+0.55, +3.87] | (1) Statement not written. (2) Not distinguished (Section 4.3) |
| | **Rules of §9.33 and of the framing review at f676539 (2026-10-20); results at dfa9541 (2026-10-21)** | | |
| M2 curated comparators on design E [post hoc] | If S-mix − HPOA has a cluster CI lower bound > 0 on both E sets, add the E values to H2 | DDXPlus-E +30.70 (cluster [+18.29, +44.86]); SymCat-E +9.14 ([+7.41, +10.88]) | Met (Section 4.3) |
| M1 HPOA + S-mix combination [development] | Write "complements" only if, in the unpublished hospital stratum (PUMCH_L, PUMCH-ADM), combination − HPOA and combination − S-mix both have macro CI lower bounds > 0 (patient-stratified, cluster and disease-equal CIs)^c^ | RRF − S-mix −4.07 [−10.08, +1.83]; edge union − S-mix +0.14 [−2.26, +2.40]; combination − HPOA +8.19 (RRF), +12.41 (union) | Not met (Section 4.5) |
| V1 BM25 sensitivity [post hoc] | If S-mix − BM25 has a CI lower bound > 0 on both E sets for all 12 v2 + v3 BM25 configurations, write "for every tested configuration" | Smallest lower bound: patient CI +16.30 / +1.47; cluster CI +6.33 / +0.52 | Met (Section 4.2) |
| P1-3 input-gold strata [post hoc] | Among DDXPlus-E patients whose input names neither the gold disease nor its direct parent: if the S-mix − S-dict CI includes 0 or is positive, state the input-structure caveat at the sealed-set level | S-mix − S-dict +1.27 [+0.09, +2.54] (cluster [−4.29, +6.21]) | Met (Section 4.6) |

^c^ The M1 and M2 rules do not name k; Top-10, the primary metric of rule D3, was used. The M1 script names Top-10, but it is not confirmed that this choice was committed before the measurement. M2 gives the same verdict at Top-1 (cluster CI lower bounds +4.06 and +0.95). At Top-1 the M1 edge union met the rule: union − HPOA +14.39 [+8.91, +19.79] (cluster [+6.86, +21.62]; disease-equal +12.11 [+3.87, +20.34]); union − S-mix +3.82 [+1.05, +6.89] (cluster [+0.16, +9.04]; disease-equal +5.02 [+1.12, +10.15]). This Top-1 result is not used as evidence that the combination complements HPOA.

### 4.2 같은 문헌 BM25와의 비교(H1)

E-T3는 실행 전에 커밋한 7개 판정 가운데 부 목표다(표 6 행 11). 두 봉인 세트에서 S-mix의 Top-10은 같은 v2+v3 문헌의 BM25(S-bm25)보다 높았다(그림 2) [확증]. DDXPlus-E에서는 41.07% 대 21.77%로 +19.3%p [+17.6, +21.0]였다 [확증]. SymCat-E에서는 12.92% 대 9.93%로 +3.0%p [+2.2, +3.8]였다 [확증]. 두 차이의 Holm 보정 p는 모두 10⁻¹³보다 작았다 [확증]. 군집 CI도 두 세트 모두 0보다 컸다(DDXPlus-E [+8.8, +30.7], SymCat-E [+1.4, +4.5]) [확증-보조]. Top-1 차이는 DDXPlus-E +3.9%p [+2.6, +5.1], SymCat-E +0.4%p [−0.1, +0.8]였다 [확증-보조]. 실행 전에 커밋한 예상(개발 세트 값)에서도 S-mix − S-bm25는 DDXPlus +20.9%p [+19.2, +22.7], SymCat +3.2%p [+2.5, +4.0]였다 [개발]. 9개 개발 세트의 매크로 Top-10은 S-mix 46.11%, S-bm25 27.35%였다 [개발].

**BM25 설정 민감도.** 확증 뒤에 BM25 설정을 바꿔 같은 비교를 다시 했다(V1, 표 7a) [사후]. 시험한 설정은 v2+v3 색인에서 질의 방식 2가지, 검색 깊이 2가지, 집계 방식 3가지를 조합한 12개다. BM25 함수 매개변수(k1 1.5, b 0.75)는 바꾸지 않았다. 12개 설정 가운데 질의 전체·최댓값 집계의 두 깊이(100, 1,000)는 Top-10 결과가 같았다 [사후]. 따라서 Top-10에서 서로 다른 결과는 11개다. 12개 설정 모두에서 두 세트의 S-mix − BM25 Top-10 CI 하한이 0보다 컸다 [사후]. 환자 CI와 군집 CI 모두 그랬다 [사후]. 환자 CI 하한의 최솟값은 DDXPlus-E +16.30%p, SymCat-E +1.47%p였다 [사후]. 두 최솟값은 v2+v3 색인의 12개 설정 가운데 확증 세트 E에서 BM25가 가장 높았던 설정(질의 전체, 깊이 1,000, 합산 집계)에서 나왔다 [사후]. 군집 CI 하한의 최솟값은 +6.33%p와 +0.52%p였다 [사후]. v2 색인에 설계 D 설정을 적용한 BM25(DDXPlus-E 27.43%, SymCat-E 10.55%)와 비교해도 S-mix가 높았다 [사후]. 차이는 DDXPlus-E +13.63%p [+11.80, +15.47], SymCat-E +2.37%p [+1.59, +3.15]였다 [사후]. 이때 군집 CI 하한은 +1.05%p와 +1.02%p였다 [사후]. 측정 전에 커밋한 V1 규칙에 따라, H1은 "시험한 BM25 설정 12개 모두에 대해"로 쓴다(표 6b). 이 판정은 Top-10에만 해당한다. Top-1에서는 설정에 따라 결과가 달랐다 [사후]. 예를 들어 v2 BM25 대비 DDXPlus-E Top-1 차이는 +1.03%p [−0.33, +2.50]였다 [사후]. 채점 설정의 조정 예산은 BM25 쪽(설정 12개)이 그래프 쪽(γ 4값)보다 컸다. 다만 S-mix 그래프 자체는 개발 세트에서 그래프 32개 가운데 고른 것이다(표 4) [개발].

**설계 D의 같은 비교.** 설계 D에서는 같은 성격의 비교인 T1이 미달했다(표 6 행 1) [확증]. v2 문헌 그래프 S-D와 같은 v2 문헌 BM25의 Top-10 차이는 DDXPlus +7.1%p [+5.0, +9.1], Phenopacket-new +0.0%p [−4.2, +3.8], SymCat +0.2%p [−0.6, +1.0]였다 [확증]. 두 판정을 같은 비교의 반복으로 읽지 않는다. 확증 세트가 다르다. E에는 희귀질환 세트인 Phenopacket-new가 없다. 판정 기준도 다르다. T1은 세 세트의 비열등과 두 세트 이상의 Holm 유의 우세를 요구했고, E-T3는 두 세트의 비열등과 한 세트 이상의 우세를 요구했다. 비교한 BM25의 문헌도 다르다.

**문헌 확장과 BM25 희석.** 같은 BM25 설정에서 색인 문헌을 늘리면 Top-10이 내려갔다 [사후]. 같은 BM25가 더 큰 코퍼스에서 낮아지는 이 현상을 희석이라 부른다(3.10절). 정보검색에서는 질의와 관련 없는 청크가 늘어 상위 결과를 차지하는 것으로 설명하지만, 이 연구는 그 기전을 따로 재지 않았다. 설계 D의 BM25 설정을 고정하고 색인 문헌만 v2에서 v2+v3로 늘리면, DDXPlus-E Top-10은 27.43%에서 21.47%로 내려갔다(−5.97%p [−7.63, −4.37], 군집 CI [−18.57, +4.83]) [사후]. SymCat-E는 10.55%에서 8.48%로 내려갔다(−2.06%p [−2.77, −1.34], 군집 CI [−3.28, −0.88]) [사후]. 이 하락은 설계 D의 설정 하나에서 잰 것이다. v2+v3 색인의 12개 설정 가운데 DDXPlus-E에서 가장 높은 값(23.03%)도 v2 색인의 값보다 낮았다 [사후]. SymCat-E에서는 v2+v3 색인의 가장 높은 값(10.69%)이 v2 색인의 값(10.55%)과 비슷했다 [사후]. v2 색인 BM25의 DDXPlus-E 값(27.43%)은 설계 D 확증 세트에서의 BM25 값(28.0%)과 비슷했다 [사후]. 따라서 DDXPlus에서 설계 D와 E 사이의 BM25 하락은 환자 표본의 차이로는 설명되지 않았고, 같은 환자에서 색인 문헌을 바꿨을 때 나타났다 [사후]. 다만 이 하락의 군집 CI는 0을 포함했다. 같은 환자에서 DDXPlus-E의 E-T3 격차(+19.30%p)는 네 단계 차이의 합으로 나뉜다(표 7b) [사후]. 첫 단계는 v2+v3 색인에서 BM25 설정의 차이로, −0.30%p [−1.87, +1.30]다 [사후]. 둘째 단계는 희석으로, +5.97%p [+4.37, +7.63]다 [사후]. 셋째 단계는 v2 색인 BM25 대비 S-D의 차이로, +7.10%p [+5.17, +9.07]다 [사후]. 넷째 단계는 S-D 대비 S-mix의 차이로, +6.53%p [+4.63, +8.47]다(E-T1) [확증]. BM25 쪽 하락(v2 색인·설계 D 설정 27.43% → v2+v3 색인·선택 설정 21.77%)은 5.66%p로, 격차의 29%다 [사후]. 정답 질환이 49개인 DDXPlus-E에서는 각 단계의 군집 CI가 0을 포함했다(1–3단계 [사후], 4단계 [확증-보조]). SymCat-E 격차(+2.99%p)를 같은 방식으로 나누면 대부분 S-D 대비 S-mix의 몫이었다 [사후]. 그 몫은 E-T1의 +2.43%p [+1.73, +3.12]이고 [확증], 군집 CI는 [+1.04, +3.87]이었다 [확증-보조].

**비교 범위.** 이 절에서 비교한 검색기는 어휘 검색인 BM25뿐이다. 밀집 검색기 MedCPT는 v1 단계에서만 평가했고, v2+v3 문헌에서는 다시 고르지 않았다(3.7절). 따라서 이 절의 결과는 BM25 대비의 결과다.

Table 7. S-mix versus same-corpus BM25 on the sealed design E sets: configuration sensitivity (a) and decomposition of the E-T3 gap (b). Lineage Top-10 (%). S-mix: DDXPlus-E 41.07, SymCat-E 12.92. Differences S-mix − BM25 in percentage points with [patient-paired 95% CI] [gold-disease cluster 95% CI]. The row "joint, 1000, RRF" is the frozen S-bm25 of E-T3 (patient CI confirmatory, cluster CI auxiliary). Every other row is a post hoc analysis of the already used design E sets. BM25 k1 = 1.5 and b = 0.75 in every configuration. Dev macro = macro Top-10 over the nine development sets, the selection criterion.

(a) Configuration sensitivity.

| Index; query mode, depth, aggregation | Dev macro | DDXPlus-E BM25 | S-mix − BM25 | SymCat-E BM25 | S-mix − BM25 |
|---|---|---|---|---|---|
| v2 + v3; joint, 100, max | 19.55 | 16.47 | +24.60 [+22.83, +26.47] [+13.82, +35.98] | 7.20 | +5.72 [+4.96, +6.47] [+4.37, +7.12] |
| v2 + v3; joint, 100, sum | 25.20 | 22.27 | +18.80 [+17.07, +20.63] [+8.17, +30.54] | 8.91 | +4.01 [+3.24, +4.76] [+2.61, +5.45] |
| v2 + v3; joint, 100, RRF | 24.62 | 22.37 | +18.70 [+17.03, +20.53] [+7.99, +30.39] | 8.80 | +4.11 [+3.35, +4.89] [+2.71, +5.53] |
| v2 + v3; joint, 1000, max | 19.55 | 16.47 | +24.60 (Top-10 result identical to joint, 100, max) | 7.20 | +5.72 (identical to joint, 100, max) |
| v2 + v3; joint, 1000, sum | 25.56 | 23.03 | +18.03 [+16.30, +19.77] [+8.07, +27.98] | 10.69 | +2.22 [+1.47, +3.00] [+0.60, +3.83] |
| **v2 + v3; joint, 1000, RRF (S-bm25, E-T3)** | **27.35** | **21.77** | **+19.30 [+17.60, +21.03] [+8.79, +30.74]** | **9.93** | **+2.99 [+2.22, +3.77] [+1.43, +4.52]** |
| v2 + v3; per concept, 100, max | 20.94 | 16.77 | +24.30 [+22.40, +26.23] [+13.04, +36.63] | 8.91 | +4.01 [+3.24, +4.80] [+2.58, +5.59] |
| v2 + v3; per concept, 100, sum | 13.26 | 14.60 | +26.47 [+24.57, +28.33] [+13.52, +39.39] | 9.43 | +3.48 [+2.68, +4.29] [+1.61, +5.31] |
| v2 + v3; per concept, 100, RRF | 13.31 | 13.87 | +27.20 [+25.27, +29.07] [+14.12, +39.98] | 8.92 | +3.99 [+3.22, +4.80] [+2.23, +5.77] |
| v2 + v3; per concept, 1000, max | 24.07 | 21.47 | +19.60 [+17.63, +21.67] [+6.33, +33.41] | 8.48 | +4.44 [+3.63, +5.19] [+3.00, +5.94] |
| v2 + v3; per concept, 1000, sum | 15.56 | 15.07 | +26.00 [+24.10, +27.87] [+13.23, +38.56] | 10.52 | +2.40 [+1.58, +3.23] [+0.52, +4.20] |
| v2 + v3; per concept, 1000, RRF | 13.80 | 13.60 | +27.47 [+25.60, +29.37] [+14.79, +40.19] | 9.82 | +3.10 [+2.32, +3.93] [+1.19, +4.91] |
| v2; per concept, 1000, max (design D BM25) | 24.46 | 27.43 | +13.63 [+11.80, +15.47] [+1.05, +26.96] | 10.55 | +2.37 [+1.59, +3.15] [+1.02, +3.81] |

Holm-adjusted p < 3.8 × 10⁻⁸ in every row (largest 3.74 × 10⁻⁸; Holm across the two sets of one configuration). Top-1 results depend on the configuration (supplement).

(b) Decomposition of the E-T3 gap on the same patients (post hoc, except step 4). Each step is B − A; the four steps add up to the gap.

| Step | DDXPlus-E: A → B, difference [patient CI] [cluster CI] | SymCat-E: A → B, difference [patient CI] [cluster CI] |
|---|---|---|
| 1. BM25 configuration on the v2 + v3 index (design D configuration − selected configuration) | 21.77 → 21.47, −0.30 [−1.87, +1.30] [−10.56, +9.27] | 9.93 → 8.48, −1.45 [−2.13, −0.78] [−2.66, −0.27] |
| 2. Dilution (design D configuration: v2 index − v2 + v3 index) | 21.47 → 27.43, +5.97 [+4.37, +7.63] [−4.83, +18.57] | 8.48 → 10.55, +2.06 [+1.34, +2.77] [+0.88, +3.28] |
| 3. S-D − v2 BM25 | 27.43 → 34.53, +7.10 [+5.17, +9.07] [−7.77, +20.37] | 10.55 → 10.49, −0.05 [−0.79, +0.72] [−1.47, +1.33] |
| 4. S-mix − S-D (E-T1; patient CI confirmatory, cluster CI auxiliary) | 34.53 → 41.07, +6.53 [+4.63, +8.47] [−6.07, +19.33] | 10.49 → 12.92, +2.43 [+1.73, +3.12] [+1.04, +3.87] |
| Sum = S-mix − S-bm25 (E-T3) | +19.30 | +2.99 |

### 4.3 같은 랭커 지식원 교체(H2)

표 8은 같은 랭커 지식원 교체 평가의 전체 값이다. 그래프 지식원(S-mix, S-dict, 순수 사전 그래프, S-D, HPOA, HPOA-LOPO, PrimeKG, MeSH 공출현, SemMedDB)은 같은 1-홉 Cypher 랭커 코드로 같은 43,575개 후보를 채점했다. 채점 설정은 지식원별로 개발 세트에서 조정했다(표 4). 표의 BM25 두 줄은 같은 후보 위의 텍스트 검색 기준선이며, 랭커가 다르다(3.7절). 개발 9개 세트에서 S-mix의 Top-10은 DDXPlus 40.63%, SymCat 13.34%, 9개 세트 매크로 46.11%였다 [개발]. 같은 순서로 HPOA는 10.30%, 3.60%, 52.45%였다 [개발]. PrimeKG는 11.27%, 5.56%, 33.52%였다 [개발]. MeSH 공출현은 24.97%, 7.70%, 9.13%였다 [개발]. 설정을 다시 고른 SemMedDB는 45.33%, 12.28%, 23.20%였다 [개발]. 같은 문헌의 BM25(S-bm25)는 9개 세트 매크로 27.35%였다 [개발]. 순수 사전 그래프는 39.53%, 14.41%, 42.38%였다 [개발]. HPOA는 9개 세트 매크로에서 표 8의 지식원 가운데 가장 높았다 [개발]. 그러나 흔한 질환 두 세트에서는 가장 낮았다 [개발].

**봉인 세트 E의 큐레이션 비교군.** 확증 뒤에 설계 D와 같은 동결 설정의 HPOA·PrimeKG 비교군을 확증 세트 E에서 한 번 돌렸다(M2, 표 6b) [사후]. 두 비교군 모두 실행 전에 개발 세트의 저장 순위를 모든 환자에서 재현했다 [구조 검사]. HPOA의 Top-10은 DDXPlus-E 10.37%, SymCat-E 3.78%였다 [사후]. PrimeKG는 11.33%, 5.82%였다 [사후]. S-mix − HPOA는 DDXPlus-E에서 +30.70%p [+28.70, +32.67]였다(그림 2) [사후]. 이 차이의 군집 CI는 [+18.29, +44.86], 질환 등가중 차이는 +34.50%p [+22.53, +46.56]였다 [사후]. SymCat-E에서는 +9.14%p [+8.36, +9.89]였다 [사후]. 군집 CI는 [+7.41, +10.88], 질환 등가중 차이는 +9.08%p [+7.50, +10.82]였다 [사후]. 두 세트 모두 군집 CI 하한이 0보다 컸으므로, 측정 전에 커밋한 M2 규칙을 충족했다 [사후]. S-mix − PrimeKG는 DDXPlus-E +29.73%p [+27.77, +31.77](군집 CI [+16.42, +44.73]), SymCat-E +7.10%p [+6.31, +7.87]([+5.33, +8.88])였다 [사후]. S-dict, 순수 사전 그래프, S-D도 두 비교군보다 높았고, 군집 CI도 모두 0보다 컸다 [사후]. HPOA 대비 차이는 S-dict가 DDXPlus-E +33.13%p [+31.17, +35.10](군집 CI [+20.37, +46.14]), SymCat-E +10.30%p [+9.47, +11.08]([+8.42, +12.20])였다 [사후]. 순수 사전 그래프는 +30.03%p [+28.03, +32.07]([+17.53, +43.55])와 +10.22%p [+9.41, +11.01]([+8.27, +12.13])였다 [사후]. S-D는 +24.17%p [+22.33, +26.13]([+14.20, +35.09])와 +6.71%p [+6.02, +7.42]([+5.18, +8.29])였다 [사후]. 개발값도 비슷했다. 개발 세트의 S-mix − HPOA는 DDXPlus +30.33%p [+28.43, +32.27], SymCat +9.74%p [+8.98, +10.52]였다 [개발]. 설계 D 봉인 세트에서 S-D − HPOA는 DDXPlus +24.8%p, SymCat +7.3%p였다 [봉인 기술].

**커버리지와 프로필 두께.** 표 9는 이 차이를 HPOA의 계보 커버리지와 프로필 두께로 나눈다 [사후]. HPOA 계보 커버 층에서 S-mix − HPOA는 DDXPlus-E +21.70%p [+18.40, +24.83](군집 CI [+4.27, +41.47]), SymCat-E +6.83%p [+5.42, +8.20]([+4.32, +9.59])였다 [사후]. 비커버 층에서는 DDXPlus-E +36.80%p [+34.62, +38.93], SymCat-E +10.68%p [+9.81, +11.58]였다 [사후]. 비커버 층에서 HPOA의 Top-10은 0.00%와 0.25%였다 [사후]. 프로필 두께는 정답 키마다 HPOA가 붙인 소견 수의 최댓값으로 정했다. 층 경계(0, 1–9, 10–29, ≥ 30)는 개발 결과를 본 뒤 정했고, E 측정 전에 고정했다. SymCat-E에서 차이는 0층 +9.71%p, 1–9층 +9.36%p, 10–29층 +8.68%p였다(CI는 표 9) [사후]. 10–29층에서 군집 CI([+0.39, +19.51])는 0보다 컸지만, 질환 등가중 CI([−0.09, +17.99])는 0을 포함했다 [사후]. SymCat-E의 소견 30개 이상 층에서는 +0.70%p [−0.93, +2.33], 군집 CI [−2.05, +3.78](질환 45개)였다 [사후]. DDXPlus-E의 30개 이상 층에서는 −1.85%p [−10.49, +6.79], 군집 CI [−46.15, +36.22](질환 5개)였다 [사후]. 개발 SymCat의 같은 층도 +2.12%p [−0.24, +4.48], 군집 CI [−2.68, +8.02](질환 45개)였다 [개발]. 따라서 HPOA가 소견 30개 이상을 붙인 질환에서는 차이를 확인하지 못했다. 이 결과는 그 층에서 두 지식원이 같다는 뜻이 아니다. DDXPlus-E의 30개 이상 층은 질환이 5개라 CI가 넓다. SymCat-E의 같은 층(질환 45개)은 군집 CI 상한(+3.78%p)이 세트 전체 차이(+9.14%p)보다 작았다 [사후]. 층 사이의 차이를 검정하지는 않았고, 동등성을 보인 것도 아니다.

**문헌 기반 비교군.** 흔한 질환 두 개발 세트에서는 시험한 문헌 기반 지식원이 모두 HPOA보다 높았다(표 8a) [개발]. LLM 추출이 없는 MeSH 공출현도 개발 DDXPlus에서 24.97%로, HPOA(10.30%)보다 높았다 [개발]. 그래서 문헌 그래프 고유의 근거는 문헌 기반 비교군과의 차이에서 찾아야 한다. 설계 D 봉인 세트에서 S-D − MeSH 공출현 Top-10은 DDXPlus +9.6%p, Phenopacket-new +7.7%p, SymCat +3.6%p였다 [봉인 기술]. 세 차이는 모두 Holm 유의했다 [봉인 기술]. 같은 문헌 BM25와의 비교는 4.2절에 적었다 [확증].

**SemMedDB.** SemMedDB는 HPOA·PrimeKG와 같은 격자에서 설정을 다시 골랐다(평가 39회, 서로 다른 설정 28개; 3.7절). 개발 매크로 Top-10은 23.20%로, S-dict(42.88%)보다 19.68%p 낮았다 [개발]. 설계 D의 동결 채점을 그대로 쓰면 17.53%였다 [개발]. 측정 전에 커밋한 P0-5 규칙은 SemMedDB가 S-dict 이상일 때만 "공개 규칙 기반 자원이 같거나 높다"를 쓰게 한다(표 6b). 그래서 그 문장은 쓰지 않는다. 그러나 흔한 질환 세트의 결과는 달랐다. 개발 DDXPlus에서 SemMedDB는 45.33%로, S-mix(40.63%)와 S-dict(42.27%)보다 높았다 [개발]. S-mix − SemMedDB는 −4.70%p [−6.63, −2.73]였고, 군집 CI([−15.27, +7.15])는 0을 포함했다 [개발]. S-dict − SemMedDB는 −3.07%p [−4.97, −1.10]였고, 군집 CI는 [−14.53, +9.16]이었다 [개발]. 개발 SymCat에서 S-mix − SemMedDB는 +1.06%p(군집 CI [−0.65, +2.71])였다 [개발]. 같은 세트에서 S-dict − SemMedDB는 +2.18%p(군집 CI [+0.55, +3.87])였다 [개발]. 측정 전에 커밋한 둘째 P0-5 규칙(3.10절)은 흔한 질환 두 세트의 S-dict − SemMedDB 군집 CI로 판정한다. DDXPlus에서는 군집 CI가 0을 포함했고, SymCat에서는 하한이 0보다 컸다. 그래서 이 결과는 규칙의 "SemMedDB와 구별되지 않았다"에 해당한다(표 6b). 이 규칙에 따라 이 절의 우세 문장은 HPOA, PrimeKG, MeSH 공출현 대비로만 쓴다. SemMedDB 그래프는 UMLS CUI 키를 가진 질환 35,281개에만 연결될 수 있다 [측정]. SemMedDB와 MeSH 공출현은 확증 세트 E에서 평가하지 않았다.

**설정과 데이터 구성.** HPOA 설정은 9개 세트 매크로 Top-10으로 골랐다(D3 규칙). 세트별로 가장 좋은 설정을 고르면 HPOA의 개발 Top-10은 DDXPlus 16.07%, SymCat 6.50%였다 [개발]. 따라서 개발 격차 가운데 DDXPlus 약 6%p, SymCat 약 3%p는 HPOA 설정의 선택에서 온다 [개발]. 같은 1-홉 랭커에서는 HPOA에 β와 γ가 효과가 없고, 원문 길이 항은 상수가 된다. HPOA의 빈도 주석도 쓰지 않는다(3.7절). 그래서 같은 1-홉 랭커의 HPOA 값은 큐레이션 지식의 상한이 아니다. 두 확증 세트는 모두 합성 환자다. DDXPlus의 지식 베이스는 역학 연구, 질환별 논문, 메타분석을 포함한 의학 논문 2만 편 이상을 모아 만들었다[ref: DDXPlus, 3.1절]. 그래서 이 구성은 문헌 그래프에 유리할 수 있다. DDXPlus-E 환자의 23.9%(717명)는 입력에 정답 질환이나 그 직속 상위 질환을 이미 담고 있다(3.6절) [사후]. 이 층으로 S-mix − HPOA를 나누어 보지는 않았다.

**비출판 병원 증례.** PUMCH_L(988명, 정답 질환 100개)은 출판되지 않은 한 병원의 희귀질환 증례다 [개발]. 이 세트에서도 S-mix의 Top-10은 같은 랭커의 HPOA보다 높았다(67.11% 대 53.24%) [개발]. 환자 가중 차이는 +13.87%p [+10.83, +16.90]였고, 군집 CI는 [+1.86, +25.85]였다 [개발]. 정답 질환마다 1표를 준 질환 등가중 차이는 +1.51%p [−7.60, +10.95]로 0과 구별되지 않았다 [개발]. 환자 가중 값과 질환 등가중 값이 갈렸으므로, 이 결과를 H2와 독립된 두 번째 증거로 쓰지 않는다.
<!-- [커밋 필요] 우세가 HPOA 프로필이 얇은 비멘델성 희귀질환 세 개(POEMS, EGPA, 카르시노이드)에 몰려 있다는 문장은 0.3 #1(adv/pumch_clusters.json)과 #2(질환 키 정의)를 처리한 뒤 여기에 넣는다. -->
같은 병원의 PUMCH-ADM(75명, 정답 질환 16개)에서 차이는 +10.67%p [−1.33, +22.67]였다 [개발]. 군집 CI는 [−4.69, +26.67], 질환 등가중 차이는 +6.51%p [−6.46, +20.26]였다 [개발]. 설계 D의 T3에서는 v2 문헌 그래프 S-D가 두 세트 모두에서 HPOA보다 낮았다(PUMCH_L −2.5%p [−5.6, +0.5], PUMCH-ADM −1.3%p [−13.3, +10.7]; 표 6 행 3) [개발].

Table 8. Same-ranker knowledge-source comparison, development (a) and sealed sets (b). Every graph source is scored by the same 1-hop Cypher ranker code over the same 43,575 candidate diseases; scoring settings were tuned per source on the nine development sets (budget in Table 4). The two BM25 rows are text-retrieval baselines over the same candidates, with a different ranker (Section 3.7). Lineage Top-10 (%). Development values include selection optimism. Superscripts in panel b: C, confirmatory system of the design; S, design D comparator run once without a pre-committed criterion (sealed description); P, post hoc analysis of the already used design E sets. Blank = not evaluated.

(a) Development sets (all values development).

| Source | DDXPlus | SymCat | RAMEDIS | LIRICAL | HMS | MME | PUMCH_L | PUMCH-ADM | Phenopacket-store | Macro Top-10 / Top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| S-mix | 40.63 | 13.34 | 57.21 | 38.38 | 44.32 | 55.00 | 67.11 | 68.00 | 31.03 | 46.11 / 19.66 |
| S-dict | 42.27 | 14.46 | 47.92 | 32.70 | 43.18 | 52.50 | 64.78 | 60.00 | 28.08 | 42.88 / 18.11 |
| Pure dictionary graph | 39.53 | 14.41 | 47.92 | 33.24 | 42.05 | 52.50 | 63.77 | 60.00 | 28.00 | 42.38 / 17.75 |
| S-D | 33.03 | 10.68 | 31.25 | 23.24 | 28.41 | 25.00 | 50.71 | 56.00 | 16.73 | 30.56 / 12.64 |
| S-bm25 (v2 + v3 BM25)^c^ | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | 27.35 / 11.04 |
| BM25, v2 (design D) | 26.47 | 10.40 | 12.82 | 17.84 | 22.73 | 22.50 | 49.19 | 42.67 | 15.56 | 24.46 / 8.75 |
| HPOA | 10.30 | 3.60 | 59.94 | 74.59 | 44.32 | 95.00 | 53.24 | 57.33 | 73.69 | 52.45 / 30.37 |
| HPOA-LOPO^a^ | 10.70 | 3.60 | 59.46 | 42.70 | 43.18 | 95.00 | 52.83 | 61.33 | 25.45 | 43.81 / 22.42 |
| PrimeKG | 11.27 | 5.56 | 51.12 | 43.78 | 32.95 | 55.00 | 38.77 | 38.67 | 24.60 | 33.52 / 15.18 |
| MeSH co-occurrence | 24.97 | 7.70 | 2.24 | 4.59 | 5.68 | 2.50 | 16.09 | 14.67 | 3.75 | 9.13 / 2.61 |
| SemMedDB (re-selected)^b^ | 45.33 | 12.28 | 28.69 | 9.19 | 14.77 | 32.50 | 18.62 | 37.33 | 10.09 | 23.20 / 7.15 |

(b) Sealed sets.

| Source | D: DDXPlus (3,000) | D: Phenopacket-new (261) | D: SymCat (7,482) | E: DDXPlus-E (3,000) | E: SymCat-E (7,463) |
|---|---|---|---|---|---|
| S-mix | | | | 41.07^C^ | 12.92^C^ |
| S-dict | | | | 43.50^C^ | 14.08^C^ |
| Pure dictionary graph | | | | 40.40^P^ | 14.00^P^ |
| S-D | 35.2^C^ | 9.2^C^ | 11.1^C^ | 34.53^C^ | 10.49^C^ |
| S-bm25 (v2 + v3 BM25) | | | | 21.77^C^ | 9.93^C^ |
| BM25, v2 (design D) | 28.0^C^ | 9.2^C^ | 10.9^C^ | 27.43^P^ | 10.55^P^ |
| HPOA | 10.3^S^ | 36.0^S^ | 3.8^S^ | 10.37^P^ | 3.78^P^ |
| PrimeKG | 12.0^S^ | 31.8^S^ | 5.6^S^ | 11.33^P^ | 5.82^P^ |
| MeSH co-occurrence | 25.6^S^ | 1.5^S^ | 7.5^S^ | | |

^a^ HPOA after deleting, from the whole resource, every annotation that cites a case paper of the benchmarks; complete only for Phenopacket-store, partial for LIRICAL, not applicable to sets without PMIDs (Section 4.5). Scoring settings were not re-selected after the deletion.
^b^ Re-selected on the same grid as HPOA and PrimeKG (39 evaluations, 28 distinct configurations). With the frozen design D scoring: macro 17.53 / 7.60, DDXPlus 35.13, SymCat 14.52. Only diseases with a UMLS CUI key (35,281) can be reached.
^c^ n.r. = not reported. Only the macro value of the frozen selection record is given; the committed pre-run expectations give the development differences S-mix − S-bm25 on DDXPlus (+20.9 [+19.2, +22.7]) and SymCat (+3.2 [+2.5, +4.0]).

Table 9. S-mix minus HPOA (and PrimeKG) on the sealed design E sets by HPOA lineage coverage and HPOA profile size (post hoc). Lineage Top-10 (%). Lineage coverage: the gold disease or an IS_A descendant (depth 3) has at least one HPOA finding edge. Profile size: maximum over the gold keys of the number of HPOA finding annotations (exact key), so the size-0 stratum also holds lineage-covered patients (DDXPlus-E 980, SymCat-E 1,847). The size boundaries were chosen after the development results and fixed before the design E measurement. Differences in percentage points: [patient-paired CI] [cluster CI]; disease-equal = mean over gold diseases of the per-disease difference, with a disease-resampling CI. Strata rows are not adjusted for multiplicity. n.a. = not applicable (one disease).

| Set; stratum | Patients (diseases) | HPOA / S-mix | S-mix − HPOA [patient CI] [cluster CI] | Disease-equal [CI] | S-mix − PrimeKG [cluster CI] |
|---|---|---|---|---|---|
| DDXPlus-E; all | 3,000 (49) | 10.37 / 41.07 | +30.70 [+28.70, +32.67] [+18.29, +44.86] | +34.50 [+22.53, +46.56] | +29.73 [+16.42, +44.73] |
| DDXPlus-E; HPOA lineage covered | 1,212 (21) | 25.66 / 47.36 | +21.70 [+18.40, +24.83] [+4.27, +41.47] | +28.75 [+9.51, +48.27] | +23.68 [+0.17, +49.75] |
| DDXPlus-E; not covered | 1,788 (28) | 0.00 / 36.80 | +36.80 [+34.62, +38.93] [+20.29, +54.13] | +38.81 [+23.82, +54.03] | +33.84 [+18.46, +50.06] |
| DDXPlus-E; profile 0 | 2,768 (43) | 8.67 / 41.00 | +32.33 [+30.46, +34.28] [+19.03, +47.35] | +36.85 [+24.70, +49.99] | +29.88 [+14.83, +45.33] |
| DDXPlus-E; profile 1–9 | 70 (1) | 25.71 / 67.14 | +41.43 [+30.00, +52.86] [n.a.] | n.a. | +40.00 [n.a.] |
| DDXPlus-E; profile 10–29 | 0 (0) | — | — | — | — |
| DDXPlus-E; profile ≥ 30 | 162 (5) | 32.72 / 30.86 | −1.85 [−10.49, +6.79] [−46.15, +36.22] | +12.88 [−32.38, +65.51] | +22.84 [−17.46, +74.02] |
| SymCat-E; all | 7,463 (760) | 3.78 / 12.92 | +9.14 [+8.36, +9.89] [+7.41, +10.88] | +9.08 [+7.50, +10.82] | +7.10 [+5.33, +8.88] |
| SymCat-E; HPOA lineage covered | 2,989 (321) | 9.07 / 15.89 | +6.83 [+5.42, +8.20] [+4.32, +9.59] | +6.74 [+4.22, +9.47] | +6.69 [+4.13, +9.43] |
| SymCat-E; not covered | 4,474 (439) | 0.25 / 10.93 | +10.68 [+9.81, +11.58] [+8.62, +12.85] | +10.80 [+8.59, +12.91] | +7.38 [+5.12, +9.89] |
| SymCat-E; profile 0 | 6,321 (637) | 3.83 / 13.54 | +9.71 [+8.88, +10.55] [+7.90, +11.53] | +9.73 [+7.96, +11.54] | +7.15 [+5.31, +9.06] |
| SymCat-E; profile 1–9 | 470 (52) | 5.74 / 15.11 | +9.36 [+5.74, +13.19] [+2.22, +16.95] | +9.13 [+2.36, +16.66] | +10.00 [+2.99, +17.77] |
| SymCat-E; profile 10–29 | 242 (26) | 2.48 / 11.16 | +8.68 [+4.55, +12.81] [+0.39, +19.51] | +8.03 [−0.09, +17.99] | +9.50 [+1.63, +20.00] |
| SymCat-E; profile ≥ 30 | 430 (45) | 1.63 / 2.33 | +0.70 [−0.93, +2.33] [−2.05, +3.78] | +0.49 [−2.17, +3.60] | +1.86 [−0.47, +4.95] |

### 4.4 LLM 없는 재구축과 비용(H3)

**구축 비용.** 표 10은 수집된 문헌에서 그래프 적재까지의 일회성 구축 비용을 보인다. 문헌 수집은 모든 경로에 같으므로 이 표에서 뺐다. S-mix의 추출부터 적재까지는 약 136시간이 걸렸다 [측정]. 그 대부분은 LLM 생성 127.4 GPU-시간이었다(RTX 4090 1장) [측정]. 합계는 모델 적재를 포함한 GPU 프로세스 시간 129.3시간에 구절 정규화 6.7 CPU-시간 이하, 구절 안 사전 재연결 3.5분, 그래프 빌드 9.2분을 더한 값이다 [측정]. 구절 정규화 시간은 상한이다. S-dict는 약 0.45시간이 걸렸다. 그중 그래프 빌드 20.6분과 v2 사전 매칭 2.2분은 실측이다 [측정]. v3 사전 매칭 3.9분은 1/20 표본 벤치마크로 낸 추정이다. v2 문헌의 LLM 추출은 v3보다 느린 실행 설정(eager 모드, 배치 64)으로 돌았다. v3 설정이었다면 LLM 생성 시간은 약 86.5시간이었을 것이다. 이 값은 v2 출력 토큰 수를 v3의 평균 처리 속도로 나눠 낸 추정이다. 추론은 모든 경로에서 LLM 없이 같은 Cypher 질의로 했다. 확증 실행에서 DDXPlus-E 환자당 순위화 시간은 S-mix 87 ms, S-dict 155 ms였다 [측정]. 간선이 S-mix의 2.8배인 S-dict가 환자당 더 오래 걸렸다 [측정].

**LLM 없는 사전 경로.** S-dict는 LLM 산출물과 완전히 독립적이지 않았다. CUI가 없는 사전 항목을 LLM 구절로 만든 구절 표로도 연결했기 때문이다(3.3절). 1/20 표본에서 사전 항목의 4.6%는 이 표로만 개념을 얻었다 [측정]. 전체 빌드에서는 근거 언급의 8.8%가 이 경로로 들어왔다 [측정]. 그래서 구절 표 없이 다시 빌드한 순수 사전 그래프를 평가했다(P0-2) [개발]. 순수 사전 그래프의 개발 매크로 Top-10은 42.38%였다 [개발]. S-dict(42.88%)와의 차이는 −0.50%p [−0.96, −0.07]였다 [개발]. 측정 전에 커밋한 P0-2 규칙(차이 1%p 안)을 충족했으므로, 비용 비교는 "LLM 없는 사전 경로"의 비용으로 쓴다(표 6b). 감소는 DDXPlus에 몰렸다. 순수 사전 그래프 − S-dict는 개발 DDXPlus에서 −2.73%p [−3.40, −2.00]였고 [개발], DDXPlus-E에서 −3.10%p [−3.77, −2.40]였다 [사후]. SymCat에서는 차이가 없었다. 개발 SymCat에서 −0.05%p [−0.32, +0.20]였고 [개발], SymCat-E에서 −0.08%p [−0.32, +0.16]였다 [사후]. 흔한 질환 두 세트의 매크로는 개발 −1.39%p [−1.78, −1.03]였다 [개발]. 확증 세트 E에서는 −1.59%p [−1.96, −1.24]였다 [사후]. 확증 세트 E에서 순수 사전 그래프의 Top-10은 DDXPlus-E 40.40%, SymCat-E 14.00%였다 [사후]. 두 세트 매크로의 S-mix − 순수 사전 그래프는 −0.21%p [−0.84, +0.45]였다 [사후]. 세트별로 보면 DDXPlus-E에서는 +0.67%p [−0.47, +1.87](군집 CI [−3.94, +5.16])로 차이가 없었다 [사후]. SymCat-E에서는 −1.09%p [−1.65, −0.48](군집 CI [−2.09, −0.09])로 순수 사전 그래프가 높았다 [사후].

**확증 판정과의 관계.** E-T2의 S-dict 비교는 미달이었다(표 6 행 9) [확증]. S-mix − S-dict 매크로 Top-10은 −1.80%p [−2.44, −1.12]로, lineage 판정에서는 사전 그래프가 높았다 [확증]. DDXPlus-E Top-1은 S-dict 17.63%, S-mix 12.43%였다 [확증-보조]. 이 비교를 판정 규칙과 입력 구조로 다시 읽은 결과는 4.6절에 둔다.

**단서.** 이 비용 비교의 단서는 다섯 가지다. 첫째, 문헌 수집 시간이 빠져 있다. v3 수집의 벽시계 시간은 약 25.2시간이었고, 그중 약 9시간은 중단이었다 [측정]. 수집의 CPU 시간은 기록하지 않았고, v2 수집 시간은 따로 떼어 낼 수 없다. 둘째, 사전 경로에도 UMLS 라이선스가 필요하다. 셋째, 수집 검색어와 주제 판정에 HPO genes_to_disease의 유전자–질환 연결을 쓴다(3.1절). 그래서 이 경로를 "공개 문헌만으로 만든다"고 쓰지 않는다. 넷째, 빌드 메모리는 관측한 순간값으로 17–20 GiB였다 [측정]. 다섯째, 순수 사전 그래프의 빌드 시간은 따로 재지 않았다. 빌더는 S-dict와 같다. 사전 경로의 정확도가 충분했다고도 쓰지 않는다. S-dict의 DDXPlus-E Top-1(17.63%)도 E-T4 기준인 18%보다 낮았다(기준을 사후에 적용) [확증-보조].

Table 10. One-time build cost and accuracy of the LLM-free and LLM paths. Cost from the collected literature to the loaded graph; literature collection is excluded because it is identical for every path (v3 collection about 25.2 h wall clock including an interruption of about 9 h; no CPU-time record; the v2 collection cannot be separated). Hardware: one NVIDIA RTX 4090 24 GB, Intel i7-13700K, WSL2. Status: cost columns are measurements, except the values marked as estimates in footnotes a, c and d; development macro values include selection optimism; design E macro values are confirmatory except for the pure dictionary graph (post hoc).

| System | GPU, generation / process (h) | CPU extraction steps | Graph build (min) | Extraction to load | LLM output used | HAS_FINDING edges | Ranking per patient, DDXPlus-E / SymCat-E (ms) | Dev macro Top-10 / Top-1 (%) | Design E macro Top-10 (%) |
|---|---|---|---|---|---|---|---|---|---|
| S-dict | 0 | UMLS dictionary ≈ 6.1 min^a^ | 20.6 | ≈ 0.45 h | Phrase table for 4.6% of dictionary items (8.8% of mentions)^b^ | 14,506,788 | 155.0 / 60.4 | 42.88 / 18.11 | 28.79 |
| Pure dictionary graph | 0 | UMLS dictionary ≈ 6.1 min^a^ | not timed (same builder) | not timed | None | 13,883,910 | not measured | 42.38 / 17.75 | 27.20 |
| S-llm | 127.4 / 129.3^c^ | Phrase normalization ≤ 6.7 h^d^ | 6.0 | ≈ 136.1 h | All | 3,672,106 | 62.3 / 26.4 | 44.90 / 19.30 | 25.35 |
| S-mix | 127.4 / 129.3^c^ | Normalization ≤ 6.7 h^d^ + relinking inside spans 3.5 min | 9.2 (50.1 with Mention nodes) | ≈ 136.2 h | All | 5,221,285 | 87.0 / 35.8 | 46.11 / 19.66 | 26.99 |
| S-union | 127.4 / 129.3^c^ | Normalization ≤ 6.7 h^d^ + UMLS dictionary ≈ 6.1 min | 26.6 | ≈ 136.5 h | All | 15,396,388 | 158.0 / 61.2 | 43.77 / 17.59 | 29.66 |

^a^ v2 2.2 min measured; v3 3.9 min estimated from a 1/20 benchmark.
^b^ 1/20 sample of literature files (items); the mention share is from the full builds with and without the phrase table.
^c^ v2 literature 66.86 h and v3 literature 60.57 h of generation. The v2 run used slower settings (eager mode, batch 64); with the v3 settings the generation total is estimated at about 86.5 h.
^d^ Upper bound; about 1.0 h for phrases inherited from an earlier table is estimated and not included.
Builder memory was observed only as instantaneous values (17.3 GiB for S-dict, 19.7 GiB for S-union). Ranking times are from the confirmation run logs; the pure dictionary graph was not timed under the same conditions.

<!-- 작성 메모(본문 아님):
1) [커밋 필요] 두 자리를 HTML 주석으로만 남겼다: 4.1의 프로필 두께 중앙값(SP adv/profile_size.py, 결과 파일 없음)과 4.3의 PUMCH_L 우세가 세 질환에 몰렸다는 문장(SP adv/pumch_clusters.json). 개요 0.3 #1을 처리한 뒤 넣는다. 자기 인용률 72.8%는 4.5절(RES-B) 몫이다.
2) PUMCH_L 질환 등가중 +1.51 [−7.60, +10.95]와 PUMCH-ADM +6.51 [−6.46, +20.26](군집 CI [−4.69, +26.67])은 커밋된 pilot/cache/v3/framing/m1/stats.json(rows, HPOA − S-mix, k 10)의 부호를 바꾼 값이다. m1_combine.py와 m2_curated_e.py는 모두 variants_analysis.gold_clusters(사상된 정답 키의 정렬 튜플)를 쓰므로 질환 키 정의가 같고, MET-B 3.8의 정의와도 같다. 원 정답 ID 튜플 기준 −0.9 [−9.5, +8.1]은 미커밋이라 쓰지 않았다.
3) 표 5b의 E 계보 커버리지는 HPOA·PrimeKG만 커밋되어 있다(framing/m2/stats.json coverage). S-mix·S-dict·순수 사전·S-D의 E 계보 커버리지는 계산하지 않았으므로 n.c.로 두었다.
4) S-bm25의 개발 세트별 값과 "9/9 세트 우세(부호검정 p = .0039)" 문장을 뺐다. 근거 파일 pilot/cache/v3/retrieval/dev_bm25.json(디렉터리 전체가 git 미추적)이 커밋되지 않았기 때문이다(framing review 5.1 #0). 남긴 값은 커밋된 매크로 27.35/11.04(frozen_config_E.json, v1_result.json dev_macro_v3)와 E4에 커밋한 개발 차이 +20.9 [+19.2, +22.7], +3.2 [+2.5, +4.0]뿐이다. DIS 5.3도 S-bm25 개발값 19.73·10.12를 인용하므로 같은 처리가 필요하다.
5) PPR-H4(표 6 행 6)는 점추정만 적었다. v3 표 7의 CI는 커밋된 결과 파일에서 출처를 찾지 못했다. 점추정과 "Holm 유의"(HPOA DDXPlus Top-1 +1.8 포함)는 docs/research_design.md D9 E-H4 확증 항목이 근거다. 가설 이름은 MET-B 3.6·3.8의 PPR-H1·PPR-H4로 맞췄다.
6) 순수 사전 그래프의 빌드 시간(개요 0.3 #4)과 수집 비용(0.3 #5, V5 미실행)은 측정하지 않았다. 표 10에 "not timed", 캡션에 수집 벽시계만 적었다. 86.5시간과 v3 사전 매칭 3.9분은 pilot/cache/v3/review/cost.md가 "추정"으로 적은 값이라 [측정]을 붙이지 않았다. 136.2시간 합계는 cost.md §3대로 GPU 프로세스 시간 129.3시간을 쓴 값이다.
7) 코퍼스 수치: 882,014·4,122·35,308·938은 커밋된 pilot/cache/v3/maint/generic/audit_v3.json, 32,545·1,119·136,373·2,663은 audit_v2.json, 17,654,170·6,518,369·43.0%는 pilot/cache/v3/maint/link_loss.json이 근거다. 34,370 = 35,308 − 938, 29,882 = 32,545 − 2,663이다. pilot/scripts/generic_doc_audit.py는 논문 수와 일반 문헌을 비인간 질환 파일까지 넣어 세고(42행), 문헌 있는 질환 수는 규칙 적용 전에 센다(58행). v3 새 연결 구성은 v3 근거 기록(draft/paper_ko_v3/evidence_log.md RES-A)의 값을 옮겼다.
8) DDXPlus 지식 베이스 문장은 원 논문(Fansi Tchango et al. 2022, arXiv 2205.09148) 3.1절 "constructed by compiling over 20,000 medical papers including epidemiological studies, disease specific articles and meta-analysis papers"와 대조했다. 참고문헌 항목에서 쪽수를 붙인다.
9) 개요 0.3 #8(E의 입력 명시 층으로 S-mix − HPOA 층화)은 하지 않았다. 본문에는 "나누어 보지는 않았다"만 썼고, 개발값 +29.1은 미커밋이라 쓰지 않았다.
10) 설계 E의 판정 기준은 5ce9c3d, E-T3·E-T5 예상은 61e3118·0cb1c59에서 적었고, 동결 커밋 49901bc(23:13:03)가 실행 시작(23:13) 직전이다. 설계 D의 목표는 3fbd29a(10-12 15:24)에 커밋했다. T3 기준값 53.6/53.3은 D4, 재판정 53.2/57.3은 D9가 근거다.
11) 방법 절 참조: 3.2(일반 문헌 규칙), 3.3(규칙 5, 순수 사전 정의), 3.4(무증상 실패), 3.6(입력 명시 층, PPR-H1·PPR-H4), 3.7(표 4 튜닝 예산, BM25 기준선, HPOA에서 β·γ가 효과 없음), 3.8(지위 표기, 군집 CI, 질환 등가중), 3.10(확증 이후 규칙, 희석 정의, P0-4 해석 규칙, M1·M2의 k).
12) P0-4 해석 규칙의 기준(Phenopacket-store 질환 등가중 HPOA − S-mix)은 커밋된 산출물에 없다(p0/hpoa_lopo/stats.json에 disease_equal 없음). 표 6b에 환자 가중으로 보고했음을 적었다.
13) M1은 Top-1에서 간선 합집합이 규칙을 충족한다(m1/stats.json macro, hospital, union − HPOA·union − S-mix, k 1). §9.35대로 보완의 근거로 쓰지 않고 표 6 각주 c에만 적었다. V1 Holm p 최댓값은 3.74 × 10⁻⁸(v3_joint_k1000_sum, SymCat-E)이다.

결정이 필요한 것(사용자):
(가) pilot/cache/v3/retrieval/dev_bm25.json을 커밋할지. 커밋하면 표 8a의 S-bm25 세트별 값과 "개발 9/9 우세" 문장을 되살릴 수 있고, DIS 5.3의 19.73·10.12도 그대로 둘 수 있다. 커밋하지 않으면 DIS 5.3도 고쳐야 한다.
(나) P0-4의 질환 등가중 값을 계산해 커밋할지. 하지 않으면 해석 규칙을 커밋한 기준이 아닌 환자 가중 값으로 읽었다는 단서를 RES-B 4.5와 DIS에 그대로 둔다.
(다) [커밋 필요] 두 문장(프로필 두께 중앙값, PUMCH_L 세 질환 집중)의 스크립트·결과를 커밋해 넣을지, 빼고 갈지.
(라) M1의 k 문제: Top-10 판정(보완 아님)을 유지할지. Top-1 합집합 통과는 각주에만 두었다.
(마) PPR-H4의 CI를 커밋된 파일로 다시 낼지, 점추정만 둘지.
-->
