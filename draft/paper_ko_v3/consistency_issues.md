# 절 간 일관성 검토 (v3)

## 1. [imprecise] ABS-CON

- 인용: 개발 세트의 Top-10에는 수집 깊이가 v2 문헌 위의 추출·연결 변경(동결 그래프 대비 최대 +4.1%p)보다 크게 작용했다.
- 문제: '추출·연결 변경'이라는 말이 두 가지 뜻으로 쓰인다. 최대 +4.1%p를 낸 그래프는 v2unionu_ns(34.66 − 30.56)다. 이 그래프는 추출·연결뿐 아니라 빌더 규칙(등위 분해, HPO 이름 연결, 원인 유전자 제목 주제 인정, 자기 간선 제거)도 S-D와 다르다. MET-B 3.6은 빌더 규칙을 수집 깊이·추출·연결과 별개의 축으로 나눈다. 같은 표현이 INT 기여 4('추출·연결 방식만 바꾼 변형의 최대 효과(동결 대비 +4.1%p)'), RES-A 4.3, DIS에도 반복된다.
- 수정: 'v2 문헌 위에서 추출·연결 방식이나 빌더 규칙을 바꾼 변형(동결 그래프 대비 최대 +4.1%p, v2unionu_ns)'처럼 빌더 규칙 변경까지 포함됨을 밝힌다.
- 근거: pilot/cache/v2/devD/variants/analysis.json dev: v2unionu_ns 34.66 / base_nm 30.56; pilot/scripts/run_graph_variants.sh: base_nm FLAGS="", v2unionu는 ALL(--split-coord --hpo-names --topic-genes)에 _ns로 --no-self 추가; MET-B 3.6의 32개 구성 서술

## 2. [imprecise] RES-B

- 인용: 표 10은 v2 문헌에서 추출·연결 방식만 바꾼 그래프 18개(기준 1개와 변형 17개)를 보여 준다.
- 문제: 표 10의 18개 가운데 한정어 제외(_nq) 4개, 자기 간선 제거(_ns) 7개, base_split_tg(원인 유전자 제목 주제 인정)는 추출이나 연결이 아니라 빌더 규칙을 바꾼 변형이다. 이를 '추출·연결 방식만 바꾼'이라고 쓰면 MET-B 3.6의 정의(수집 깊이, 추출·연결 방식, 빌더 규칙 중 하나 이상을 바꾼 그래프)와 어긋난다.
- 수정: 'v2 문헌에서 추출·연결 방식이나 빌더 규칙을 바꾼 그래프 18개'
- 근거: 표 10의 '_nq', '_ns' 행; pilot/scripts/run_graph_variants.sh의 접미어 정의(_ns → --no-self, _nq → --no-qualifiers); MET-B 3.6

## 3. [imprecise] ABS-CON

- 인용: 이 빌드의 개발 DDXPlus 3,000명 순위는 근거 언급 없이 적재한 빌드와 같았다.
- 문제: 실제로 비교한 것은 전체 후보 순위가 아니다. graph_dx.evaluate가 기록하는 ranks_lineage만 비교했다. 이 값은 환자마다 lineage 정답이 Top-10 안 몇 번째에 있는지이고, Top-10 밖이면 None이다. Top-10 아래 순위와 정답이 아닌 질환의 순서는 비교하지 않았다. 같은 표현이 INT, MET-A 3.4, MET-B 3.6, RES-A 4.1, RES-B 4.7, DIS 5.1에도 있다. 지문이 같으므로 결론은 맞지만, 서술이 검사 범위보다 넓다.
- 수정: '개발 DDXPlus 3,000명 모두에서 정답 질환의 Top-10 내 위치(lineage)가 근거 언급 없는 빌드와 같았다'
- 근거: pilot/cache/v2/devD/e/dev_ddx_smix_mentions.json과 variants/dev_v3llmd_ns_ot0.5.json의 ranks_lineage 3,000/3,000 일치; pilot/scripts/graph_dx.py의 evaluate()는 ranked = dx.rank(ev, max(KS))에서 정답 위치만 저장한다

## 4. [imprecise] MET-A

- 인용: 설계 E는 모든 그래프 시스템을 확증 실행 직전에 같은 비교 인스턴스에 다시 빌드하고, 지문과 적재된 HAS_FINDING·Disease 수가 동결 커밋의 빌드 요약과 같을 때만 평가하도록 정했다.
- 문제: 코드의 동작과 다르고, MET-B 3.6의 서술과도 다르다. run_confirm_E.sh는 빌드 요약의 지문과 적재된 HAS_FINDING 수만 frozen_config_E.json과 대조한다. Disease 수는 동결 요약과 대조하지 않는다. 빌더가 후보 질환 수(universe)와 다르면 멈출 뿐이다. MET-B 3.6은 이를 정확히 적었다('…HAS_FINDING 수가 동결 기록과 같을 때만 평가한다. Disease 노드 수가 후보 질환 수(43,575)와 다르면 빌더가 멈춘다').
- 수정: '지문과 적재된 HAS_FINDING 수가 동결 기록과 같을 때만 평가하고, Disease 노드 수가 후보 질환 수와 다르면 빌더가 멈춘다'
- 근거: pilot/scripts/run_confirm_E.sh의 load()는 fingerprint와 loaded_edges(HAS_FINDING 개수)만 비교한다; build_graph_v2.py는 n_dz != len(uni)이면 sys.exit

## 5. [imprecise] MET-B

- 인용: - E-T2(주): S-mix − S-llm, S-mix − S-dict, S-mix − S-union 각각의 매크로 Top-10 차이에서 95% CI 하한 > −1%p(비열등). 세 비교는 따로 판정한다.
- 문제: '주'라는 말이 두 가지 뜻으로 쓰인다. 목록에서는 E-T2의 세 비교 전체를 '(주)'로 표시한다. 그런데 바로 아래 문단('주 목표는 E-T1과 E-T2(S-llm 비교)이고, 나머지는 보고용 부 목표다')과 표 15b('E-T2 (primary)'는 S-llm 행에만 표시)는 S-llm 비교만 주 목표로 둔다. S-dict·S-union 비교가 주 목표인지 부 목표인지 독자가 판단하기 어렵다.
- 수정: 'E-T2(S-llm 비교는 주, S-dict·S-union 비교는 부)'처럼 목록에서부터 구분한다.
- 근거: MET-B 3.6 E-T2 항목과 E0 문단; RES-B 표 15b; docs/research_design.md E4(같은 이중 표기)

## 6. [imprecise] DIS

- 인용: 일반 문헌 규칙(20질환 이상에 나열된 문헌 제외)
- 문제: MET-A 3.2의 정의와 다르다. MET-A는 '질환별 수집 목록에 모두 20회 이상 나타난 논문(한 질환 목록에 두 번 나오면 2회로 센다)'으로 정의하고, 표 문서는 원 논문과 함께 센다. RES-B 4.4.6과 4.5.1도 '질환 파일에 20회 이상 나열'로 쓴다. 코드는 나열 횟수를 세며, 비인간 질환 파일까지 포함한다. 따라서 '20질환 이상'은 이 규칙을 다르게 기술한 것이다.
- 수정: '일반 문헌 규칙(질환 파일에 모두 20회 이상 나열된 논문 제외)'
- 근거: pilot/scripts/build_graph_v2.py: use[article(doc_id)] += 1이 질환 파일마다 문서 다중도대로 더해지고 generic = n >= 20; MET-A 3.2; RES-A 표 2 각주 e

## 7. [imprecise] RES-B

- 인용: 큐레이션 대조에서는 HPO 주석(phenotype.hpoa, 2026-02-16)이 있는 질환 12,296개의 비교 가능한 간선 194,318개를 큐레이터 표현형과 대조했다
- 문제: 같은 HPOA 2026-02-16판의 질환 수가 MET-B 3.7과 표 5에서는 '소견이 있는 질환 12,251개'로 나온다. 차이 45개가 어디서 오는지(개념 사상이 안 되는 질환인지, 다른 질환 집합인지) 원고 어디에도 설명이 없다. 독자에게는 한 수량이 두 값으로 보인다.
- 수정: 두 수의 정의를 각각 밝힌다. 예: '주석이 있는 질환 12,296개(그중 개념으로 사상되는 소견이 있는 질환 12,251개)'. 아니면 한쪽 기준으로 통일한다.
- 근거: docs/research_design.md §9.16(12,296), §9.22(12,251); MET-B 3.7과 표 5

## 8. [imprecise] DIS

- 인용: 확증에서 gemma 그래프의 PPR은 1-홉 집계보다 Top-10이 DDXPlus에서 15.8%p, SymCat에서 1.6%p 낮았다(Holm 유의).
- 문제: DIS는 S-D를 'gemma 그래프'라고 부른다(PPR 문단 세 곳, 큐레이션 비교 문단의 'D 확증에서 gemma 그래프의 Top-10은 DDXPlus 35.2%'). 그런데 S-mix, S-llm, S-union도 gemma-4-E4B로 추출한 그래프다. 다른 절(RES-A 4.2, 표 6·7)은 같은 시스템을 S-D로 쓴다. 독자가 v2+v3 gemma 그래프의 결과로 오인할 수 있다.
- 수정: 'S-D 그래프(v2 문헌 gemma 그래프)'로 고친다.
- 근거: RES-A 4.2와 표 6('Literature graph S-D'); MET-B 3.6의 시스템 정의(S-mix·S-llm도 gemma 구절)

## 9. [imprecise] DIS

- 인용: 미래·기주석 증례에서 HPOA 2022-12 대비 Top-10 격차는 UMLS 사전이 −15.6%p에서 −8.1%p로, LLM이 −16.1%p에서 −7.3%p로 줄었고
- 문제: UMLS 쌍(v2umls_ns → v3umls_ns)은 빌더 규칙이 같다. LLM 쌍은 v2 기준이 base_split_tg(자기 간선 유지, v2 구절 표)이고 v3가 v3llm_ns(자기 간선 제거, v3 구절 표)여서 빌더 규칙과 구절 표가 함께 바뀐다. RES-B 4.5는 이 차이를 밝혔지만, DIS는 두 쌍을 같은 종류의 수집 깊이 효과로 나란히 제시한다.
- 수정: LLM 쌍에 '(v2 기준은 자기 간선을 두고 v2 구절 표를 쓴 그래프)'를 덧붙이거나, 깊이 효과의 근거로 UMLS 쌍만 쓴다.
- 근거: analysis.json temporal: base_split_tg_y2022_ot0.0 −16.1, v3llm_ns_y2022_ot0.0 −7.3; base_split_tg_ns_y2022 빌드는 없음(variants 디렉터리); RES-B 4.5

## 10. [imprecise] INT

- 인용: 자기 간선은 질환에서 자기 개념이나 상위 질환 개념으로 가는 소견 간선이다.
- 문제: 빌더(--no-self)는 질환 자신의 CUI와 질환 IS_A의 직접 상위 질환만 제거한다. MET-A 3.4는 '직접 상위 질환의 개념'으로 정확히 쓴다. 반면 INT와 RES-B 4.4.4('자기 개념과 상위 질환 개념으로 가는 간선')는 '직접'을 빠뜨려 모든 조상이 제거 대상으로 읽힌다.
- 수정: '질환에서 자기 개념이나 직접 상위 질환의 개념으로 가는 소견 간선'
- 근거: pilot/scripts/build_graph_v2.py: self_cuis[c].add(p) for c, p in v2_ids.disease_isa()(직접 부모 쌍)와 self_cuis[k].add(k); MET-A 3.4

## 11. [style] RES-A

- 인용: 사람 질환의 질환–문헌 연결은 246,203건, 고유 문헌은 123,772편이었다.
- 문제: MET-A 3.2는 문서(출처 단위), 논문(원 논문), 문헌(수집 문서 전체)을 구분해 정의한다. 그런데 4.1의 v2 단락은 '문헌'을 셀 수 있는 단위로 쓴다('고유 문헌 136,373편', '일반 문헌 1,119편'). v3 단락은 같은 수량을 '문서 ID', '논문 단위'로 센다. 표 2도 'Unique documents'와 'Generic articles'로 단위를 나눈다. 같은 절 안에서 집계 단위가 섞인다.
- 수정: v2 수치도 정의에 맞춰 '고유 문서 123,772개', '일반 문헌 규칙으로 제외된 논문 1,119편'처럼 단위를 맞춘다.
- 근거: MET-A 3.2 첫 문단의 용어 정의; RES-A 표 2 행 이름('Unique documents', 'Generic articles excluded')
