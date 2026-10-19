# 선행연구: "일반 방법 A의 의료 감별진단 전이" 방향 (2026-10-12)

> 조사 에이전트(WebSearch) 결과 정리. 표의 일부 원 논문 URL은 에이전트가 직접 열지 못함(아래 "미확인"). 투고 직전 arXiv·Scholar 재검색 필요.

## 결론
- **HippoRAG/PPR을 환자 소견 → 질환 순위화에 적용한 연구는 찾지 못함.**
- 부분 선행(반드시 인용): PPR/RWR을 희귀질환에 쓴 연구는 있으나 진단 순위화 자체가 아님.
  - RareBench(KDD 2024): 큐레이션 표현형–질환 그래프에서 IC 가중 Node2vec random walk → few-shot 예시 검색용 환자 임베딩. https://arxiv.org/html/2402.06341v2
  - MultiXrank RWR(medRxiv 2025): 다층망, 목적은 새 표현형 발굴. https://www.medrxiv.org/content/10.1101/2025.04.02.25325098v1.full.pdf
  - 유전자 우선순위화: PhenoRank https://pubmed.ncbi.nlm.nih.gov/29360927, BiRW https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4416812/
- KG 기반 진단은 다수지만 거의 모두 추론 시 LLM 사용: MedRAG https://arxiv.org/abs/2502.04413, LogosKG https://pmc.ncbi.nlm.nih.gov/articles/PMC12870703/,
  RareSeek-R1 https://arxiv.org/pdf/2511.14638, RareDx https://arxiv.org/abs/2609.35549, GraphRareBench https://arxiv.org/abs/2607.24878.
- 의료 QA의 KG-RAG는 이미 붐빔: MIRAGE https://arxiv.org/abs/2402.13178, MedGraphRAG https://aclanthology.org/2025.acl-long.1381.pdf.
- 일반 도메인 경고: GraphRAG-Bench — 그래프 방법이 single-hop에서 vanilla RAG보다 못한 경우가 많음. https://arxiv.org/abs/2506.05690
- **빈자리**: 문헌 OpenIE KG(문장 단위 출처) + LLM 없는 PPR + 43k 열린 집합 + 같은 코퍼스 BM25/MedCPT 대조 + 흔한·희귀 9개 벤치마크.

## 후보 비교
| A | 일반 근거 | 진단 적용 선행 | 비용 | 시험되는 질문 |
|---|---|---|---|---|
| HippoRAG/PPR | https://arxiv.org/abs/2405.14831, HippoRAG 2 https://arxiv.org/abs/2502.14802 | 같은 형태 없음(부분 선행 위) | 매우 낮음(새 IE 불필요) | 소견 간 다중 홉(동의어·계층·동반 소견)이 입도 불일치를 메우는가, 허브 질환 쏠림을 키우는가 |
| GraphRAG 요약 | https://arxiv.org/abs/2404.16130 (미확인) | LLM 결합형만 | 높음 | — |
| PRF(Rocchio/RM3) | IR 교과서 | 질환 순위화 적용 못 찾음 | 낮음 | 결측 소견 보충이 커버리지 교란을 줄이는가 |
| SPLADE/ColBERT | (미확인) | 진단 순위화 적용 못 찾음 | 중간 | 학습형 어휘 확장 vs UMLS 정규화 |
| RRF/결합 | — | 있음(2026) | 매우 낮음 | 새로움 낮음 |

## 결정 (2026-10-12)
1순위 HippoRAG식 PPR 전이 연구로 진행. 주장은 "PPR 최초"가 아니라 **"HippoRAG의 이점이 의료 열린 집합 감별진단으로 전이되는가"**로 한정하고 위 부분 선행을 인용.
대안: PRF, SPLADE/ColBERT.

## 미확인
GraphRAG·LightRAG·ToG·SPLADE·ColBERTv2·RRF 원 논문 URL, MultiXrank 본문(403), medRxiv "RAG vs GraphRAG complex clinical cases"(403), RareDx 벤치 구성, MedGraphRAG의 진단 과제 포함 여부.
