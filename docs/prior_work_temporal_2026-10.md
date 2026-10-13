# 선행연구: 진단 지식원의 시간 노후화 (2026-10-13)

> 조사 에이전트(WebSearch) 결과. "미확인"은 원문을 열지 못한 항목 — 투고 전 재확인.

## 결론
문헌 KG·큐레이션 HPOA·PrimeKG·LLM을 phenopacket-store 시간축에 놓고 cutoff 이후 증례에서의 성능 감소를 비교한 연구, 그리고 phenopacket-store 신규 배포로 갱신하는 희귀질환 living benchmark는 찾지 못함.

## 가장 가까운 선행
- Jiang et al. 2026, "Learning to Fuse LLMs with Ontology Rankers for Rare-Disease Diagnosis" https://arxiv.org/html/2609.02473v1
  - 0.1.27에서 증례 논문 = HPOA 근거 논문 74.6%, 제거 시 Phenomizer R@1 0.448→0.122.
  - 분할은 출판물 단위이고 시간 분할이 아님.
- LiveMedBench https://arxiv.org/abs/2602.10367
  - 매주 갱신하며, 38개 LLM 중 84%가 cutoff 이후 성능 하락.
  - 희귀질환·KG 비교는 없음.

## 축별
- **표현형 기반 진단의 시간 분할**: 정량 연구 없음.
  - AMELIE(2011 KB 제한 점검, 본문 미확인) https://www.biorxiv.org/content/10.1101/171322.full.pdf
  - Jacobsen 2022(KB 유지보수에 대한 정성적 경고, 미확인) https://pmc.ncbi.nlm.nih.gov/articles/PMC9487604/
  - arXiv 2606.16149(Orphanet 미매핑을 신규 질환의 대리지표로 사용) https://arxiv.org/pdf/2606.16149
- **KG 시간 분할**: 모두 링크 예측이고 진단 과제가 아님.
  - PrimeKG-CL https://arxiv.org/abs/2605.10529
  - 약리 KG https://arxiv.org/html/2603.01537v1
  - COVID KG https://arxiv.org/pdf/2010.09600
  - HPO 주석 예측 https://pmc.ncbi.nlm.nih.gov/articles/PMC9590544
- **LLM 희귀질환과 오염**
  - Reese(0.1.14, Eur J Hum Genet 2026) https://escholarship.org/content/qt64v8j4b5/qt64v8j4b5.pdf
  - RareArena(cutoff 전후 분석 여부 미확인)
  - DiagnosisArena https://arxiv.org/pdf/2505.14107
  - 메타분석(누설 위험 고위험) https://www.nature.com/articles/s41746-026-03109-z
- **의료 living/시간 벤치**: 모두 일반 임상이나 가이드라인 QA.
  - LiveClin https://arxiv.org/abs/2602.16747
  - TempoMed-Bench(LLM 지식이 선형적으로 서서히 감소) https://arxiv.org/pdf/2605.13045
  - DriftMedQA https://arxiv.org/abs/2505.07968
- **phenopacket-store를 벤치마크로 사용**: 모두 단일 스냅샷.
  - Danis 2025 https://pmc.ncbi.nlm.nih.gov/articles/PMC11160806/
  - PhEval https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11195176/

## 필요한 차별화
1. 지식원 유형 간 노후화 곡선 비교.
2. 증례 시점(오염)과 질환 등재 시점(공백)을 분리.
3. Jiang의 출처 누설을 시간으로 일반화.
4. TempoMed의 "완만한 감소"와 큐레이션의 "계단형"을 대비.
5. living benchmark는 희귀질환·HPO 구조화 입력·출처 기록으로 구분.
