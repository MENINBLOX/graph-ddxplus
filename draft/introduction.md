# Introduction

> **잠정(provisional) 작업 초안.** 연구 정체성이 확정되기 전 논증을 압박-테스트하기 위한 초안이며, Method·Results가 안정된 뒤 최종 확정한다. 프레이밍은 순수 감별진단 논문과 향후 자동 문진(interactive history-taking) 응용 **양쪽을 닫지 않도록** 유지한다(§5). 문장별 주장은 검증된 출처에 근거하며, 미구현 기능(간선 단위 provenance materialization)에 의존하는 주장은 설계 수준으로만 진술한다.

## 1. 배경: 감별진단과 그 오류의 부담

감별진단(differential diagnosis)은 환자의 소견에서 가능한 질환을 열거하고 순위화하는 임상 추론의 핵심 단계다. 이 단계의 실패는 측정 가능한 대규모 위해로 이어진다. 미국 국립학술원은 진단 오류를 "환자의 건강 문제에 대한 정확하고 시의적절한 설명을 확립하거나 전달하지 못하는 것"으로 정의하고, 매년 외래 성인의 약 5%가 진단 오류를 경험하며 대부분의 사람이 일생에 최소 한 번 이를 겪을 것으로 추정하였다[1]. 후속 역학 추정은 미국에서 매년 약 795,000명이 진단 오류로 영구적 장애 또는 사망에 이른다고 보고하였다[2]. 부담은 희귀질환에서 특히 두드러진다. 희귀질환 환자는 평균 수년에 걸쳐 다수의 임상의를 거치고 여러 차례의 오진을 경험하는 이른바 "진단 여정(diagnostic odyssey)"을 겪는다[3]. 따라서 감별진단을 지원하는 계산적 방법은 분명한 임상적 동기를 갖는다.

## 2. 현행 계산적 접근과 그 제약

자동 감별진단 방법은 대체로 세 패러다임으로 나뉘며, 각 패러다임은 임상 배치(deployment)와 신뢰(trust)의 관점에서 서로 다른 제약을 갖는다. 첫째, 지도학습 모델은 벤치마크의 레이블 분포에 맞추어 학습되어 높은 정확도를 보이나[4], 그 분포와 disease–symptom 레이블에 종속되므로 다른 벤치마크·인구집단으로의 전이가 제한되고 학습된 연관을 출처로 되짚기 어렵다. 둘째, agentic 대형언어모델(LLM) 시스템은 도구 사용·검색·반복 질의로 강한 정확도와 일부 추론 흔적을 제공하나[5][6], 추론 시점에 LLM을 호출하므로 비용·비결정성·재현성의 부담이 있고, 생성된 "추론"이 출처 문헌에 근거(source-grounded)한다는 보장이 없다. 셋째, 추론 시점에 LLM을 zero-shot으로 사용하는 방법은 학습이 불필요하나 희귀질환에서 성능이 급락하고 환각과 출처 부재의 문제를 공유한다[7]. 주목할 점은 최신 agentic 시스템조차 "추적 가능한 추론(traceable reasoning)"을 핵심 강점으로 내세운다는 사실로[5], 이는 진단 AI에서 추적성이 이미 인정된 요구임을 방증한다.

## 3. 문제 정의: 무엇이 빠져 있는가

이 관점에서, 다음 네 성질을 **동시에** 만족하는 방법은 아직 정립되지 않았다. (i) 벤치마크별 학습 없이 이종 벤치마크를 가로질러 작동하고(transfer), (ii) 모든 주장을 인용 가능한 출처에 근거시키며(traceable), (iii) 독자적 스키마를 새로 만들지 않고 기존 임상 표준 용어체계를 따르고(standards-grounded), (iv) 추론 시점에 LLM을 호출하지 않아 저비용·결정론적(deterministic)인 방법이다. 본 연구는 이 공백을 겨냥한다.

## 4. 접근: 왜 agent가 아니라 KG인가

본 연구는 지식 표현·추론의 기질(substrate)로 단일 지식 그래프(knowledge graph, KG)를 택한다. LLM은 KG 구축(정보추출) 시점에만 사용하며, 이는 사람이 교과서를 읽어 지식을 정리하는 것과 동일한 인식론적 위상을 갖는다. 진단 시점에는 KG와 단일 결정론적 알고리즘만 동작하고 LLM을 호출하지 않는다. agent가 아니라 KG를 택한 이유는 네 가지다. 첫째, **결정론성·재현성**: 동일 입력에 동일 출력이 보장되어 감사(audit)와 규제 검토에 부합한다. 둘째, **추적성**: 모든 disease–phenotype 간선은 정보추출이 읽은 특정 출처 문헌에서 유래하므로, 순위의 근거를 "어떤 주장이 어떤 문헌에서 왔는지"까지 원리적으로 되짚을 수 있다 — 이는 추론을 모델 내부에서 생성하는 agent가 구조적으로 제공하기 어려운 성질이다. 셋째, **비용**: 지식을 한 번 구축하면 추론은 그래프 연산만으로 수행되어 로컬 소형(약 8B) 모델과 단일 GPU로 충분하다. 넷째, **검사가능성**: 지식이 그래프로 노출되어 임상 전문가가 직접 점검·교정할 수 있다.

## 5. 왜 속성(attribute)인가: cross-benchmark interlingua

본 연구의 과학적 핵심은 또 하나의 KG 진단기를 더하는 것이 아니라, 증상을 임상 표준 속성으로 특성화하여 이를 벤치마크 간 상호언어(interlingua)로 삼는 데 있다. 병력청취(history taking)에서 임상의는 각 증상을 위치·성상·정도·시간·악화/완화 인자 등 표준 축으로 특성화한다. 본 연구는 이 축을 증상 특성화의 교과서 표준인 Bates의 seven attributes[8]에 앵커하여 정보추출 단계에서 추출하고, UMLS·HPO·SNOMED 표준 개념으로 정규화하여[9][10][11] disease–phenotype 간선의 수식 진술(qualified statement)로 부착한다. 이 부착 방식은 속성을 별도 노드로 승격하지 않는 GA4GH Phenopacket 스키마와 정합한다[12]. 표준으로 정규화된 속성은 서로 다른 인코딩을 쓰는 이종 벤치마크를 가로질러 동일한 의미 단위로 정합될 수 있어, 단일 KG가 다수 벤치마크로 전이되는 통로가 된다. 속성이 곧 병력청취가 이끌어내는 정보라는 점에서, 이 표현은 향후 자동 문진(interactive history-taking) 응용의 자연스러운 기반이기도 하다. 다만 공개 벤치마크가 속성을 질의 측면에서 분리 가능한 형태로 제공하는 경우는 제한적이므로, 본 연구는 속성의 인과적 기여를 그러한 분리가 가능한 무대에서 통제 실험으로 검증하고 나머지는 한계로 명시한다(Method §4).

## 6. 왜 감별진단과 최종진단을 함께 보는가

본 연구는 후보 질환의 순위(감별진단)와 최상위 질환(최종진단)을 함께 보고한다. 순위는 각 후보에 근거와 출처를 부착한 형태로 제시되어, 결과물이 단일 정답이 아니라 근거가 추적되는 후보 목록이라는 본 접근의 성격을 반영한다. 최상위 순위 정확도(GTPA@1)는 외부 방법 대비 우리의 위치를 정직하게 보고하기 위한 지표로 병기하되, 세팅이 다른 방법(지도학습·agentic)과의 비교는 방법론적 상한으로만 인용한다(Method §5.3). 본 연구는 결과물의 성격(근거-추적 후보 목록)을 기술하는 데 그치며, 특정 임상 워크플로에서의 역할을 규정하지 않는다.

## 7. 기여

본 연구의 기여는 세 축이며, 절대 정확도의 SOTA를 주장하지 않는다.

1. **표준-정규화 속성 정보추출**: 로컬 소형 LLM으로 질환 텍스트에서 finding과 임상 표준 속성(Bates seven attributes 앵커)을 추출하고 UMLS/HPO/SNOMED로 정규화한다(Method §2).
2. **속성 = cross-benchmark interlingua**: 표준으로 정규화된 속성이 이종 벤치마크를 가로지르는 상호언어로 작동함을 통제 ablation으로 인과적으로 검증한다(Method §4, §6.1).
3. **cross-benchmark robustness + traceable provenance + 저비용**: 단일 KG와 단일 알고리즘(벤치마크별 튜닝 없음)으로 이종 벤치마크를 가로질러 작동함을 보이고, 추론시 LLM 없이 로컬 8B·단일 GPU로 수행한다(Method §3, §5, §6.2).

세팅이 동일한(추론시 LLM 미사용) 비교에서 본 방법은 지식기반 최고 성능과 경쟁하거나 이를 상회하며, 남은 절대 정확도 격차는 본 연구가 명시적으로 제약으로 배제한 다른 패러다임(지도학습·agentic)에서 비롯된다. 즉 본 연구의 강점은 최고 점수가 아니라 **전이·추적성·비용**에 있다.

---

## References

*새로 검증한 출처(intro 전용)와 Method 초안과 공유하는 출처를 함께 표기한다. 최종 통합 시 번호를 단일화한다.*

[1] National Academies of Sciences, Engineering, and Medicine. *Improving Diagnosis in Health Care.* Washington, DC: The National Academies Press; 2015. doi:10.17226/21794. PMID: 26803862.
-> 인용 이유: 감별진단 오류가 대규모·측정된 공중보건 문제임을 정의·정량화(외래 성인 ≈5%/년, 일생 최소 1회)한 표준 근거. (2026-07 NAP/PubMed 대조 확인.)

[2] Newman-Toker DE, Nassery N, Schaffer AC, et al. Burden of serious harms from diagnostic error in the USA. *BMJ Qual Saf.* 2024;33(2):109–120. doi:10.1136/bmjqs-2021-014130. PMID: 37460118.
-> 인용 이유: 진단 오류의 연간 심각 위해 규모(≈795,000명 영구장애/사망) 근거. (2026-07 PubMed 원문 대조로 서지·DOI 확인.)

[3] Marwaha S, Knowles JW, Ashley EA. A guide for the diagnosis of rare and undiagnosed disease: beyond the exome. *Genome Med.* 2022;14(1):23. doi:10.1186/s13073-022-01026-w. PMID: 35220969.
-> 인용 이유: 희귀질환 진단여정(수년·다수 임상의·2–3회 오진) 근거 = 희귀질환 포함 scope 확장의 임상적 동기. (2026-07 Genome Medicine 대조 확인.)

[4] Ghosh S, et al. DDxT: Deep Generative Transformer Models for Differential Diagnosis. NeurIPS 2023 DGM4H workshop; *arXiv:2312.01242.* (Method §16과 공유.)
-> 인용 이유: 지도학습 감별진단 패러다임(벤치마크 분포 종속·전이 제약)의 대표 사례.

[5] Zhao et al. An agentic system for rare disease diagnosis with traceable reasoning (DeepRare). *Nature* (2025); doi:10.1038/s41586-025-10097-9. (Method §14와 공유.)
-> 인용 이유: agentic LLM 패러다임의 최신 사례이자, agentic 시스템조차 "traceable reasoning"을 강점으로 내세운다는 근거(추적성이 인정된 요구임을 방증).

[6] Rose et al. MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis. ACL 2025; *arXiv:2502.19175.* (Method §18과 공유.)
-> 인용 이유: agentic 감별진단 프레임워크 사례(추론시 LLM·도구 반복의 비용·비결정성 제약).

[7] Chen et al. RareBench: Can LLMs Serve as Rare Diseases Specialists? KDD 2024; *arXiv:2402.06341.* (Method §19와 공유.)
-> 인용 이유: 추론시 LLM(zero-shot)이 희귀질환에서 급락함을 보인 벤치마크 근거이자 본 연구 평가 대상.

[8] Bickley LS, Szilagyi PG. *Bates' Pocket Guide to Physical Examination and History Taking.* 7th ed. Wolters Kluwer/Lippincott; 2013. Ch.3, "The Seven Attributes of a Symptom", p.38. (Method §1과 공유.)
-> 인용 이유: 속성 집합을 임의 선택이 아니라 증상 특성화의 교과서 표준에 앵커한 근거.

[9] Bodenreider O. The Unified Medical Language System (UMLS). *Nucleic Acids Res.* 2004;32(D):D267–D270. (Method §5와 공유.)
-> 인용 이유: 개념형 속성·finding을 표준 용어체계(UMLS)로 정규화하는 근거.

[10] Gargano MA, et al. The Human Phenotype Ontology in 2024. *Nucleic Acids Res.* 2024;52(D1):D1333–D1346. (Method §4와 공유.)
-> 인용 이유: severity·timing·aggravating·relieving를 HPO 표준 term으로 정규화하는 근거.

[11] SNOMED International. *SNOMED CT Editorial Guide — Body Structure* (123037004). (Method §6와 공유.)
-> 인용 이유: location(해부학적 부위)을 SNOMED Body structure로 정규화하는 근거.

[12] Jacobsen JOB, et al. The GA4GH Phenopacket schema. *Nat Biotechnol.* 2022;40(6):817–820. (Method §2와 공유.)
-> 인용 이유: 속성을 별도 노드가 아니라 phenotype 부착 수식자로 두는 표현 방식의 국제 표준 선례.

[13] Fansi Tchango A, et al. DDXPlus: A New Dataset for Automatic Medical Diagnosis. NeurIPS 2022 D&B; *arXiv:2205.09148.* (Method §20과 공유.)
-> 인용 이유: 흔한 질환 감별진단 벤치마크(본 연구 평가 대상이자 최초 동기).
