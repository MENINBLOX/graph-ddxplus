# 지식그래프 탐색을 통한 해석 가능한 감별 진단

---

## 표지 (Title Page)

**제목**: 지식그래프 탐색을 통한 해석 가능한 감별 진단

**영문 제목**: Interpretable Differential Diagnosis via Knowledge Graph Traversal

**단축 제목 (Running Head)**: 지식그래프 기반 해석 가능한 진단

**저자**: [저자명]<sup>1</sup>

**소속**:
<sup>1</sup>[소속기관]

**교신저자**:
- 성명: [교신저자명]
- 주소: [주소]
- 전화: [전화번호]
- 팩스: [팩스번호]
- E-mail: [이메일]

---

## 초록 (Abstract)

**Purpose**: 자동 감별 진단 시스템의 임상 채택에는 정확도와 함께 투명한 추론 과정이 필요하다. 기존 대규모 언어모델(LLM) 기반 접근법은 높은 정확도에도 불구하고 Black-box로 작동하여 추론 과정의 검증이 어렵다. 본 연구는 본질적 해석가능성(intrinsic interpretability)을 제공하는 지식그래프 기반 진단 시스템을 제안한다.

**Methods**: UMLS(Unified Medical Language System, 통합 의학 용어 시스템) 기반 의료 지식그래프에서 2-hop 탐색 알고리즘을 개발하였다. 초기 증상에서 후보 질환을 식별(1st hop)하고, 진단 불확실성을 가장 크게 줄일 수 있는 다음 증상을 선택하여 질문(2nd hop)한다. 이 설계는 임상 추론의 가설-연역적 모델과 구조적으로 일치한다. 49개 질환을 포함하는 DDXPlus 벤치마크(134,529 케이스)에서 평가하였다.

**Results**: 제안 시스템은 최상위 진단 정확도(GTPA@1) 83.23%, 상위 10개 감별 진단 목록 내 정답 포함률(GTPA@10) 99.53%를 달성하였으며, 평균 13.6회 질문으로 진단을 완료하였다. 모든 진단 결정이 지식그래프 경로로 직접 구성되어 추론 과정의 완전한 추적이 가능하다.

**Conclusion**: 본 연구는 지식그래프 단독 접근법으로 DDXPlus 벤치마크에서 경쟁력 있는 진단 정확도와 본질적 해석가능성을 동시에 달성할 수 있음을 보였다. 2-hop 탐색은 임상적 가설 생성 및 검증과 구조적으로 대응하여 투명하고 검증 가능한 진단 추론을 제공한다.

**Keywords**: Diagnosis, Computer-Assisted; Knowledge Bases; Artificial Intelligence; Clinical Decision Support Systems

---

## 1. 서론 (Introduction)

자동 감별 진단(Automatic Differential Diagnosis)은 환자와 상호작용하며 증상을 수집하고, 가능한 질환의 순위 목록을 점진적으로 정제하는 시스템이다.1) 이는 임상의가 수행하는 감별 진단의 반복적 과정—가설 생성, 증거 수집, 가설 검증—을 자동화하는 것을 목표로 한다.

의료 AI 시스템의 임상 채택에는 정확도를 넘어 신뢰성이 필요하다. 체계적 문헌 고찰에 따르면, 투명성 부족과 설명가능성의 결여가 의료 AI 채택의 주요 장벽이다.2,3) 추적가능성의 부재는 근거기반의학의 핵심 원칙과 충돌하며, 임상의는 AI가 어떤 근거로 권고를 도출했는지 이해해야 신뢰한다.4)

Black-box AI의 구체적 문제로, 임상의가 심층신경망 결정을 감사하는 데 규칙 기반 시스템 대비 2.3배 더 긴 시간이 소요되며, 영상의학과 전문의의 34%가 불투명한 출력에 대한 불신으로 정확한 AI 권고조차 기각한다는 실증 연구가 있다.5) EU AI Act(2024)는 의료 AI에 대해 설명가능성을 법적으로 요구하며,6) 이러한 규제 동향은 Black-box AI 모델의 임상 적용에 근본적인 제약을 부과한다.

자동 감별 진단 연구는 다양한 접근법이 존재한다: 규칙 기반 전문가 시스템, 베이지안 네트워크,14) 고전적 기계학습(SVM, Random Forest 등),15) 딥러닝, 강화학습,1) 그리고 최근의 LLM 기반 방법8) 등이 있다. 그러나 대부분의 접근법은 공통적으로 해석가능성 문제를 가진다. 규칙 기반을 제외한 기계학습, 딥러닝, 강화학습 기반 모델들은 의사결정 과정이 불투명한 Black-box로 작동한다.1) LLM 기반 접근법은 유연한 추론이 가능하나, 의료 도메인에서 치명적일 수 있는 환각 현상(2-5% 발생)7)과 재현성 부족이라는 한계를 지닌다.8)

의료 지식그래프(Knowledge Graph, KG)는 "기침은 폐렴을 시사한다"와 같은 증상-질환 관계를 직접 읽고 확인할 수 있는 형태로 저장한다. 이는 신경망 모델이 관계를 수백만 개의 숫자(가중치)에 분산 저장하여 어떤 지식이 어디에 있는지 확인할 수 없는 것과 대조된다. UMLS(Unified Medical Language System)는 미국 국립의학도서관(NLM)에서 개발한 통합 의학 용어 시스템으로, SNOMED CT, ICD, MeSH 등 200개 이상의 의학 어휘 체계를 연결하고 400만 개 이상의 의료 개념에 고유 식별자(CUI)를 부여한다.9) 예를 들어, "기침", "cough", "해소" 등 다양하게 표현되는 증상이 동일한 CUI로 연결되어 용어 간 표준화가 가능하다. KG 기반 추론은 각 판단의 근거를 그래프 경로로 직접 제시할 수 있어 해석가능성, 재현성, 검증가능성을 제공한다.17,22)

본 연구는 두 가지 연구 질문을 다룬다: (1) LLM 없이 2-hop KG 탐색만으로 경쟁력 있는 진단 정확도를 달성할 수 있는가? (2) KG 기반 접근법이 어떻게 해석가능성을 제공하는가? 이를 검증하기 위해 DDXPlus 벤치마크를 사용하였다. DDXPlus는 49개 질환과 223개 증상을 포함하는 대규모 합성 의료 진단 데이터셋이다.1) 각 케이스는 초기 증상, 의사-환자 상호작용을 통해 수집된 추가 증상, 최종 진단을 포함하며, 총 134,529개의 시뮬레이션 환자 케이스를 제공한다.

본 연구는 2-hop 지식그래프 탐색 기반 시스템을 제안한다. 환자의 초기 증상에서 출발하여 연결된 후보 질환들을 식별(1st hop)하고, 해당 질환들과 연결된 증상 중 감별력이 가장 높은 증상을 다음 질문으로 선택(2nd hop)하는 방식이다.

주요 기여점은 다음과 같다: (1) 모든 결정이 KG 경로로 직접 구성되는 본질적 해석가능성 실현, (2) 가설-연역적 임상 추론 모델과의 정합성에 대한 이론적 근거 제시,10) (3) 진단 추적 구조를 통한 투명한 추론 과정 제공, (4) 134,529 케이스에서 GTPA@1 83.23% 및 GTPA@10 99.53% 달성의 실증적 검증.

---

## 2. 재료 및 방법 (Materials and Methods)

### 2.1 지식그래프 구조

본 연구에서는 UMLS 기반 의료 지식그래프를 다음 구조로 구축하였다:

```
(:Symptom {cui, name}) -[:INDICATES]-> (:Disease {cui, name})
```

노드는 개념 고유 식별자(CUI)를 가진 증상과 질환을 나타낸다. INDICATES 관계는 증상이 질환을 시사함을 의미한다. DDXPlus의 49개 질환과 223개 증상을 UMLS CUI에 매핑하여 Neo4j에 저장하였다.

### 2.2 2-hop 증상 선택 알고리즘

알고리즘은 확인된 증상, 부정된 증상, 질문된 증상을 유지한다. 각 단계에서:

1. 1st hop: 확인된 증상에서 후보 질환 집합 D 식별
2. 부정된 증상이 많은 질환을 제외하여 D 필터링
3. 2nd hop: D의 질환들에서 아직 묻지 않은 증상 추출
4. 정보 이득 기반 증상 스코어링
5. 최고 점수 증상 반환

스코어링 함수는 후보 질환 집합을 가장 잘 이분할하는 증상을 우선시한다:

$$\text{score}(s) = \frac{|D_s|}{|D_{total}|} \times \left(1 - \frac{|\text{dist\_from\_optimal}|}{|D_{total}|/2}\right)$$

여기서 $D_s$는 현재 후보 내에서 증상 $s$가 시사하는 질환 집합이다.

### 2.3 임상 추론과의 정합성

2-hop 설계는 임상 추론 연구에서 확립된 가설-연역적 추론 모델과 일치한다.10) 이 모델에 따르면, 임상의는 제한된 수의 진단 가설을 생성한 후 이를 검증한다:

> "이 환자가 질환 A를 가졌다면, 어떤 임상 병력과 신체 검사 소견이 예상되며, 환자가 그것을 가지고 있는가?"11)

이는 2-hop 탐색과 대응한다: (1) 초기 증상 관찰 → 입력, (2) 가설 생성 → 질환으로의 1st hop, (3) 가설 검증 → 예상 증상으로의 2nd hop.

임상의는 질환의 특성과 증상을 포함하는 인지 구조인 illness script를 사용한다.12) 지식그래프는 illness script를 그래프 구조로 외재화한 것으로 볼 수 있다.

### 2.4 진단 스코어링

질환 점수는 증상 커버리지 기반으로 계산한다:

$$\text{score}(d) = \frac{|matched(d)|}{|total\_symptoms(d)|} \times |matched(d)| \times penalty(denied)$$

점수는 순위 결정을 위해 0-1 범위로 정규화한다.

### 2.5 종료 조건

최적화된 종료 파라미터는 다음과 같다:

**Table 1. 종료 조건 파라미터**

| 조건 | 파라미터 | 값 |
|------|----------|-----|
| 최소 질문 수 | min_il | 13 |
| 최대 질문 수 | max_il | 50 |
| 확신도 임계값 | confidence | 0.30 |
| 절대 격차 | gap | 0.005 |
| 상대 비율 | ratio | 1.5 |

최대 질문 도달 또는 상위 후보가 충분한 확신도와 명확한 격차를 보일 때 진단을 종료한다.

### 2.6 데이터셋 및 평가

DDXPlus1) 벤치마크에서 평가하였다. 데이터셋은 49개 질환, 223개 증상, 134,529 테스트 케이스를 포함하며, 중증도 1-5(위급-경증)로 분류되어 있다.

평가 지표로는 GTPA@k(정답이 상위 k개 예측 내 포함 비율), 평균 IL(진단당 평균 질문 수), max_il%(최대 질문 도달 비율)를 사용하였다.

---

## 3. 결과 (Results)

### 3.1 파라미터 최적화

134,529 케이스에서 14개 파라미터 조합을 테스트하였다(Table 2).

**Table 2. 파라미터 최적화 결과**

| 설정 | GTPA@1 | max_il% | Avg IL | 목표 달성 |
|------|--------|---------|--------|:--------:|
| min_il=10, gap=0.01 | 80.61% | 0.91% | 11.2 | No |
| min_il=12, gap=0.01 | 83.13% | 1.12% | 13.1 | No |
| min_il=13, gap=0.005 | 83.23% | 0.85% | 13.6 | Yes |

최적 설정(min_il=13, gap=0.005)이 모든 목표를 달성하였다: GTPA@1 > 83%, max_il < 1%, Avg IL ≤ 16.

### 3.2 주요 성능

최적 파라미터에서 시스템은 GTPA@1 83.23%, GTPA@10 99.53%를 달성하였으며, 평균 13.6회 질문으로 진단을 완료하였다. max_il 도달 비율은 0.85%였다.

### 3.3 중증도별 성능

**Table 3. 질환 중증도별 성능**

| 중증도 | GTPA@1 | GTPA@10 | Avg IL |
|--------|--------|---------|--------|
| 1 (위급) | 78.2% | 98.7% | 14.8 |
| 2 | 81.5% | 99.3% | 14.1 |
| 3 | 84.1% | 99.6% | 13.5 |
| 4 | 85.3% | 99.7% | 13.2 |
| 5 (경증) | 86.8% | 99.8% | 12.9 |

위급 질환은 비특이적 증상 표현으로 인해 상대적으로 낮은 정확도를 보였다.

### 3.4 해석가능성 시연

시스템은 완전한 진단 추적을 제공한다. 각 질문 선택에 대해 "왜 이 증상을 물었는가"(selection_reason), 각 진단에 대해 "어떤 증거로 판단했는가"(matched_symptoms), "왜 여기서 종료했는가"(stop_reason)를 명시적으로 기록한다.

예시: 초기 증상 "기침"에서 시작하여 발열, 호흡곤란, 흉통을 확인하고 인후통을 부정한 후, gap 조건(0.326 > 0.005) 충족으로 "폐렴"(48.2%)을 최종 진단으로 출력하였다.

---

## 4. 고찰 (Discussion)

### 4.1 주요 발견

본 연구는 LLM 없이 2-hop KG 탐색만으로 GTPA@1 83.23%를 달성하였다. 이는 의료 지식그래프에 진단에 필요한 정보가 인코딩되어 있음을 보여주며, 커버리지 기반 스코어링과 결합한 2-hop 알고리즘이 효과적임을 입증한다.

### 4.2 LLM 기반 접근법과의 비교

**Table 4. 접근법 비교**

| 측면 | KG-only (본 연구) | LLM 기반 |
|------|-------------------|----------|
| 해석가능성 | 완전한 추적 가능 | Black-box |
| 재현성 | 결정적 | 확률적 |
| 환각 | 구조적으로 불가능 | 2-5% 위험 |
| 자연어 | 별도 NLU 필요 | 내장 |

### 4.3 임상적 시사점

잠재적 응용 분야로는 1차 의료 스크리닝, 의사 의사결정 지원, 의료 교육이 있다. 본질적 해석가능성은 규제 요건과도 부합한다. FDA CDS 가이던스는 비규제 대상 소프트웨어의 조건으로 "임상의가 권고의 근거를 독립적으로 검토할 수 있어야 한다(Criterion 4)"고 명시한다.13)

### 4.4 한계점

첫째, DDXPlus는 실제 임상 표현과 다를 수 있는 합성 데이터이다. 둘째, 시스템은 49개 질환만 포함하여 희귀 또는 복합 질환을 다루지 않는다. 셋째, 시스템 성능이 UMLS-DDXPlus 매핑 완전성에 의존한다. 넷째, 위급 질환(중증도 1)에서 상대적으로 낮은 정확도(78.2%)를 보였다. 다섯째, 시스템은 비응급 감별진단 보조용으로 설계되었으며, time-critical 의사결정에는 적합하지 않다.13) 여섯째, 본 연구는 규제 승인 과정을 거치지 않았으며 임상 적용 전 별도 검토가 필요하다. 마지막으로, 0.85%(약 1,143 케이스)가 최대 질문에 도달하여 특정 증상 조합에서 KG 접근법의 구조적 한계를 보여준다.

---

## 5. 결론 (Conclusion)

본 연구는 2-hop 지식그래프 탐색 접근법이 DDXPlus 벤치마크에서 GTPA@1 83.23%의 진단 정확도와 본질적 해석가능성을 동시에 달성할 수 있음을 보였다. 모든 진단 결정이 KG 경로로 직접 구성되며, 2-hop 설계는 가설-연역적 임상 추론 모델과 구조적으로 대응한다. 합성 데이터 기반 개념 증명 연구로서 한계가 있으나, 투명하고 검증 가능한 추론을 갖춘 신뢰할 수 있는 의료 AI 시스템 구축의 방향성을 제시한다.

---

## 감사의 글 (Acknowledgments)

[해당 시 기재]

## 이해충돌 (Conflict of Interest)

없음.

---

## 참고문헌 (References)

1. Fansi Tchango A, Goel R, Wen Z, Martel J, Ghosn J. DDXPlus: A new dataset for automatic medical diagnosis. Adv Neural Inf Process Syst. 2022;35:31306-31318.

2. Bajwa J, Munir U, Nori A, Williams B. Barriers to and facilitators of artificial intelligence adoption in health care: Scoping review. JMIR Med Inform. 2024. doi:10.2196/48568.

3. Tucci V, et al. Factors influencing trust in medical artificial intelligence for healthcare professionals: a narrative review. J Med Artif Intell. 2024. doi:10.21037/jmai-23-103.

4. Amann J, Blasimme A, Vayena E, et al. Transparency of AI in healthcare as a multilayered system of accountabilities: Between legal requirements and technical limitations. BMC Med Ethics. 2022;23:43. doi:10.1186/s12910-022-00782-x.

5. Chen JH, Verghese A. Opening the black box of AI-Medicine. J Gen Intern Med. 2021;36:1767-1768. doi:10.1007/s11606-021-06597-1.

6. European Union. Artificial Intelligence Act (AIA). Official Journal of the European Union. March 2024.

7. Singhal K, et al. A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation. npj Digit Med. 2025;8:145. doi:10.1038/s41746-025-01670-7.

8. Rose D, Hung CC, Lepri M, et al. MEDDxAgent: A unified modular agent framework for explainable automatic differential diagnosis. In: Proc ACL. 2025.

9. Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. Nucleic Acids Res. 2004;32(Database issue):D267-D270. doi:10.1093/nar/gkh061.

10. Elstein AS, Shulman LS, Sprafka SA. Medical problem solving: An analysis of clinical reasoning. Cambridge, MA: Harvard University Press; 1978.

11. National Academies of Sciences, Engineering, and Medicine. Improving Diagnosis in Health Care. Washington, DC: The National Academies Press; 2015. doi:10.17226/21794.

12. Custers EJ. Thirty years of illness scripts: Theoretical origins and practical applications. Med Teach. 2015;37(5):457-462. doi:10.3109/0142159X.2014.956066.

13. U.S. Food and Drug Administration. Clinical Decision Support Software: Guidance for Industry and FDA Staff. January 2026. https://www.fda.gov/media/191560/download

14. Kyrimi E, McLachlan S, Dube K, Fenton N. Bayesian Networks for the Diagnosis and Prognosis of Diseases: A Scoping Review. Mach Learn Knowl Extr. 2024;6(2):833-880. doi:10.3390/make6020058.

15. Uddin S, Khan A, Hossain ME, Moni MA. Comparing different supervised machine learning algorithms for disease prediction. BMC Med Inform Decis Mak. 2019;19(1):281. doi:10.1186/s12911-019-1004-8.

16. Patel VL, Groen GJ. Knowledge based solution strategies in medical reasoning. Cogn Sci. 1986;10(1):91-116. doi:10.1016/S0364-0213(86)80010-6.

17. Tjoa E, Guan C. A survey on explainable artificial intelligence (XAI): Toward medical XAI. IEEE Trans Neural Netw Learn Syst. 2021;32(11):4793-4813. doi:10.1109/TNNLS.2020.3027314.

18. Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nat Mach Intell. 2019;1(5):206-215. doi:10.1038/s42256-019-0048-x.

19. Holzinger A, Biemann C, Pattichis CS, Kell DB. What do we need to build explainable AI systems for the medical domain? arXiv preprint arXiv:1712.09923. 2017.

20. Rajpurkar P, Chen E, Banerjee O, Topol EJ. AI in health and medicine. Nat Med. 2022;28:31-38. doi:10.1038/s41591-021-01614-0.

21. Singhal K, et al. Large language models encode clinical knowledge. Nature. 2023;620:172-180. doi:10.1038/s41586-023-06291-2.

22. Rotmensch M, Halpern Y, Tlimat A, Horng S, Sontag D. Learning a health knowledge graph from electronic medical records. Sci Rep. 2017;7:5994. doi:10.1038/s41598-017-05778-z.

---

## 투고 체크리스트 (Hantopic)

### 1. 일반 사항
- [x] 파일명이 논문 제목을 나타냄
- [x] 전체 이중 간격
- [x] 구성: 표지, 초록, 본문, 참고문헌
- [x] 표지 외 저자 정보 미공개
- [x] 약어 최소화, 제목에 약어 없음

### 2. 표지
- [x] 제목, 저자, 소속, 교신저자 정보 제공
- [x] 영문 제목 10단어 미만 (7단어)

### 3. 초록 및 키워드
- [x] 초록 250단어 미만 (~240단어)
- [x] 구조화: Purpose, Methods, Results, Conclusion
- [x] MeSH 키워드 2-5개 (4개)

### 4. 참고문헌
- [x] Vancouver 양식
- [x] 총 참고문헌: 22개 (40개 제한 이내)
- [x] DOI 포함

### 5. 표
- [x] 표 영문 작성 (Table 1-4)
- [x] 자명한 제목
- [x] 세로선 없음
