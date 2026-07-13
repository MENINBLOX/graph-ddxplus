# 속성 선택 (①) — 확정 문서

> 상태: **확정(settled)**. 이 문서는 "어떤 속성을 KG에 둘 것인가"의 학술적 근거를 종결한다.
> 핵심 입장: **속성 선택은 논문의 *정당성 토대*이지 *기여*가 아니다.** 기여는 IE 품질 검증 + 표준 정규화 cross-benchmark 전이에 있다. 따라서 본 항목은 논문에서 한 문단 + 표 1개로 다룬다.

---

## 1. 결정

증상 특성화의 **교과서 정전인 Bates의 "seven attributes of a symptom"** (Bickley & Szilagyi, *Bates' Guide to Physical Examination and History Taking*)을 속성 스키마의 **단일 앵커**로 채택한다. 병력청취 mnemonic(OLDCARTS/OPQRST/SOCRATES/LOCATES/COLDSPA)과 계산 프레임워크(NLICE)는 **보강 근거**로 인용한다. 각 속성은 **표준 온톨로지(HPO/UMLS/SNOMED)에 정규화 가능한 범위에서** 매핑한다(교수님 지침: 프레임워크 + 온톨로지 이중 근거).

자작 "≥2 프레임워크 ∩ 온톨로지" 교집합 규칙은 post-hoc 의심을 부르므로 **채택하지 않는다.** 단일 권위 표준(Bates)을 채택하는 편이 깨끗하다.

## 2. 확정 스키마

| Bates 속성 | KG 속성 슬롯 | 표준 타깃 | 비고 |
|---|---|---|---|
| (1) Location | **location** | UMLS Anatomy(T023/T029) / SNOMED 123037004 | **radiation을 `radiates-to` 방향 qualifier로 포함** (§4) |
| (2) Quality | **character** | HPO HP:0025280(부분) / 개별 phenotype / 문자열 | 통제어휘 부족분은 문자열 보존(비정규화) |
| (3) Quantity/Severity | **severity** | HPO HP:0012824 (ordinal: Mild/Mod/Severe...) | 깨끗이 정규화 |
| (4) Timing | **timing** = { onset-pace, duration, frequency } | HPO HP:0011008(Acute/Insidious), 외 | duration은 **onset이 아니라 Timing 하위 필드** (§4) |
| (5) Setting | (context) | — | 유발상황과 겹침, 선택적 |
| (6) Aggravating/Relieving | **aggravating** + **relieving** | HPO HP:0025204/0025285, HP:0025254 | SOCRATES "E"·OPQRST "P"가 짝으로 둠 |
| (7) Associated | — (속성 아님) | — | KG의 **공존 finding**으로 구조 표현 |

**핵심 슬롯(6)**: location(+radiation), character, severity, timing, aggravating, relieving.
**제외/분리**: associated(구조적 공존 finding), frequency-of-phenotype-in-disease(= **edge prior P(E\|D)**, 환자 보고 속성 아님).

## 3. 정당성 (cherry-pick 공격 무력화)

1. **앵커 = 교과서 정전**: "왜 이 속성?" → "Bates seven attributes(증상 특성화 표준)를 채택". 자의성 공격 차단.
2. **Robustness check**: Bates·COLDSPA를 추가해도 합의 코어(location/character/severity/timing/aggravating/relieving)가 **6-7/7로 유지**됨을 보고. 5개 프레임워크로 한정한 게 결론을 바꾸지 않음을 명시.
3. **온톨로지 근거 분리**: 온톨로지 가용성은 *임상 타당성*이 아니라 *표현/정규화 방식*을 정함을 명시(임상 스키마=Bates, 온톨로지=구현).
4. **NLICE 플래그**: 임상 mnemonic 아닌 계산/합성(Synthea) 프레임워크임을 명시(provenance 구분).

## 4. 흡수(absorption) 항목 — 학술 방어

| 흡수 | 방어 | 조건 |
|---|---|---|
| **radiation → location** | OPQRST(Region/Radiation 묶음)·Bates(location 내 radiation) 선례 | ⚠️ **방향(`radiates-to`) 관계 보존 필수.** 평평한 위치집합으로 뭉개면 "흉통→양팔 방사"(Panju LR+ 7.1, MI 최강 rule-in)와 "독립 흉통+양팔통"이 구분 불가 → 신호 소실. 방향 qualifier는 first-class·queryable로 유지 |
| **duration → Timing 하위 필드** | Bates가 onset+duration+frequency를 **"Timing" 축에 묶음** | ⚠️ **onset에 넣지 말 것**(onset=시작 양상 vs duration=지속기간, 직교). duration은 독립적으로 감별을 가름(기침 급/아급/만성<3·8주, 통증 acute/chronic >3개월). Timing의 **별도 하위 필드**로 보존하거나 numeric·온톨로지 부재로 honest limitation 처리 |

(제 이전 초안의 "duration→onset 흡수"는 **폐기**: 모든 프레임워크가 onset/duration을 분리하며, duration은 감별 동인이다.)

## 5. 범위 명시 (중요)

- 본 속성 선택은 **정당성 토대**다. **기여가 아니다.** Bates 표준 채택은 올바른 관행이지 novelty가 아니다.
- 논문에서의 분량 = **Methods의 한 소절 + 표 1개**(Bates 인용). 길게 방어하지 않는다.
- 논문의 기여는 별도: (B) IE 품질 검증, (②) 표준 정규화 interlingua, (③) cross-benchmark 전이.

## 참고문헌
- Bickley LS, Szilagyi PG. *Bates' Guide to Physical Examination and History Taking* — "seven attributes of a symptom"(Location/Quality/Quantity/Timing[onset·duration·frequency]/Setting/Aggravating·Relieving/Associated).
- OLDCARTS·OPQRST·SOCRATES·LOCATES·COLDSPA (병력청취 mnemonic, 교육 표준).
- NLICE: Guo et al., arXiv:2401.13756 (2024) — 계산/합성 프레임워크.
- Panju AA et al., *JAMA* 1998;280:1256 — radiation LR+ 7.1(양팔), character(sharp 0.3) 등.
- HPO: HP:0012824/0011008/0025204/0025285/0025254/0025280 (OLS4 검증). UMLS Anatomy T023/T029; SNOMED 123037004.
- 온톨로지 정규화 선행: SemEval-2015 T14, Phenopackets(Jacobsen Nat Biotechnol 2022), OMOP CDM.
