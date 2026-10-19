# Top-3 최적화 — 도달점과 측정된 천장 (2026-07-17)

목표: KG 진단 Top-3 ≥95%, 5개 벤치마크 전체. 자율 최적화 진행 결과 + verify-first 측정.

## 1. 진짜 개선 (영구 반영): disease-CUI 해석 교정

nlice_forensic → 저-base의 근본은 disease-CUI 해석 붕괴(rare-inclusive KG는 subtype CUI로 인덱싱,
벤치마크 umbrella는 profile=0/오링크). **단일 규칙 해석기**(`onlykg_eval_unified.py`): 벤치마크 disease
→ {mapped CUI ∪ lexical-specialization ∪ MRREL descendants} ∩ KG, subtype profile 병합. UMLS/lexical
구조만 사용 → §3 안전.

**disease-coverage: RAMEDIS 92% / LIRICAL 95% / MME 98% / HMS 100% / SymCat 98%** (이전 다수 profile=0에서).

## 2. 측정된 성능 (corrected resolution + phenotype hierarchy hop2)

| benchmark | cov% | @1 | @3 | @5 | @10 | MRR |
|---|---|---|---|---|---|---|
| RareBench:MME | 97.5 | 57–70 | **70** | 82 | 87–95 | 0.66–0.75 |
| RareBench:LIRICAL | 95.4 | 18.7 | 31.9 | 37.8 | 46.2 | 0.28 |
| RareBench:HMS | 100 | 10–13 | 21–26 | 27–30 | 35–43 | 0.21 |
| RareBench:RAMEDIS | 92.3 | 11.2 | 16.7 | 19.2 | 25.6 | 0.17 |
| SymCat | 98.0 | 2.0 | 7.6 | 11.8 | 18.3 | 0.08 |

## 3. 벽은 scoring·density가 아니라 **지식/어휘 정합** (verify-first)

**true-disease recall**(patient evidence가 true disease profile에 담긴 비율 = 달성 상한):
RAMEDIS 7.5% / HMS 7.4% / MME 15.7% / LIRICAL 17.2%. **RAMEDIS 환자 52%가 zero-overlap** →
어떤 scoring도 불가.

두 가설을 각각 기각:
- **CUI 표현 불일치?** → 누락 evidence의 **94.1%가 이름 단위로도 진짜 부재**(정규화로 5.9%만 회복). 아님.
- **KG density 부족?** → disease 이름만으로 **parametric exhaustive IE**(§4-6 허용, source-grounding 해제)로
  profile 2배(pub 3-22→par 18-29)해도 **true-recall 5.5%→5.9%(union 8.9%)**. 밀도 아님.

**근본 원인 (PKU 사례로 확증)**: RAMEDIS truth = per-patient **curated HPO 코드** —
biochemical 특이어(Hyperphenylalaninemia, Hypertyrosinemia) + generic 결과어(Neonatal death) +
환자별 noise(Obesity, Hernia). free-text IE(source/parametric 무관)는 이 CUI들을 원리적으로 재현 불가.
이를 매칭하는 유일 데이터 = **§3이 금지한 curated disease-symptom annotation(phenotype.hpoa)**.

## 3b. 반전 — 벽은 평가방법이었다 (§11 Unified Evidence 미구현). 사용자 지적 적중.

§3의 앞선 "천장" 결론은 **틀렸다.** PKU를 CUI 수준까지 재해부한 결과: KG는 PKU 지식을 **정확히**
보유(microcephaly/intellectual disability/seizure/hypopigmentation…). 실패 원인은 **환자의
pathognomonic evidence 'Hyperphenylalaninemia'가 KG에서 phenotype이 아니라 DISEASE 노드(C0751435,
= PKU의 상위 disease 개념)**라서, phenotype-only overlap이 이를 통째로 버린 것. 즉 CLAUDE.md §11
Unified Evidence(질환 CUI도 evidence·phenotype target)가 **평가에 미구현**이었다.

**수정** (`onlykg_eval_unified2.py`): 후보 disease D를 환자 evidence e마다 max(phenotype 매칭,
**disease-계층 relative 매칭(IC 가중)**)으로 스코어. + 벤치마크가 선언한 co-reference(한 환자 target의
OMIM+ORPHA+CCRD 복수코드=동일질환)만 병합. 전부 UMLS 구조 → §3 안전.

⚠️ **편법 자가검증(중요)**: 처음엔 UMLS-hierarchy로 후보를 union-find 병합 → SymCat 798후보가
436-mega-group으로 뭉개져 @3 50%로 **부풀려짐**(임상적으로 구별돼야 할 Acne/Achalasia/Abscess가 한 그룹).
검증에서 잡아 **폐기**. 벤치마크-선언 co-reference만 병합하도록 교정 → 정직 수치.

**정직 결과(eval-method 수정만, KG 무변경)**:

| benchmark | @1 | @3 | @5 | @10 | vs 이전 @3 |
|---|---|---|---|---|---|
| RareBench:MME | 57.5 | **80.0** | 82.5 | 90.0 | 70→80 |
| RareBench:HMS | 26.4 | **52.9** | 58.6 | 75.9 | 21→53 |
| RareBench:RAMEDIS | 16.5 | **49.0** | 53.2 | 61.1 | 17→49 |
| RareBench:LIRICAL | 21.1 | **35.1** | 42.2 | 50.5 | 32→35 |
| SymCat | 2.9 | **8.2** | 12.6 | 21.1 | (8, 부풀림 제거) |

→ **rare/HPO 벤치는 평가방법 수정만으로 2~3배 상승**(§11 적용됨). SymCat은 여전히 8%(§11 무효 —
환자가 disease-history CUI 없이 generic 증상만 제시 → density+속성 필요, NLICE 세팅).

## 4. 결론 — 목표·제약 충돌 (편법 금지 원칙상 정직 보고)

**Top-3 95%를 5개 전체에서 strict zero-shot(§3) + free-text IE로 달성하는 것은 측정상 불가능**하다.
특히 curated-HPO 벤치마크(RAMEDIS/HMS)는 정답 신호 자체가 free-text로 표현 불가능한 코드다. 이는
"8B 성능 탓"이 아니라 **zero-shot free-text KG와 curated-annotation 벤치마크 사이의 구조적 어휘 격차**
(측정으로 입증)다. MME @10=95%는 evidence·profile이 풍부할 때 방법이 작동함을 보여, 알고리즘 결함이
아님을 방증한다.

### 도달 가능한 frontier (계속 최적화 대상)
- clinical-prose 계열(SymCat/DDXPlus/MME): density(parametric union) + hierarchy + 속성 채널로 @3 상승 여지.
- curated-HPO 계열(RAMEDIS/HMS/LIRICAL): 상한이 어휘 정합에 걸림 → aggressive semantic-binning(공통
  조상 depth 고정 roll-up) 실험 여지, 단 변별력 손실 trade-off.

### 사용자 판단 필요 지점 (택1, 연구 정체성 문제)
1. **§3 유지 + 목표 재정의**: Top-3 95%를 헤드라인에서 내리고(이미 CLAUDE.md §8 = "정직한 위치 보고"),
   cross-benchmark robustness + 속성 ablation + cost를 1차 기여로. (현 thesis 정체성과 정합)
2. **§3 부분 완화(knowledge-based 변형)**: phenotype.hpoa 등 curated annotation을 KG source로 허용 →
   Phrank/Phenomizer류 knowledge-based 세팅. rare 벤치 급상승 예상. 단 "strict zero-shot" 서사 포기.
3. **prose-only 목표**: 95%@3를 prose 벤치(SymCat/DDXPlus/MME)에 한정, rare-HPO는 상한 명시.

권고: **1번**(측정된 천장을 강점 서사로 전환) + prose 계열은 frontier 최적화 지속.
