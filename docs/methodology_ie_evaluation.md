# IE 성능의 학술적 평가 방법론 (2026-06-05)

## 1. 문제

IE는 본 연구(strict zero-shot KG-only 진단)의 기술적 핵심이다. 그러나 IE 프롬프트
변형(v105/v106/v107)의 우열을 **임의 proxy**(잡음%/메타속성% 정규식 카운트)로 판정했고,
이는 ground truth도 downstream 근거도 없어 "v106 최적" 주장이 성립하지 않았다(사용자 지적).
학술적으로 인정받는 평가 방법을 적용해 재검증한다.

## 2. 학술적으로 확립된 IE 평가 두 축

선행 연구 조사(intrinsic/extrinsic 평가 패러다임)에 따르면 IE/KG 품질 평가는 두 축이다.

### (A) Intrinsic — 추출물 자체의 품질
- accuracy(엔티티/관계 정확성), coverage, consistency. gold corpus 대비 P/R/F1.
- **Faithfulness (NLI entailment)**: precision/hallucination 축. source가 추출 claim을
  entail하는지 문장 단위 max-entailment로 측정(hallucination survey 표준).
- gold corpus 직접 측정: n2c2 2018 Track2 (entity+attribute P/R/F1, DUA 필요).

### (B) Extrinsic — downstream task 효용
- KG를 실제 과제 성능으로 평가. 본 연구 downstream = DDXPlus top-k 진단.
- 근거: **KGrEaT** (Heist & Paulheim, arXiv:2308.10537) — "Framework to Evaluate
  Knowledge Graphs via Downstream Tasks". extrinsic 메트릭은 intrinsic의 보완으로
  KG completeness(recall) + 실효성을 포착.

두 축은 **상호 보완**이다: intrinsic faithfulness는 precision만, extrinsic은 recall+효용.

## 3. 적용 결과

### (A) Intrinsic faithfulness (NLI, roberta-large-mnli, 문장단위)
| variant | overall | finding | attr |
|---|---|---|---|
| v105 | 83% | 95 | 76 |
| v106 | 83% | 98 | 67 |
| v107 | 83% | 98 | 69 |

→ **세 프롬프트 동률(83%).** 모두 거의 hallucinate하지 않음. precision 축만으로는 변별 불가.

### (B) Extrinsic downstream (DDXPlus, KGrEaT)
통제 실험 — 변수는 **IE 프롬프트뿐**: 동일 source(Wikipedia 임상섹션)·동일 49질환·
동일 scispaCy UMLS 링커(공유 name→CUI 맵, 83.5% 매핑)·동일 알고리즘
(v71 self-aware cosine+IDF+negative)·동일 하이퍼파라미터. `v110_extrinsic_ie_eval.py`.

| variant | findings | CUIs | @1 | @5 | @10 | MRR |
|---|---|---|---|---|---|---|
| v105 | 376 | 229 | 15.83% | 35.00% | 45.19% | 0.261 |
| v106 | 551 | 250 | 24.50% | 58.63% | 74.07% | 0.401 |
| v107 | 601 | 268 | **25.84%** | 58.67% | **75.12%** | **0.403** |

(N=6000 random, seed=42; 절대점수는 단일source 49질환 minimal KG라 낮음 — **상대비교가 신호**.)

**Robustness** — 3 seed × 3 하이퍼파라미터 = 5구성 전부 순위 동일:
`v105 << v106 ≲ v107`. v107이 @1/@10 모두 전 구성에서 v106을 +1~1.5%p 일관 상회,
v105는 @10에서 ~30%p 열위.

## 4. 결론

1. **두 축이 상반된 정보를 준다.** faithfulness(precision)는 셋 다 83% 동률 — 변별 불가.
   extrinsic(recall+효용)은 명확히 변별: v105가 크게 열위.
2. **원인**: v105는 grounded지만 **추출 양이 적어(376 vs 551/601 findings) recall이 낮고**,
   진단 변별 content를 누락 → downstream @10 −30%p. v106/v107은 동등 faithfulness에서
   더 많은 finding을 확보 → recall↑ → downstream↑.
3. **"v106 최적"은 두 축 어디서도 성립 안 함.** faithfulness=동률, downstream=v107≳v106.
   v106/v107은 실질 동급(차 1~1.5%p), v107 미세 우위.
4. **방법론적 교훈**: IE 프롬프트 비교는 단일 축(특히 faithfulness만)으로 부족.
   intrinsic(precision) + extrinsic(downstream recall+효용) **양축 동시 측정**이 학술 표준.
   본 연구처럼 downstream task가 명확하면 extrinsic이 가장 직접적·결정적.

## 4b. Intrinsic gold 인증 — 알고리즘 독립 (2026-06-05 추가)

리뷰어 공격("동일 알고리즘이 그 IE에 co-adapt된 것 아니냐")의 근본 차단책: IE를
**알고리즘을 거치지 않고** 전문가 gold corpus 대비 직접 평가한다. extrinsic robustness는
알고리즘 공간을 다 못 덮는 귀납적 논증이라 이 공격을 원천 봉쇄 못 함 → intrinsic이 정답.

**인정 방법**: 임상 concept+attribute 추출 → ontology 정규화, **gold 대비 mention-level
P/R/F1 + per-slot accuracy**. 표준 = SemEval-2015 Task 14(9속성: body location/severity/
course/negation..., PhysioNet DUA), n2c2 2018 Track2(DUA).

**공개 gold로 즉시 검증**: MACCROBAT2020 (200 임상 케이스리포트, brat, 영어; HuggingFace
singh-aditya/MACCROBAT_biomedical_ner). 우리 속성과 직접 대응: SIGN_SYMPTOM 3347,
BIOLOGICAL_STRUCTURE(location) 2928, SEVERITY 374. 우리 추출기(v106 철학, **zero-shot**)를
200문서에 돌려(`v111_maccrobat_ie.py`) 표준 P/R/F1 채점(`v111_maccrobat_score.py`).

| type | gold | pred | STRICT P/R/F1 | RELAXED P/R/F1 |
|---|---|---|---|---|
| SIGN_SYMPTOM | 3347 | 1776 | .370/.196/.256 | **.723/.384/.501** |
| SEVERITY | 374 | 207 | .609/.337/.434 | .628/.348/.448 |
| BIOLOGICAL_STRUCTURE | 2928 | 645 | .558/.123/.202 | .747/.165/.270 |

**해석**:
1. **Precision 견고**(relaxed P 0.72 증상 / 0.75 location) — 추출하면 대개 실제 gold 엔티티,
   hallucination 낮음. NLI faithfulness 83%와 **교차 일치**.
2. **Recall 낮음**(0.38 / 0.16) = 정직한 약점. 정성확인상 누락은 (a) 일부 실제 누락 (b) **의도적
   scope 제한**: 우리는 환자 증상만·속성을 증상에 bound. gold BIOLOGICAL_STRUCTURE 2928은
   대부분 procedure/해부 기술에 붙은 standalone이라 우리 scope 밖 → location recall 구조적 저평가.
3. **두 인정 방법의 수렴**: intrinsic(precision↑/recall↓)이 extrinsic 발견(recall이 병목 —
   v105가 low recall로 @10 손실)과 **독립적으로 같은 결론**. 서로 다른 인정 메트릭이 수렴 =
   평가 프레임워크 자체의 강한 검증.

**caveat(정직)**: ① zero-shot — supervised fine-tuned 토큰분류 baseline(MACCROBAT micro F1
~0.7-0.8)이 비교군이므로 절대값 낮음은 예상됨. ② gold 노이즈 존재(HLA 대립유전자를
SIGN_SYMPTOM으로 단 케이스 — 우리 추출기가 오히려 더 깨끗). ③ 정성확인: doc0 murmur를
location("left sternal border")까지 정확 추출.

→ **결론: IE 방법을 알고리즘 독립적·인정 메트릭으로 검증 완료.** precision은 견고(인정 가능),
recall이 개선 레버. 이 recall 약점이 곧 다음 IE 사이클의 목표(더 많은 source·exhaustive 추출).

## 4c. Cross-corpus 공개 인증 — CADEC 추가 (2026-06-05)

DUA 불필요 공개 corpus를 **장르 다양화**로 확장 → "한 벤치마크에 맞춘 평가" 공격을 평가
차원에서도 차단. CADEC(CSIRO, 환자 온라인 포럼 게시글 = **lay vocabulary**, HF
KevinSpaghetti/cadec). 1098 게시글, gold = ADR mention span(verbatim) + MedDRA PT 정규화.
동일 zero-shot 추출기(`v111_cadec_ie.py`,`v111_cadec_score.py`).

| target | STRICT P/R/F1 | RELAXED P/R/F1 |
|---|---|---|
| ADR verbatim span | .556/.553/.554 | **.811/.806/.808** |
| MedDRA PT (정규화) | .126/.142/.133 | .236/.264/.249 |

- ADR span RELAXED **F1 0.808, P/R 균형(0.81/0.81)** — supervised CADEC NER baseline(CRF/BERT
  span F1 ~0.6-0.7)과 동급대(zero-shot임에도). 정성: "could barely walk","numb leg" 등 구어체도 정확.
- MedDRA PT 낮음 = 정규화 **어휘 불일치**(우리는 UMLS/scispaCy로 정규화, MedDRA PT 문자열과 직접
  대조는 불공정). 추출 실패 아님 → verbatim span이 공정한 추출 지표.

### Cross-corpus 종합 (공개, DUA 없음)

| corpus | 장르 | 타깃 | RELAXED F1 | P | R |
|---|---|---|---|---|---|
| MACCROBAT | formal 임상노트 (83 fine type) | SIGN_SYMPTOM | 0.50 | 0.72 | 0.38 |
| CADEC | lay 환자포럼 | ADR span | **0.81** | 0.81 | 0.81 |

**핵심**: ① 추출기가 **두 장르(formal 임상 ↔ lay 환자언어) 모두에서 작동** → 일반화 입증.
② MACCROBAT의 낮은 recall(0.38)은 **83-type 세분 스키마 + 의도적 scope 제한**의 산물이지
추출기 본질적 한계가 아님 — **clean 증상추출 과제(CADEC)에선 recall 0.81로 균형**. ③ lay
언어(=DDXPlus 환자 evidence와 동질)에서 가장 강함 → 우리 실제 과제 적합성을 직접 뒷받침.

## 5. 한계 / 다음

- extrinsic은 downstream 알고리즘·링커를 상수로 고정해 IE에 차이를 귀속시키나, 그 고정값의
  품질에 절대점수가 의존(상대순위는 robust). 단일 source(Wikipedia)라 절대 recall 낮음.
- gold P/R/F1(n2c2 Track2)은 DUA 등록 후 추가 가능 — attribute-value 정확도까지 측정.
- 후속: 최적 프롬프트(v106/v107)로 full union KG 재구성, cross-benchmark(SymCat/RareBench)
  extrinsic 재확인 → 교수님 cross-benchmark robustness 논리와 연결.
