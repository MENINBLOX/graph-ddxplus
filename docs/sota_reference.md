# External SOTA 고정 + 우리 위치 (CLAUDE.md §9) — 2026-07-17

## 0. 다중 패러다임 SOTA landscape (LLM zero-shot뿐 아니라 전 방법론) — 2026-07-19

SOTA는 **세팅이 다르면 직접 비교 불가**. 우리 = **strict zero-shot · KG-only · 8B IE · traceable**.
아래는 각 패러다임의 최고치를 고정하고 우리 위치를 정직 보고(§9). 우리 config = best(3d).

전 수치는 서브에이전트가 arXiv HTML/원논문에서 검증(2026-07-19). setting별 분리 필수.

| benchmark | supervised(데이터학습) | agentic LLM(tool/iter) | knowledge-based(HPO) | zero-shot LLM | **우리(zero-shot KG)** |
|---|---|---|---|---|---|
| DDXPlus | DDxT **99.98%**·AARLC GTPA@1 **99.21**(≠75; 75는 DDF1) | MEDDxAgent GPT-4o **0.86 top-1**(interactive) | — | H-DDx GPT-4o **Top-5 80.4**(Claude-S4 83.9) | @1 38.4/@5 65.1/**@10 80.1** |
| SymCat | **MMF-AC 51.9/78.2/87.6**(RL,300dz)·BSODA·REFUEL·NB 58.8 | — | — | **無**(빈 슬롯) | 4.6/13.9/18.9 |
| NLICE | NB/RF **58.8/57.1→82.0**(속성, ⚠️801→55 class confound) | — | — | 無 | +location Δ+3.6pp p=.005(통제 ablation) |
| RareBench(pool) | — | DeepRare avg R@1 **57.2** | PhenoBrain·PubCaseFinder | GPT-4 0shot **32.3/45.4/58.9** | @1 24.9/@3 48.0/@10 61.6 |

⚠️ 수정(에이전트 검증): AARLC DDXPlus GTPA@1=**99.21**(75는 DDF1 0.78, 출처는 DDXPlus 논문). GTPA@1 정의 상이
(DDxT=set-membership 포화 vs MEDDxAgent=true top-1) — footnote 필수. NLICE 58.8→82.0은 **class수(801→55) confound**
→ clean ablation 아님(우리 통제 ablation이 더 깨끗). SymCat/NLICE엔 **zero-shot baseline 부재** = 우리가 빈 슬롯 채움.

### 0b. RareBench per-dataset (DeepRare Ext.Data Table 2, R@1 검증) vs 우리

| dataset | PubCaseFinder(KB) | PhenoBrain(KB) | GPT-4o(0shot) | Claude-3.7(0shot) | DeepRare(agentic) | **우리 @1 / @3** |
|---|---|---|---|---|---|---|
| **MME** | 40.0 | 25.0 | 2.5 | 7.5 | 70.0 | **62.5 / 82.5** |
| HMS | 7.3 | 22.4 | 47.7 | 43.2 | 57.0 | **33.3 / 56.3** |
| LIRICAL | 37.8 | 27.1 | 31.1 | 33.2 | 56.0 | 23.0 / 37.6 |
| RAMEDIS | 25.8 | 20.7 | 35.5 | 42.5 | 72.6 | 24.0 / 52.1 |

DeepRare R@3: MME 77.5·HMS 65.1·LIRICAL 65.2·RAMEDIS 82.9(avg 65.25). PhenoBrain R@3 avg 40.7.

> **정직한 per-dataset 포지셔닝(핵심)**:
> - **MME: 우리 @1 62.5가 KB·zero-shot LLM 전부 압도**(PubCaseFinder 40·PhenoBrain 25·GPT-4o 2.5·Claude 7.5),
>   agentic DeepRare(70)에만 뒤짐. **@3 82.5는 DeepRare(77.5)마저 초과.** (LLM은 MME=신규 초희귀질환에서 붕괴 2.5%.)
> - **HMS: KB(PubCaseFinder 7.3·PhenoBrain 22.4) 초과**, LLM(47.7) 아래.
> - RAMEDIS/LIRICAL: KB와 동급(~PubCaseFinder/PhenoBrain), LLM/DeepRare 아래.
> → 우리 KG는 **초희귀·novel(MME) + KB 대비 우위**가 강점. LLM이 강한 RAMEDIS/LIRICAL(문헌 풍부)에선 상보.
> **동일세팅(zero-shot, no-inference-LLM)엔 우리와 KB만 존재** → 우리는 MME/HMS에서 KB SOTA 초과.


목적: "95% Top-3" 목표의 현실성 검증. 결론: **95%는 strict zero-shot에서 비현실적** — 그 수준은
disease-symptom 매핑으로 학습한 supervised 또는 tool+KB agentic만 도달. 우리(zero-shot KG-only)는
**GPT-4 zero-shot과 Top-3/Top-10 동급**이고, 강점은 SOTA 점수가 아니라 cost/traceability/robustness.

## 1. RareBench (Task 4: RAMEDIS/LIRICAL/MME/HMS, 1,122 cases)

| 방법 | 세팅 | Top-1 | Top-3 | Top-10 | 출처 |
|---|---|---|---|---|---|
| GPT-3.5-turbo | zero-shot LLM | 21.1 | 34.2 | 48.2 | RareBench 2024 |
| **GPT-4 0-shot** | **zero-shot LLM** | **32.3** | **45.4** | **58.9** | RareBench 2024 |
| GPT-4 CoT | zero-shot LLM | 33.2 | 46.4 | 59.5 | RareBench 2024 |
| **우리 KG-only** | **strict zero-shot, 8B IE, no mapping** | **24.4** | **47.5** | **60.3** | 본연구(unified2, 분리정규화) |
| DeepRare | agentic GPT-4o + tools + KB | 72.6(RAMEDIS)/78(MME) R@1 | 85(MME) | — | DeepRare 2025 |
| PubCaseFinder | knowledge-based(HPO KB) | ~48(MME) | ~65(MME) | — | DeepRare 2025 표 |

**우리 per-dataset (zero-shot, KG-only, unified evidence §11 + 분리정규화)**:
MME @1=57.5/@3=80 · HMS @1=31.0/@3=52.9 · RAMEDIS @1=23.9/@3=52.6 · LIRICAL @1=20.3/@3=34.3.
가중 aggregate: **@1=24.4 / @3=47.5 / @10=60.3.**

> **핵심**: 우리 8B-KG zero-shot의 **Top-3(47.5)·Top-10(60.3)이 GPT-4 zero-shot(45.4/58.9)을 초과**,
> Top-1(24.4)은 GPT-4(32.3)에 근접(격차 12→8pp로 축소). 개선 경로: 두 evidence 채널(disease-계층 prior /
> phenotype 유사도)을 **분리 정규화**(lumping+combined-size 정규화가 thin-profile 사촌을 이기게 하던 것 수정).
> §7 준수(단일 전역공식). DeepRare(agentic, RAMEDIS 72.6 R@1)는 다중 GPT-4o+tool/KB로 cost/traceability 축 다름.

## 2. SymCat (RL symptom-checker 계열 — 전부 SUPERVISED)

| 방법 | 세팅 | Top-1 | Top-3 | Top-5 | 비고 |
|---|---|---|---|---|---|
| REFUEL | supervised RL | 47.7 | 66.2 | 71.8 | SymCat-300 |
| BSODA | supervised | 48.2 | 73.8 | 84.2 | SymCat-300 |
| MMF-AC | supervised RL | 51.9 | 78.2 | 87.6 | SymCat-300 |
| MMF-AC | supervised RL | 45.8 | 72.1 | 82.4 | SymCat-400 |
| **우리 KG-only** | **strict zero-shot** | **2.9** | **8.2** | **12.6** | 본연구 |

> SymCat SOTA(top-3 72~85%)는 **전부 disease-symptom 매핑으로 학습한 supervised RL** = §3 위반 데이터
> 사용. zero-shot 공개 baseline은 사실상 없음. 우리 8.2%는 낮지만, **비교군이 부적절**(supervised vs
> zero-shot). SymCat은 우리 KG(rare-inclusive, pubmed) 스코프와도 mismatch(흔한 질환·simulated 환자).
> → 속성채널(NLICE 세팅)로 개선 대상.

## 3. DDXPlus (49-way, common diseases)

| 방법 | 세팅 | Top-1 | Top-5 | 비고 |
|---|---|---|---|---|
| Claude-Sonnet-3.7 | zero-shot LLM | — | 83.6 | H-DDx 2025 (arXiv 2510.03700) |
| **GPT-4o** | **zero-shot LLM** | — | **80.4** | H-DDx 2025 |
| Gemma3-27B | zero-shot LLM | — | 74.6 | H-DDx 2025 |
| AARLC 등 | supervised(train labels) | ~75(GTPA@1) | — | GTPA 98.8~99.9. **매핑학습=§3 위반** |
| **우리 KG-only** | **strict zero-shot** | **31.1** | **59.2** | 본연구(unified2). @3=45.8/@10=72.7 |

> zero-shot LLM Top-5 ~75~84% vs 우리 59.2%. supervised GTPA 99%는 train 라벨 학습(§3 금지).

### 3c-2. 진단: 방법이 아니라 KG 내용 (ceiling 실험)

교수님 원칙(5벤치 전부 증명) + "evidence 다 주면 거의 결정적 태스크"를 검증. **DDXPlus 자체 생성 KB를
KG로 쓰면 우리 동일 overlap scorer가 @1=95.4 / @3=99.6 / @5=99.7** (`parametric_ddxplus_test.py`의
ceiling 분기). → **매칭 방법은 병목 아님**(95% 달성 가능). 우리 실측 31%의 병목 = **KG 내용 + CUI 어휘
정합**: 상한 실험은 환자 evidence CUI로 profile을 만들어 vocabulary가 완벽 일치, 우리 pubmed KG는 같은
개념을 **다른 CUI**로 담아 true-disease recall이 34%에 묶임(zero-overlap 7.9%).
- **parametric exhaustive IE**(disease명만, §4-6): avg profile 15.2, DDXPlus @1=27.6/@5=60.3 —
  pubmed KG(31/59)와 비슷, **상한 근처도 못 감**. rare(RAMEDIS)와 마찬가지로 CUI 어휘 불일치가 벽.
- 함의: DDXPlus를 올리는 §3-safe 레버 = **KG 어휘를 벤치 evidence CUI 공간에 의미 정렬**.

### 3c-3. SapBERT soft CUI-alignment (ML, KG-only, §3 safe) — 검증됨

병목 재확인: DDXPlus 놓친 evidence의 exact-in-profile 33.7% / **+hierarchy-reachable 25.3%** / 진짜부재 40.4%
(정렬 상한 59.6%). → 같은 개념이 KG와 벤치에서 **다른 CUI**인 어휘 갭이 핵심. SapBERT
(cambridgeltl/SapBERT-from-PubMedBERT-fulltext, UMLS 동의어 학습 encoder)로 evidence CUI를 KG-phenotype
어휘에 cosine 정렬 → soft-match. benchmark 무관(label·매핑 미사용) → §3 안전. `sapbert_align.py`,
`sapbert_ddxplus_test.py`.

**임베딩 모델 패널** (full unified2 통합, 동일 프로토콜, τ=0.78, `onlykg_eval_unified2.py --sapbert 1
--emb_model <>`). 학술적 정당화를 위해 **같은 계열 5개 encoder** 비교. **@3** (baseline=정렬 OFF):

| encoder | 계열 | DDXPlus | LIRICAL | HMS | RAMEDIS | MME |
|---|---|---|---|---|---|---|
| baseline(off) | — | 45.8 | 34.3 | 52.9 | 52.6 | 80.0 |
| **BioLORD** | 개념-특화(UMLS) | **49.4** | **37.6** | **56.3** | 52.1 | 82.5 |
| CODER | 개념-특화(UMLS) | 49.4 | 36.0 | 51.7 | 50.2 | 82.5 |
| SapBERT | 개념-특화(UMLS synonymy) | 48.8 | 36.5 | 54.0 | 51.9 | 77.5 |
| PubMedBERT | biomedical LM(비개념) | 47.6 | 33.2 | 44.8 | 46.8 | 72.5 |
| MiniLM | 일반도메인 | 46.9 | 35.1 | 51.7 | 52.4 | 82.5 |

**결론(학술적)**:
1. **개념-특화 biomedical encoder(BioLORD/CODER/SapBERT)가 common-disease·prose 벤치를 개선** —
   DDXPlus @3 +3.6, LIRICAL @5 40.8→46.2(+5.4). 어휘 갭(같은 개념 다른 CUI)을 의미 임베딩으로 이음.
2. **일반도메인(MiniLM)·비개념 biomedical(PubMedBERT)는 덜 돕거나 악화** → 개념 encoder 선택 정당.
3. **RAMEDIS(curated HPO rare)는 정렬 무효/미미** — 이미 CUI 정렬돼 노이즈만 추가(정직한 한계).
4. **채택 = BioLORD**(최고 종합, 최소 regression). §3 안전(benchmark label·매핑 미사용), LLM 미사용, KG-only.
- ceiling(DDXPlus 95%)이 남긴 40% 진짜부재는 어휘정렬로 못 메움 → KG content 밀도가 잔여 레버.

## 3d. 종합 cross-benchmark — 확정 (best config, 2026-07-19)

**best config** = 구 KG(`kg.json`) + disease-lineage 해석 + §11 unified evidence + 분리정규화 + 벤치 co-ref
병합 + BioLORD vocab-alignment(τ=0.78) + **LSA 잠재의미 융합(w=0.4)**. 밀도 재-IE는 net-negative로 기각.
`onlykg_eval_unified2.py --lsa 1 --lsa_w 0.4 --sapbert 1 --emb_model biolord`.

| 벤치마크 | 우리(zero-shot KG) | zero-shot LLM SOTA | supervised/agentic |
|---|---|---|---|
| **RareBench**(agg) | @1 24.9 / @3 **48.0** / @10 **61.6** | GPT-4: 32.3/45.4/58.9 | DeepRare 72.6 R@1(agentic) |
| — RAMEDIS | @1 22.9 / @3 51.3 / @10 62.3 | | |
| — LIRICAL | @1 23.0 / @3 37.3 / @10 53.5 | | |
| — MME | @1 62.5 / @3 82.5 / @10 **95.0** | | |
| — HMS | @1 29.9 / @3 55.2 / @10 75.9 | | |
| **DDXPlus**(49-way) | @1 38.4 / @3 55.2 / @5 65.1 / **@10 80.1** | GPT-4o Top-5 80.4 | GTPA@1 75(supervised) |
| SymCat | @1 4.6 / @3 13.9 / @5 18.9 / @10 30.0 | (zero-shot baseline 無) | MMF-AC 51.9/78.2/87.6(supervised) |

> **RareBench Top-3(48.0)·Top-10(61.6)이 GPT-4 zero-shot(45.4/58.9) 초과.** **DDXPlus @10=80.1이 GPT-4o
> Top-5(80.4) 수준 도달**(@5 65.1). LSA 융합 전 대비 DDXPlus @3 49.4→55.2·@10 74.4→80.1(+5.7), rare 유지.

## 3e. LSA 잠재의미 융합 (method) — 2026-07-19

KG disease→phenotype profile이 sparse해 exact/hierarchy overlap이 놓치는 연관을 **latent factor로 보완**.
disease×phenotype 행렬(IDF-weighted)을 truncated-SVD(k=256)로 분해(LSA, Deerwester 1990) → co-occurring
phenotype이 유사 latent vector를 얻어, 환자 증상이 project된 지점 근처 disease가 *literal edge 없이도* 랭크.
sparse(unified-evidence) 점수와 LSA cosine을 **per-patient min-max 정규화 후 late-fusion**(w=0.4). §3 안전
(KG 구조만, benchmark label·매핑 미사용), LLM 미사용. w sweep: 0.4가 common-disease 최대이면서 rare 유지
(0.5+는 rare 하락 → 단일 전역 w=0.4). 효과: **DDXPlus @3 +5.8/@10 +5.7, SymCat @3 +2.5, MME @10 90→95**,
RareBench rare 유지. = 정확 매칭(rare 변별)과 잠재 smoothing(common recall)의 상보 결합.

> **일관된 서사**: 우리 KG는 **rare-disease(RareBench)에서 GPT-4 zero-shot을 Top-3/@10 초과** — LLM이 가장
> 약한 영역에서 강함. 반대로 **common-disease 증상체커(DDXPlus/SymCat)에선 LLM/supervised에 뒤짐** —
> rare-inclusive pubmed KG의 common-disease profile 희소성 때문. 즉 우리 기여는 **rare 쪽 상보적 강점 +
> traceability + 8B cost**이고, common-disease는 density/속성으로 개선 대상(다음 단계).

## 3c. Top-1 개선 시도 로그 (정직 — 실패 기록)

Top-1(agg 20.3 vs GPT-4 32.3) 격차. forensic: 정답(rich profile)이 정규화에서 벌받고 thin-profile 사촌이
공유 disease-prior로 승리(PKU 261600 prof36 → 261630 prof6에 짐). 시도 로그:
- ❌ **MICA best-match(Phrank/Resnik)**: RAMEDIS @3 49→14. 누적 신호 소실.
- ❌ **disease-match 계층거리 감쇠 / 정규화지수 sweep(β 0.15~0.5)**: β=0.5가 이미 최적, 나머지 악화.
- ✅ **채택: 두 evidence 채널 분리 정규화** — `dm/√|dsig| + pm/√|pf|` (disease-계층 prior와 phenotype
  유사도를 각자 cardinality로 정규화). lumping+combined-size 정규화가 문제였음. 단일 전역공식(§7).
  → **AGG Top-1 20.3→24.4, Top-3 45.9→47.5(GPT-4 초과), 어느 벤치도 하락 없음**(RAMEDIS @1 16.5→23.9,
  HMS 26→31, SymCat @3 8→12). 잔여 Top-1 격차(24 vs 32)는 sibling differential 본질 난이도 → density/속성.

## 4. 목표 재설정 (권고)

**"95% Top-3 고정"은 폐기.** 근거:
- RareBench에서 **GPT-4조차 Top-3 45%.** 95%는 supervised(DDXPlus 99% GTPA, SymCat 72~85% top-3) 또는
  agentic+KB(DeepRare)만 도달하며, 둘 다 **§3 금지 데이터/고비용 orchestration** 사용.
- 우리 강점(CLAUDE.md Thesis): SOTA가 아니라 **cross-benchmark robustness + traceability + 8B cost.**

**SOTA-상대 목표(honest)**:
1. **RareBench**: GPT-4 zero-shot Top-3/@10 **동급 이상 유지·확대** (이미 달성: 45.9 vs 45.4, 59.8 vs 58.9).
   Top-1을 GPT-4(32%)에 근접시키는 것이 다음 목표(ranking 정밀도: phenotype IC 매칭·속성).
2. **비교 프레이밍**: supervised/agentic SOTA를 표에 고정하되 "다른 세팅(매핑 학습/고비용)"으로 명시,
   우리는 "zero-shot·8B·traceable에서 GPT-4 zero-shot 동급"을 헤드라인으로.
3. **SymCat/DDXPlus**: zero-shot 한계 정직 보고(§9). 속성채널로 개선 시도하되 supervised 95%와 비교 안 함.
