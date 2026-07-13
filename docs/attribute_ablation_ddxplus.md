# 속성(attribute) ablation — DDXPlus (2026-06-03)

교수님 자문(속성을 표준에서 정의, NLICE 방법 차용)에 따라 evidence 속성의 진단 기여를
DDXPlus에서 실측. 최대 속성 = OLDCARTS/OPQRST/SOCRATES/NLICE/LOCATES/CPX 합집합 **13개**로 IE
(`v104_attr_ie.py`, benchmark-blind, gemma-4-E4B). 단 **DDXPlus 환자측 값이 존재하는 속성은 6개**.

## 1. DDXPlus 환자측 속성 커버리지 (13 중 6)

| 활성 6 | DDXPlus 접미사 | 보유 케이스(/3000) |
|---|---|---|
| location | endroitducorps, precis | 2389 |
| onset | soudain | 2422 |
| severity | intens, prurit, sev | 2471 |
| character | carac, Aboy | 914 |
| radiation | irrad | 648 |
| timing | noct, nuit | 138 |
| **비활성 7** | duration/aggravating/relieving/associated/course/context/prior_episodes | 0 |

→ DDXPlus는 6속성만 테스트 가능. 나머지 7은 **SymCat(NLICE)·RareBench(HPO modifier)에서만** 검증 가능
(= 교수님 cross-benchmark robustness 논리).

## 2. 방법

- 속성-rich IE 1회(질환측 13속성 전부) → 채널 분리: location/radiation/character는 UMLS CUI
  (환자=value_cuis, 질환=scispaCy), severity/onset/timing은 버킷 토큰(mild/moderate/severe,
  sudden/gradual, nocturnal). base = bare 증상+병력 CUI(속성값 제거).
- 가산적 scoring: 채널별 IDF² overlap / 고정 질환 norm → 부분집합 점수 = base + Σ 선택 속성.
- 전 부분집합(2⁶=64) 평가. (`v104_build_attr_vectors.py`, `v104_ablation.py`)
- 주의: 이 base는 단일 v104 IE만(전체 union 아님)이라 절대점수 낮음(@1 12%). **상대 기여가 결론.**

## 3. 결과 (n=3000)

| k | best @1 | best @10 | 최적 속성 |
|---|---|---|---|
| 0 | 12.03 | 41.33 | base |
| 1 | 15.87 | 57.57 | **location** |
| 2 | 16.37 | 59.27 | location+severity |
| 4 | 16.33 | 63.67 | location+severity+character+timing |
| 6 | 14.73 | **64.17** | 전체 |

**속성 추가가 @10을 41.3→64.2% (+22.8%p)** 향상.

### 속성별 변별 기여
| 속성 | base+1 @10 | leave-one-out Δ@1 | 평가 |
|---|---|---|---|
| **location** | **57.6** (+16.2%p) | **+4.20** | 압도적 1위 |
| severity | 44.5 | @10 기여(−1.07 trade) | 보조 |
| onset | 42.2 | @10 기여(−1.00 trade) | 보조 |
| character | 46.5 | ~0 | 약 |
| radiation | 42.3 | ~0 | 미약(17질환) |
| timing | 42.2 | 0.00 | 무효(138케이스) |

## 4. 결론

1. **location이 압도적 지배 속성** — @10 단독 +16%p, leave-one-out @1 −4.2. Panju(JAMA 1998)의
   "흉통 위치/방사가 진단을 가른다"와 anatomical-IE 발견(hernia 59→24)을 정량 확증.
2. **@1 vs @10 trade-off**: @10은 전체 6속성 최선(64.2), @1은 소수(location+severity, k=2)에서 최선(16.4).
   잡음 속성(radiation/timing)이 @1 희석.
3. **timing은 DDXPlus에서 무효**(환자 138케이스). radiation도 미약(17질환만 IE 보유).
4. 비활성 7속성은 DDXPlus 구조상 테스트 불가 → **SymCat/RareBench cross-benchmark에서 13속성 ablation 필요**.

## 5. 다음
- 활성 6속성을 전체 union KG(현 best @1 47.7/@10 94.4)에 qualified-edge로 통합해 @10 추가 검증.
- SymCat(NLICE 5속성)·RareBench(HPO onset/severity)로 나머지 7속성 ablation → cross-benchmark robustness 입증.
- 교수님 지침대로 속성=qualified statement(노드 아님), HPO 정규화(HP:0012823/HP:0003674/HP:0012824), numeric→edge weight 폐기.
