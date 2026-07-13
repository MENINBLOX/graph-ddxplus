# Cross-Benchmark Forensic Synthesis — v59 cosine

5 벤치마크에서 각 10 success + 10 failure 케이스 분석 결과 통합.
DDXPlus는 별도 `forensic_ddxplus_cases.md` 참고.

## v59 cosine performance summary (이전 보고)

| Benchmark | @1 | 평균 patient CUI count |
|---|---|---|
| DDXPlus 134K | 52.96% | ~13 |
| SymCat | 20.62% | ~4 (low!) |
| RAMEDIS | 27.76% | ~4 |
| HMS | (rerun needed) | ~10 |
| MME | (rerun needed) | ~3 |
| LIRICAL | 17.17% | ~6 |

## 실패 원인 — 벤치마크별 다른 mechanism

### 1. DDXPlus — Profile size dilution (cosine으로 해소)

- 호흡기 cluster 내 Bronchitis (110 CUIs profile) vs Bronchiolitis (82 CUIs)
- 같은 patient overlap이라도 작은 profile이 더 높은 평균 P(E|D) → NB에서 dilution
- **Cosine은 양쪽 normalize → 해소**
- v54 NB 46.89% → v59 cosine 52.96% (+6%p)

### 2. SymCat — 평균 patient CUI 매우 적음 (~4개)

| 분류 | avg patient CUI | true_score |
|---|---|---|
| Success | 4.2 | 0.4320 |
| **Failure** | **1.5** | **0.0000** |

**핵심 패턴**: 실패는 **patient CUI가 너무 적어** (1.5개) 매칭 자체가 안 됨.
- SymCat 환자는 binomial sampling으로 시뮬레이션 → 가끔 1-2 symptom만 추출
- patient ∩ profile = 0 인 경우 → score 0 → 무작위 rank

**해결 방향**: 
- SymCat 평가에서 최소 patient CUI 기준 적용 (≥5 CUI 필터)
- 또는 시뮬레이션 시 minimum symptom 강제

### 3. RAMEDIS — 부분적 profile coverage

| 분류 | avg patient CUI | true_score |
|---|---|---|
| Success | 3.8 | 0.1654 |
| Failure | 5.0 | 0.0294 |

**핵심 패턴**: 실패도 patient CUI는 많은데 (5개) **true 점수가 0.03** → 거의 매칭 안 됨.
- RAMEDIS true disease의 KG profile이 patient HPO와 vocabulary mismatch
- HPO codes (specific phenotypes)가 KG의 PubMed-extracted CUI와 다름

**해결 방향**: HPO ontology의 description text를 IE source에 추가

### 4. HMS / MME / LIRICAL — **True disease가 KG에 없음** (profile=0)

LIRICAL failure 샘플 분석:
```
Failure #3 (Hyper-IgE syndrome): top-1 profile=159, true=0
Failure #5 (DYRK1A syndrome):    top-1 profile=115, true=0
Failure #11 (Arthrogryposis):    top-1 profile=10,  true=54
```

대부분 true disease가 KG에 edges 없음 → score=-1e9 (sentinel) → 무조건 최하위

| Benchmark | Avg failure true_score | 의미 |
|---|---|---|
| HMS | ~-2e8 | 일부 disease profile 비어있음 |
| MME | ~-1e9 | 거의 모든 failure가 profile 없음 |
| LIRICAL | ~-5e8 | 절반 정도 profile 없음 |

**해결 방향**:
- BENCHMARK_COVERAGE.md에서 확인: LIRICAL 272 truth 중 KG cover 88.6%이지만 well-covered (≥20 edges) 59.6%
- IE source 확장 (OMIM Clinical Synopsis, GeneReviews 깊이) 필요

## 통합 결론

| 벤치마크 | 주요 실패 원인 | 해결 우선순위 |
|---|---|---|
| **DDXPlus** | profile dilution | ✅ **이미 해결** (cosine) |
| **SymCat** | patient input 너무 sparse (sim) | 평가 minimum CUI 필터 |
| **RAMEDIS** | HPO ↔ PubMed vocab mismatch | HPO description IE |
| **HMS** | KG profile depth 부족 | rare disease source 확장 |
| **MME** | KG profile depth 부족 | rare disease source 확장 |
| **LIRICAL** | KG profile 자체 비어있음 | rare disease source 확장 |

### 핵심 통찰

**DDXPlus의 cosine fix는 일반 통용 안 됨**:
- DDXPlus 같이 patient CUI ≥10개 + 모든 truth가 KG에 잘 covered되어 있을 때만 cosine이 우월
- Rare disease 환경 (LIRICAL)은 cosine보다 더 깊은 문제: **KG content missing**
- 단순 알고리즘 변경으로는 rare disease 해결 불가

**향후 작업 우선순위**:
1. SymCat sparse patient 필터링 (1시간)
2. HPO description text IE pipeline (1-2일)
3. OMIM Clinical Synopsis / GeneReviews / NORD rare disease source 추가 IE (2-3일)
4. 그 후 cosine + 새 KG로 6 벤치마크 통합 평가
