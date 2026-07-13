# Recursive Bidirectional IE Methodology

## Motivation

기존 IE pipeline은 disease CUI → PubMed → phen CUI **단방향** 추출. 결과 KG의 한계:
1. **Orphan phen**: 추출된 phen CUI 중 다수가 다른 disease와 연결 없이 isolated
2. **환각 검증 불가**: gemma-4-E4B가 추출한 phen이 실제 그 disease와 연관 있는지 단방향 신호로는 검증 못함
3. **Phen-phen co-occurrence 정보 없음**: 같은 disease의 phen 간 의학적 연관 미반영

## 핵심 아이디어

모든 evidence (disease, symptom, finding 등)를 동등한 CUI 노드로 취급. 추출된 evidence를 새 seed로 재귀적 IE 수행하며 **양방향 confirmation**으로 환각 필터링.

**Depth-k에서 모든 evidence를 seed로 사용** (a 하나만이 아니라 a, b, c, d, e 모두):
```
Depth-1: A → {a, b, c, d, e}        (기존 방식: disease A seed)

Depth-2: ALL of {a, b, c, d, e}를 seed로 사용
   a → {f, g, A?, ...}              (A 다시 나오면 A↔a 양방향 ✓)
   b → {h, i, A?, ...}              
   c → {j, k, A?, ...}
   d → {l, m, ...}                  (A 안 나오면 A→d 환각 의심 ✗)
   e → {n, o, A?, ...}
   → S₂ = {f, g, h, i, j, k, l, m, n, o, ...} 모두 다음 depth seed

Depth-3: ALL of S₂를 seed로 사용
   f → {..., a?, b?, ...}           (a, b 재발견 → 양방향 확인)
   g → ...
   ...
```

### CUI 매칭 정확성 (필수)
- IE 출력 string → scispaCy `en_core_sci_lg` + UMLS linker로 CUI 매핑
- MRCONSO 기반 직접 string-match도 병행 (preferred name, synonyms)
- 양방향 confirmation은 **CUI level 매칭**만 사용 (string 매칭 노이즈 제거)
- 동일 concept의 다른 CUI 표현 → MRREL의 RT (related) 또는 same-AUI 합치기 권장

## 정형 정의

### 입력
- **Seed pool** S₀ ⊂ UMLS CUI (universal medical concepts)
  - Initial S₀ = UMLS DISO CUIs (38,456) — 기존 v13 source
- **Source corpus** C: PubMed abstracts (universal)
- **IE 모델** M: gemma-4-E4B-it (vLLM batch)
- **Normalization** N: scispaCy `en_core_sci_lg` + UMLS linker

### Depth-k 재귀 IE
```
S_k = ⋃ extract_evidence(M, search(C, name(s))) for s in S_{k-1}
E_k = {(s, e, k) : s ∈ S_{k-1}, e ∈ extract(s)}  # directed edges with depth
```

### Bidirectional confirmation
edge (a, b, depth=k) 가 confirmed 되려면:
- ∃ edge (b, a, depth=k') in E (양방향 발견)
- OR co-occurrence count (a, b) in single document > threshold

### Weight 조정
| Confirmation type | Weight multiplier |
|---|---|
| Bidirectional (A↔a in depth ≤2) | × 2.0 |
| Co-occurrence (multi-source) | × 1.0 (기본) |
| Unidirectional with high frequency | × 0.7 |
| Unidirectional with single source | × 0.3 (환각 의심) |
| Depth > 3 unidirectional | × 0.1 (very weak) |

### 평가 기준
- **Primary**: DDXPlus GTPA@1 변화량 (현재 SOTA 48.12% 대비)
- **Secondary**:
  - Bidirectional confirmation rate (전체 edges 중 양방향 비율)
  - Hallucination filtered rate (낮은 weight로 demote된 edge 비율)
  - Cross-benchmark (SymCat, RareBench) 일관성

## 단계별 실행 계획

### Phase 1: Phen seed extraction (Depth-2 준비)
- v13 KG의 49 disease 각각 top-12 signature phen CUI 추출
- 중복 제거 → **147 unique phen CUI** (예: fever, fatigue, headache 등 공유)
- UMLS preferred name 매핑

### Phase 2: Phen-seeded PubMed IE (Depth-2, ALL evidences)
- 147 phen CUI 각각으로 PubMed query (a, b, c, d, e ... 모두)
- gemma-4-E4B IE → 추출된 disease/phen CUI 수집
- scispaCy + MRCONSO로 CUI 매핑 (string 매칭 노이즈 제거)
- 결과: `edges_phen_seeded_ie.jsonl` (depth-2)

### Phase 3: Bidirectional analysis (Depth-1 ↔ Depth-2)
- Depth-1 edges (disease → phen) vs Depth-2 edges (phen → disease)
- 양방향 confirmed edge ratio 측정
- 단방향 edge 중 환각 후보 식별 (low source frequency)

### Phase 4: v18 KG build + 평가
- Bidirectional weight × 2.0
- Unidirectional × 0.3 ~ 0.7 (frequency 따라)
- DDXPlus 30K 평가 → SOTA 48.12% 대비 변화 측정

### Phase 5: Depth-3 확장 (모든 depth-2 결과 활용)
- Phase 4 결과 향상 시 진행
- Depth-2에서 발견된 모든 새 CUI를 seed로 PubMed re-crawl
- Depth-1↔3 양방향 확인 가능 (long-range hallucination 검출)
- 비용: depth-2가 100 → 500 CUI 발견하면 depth-3은 500개 추가 crawl

### Phase 6: Depth-K iteration (조건부)
- Depth-3 추가 향상 시 깊이 늘림
- Convergence: 새 CUI 비율 < 5% 또는 SOTA 향상 < 0.1%p

## 환각 검증 사례 (예상)

```
A = "HIV (initial infection)" (C0001175)
Depth-1 IE 결과: a = "fever", b = "fatigue", c = "lymphadenopathy", 
                  d = "vocal cord paralysis" (← 환각 의심)

Depth-2 IE:
  fever → 수천 disease (HIV 포함) ✓ A↔a 확인
  fatigue → 수천 disease (HIV 포함) ✓ A↔b 확인  
  lymphadenopathy → 수백 disease (HIV 포함) ✓ A↔c 확인
  vocal cord paralysis → laryngeal/thyroid disease only (HIV 없음) ✗
  
→ A→d 단방향 환각 의심 → weight demoted to 0.3
```

## 제약 조건 유지
- Universal medical vocabulary (UMLS CUI + preferred name)만 사용
- DDXPlus 49 영문 이름 직접 사용 금지
- No curated KG (HPO, Orphanet, SemMedDB) 사용 금지
- No train labels 사용

## 예상 결과 시나리오

| Scenario | Expected DDXPlus @1 | Insight |
|---|---|---|
| Bidirectional confirmed edges만으로 빌드 | ≥48.12% | 환각 제거가 ceiling 넘김 |
| 양방향 weight boost (×2) + 단방향 demote (×0.3) | +1-3%p | 일부 향상 가능 |
| Phen-phen edges 추가 (co-occurrence) | unclear | 새 신호 활용 가능 |
| No change (양방향 비율 낮음) | ≈48% | Method ineffective on closed-set |

## 실제 결과 (2026-05-13, v18 PoC)

| Configuration | DDXPlus @1 | Δ vs v13 |
|---|---|---|
| v13 baseline | 48.12% | - |
| v18 (bidirectional boost ×1.5 + 1,427 new unidir edges weight 0.3) | 47.60% | -0.52%p |
| v18b (bidirectional only ×1.5, no new edges) | 47.93% (15K) | -0.0%p (noise) |
| v18b ×2.0 | 47.57% | -0.4%p |
| v18b ×3.0 | 46.57% | -1.4%p |

**측정 데이터**:
- 147 phen CUI seeds → 2,874 d2 IE records → 1,588 resolved d2 edges (after scispaCy + MRCONSO)
- 40,728 d1 edges in v13 (depth-1: disease → seed-phen)
- **Bidirectional confirmation rate: 0.28% (114/40,728)**
- New d2 → DDXPlus 49 edges: 10 (5 already in v13, 5 newly discovered)

## 진단된 한계

### 1. PubMed search-based bidirectional의 본질적 한계
Phen이 generic할수록 (fever, fatigue, headache 등) PubMed query 결과가 광범위 disease 분포. DDXPlus 49 specific disease로 narrow 안 됨. 따라서 d2에서 same disease 재발견 비율 매우 낮음.

→ "0.28% bidirectional rate"는 환각 비율이 아니라 **closed-set 한계의 신호**.

### 2. CUI 매핑 손실
- 2,300 unique phenotype text → 808 (MRCONSO) → 924 (+ lemma) → resolve 33-40%만
- scispaCy cache 더해도 ~50% 수준

### 3. Generic phen의 noise
- IDF 재계산 시 다수 disease가 같은 generic phen 공유 → discrimination power 희석
- Bidirectional boost가 specific phen에 적용되어도 IDF 분포 변화로 다른 phen weight 감소

## 향후 방향 (long-cycle)

### Same-document co-occurrence-based bidirectional
PubMed abstract 단위에서 **모든 medical entity** 추출 (disease-anchored 아님):
1. scispaCy NER + UMLS linker로 abstract 내 모든 CUI 추출
2. abstract 안의 (entity₁, entity₂) pair = bidirectional confirmed co-occurrence
3. Co-occurrence count로 edge weight 결정
4. 단일 source / single-mention edge → 환각 후보

이는 현재 disease-anchored IE pipeline의 fundamental change 필요. 그러나 진짜 양방향 검증 가능.

### 결론

**Search-based recursive bidirectional IE는 closed-set DDXPlus 49 disease에서는 효과 없음**. Phen의 universality (generic) 때문. SOTA 48.12% 유지.

진정한 양방향 검증을 위해서는 corpus-internal co-mention 분석 필요.
