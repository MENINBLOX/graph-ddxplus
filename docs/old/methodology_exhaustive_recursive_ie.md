# Exhaustive Depth-K Recursive IE with Disease-Scoped Termination

## 1. 핵심 아이디어

각 disease A에 대해 **완전히 독립적으로** depth-k expansion 수행. 각 disease의 termination은 **자체 phen pool (A_phens) 만**으로 결정 — global phen pool 영향 받지 않음. 49 disease 모두 동등한 수준의 deep expansion 형성.

**Global phen pool 사용 안 함**: Disease Z도 Disease A처럼 충분히 deep expansion 받을 권리 있음. Disease 간 종료 조건은 독립.

**Global cache는 유지**: 같은 CUI에 대해 IE 중복 실행 방지 (실행 효율성용, 종료 결정에는 사용 안 함).

```
Disease A:
  Depth-1. A → {a, b, c, d, e}                 (A_phens=5)
  Depth-2. {a,b,c,d,e} → {f, g, h, i, a}       (A_phens=9, a는 이미 있음)
  Depth-3. {f,g,h,i} → {j, k, l, m, n}         (A_phens=14)
  ...
  Depth-50. {...} → {a, c, g, k} 모두 A_phens 내부 → 종료 (own termination)

Disease B (independent):
  Depth-1. B → {p, q, a, b, ...}               (B_phens=5)
  Depth-2. ...
  Depth-50. → 모두 B_phens 내부 → 종료          (A_phens과 무관, B 자체 expansion)

...
Disease Z:
  Depth-50. → 모두 Z_phens 내부 → 종료          (Z 고유 KG 충분히 형성)
```

**핵심 차이점**: B의 종료는 A_phens과 무관 — B 자체적으로 충분히 expansion 받음. 모든 49 disease가 자기 KG path 충분히 형성.

## 2. 정형 정의

### 2.1 자료구조

```python
# Per-disease state (independent expansion)
disease_phens[A] = set()      # A에서 발견된 모든 phen — termination 결정용
disease_edges[A] = []         # (src_cui, dst_cui, depth, abs_pmid) tuples
disease_cooccur[A] = []       # (e1, e2, pmid) — same-doc co-occurrence
disease_depth[A] = int        # 최종 도달 depth

# Global cache (efficiency only, NOT used for termination)
global_phen_cache = {}        # cui → IE 결과 (한 번만 실행, 모든 disease가 공유)
```

**중요**: `global_phen_pool` 없음. 각 disease의 종료는 오직 `disease_phens[A]` 자체 비교로만 결정. Disease 간 expansion 독립.

### 2.2 IE 실행 (per CUI, cached)

```python
def ie_extract(cui, name):
    """주어진 CUI에 대해 IE 실행 (cache 사용)."""
    if cui in global_phen_cache:
        return global_phen_cache[cui]
    
    abstracts = pubmed_search(name, retmax=20)
    extracted_cuis = set()
    cooccur_edges = []  # (e1, e2, pmid)
    
    for abs in abstracts:
        ents = scispacy_extract_cuis(abs.text)
        # Filter to medical semantic types (DISO, FNDG, SOSY, BODY)
        ents = {e for e in ents if is_medical_cui(e)}
        ents.discard(cui)
        extracted_cuis |= ents
        # Co-occurrence: 같은 abstract의 entity pair
        ent_list = list(ents)
        for i in range(len(ent_list)):
            for j in range(i+1, len(ent_list)):
                cooccur_edges.append((ent_list[i], ent_list[j], abs.pmid))
    
    result = {"extracted": extracted_cuis, "cooccur": cooccur_edges}
    global_phen_cache[cui] = result
    return result
```

### 2.3 Per-disease expansion

```python
def expand_disease(A_cui, A_name, MAX_DEPTH=200):
    A_phens = set()
    A_edges = []          # (src, dst, depth, pmid)
    A_cooccur = []        # (e1, e2, pmid)
    
    current_seed = {A_cui}
    
    for depth in range(1, MAX_DEPTH + 1):
        new_phens = set()
        for s in current_seed:
            s_name = cui_to_name[s]
            result = ie_extract(s, s_name)
            
            for e in result["extracted"]:
                if e not in A_phens and e != A_cui:
                    new_phens.add(e)
                    A_edges.append((s, e, depth, None))
            
            # Co-occurrence edges within this seed's abstracts
            for (e1, e2, pmid) in result["cooccur"]:
                # 양방향 모두 A_phens 또는 새 phen에 포함되어야 의미
                if (e1 in A_phens or e1 in new_phens) and \
                   (e2 in A_phens or e2 in new_phens):
                    A_cooccur.append((e1, e2, pmid))
        
        # ==== 종료 조건 ====
        # 1. 새 phen 없음 (전부 이미 본 것)
        if not new_phens:
            terminate_reason = "no_new_phen"
            break
        
        # 2. Relationship coverage: 새 phen들이 기존 phen과 충분히 연결됐는지
        if depth >= 5:  # 최소 5 depth는 진행
            new_with_link = sum(1 for p in new_phens 
                                if any((p, q) in A_cooccur or (q, p) in A_cooccur 
                                       for q in A_phens))
            coverage = new_with_link / len(new_phens) if new_phens else 1.0
            if coverage >= 0.8 and len(new_phens) < 5:
                # 새 phen 80%+ 이상이 기존과 연결됐고, 적은 양만 추가됨
                terminate_reason = "saturated"
                break
        
        # 3. MaxDepth 도달
        if depth == MAX_DEPTH:
            terminate_reason = "max_depth_reached"
            break
        
        A_phens |= new_phens
        current_seed = new_phens
    
    disease_phens[A_cui] = A_phens
    disease_edges[A_cui] = A_edges
    disease_cooccur[A_cui] = A_cooccur
    disease_depth[A_cui] = depth
    # global_phen_pool 사용 안 함 — 각 disease 독립적 종료
    
    return depth, terminate_reason
```

### 2.4 49 disease iteration

```python
for A in sorted(DDX_49):  # alphabetical or by priority
    depth, reason = expand_disease(A_cui, A_name)
    log(f"{A_name}: depth={depth}, reason={reason}, phens={len(disease_phens[A_cui])}")
```

## 3. 종료 조건 (3-layer)

| Layer | 조건 | 의미 |
|---|---|---|
| 1 | 새 phen 없음 | seed의 모든 추출 phen이 A_phens에 포함 |
| 2 | Saturation: 새 phen <5 AND 80% 이상이 기존과 co-occur | relationship 충분 형성 + 추가 noise만 |
| 3 | MaxDepth = 200 | 안전장치 (무한 루프 방지) |

**핵심**: 단순 phen 중복 체크가 아니라 **relationship coverage**를 함께 확인. 새 phen이 기존 phen들과 co-occurrence edge로 연결되어야 의미 있는 종료.

## 4. 비용 분석

### 4.1 IE 캐시 효과 (효율성용, termination에는 영향 없음)

- **Global cache**: 같은 CUI는 한 번만 IE 실행 (efficiency)
- 각 disease의 termination은 자체 `disease_phens[A]` 기반 — cache는 무관
- 후속 disease 실행 시간은 단축되지만 expansion 깊이는 영향 없음

### 4.2 예상 총 IE 횟수

각 disease가 독립적으로 자체 종료까지 expansion. 단 같은 CUI는 cache hit:

```
Disease 1 (e.g., Pneumonia): full expansion 50-100 depth, 1000+ unique CUI IE
Disease 2-49: 각자 자체 종료까지 (depth 비슷), 일부 CUI는 cache hit으로 query만 skip

총 unique CUI: ~5K-30K (medical phen vocabulary 크기에 의존)
- Cache hit 비율: 70-90% (49 disease가 비슷한 medical vocab 공유)
- Cache hit은 query 시간만 줄임, expansion 깊이는 동일
```

### 4.3 비용 예측

- **Disease별 expansion depth**: 평균 50-100 depth (real measurement Phase 1에서 확정)
- **PubMed query 시간**: ~20K CUI × 1.4s = 28K초 ≈ 8시간
- **vLLM IE 시간**: ~20K CUI × 20 abstracts ÷ 30/s = 13K초 ≈ 3.7시간
- **총 ~12시간** (한 번만 실행, full cache 형성)

이후 KG rebuild만 추가 (분 단위).

## 5. 노이즈 처리

### 5.1 Semantic type filter (필수)

```python
ACCEPT_TUI = {
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T047",  # Disease or Syndrome
    "T046",  # Pathologic Function
    "T037",  # Injury or Poisoning
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction
    "T023",  # Body Part, Organ, or Organ Component
}
```

다른 TUI (treatment, gene, study design 등)은 제외.

### 5.2 Edge weight

- (src, dst, depth) → weight = 1 / sqrt(depth+1)
  - depth-1: weight 0.71 (direct, strong)
  - depth-10: weight 0.30
  - depth-50: weight 0.14
  - depth-200: weight 0.07

→ Deep expansion은 자연스럽게 weight 낮아 noise 영향 감소.

### 5.3 Co-occurrence frequency
- (e1, e2) pair count ≥ N (예: 3)인 것만 KG에 추가
- Single abstract co-occurrence는 환각 가능성 → 필터

## 6. 점진적 진행 전략

### Phase 1: PoC (1 disease)
- 가장 잘 알려진 disease 1개 (e.g., Pneumonia)에 expansion
- depth, phen count, edge count 측정
- IE 캐시 vs 실시간 처리 비교
- Termination 정확히 동작하는지 확인

### Phase 2: Small batch (10 disease)
- 가장 자주 실패하는 10 disease (e.g., Pneumonia, HIV, Influenza, Acute laryngitis, Sarcoidosis 등)
- Per-disease depth 분포, 누적 캐시 효과 측정
- 시간 비용 검증

### Phase 3: Full 49 disease
- 모든 49 disease expansion
- KG build + DDXPlus 30K 평가

### Phase 4: Beyond 49 (선택)
- v13 KG의 모든 disease 또는 UMLS DISO 35K로 확장 (universal)

## 7. 기대 효과

### 7.1 직접 효과
- KG content quality: GT recall 12% → 50%+ 예상 (relationship-rich edges)
- Disease별 phen coverage: 평균 116 → 500+ (deep expansion)
- Cluster confusion 해결: 같은 cluster의 disease들이 서로 다른 deep phen path 형성

### 7.2 측정 지표
- @1, @3, @10
- MRR
- GT phen recall, our specificity
- Cluster confusion rate (HIV→Ebola 등)

### 7.3 예상 SOTA
- 현재 53.43% → **60-70%** 예상 (depth-3 결과 후 재추정)
- 80% 목표까지 잔여 26.57%p → 줄일 수 있는 path

## 8. 제약 조건 유지

- ✅ Universal medical vocabulary (UMLS CUI)
- ✅ No curated KG (HPO, Orphanet, SemMedDB) 사용 안 함
- ✅ DDXPlus 49 영문 이름 IE prompt 직접 사용 안 함 (UMLS preferred name만)
- ✅ Train labels 사용 안 함 (zero-shot)
- ✅ scispaCy + UMLS linker만 사용 (universal NER)
- ✅ gemma-4-E4B-it for IE (universal LLM)

## 9. 다음 액션

1. **Phase 1 PoC 즉시 시작**: 1 disease (Pneumonia) full expansion
2. depth, phen count, time 측정
3. Phase 2/3 시간 비용 검증 후 실행 결정
