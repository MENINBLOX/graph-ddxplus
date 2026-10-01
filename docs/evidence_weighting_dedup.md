# Evidence Over-representation & Weighting (#8) — 학술 방어 가능한 방법

**질문**: IE 후 "중복이 많은 것들"(과대표현 evidence)을 어떻게 제거/처리하는가?
hand-built stop-list(현 `kg_diagnose_v41` STOPWORDS)은 너무 단순 → 인용 가능한 표준 방법으로 대체.

**원칙**: 과대표현 evidence 는 **하드 제거(임계치)** 하지 않고 **연속 가중치로 down-weight**.
(임계치 폐기 = UMLS linking 에서 이미 채택한 정책과 일관, [[ie-umls-linking-2026-07]].)
구조적 noise(qualifier/demographic)만 semantic-type 로 제거.

---

## 1. 세 가지 "중복"을 구분

| 유형 | 예 | 처리 | 근거 |
|------|-----|------|------|
| (a) disease 내 중복 mention | 한 disease 소스에서 "pain" 3회 | CUI **set-dedup** | 정규화 단계에서 이미 처리 |
| (b) 구조적 noise CUI | Woman, year, Patient, Severe(단독), Age | **semantic-type stop-list 제거** | CLAUDE.md §12; MetaMap/cTAKES semantic filtering |
| (c) 과대표현 *유효* phenotype | Pain, Fever (deg>1000, 의학적으로 정상) | **제거 안 함 → IDF/IC down-weight** | Spärck Jones 1972; Resnik 1995; Phrank 2019 |

(b)와 (c)의 구분이 핵심: Pain 은 hub 지만 유효 임상 개념 → 제거하면 정보 손실.
Woman/year 는 hub 이면서 무의미 → 제거. **빈도만으로 자르면 (c)까지 날아간다** → 빈도 임계치 금지.

## 2. (b) 구조적 noise = semantic-type stop-list (threshold-free)

- UMLS semantic type(TUI)로 판별. Demographic/temporal/qualifier 타입 = 제거:
  - 예: T098(Population Group=Woman/Patient), T079(Temporal=year/day), T080(Qualitative),
    T081(Quantitative), T170(Intellectual Product), age/sex lexical.
- **임계치 아님** = 타입 membership(결정론적). MetaMap 의 semantic-type 필터, cTAKES 와 동일 계열.
- 유효 임상 타입(T184 Sign/Symptom, T047 Disease, T033 Finding …)만 evidence 로 인정
  = 이미 IE 정규화의 clin-type 필터(`FINDING_TUI`)와 동일 장치. KG 빌드 시 재적용.

## 3. (c) 과대표현 phenotype = Inverse Document Frequency / Information Content

disease–phenotype bipartite KG 를 "문서(disease) × 용어(phenotype)" 행렬로 보면:

- **IDF**(Spärck Jones 1972): `idf(p) = log( N_disease / df(p) )`, df(p)=phenotype p 를 가진 disease 수.
  Pain(거의 모든 disease) → idf≈0(기여 0), 희귀 phenotype → 높은 idf. **연속·임계치 없음.**
- **Information Content**(Resnik 1995): `IC(p) = -log( freq(p)/N )`. IDF 와 동형.
  희귀질환 진단 표준(Köhler 2009 Phenomizer, **Jagadeesh 2019 Phrank**, Bauer 2012)이
  HPO IC 로 phenotype 유사도를 가중 → 우리 rare-inclusive 세팅과 정확히 부합.
- **PMI**(disease-phenotype 연관강도, 선택): `pmi = log p(d,p)/(p(d)p(p))`.

### 진단 알고리즘 통합 (CLAUDE.md §7 단일 algorithm)
현 v41 Naive-Bayes log-likelihood 에 phenotype 별 IDF/IC 가중을 곱해 **weighted log-likelihood**:
```
score(d) = Σ_{p∈evidence}  w(p) · log P(p | d)      # w(p) = idf(p) 또는 IC(p)
```
- Naive-Bayes 자체도 흔한 evidence 를 부분적으로 약화하나, 명시적 IC 가중이 **희귀-phenotype
  변별력**을 살려 rare benchmark 에서 특히 유효(Phrank 의 핵심 결과).
- **hand-built STOPWORDS 삭제** → (b) semantic-type 필터 + (c) IC 가중으로 대체.
  = 재현 가능·인용 가능·모든 benchmark 동일 적용(per-benchmark tuning 없음, §7 준수).

## 4. 왜 학술적으로 방어되는가
- IDF/IC/semantic-type 필터는 IR·생의학 phenotype 진단의 **표준 인용 문헌**(1972~2019).
- 전부 **threshold-free 연속 가중** 또는 **결정론적 타입 membership** → magic number 없음.
- 단일 KG·단일 algorithm 에 동일 적용 → strict zero-shot·cross-benchmark 제약 유지.

## 5. 검증
- KG 빌드 후 `kg_topology_inspect.py`(CLAUDE.md §12): hub CUI 목록에서 (b) noise 가
  제거됐고 (c) 의학 hub(Pain/Fever)만 남는지 확인. IDF 분포로 과대표현 tail 확인.
- 효과 = **진단 성능(GTPA@1)** ablation: {stopword-list} vs {semantic-filter + IC-weight}.
