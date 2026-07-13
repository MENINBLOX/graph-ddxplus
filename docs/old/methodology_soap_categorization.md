# IE Methodology — SOAP Categorization (Subjective/Objective)

## 1. 동기

벤치마크별로 environment input의 evidence 종류가 다름:

| Benchmark | Input | Evidence type |
|---|---|---|
| DDXPlus | 환자 자가 응답 questionnaire | **Subjective only** (110 symptom + 113 history, no lab/imaging) |
| SymCat | Lay symptom slug | **Subjective only** (474 lay symptoms) |
| NLICE | SymCat-derived synthetic narrative | **Subjective only** |
| RareBench HPO | HPO codes | **Mixed** (HPO 내부에 sign + lab + imaging 모두) |
| AMELIE | HPO codes + variants | **Mixed** (≈ RareBench) |
| ER-Reason | Clinical narrative | **Full SOAP** (subjective HPI + objective exam/lab/imaging) |

→ KG 구축 시 evidence 종류 명시 필요. eval 시 fair comparison 위해.

## 2. 카테고리 설계 (2-tier)

### Level 1: SOAP 표준 (Weed 1968)

| Level 1 | 정의 |
|---|---|
| **subjective** | 환자가 직접 지각/보고 가능 |
| **objective** | 의료진의 측정/관찰/검사 결과 |

### Level 2: 세분화 (KG inspection / optional ablation)

```
subjective:
  ├── symptom              # chest pain, fatigue, blurred vision
  ├── history              # smoking, prior MI, recent travel
  └── demographic          # age, sex, ethnicity, pregnancy

objective:
  ├── physical_sign        # jaundice, edema, murmur (PE finding)
  ├── lab_finding          # elevated troponin, leukocytosis
  └── imaging_finding      # consolidation on CXR, ST elevation
```

## 3. Eval 정책 — 단일 알고리즘, 벤치마크-blind

**원칙**: 벤치마크별로 eval mode를 다르게 설정하지 않음 (cherry-picking 의심).

**기본**: 모든 벤치마크에 KG 전체 (subjective + objective) 사용. 환자 input이 자연스럽게 결정:
- DDXPlus 환자 input은 subjective CUI만 → subjective KG edges에만 자연스레 match
- ER-Reason 환자 input은 양쪽 → 양쪽 모두 match

**Ablation (선택사항)**: 별도 reporting으로 `--eval_mode subjective` vs `comprehensive` 비교.
이는 정당한 ablation — "subjective-only eval이 input data structure에 매칭됨" 으로 설명.

## 4. IE 프롬프트 v4 (`pilot/scripts/medkg_ie_universal_v4.py`)

Few-shot examples (3개, benchmark-blind):
1. **Iron deficiency anemia** — subjective(fatigue) + objective(low Hb)
2. **Sarcoidosis** — subjective(cough) + objective(hilar lymphadenopathy)
3. **Pulmonary embolism** — subjective(chest pain) + objective(D-dimer, CT)

각 예시가 6 sub-category 모두 등장 → 균형 학습. DDXPlus 49 / SymCat 50 / RareBench / ER-Reason 의 dominant disease와 직접 overlap 없음.

## 5. 출력 포맷

```
CAT: <level1>.<level2>: <evidence text>
```

예시:
```
CAT: subjective.symptom: chest pain
CAT: subjective.history: smoking history
CAT: objective.physical_sign: jaundice
CAT: objective.lab_finding: elevated troponin
CAT: objective.imaging_finding: ST elevation on ECG
```

KG edge에 `level1`, `level2` 속성 저장 → eval 시 mode filter 가능.

---

# IE Methodology — "꼬리물기" Recursive Expansion (검증 완료, marginal gain)

## 1. 핵심 아이디어

각 seed CUI에서 IE 수행 → 추출된 새 CUI를 다음 seed로 → 새 CUI 없을 때까지 반복 → 다음 원래 seed로.

```
for each seed S (UMLS DISO CUI):
    queue = [S]; visited_local = {S}
    while queue:
        cui = queue.pop()
        if not in /mnt/medkg/pubmed_alt: crawl(cui)
        evidences = IE(cui's PubMed corpus)
        for ev_cui in evidences:
            if ev_cui not in visited_local:
                visited_local.add(ev_cui)
                queue.append(ev_cui)  # 꼬리에서 새 꼬리
    # local convergence 후 다음 seed
```

## 2. 검증 결과

| 시도 | Task | DDXPlus @1 | 결과 |
|---|---|---|---|
| Recursive bidirectional IE | #163-167 (v18) | ~48.45% | 직접 IE 대비 +0.33%p (marginal) |
| BFS exhaustive depth-10 | #181 (Phase 3) | 비슷 | 시간 비용 대비 효과 미미 |
| Same-document scispaCy co-occurrence | #167-169 | 49.27% | +0.82%p (작음) |

**결론**: 꼬리물기 / 재귀 IE는 KG depth는 증가시키지만 **discriminative power는 비례 증가 안 함**. 새로 발견되는 CUI 대부분이 이미 일반적 의학용어 (Pain, Lung, Patient 등) — heavy-tail noise 증가.

## 3. 학술적 가치

직접 사용은 권장하지 않음. 하지만 ablation/contrast로 기록 가치 있음:
- "Naive recursive expansion did not significantly improve KG quality (49.27% vs 48.45% baseline)"
- KG quality bottleneck은 quantity가 아닌 **extraction precision**임을 입증

## 4. 결합 방안 (향후)

꼬리물기를 SOAP 카테고리와 결합하면:
- subjective CUI에서 → 새 subjective CUI만 follow (lay vocab 깊이)
- objective CUI에서 → 새 objective CUI follow (clinical depth)
- 결과: 카테고리별 specialized subgraph

이론적으로 흥미롭지만 실증 효과 미미 (#181 결과 기반).
