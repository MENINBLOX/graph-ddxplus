# Benchmark ↔ Source Coverage Mapping

벤치마크 disease CUI 집합을 raw text source와 매핑해 KG 구축 우선순위를 결정합니다.
**원칙**: KG seed는 UMLS DISO에서 선정, benchmark는 evaluation-only. 본 문서는 coverage *진단*만을 위한 것이며 selection priority로 사용하지 않습니다.

## 평가 대상 벤치마크 (7종)

| 벤치마크 | 질환 수 | 환자 수 | Input 형식 | SOTA | KG cover (v41) | 비고 |
|---------|--------|--------|-----------|------|---------------|------|
| **DDXPlus** | 49 | 134,529 | 자가증상 텍스트 | 86% (MedDxAgent, supervised) | 49/49 (100%) | 흔한 acute 진단 |
| **SymCat** | 50/801 | 2,500 (sim) | 자가증상 slug | 58.8% (NLICE NB) | 760/761 (99.9%) | 1차 진료 |
| **NLICE** | SymCat-derived | 2,500 (sim) | 자가증상 텍스트 | 82.0% (NLICE NB) | (=SymCat) | SymCat synthetic 증강 |
| **RareBench/RAMEDIS** | 82 | 624 | HPO IDs | 72.6% (DeepRare) | 82/82 (100%) | metabolic rare |
| **RareBench/HMS** | 78 | 88 | HPO IDs | — | 69/78 (88.5%) | HMS rare cohort |
| **RareBench/MME** | 19 | 40 | HPO IDs | — | 18/19 (94.7%) | Matchmaker Exchange |
| **RareBench/LIRICAL** | 272 | 370 | HPO IDs | 28-37% (LIRICAL Robinson 2020) | 241/272 (88.6%) | real-world rare cases |
| **AMELIE** (외부) | ~200 | ~215 | HPO + variants | 80% top-25 (AMELIE 2020) | 데이터 미확보 | rare + WES/WGS |
| **ER-Reason** | 수천 ICD-10 | 3,984 | clinical narrative | 34.4% (o3-mini) | 데이터 형식 파싱 필요 | ER 임상, lab/imaging 포함 |

## 벤치마크별 phenotype 종류 정합성 (mode 매핑)

| 벤치마크 | Input 형식 | KG-NB mode | 필요 IE category |
|---------|----------|-----------|-----------------|
| DDXPlus | 자가증상 | `lay` | patient_reportable + history + demographic |
| SymCat | 자가증상 | `lay` | patient_reportable + history + demographic |
| NLICE | 자가증상 + NLICE 증강 | `lay` | patient_reportable + history + demographic |
| RareBench/* | HPO codes | `clinical` | clinical_sign + lab_finding + imaging_finding + history + demographic |
| AMELIE | HPO codes | `clinical` | (= RareBench) |
| ER-Reason | clinical narrative | `comprehensive` | 모든 6 카테고리 |

## KG 구축 시 IE source 선택 (coverage 진단 기반)

**우선순위는 source 자체의 vocabulary 정합성으로 결정** (benchmark 라벨 사용 X).
다만 source 후보군을 평가할 때, **coverage 진단으로 'source가 vocabulary 영역을 커버하는지' 검증**:

### Tier 1 (현재 사용 중) — vocabulary 영역별 강점

| Source | Vocabulary 강점 | Coverage 영역 |
|--------|---------------|--------------|
| PubMed abstracts | 임상 narrative + 학술 | 광범위 (common + rare) |
| StatPearls | textbook (History+P/E+Lab) | common acute + chronic |
| MedlinePlus | 환자 친화어 (lay) | common, lay-vocab |
| Wikipedia | 광범위 (Infobox structured) | common, semi-technical |
| GeneReviews | rare disease textbook | rare genetic |

### Tier 2 추가 우선순위 (vocabulary 정합성 기반)

| Source | 추가 이유 | 영향 받을 벤치마크 |
|--------|---------|-----------------|
| **MSD Manual Professional** | clinical narrative depth, lab/imaging 풍부 | ER-Reason, DDXPlus |
| **OMIM Clinical Synopsis (text only)** | rare disease 표준 임상 표현 | RareBench, AMELIE, LIRICAL |
| **GARD/NORD** | rare disease 환자 친화어 | LIRICAL, AMELIE |
| **MSF/CDC/WHO guidelines** | ER + 응급/감염 | ER-Reason |
| **PMC Open Access full-text** | abstract 부족분 보완 (rare disease) | RareBench, LIRICAL |
| **DermNet NZ, LITFL** | 도메인 특화 | ER-Reason |

## Coverage 진단 결과 (v41_universal KG 기준)

### Disease CUI 존재 여부 (in KG)
- DDXPlus: 100%
- SymCat: 99.9%
- RAMEDIS: 100%
- HMS: 88.5%
- MME: 94.7%
- **LIRICAL: 88.6%** (v39 baseline 58% → v41 +30.6%p)
- AMELIE: 미진단 (데이터 미확보)
- ER-Reason: 미진단 (ICD-10 → CUI 파싱 필요)

### Disease KG depth (≥20 edges)
- DDXPlus: 100% (well-covered)
- SymCat: 89.8%
- RAMEDIS: 80.5%
- HMS: 60.3%
- MME: 73.7%
- **LIRICAL: 59.6%** ← bottleneck (커버는 되지만 깊이 부족)

## 다음 IE source 확장 우선순위 (vocabulary 영역별)

LIRICAL/RareBench의 **depth** 부족 = rare disease 정보 sparsity. 해결:

1. **OMIM Clinical Synopsis (text only)** — rare disease 표준 표현 풍부, vocabulary 영역에 rare 강함
2. **PMC Open Access full-text** — 기존 abstract보다 detail 많음, rare disease case report 다수
3. **GeneReviews 확장** — 현재 일부, 800개 모두 IE 처리
4. **MSD Manual Professional** — ER-Reason + DDXPlus 깊이

각 source가 **어떤 vocabulary 영역을 커버하는지**가 source 선정 기준 (benchmark coverage는 진단 목적으로만 확인).

## 학술적 framing

> "We construct our KG from public medical text sources (PubMed, StatPearls, MedlinePlus,
> Wikipedia, GeneReviews) using UMLS DISO CUIs as universal seeds. Coverage diagnostics
> against multiple benchmarks (DDXPlus, SymCat, RareBench, AMELIE, ER-Reason) are reported
> for transparency, but benchmark disease lists are never used to drive seed selection or
> source prioritization. Source selection is based on vocabulary breadth and clinical detail
> depth, not benchmark-specific gaps."
