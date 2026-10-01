# Extrinsic Diagnosis — first cross-benchmark baseline (2026-07-16)

**속성-KG(v157, IE-only, strict zero-shot) → onlykg Stage-1 진단.** verify-first 실측 baseline.
헤드라인 아님(CLAUDE.md §8: GTPA@1 은 "정직한 위치 보고"). 개선 대상 baseline.

## KG (pilot/cache/prod/kg.json)
- 35,193 disease · 18,866 phenotype · 234,401 edge. IE-only(pubmed+sections, hpoa 미사용=§3 준수).
- §12 topology: **disease degree median=6**(목표 ≥20), ≤5 evidence **47.9%**(목표 <10%), singleton 40.5%.
- hub 16개 전부 정상 의학개념(pain/seizure/fever…), demographic/temporal noise 0(STOP_TUI 10,925 제거).
- **핵심 한계 = degree 낮음**(pubmed abstract 는 curated symptom-section 보다 증상 밀도 낮음: 8.6 finding/disease).

## Baseline (onlykg Stage-1, hop2=0, IDF edge weight)

| benchmark | patients | KG target-coverage | GTPA@1 | @3 | @5 | @10 | MRR |
|-----------|----------|--------------------|--------|----|----|-----|-----|
| RareBench | 1,121 | 430/440 (98%) | **9.19%** | 17.0% | 21.2% | 27.6% | 0.157 |
| SymCat(sim) | 2,431 | 760 cand | **1.52%** | 6.0% | 8.0% | 11.9% | 0.058 |
| DDXPlus | — | 191/335 evidence in KG | TODO | | | | |

## 진단(왜 낮은가) — verify-first

- **namespace 아님**: benchmark evidence CUI ∩ KG phenotype = SymCat **73%**, DDXPlus 57%, RareBench(전체 HPO) 26%(환자별은 더 높음). SymCat 은 매칭 substrate 충분한데도 1.5% → **matching 문제 아니라 disease profile 변별력 문제**.
- **근본 = degree 낮음 + 흔한 증상 저-IDF**: disease당 ~6 edge, common symptom(cough/fever)은 IDF 낮아 가중 약함 → 760 후보 중 변별 실패. rare(RareBench)는 target-coverage 98%로 좋으나 profile 희소.

## 개선 레버 (strict zero-shot 준수, hpoa 금지)
1. **source 밀도↑**: disease당 pubmed abstract 수↑ + symptom-section 우선 선택(v156 word-budget/cue 튜닝). curated section 있으면 8.6→16 finding.
2. **IE 완전성↑**: exhaustive directive 강화, umbrella term 억제.
3. **scoring**: hop2 hierarchy edge(UMLS/HPO IS-A, 벤치마크 무관 = 허용) 로 phenotype 확장 → 희소 profile 보완. IDF/IC 가중 정교화(#8).
4. **속성 ablation(헤드라인)**: base vs +attribute leave-one-out delta(v104 harness). character 등은 KG+LLM 변형.

## 다음
- DDXPlus baseline 완료 → cross-benchmark 표 완성.
- v156 source 밀도 튜닝 + hop2 hierarchy → re-KG → delta 측정.
- 속성 채널을 scoring 에 연결(v104) → leave-one-out ablation(진짜 기여축).
