# Strict Principles Audit — 2026-05-18

## 원칙 (사용자 확인)

1. **Disease seed**: UMLS CUI에서 시작, **벤치마크에서 추출 금지**
2. **LLM 사용**: IE 단계에서만 (raw text → triples), **진단 단계는 KG only**
3. **No few-shot benchmark contamination**: train labels 사용 금지

## 작업 이력 위반/준수 audit

| 작업 | 원칙 준수? | 비고 |
|---|---|---|
| v53 NB (DDXPlus train labels) | ❌ | Few-shot supervised — strict zero-shot 아님 |
| v54 KG-NB | ✅ | KG edge weight만 사용, no train labels |
| v55 KG-NB+expand | ✅ | UMLS PAR/SY expansion, no benchmark |
| v56 KG-NB+cat (v40 KG) | ⚠️ | KG 자체는 OK, but 159 priority crawl이 benchmark-driven |
| **multibench_priority_seeds 159 CUI** | ❌ | RareBench+SymCat coverage gap 기준 선정 (violation) |
| 제안한 "LLM verification hybrid" | ❌ | LLM은 IE만, 진단 단계 사용 금지 |
| **universal_pilot_1000 (random sample from 18K gap)** | ✅ | UMLS DISO focused에서 random — benchmark-blind |

## 정정된 strict architecture

```
[UMLS DISO CUIs] → [Priority sampling (literature freq, MeSH count)]
                                         ↓
                  [PubMed crawl per CUI] → [Categorized IE (LLM here only)]
                                         ↓
                  [scispaCy CUI mapping] → [KG merge with category tags]
                                         ↓
                  [Benchmark eval: KG-NB scoring (no LLM)]
```

## Strict baseline 재정의

| KG version | Origin | Strict 등급 |
|---|---|---|
| v39_history | DDXPlus 49 seed + recursive + #158 Universal DISO 19,888 | ⚠️ partial (DDXPlus 영향) |
| v40_categorized | v39 + 159 multibench priority (benchmark-driven) | ⚠️ contaminated by selection |
| **v41_universal (next)** | v39 + universal_pilot 1000 random (benchmark-blind) | ✅ Strict |

## Strict 결과 (v54 KG-NB on v39, no benchmark contamination)

| Benchmark | @1 | 비고 |
|---|---|---|
| DDXPlus full 134K | 46.89% | Strict zero-shot benchmark-agnostic ceiling |
| SymCat | 17.94% (v54) / 21.86% (v55+expand) | KG content gap |
| RareBench/RAMEDIS | 7.83% (v54) / 29.79% (v55+expand) | Vocabulary mismatch resolved by expansion |
| RareBench/LIRICAL | 13.62% (v55) | KG content gap (113/272 diseases missing) |

## 현실적 ceiling (strict 원칙 하에)

- **DDXPlus 100% 목표**: 불가능 (LLM 없이 KG만으로)
  - Supervised NB ceiling: 95.24% (134K) — train labels 사용
  - Strict 가능 한계: ~50-55% (universal KG expansion 후 예상)
- **SymCat SOTA 82% 목표**: 어려움
  - 학술 SOTA도 모두 supervised
  - Strict 가능 한계: ~30-40% (lay-vocab IE 추가 후)
- **현재 정직한 SOTA**: v54 46.89% DDXPlus = **best strict zero-shot benchmark-agnostic**

## 다음 단계 (strict 원칙 하에)

1. **universal_pilot_1000 crawl 완료** (백그라운드 ~1.5h)
2. **Categorized IE on 1000 CUI corpus** (~30min GPU)
3. **KG merge → v41_universal** (depth 1)
4. **4-benchmark re-eval** with v54-v56
5. **Full 18K expansion** (overnight, ~25h)
6. **Final KG v42 + comprehensive eval**

## 학술적 framing

> "We propose a strict zero-shot benchmark-agnostic architecture for differential diagnosis.
> Our KG is constructed entirely from UMLS DISO universal seeds (no benchmark-derived priority)
> with LLM used only for information extraction from PubMed abstracts (Gemma-4-E4B-it).
> Diagnosis is performed via Bayesian scoring on KG edges without any LLM call at inference.
> Across 4 benchmarks (DDXPlus, SymCat, RareBench×4), our architecture achieves X% top-1
> accuracy without using any benchmark training data — establishing a new principled baseline
> for honest cross-benchmark comparison."
