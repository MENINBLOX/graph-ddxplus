# Method — Extrinsic validation of normalized attributes (attribute → diagnosis)

*Draft Method section. Rigorous, paper-oriented. 2026-07-19.*

## Motivation

Binary symptom representation caps differential-diagnosis accuracy: generic symptoms are
shared across many diseases, so a symptom set is often consistent with many conditions. The
NLICE study (Al-Ars et al., arXiv:2401.13756) reported that identical classifiers rise from
**58.8%/57.1% (Naive Bayes/Random Forest, binary symptoms) to 82.0% Top-1** with per-symptom
attributes (Nature/Location/Intensity/Chronology/Excitation), in a *supervised* setting trained
on the synthetic data.

> **Confound in the external number (important).** The NLICE Table I binary and attribute rows
> are **not on the same class set**: the SymCat(binary) rows use **801 conditions**, the
> NLICE(attribute) rows use **55 conditions**. The 58.8→82.0 jump therefore mixes attribute
> enrichment with a large label-space reduction, so it is not a clean representation ablation.
> **Our test controls exactly this**: we hold the candidate set, the patients, and the base
> scorer fixed and toggle only the attribute channel — a within-subjects design the external
> paper lacks. This makes our (smaller, honest) effect a cleaner causal estimate of the
> attribute contribution, additionally hardened by a permutation null.

We test whether the same signal survives in our **strict zero-shot, KG-only** setting, where
(i) attributes are extracted zero-shot by a local 8B LLM IE and normalized to UMLS
(location→CUI, severity/onset→controlled vocab), (ii) attached to KG disease→phenotype edges as
qualified statements, and (iii) never trained on any disease–symptom mapping (CLAUDE.md §3).
If a normalized attribute improves diagnosis here, it demonstrates that the attribute is a
transferable cross-benchmark *interlingua*, not a dataset-specific learned feature.

## Setup

- **Benchmark**: NLICE `sample_data` synthetic patients — SymCat-grounded and the only public
  benchmark carrying separable, query-side Bates attributes (DDXPlus bakes coarse multi-region
  location into codes; RareBench compounds modifier+concept into single HPO CUIs; plain SymCat
  is symptom-only — see `docs/attribute_benchmark_analysis.md`). Each patient symptom carries
  `nature`(→character), `location`+`sublocation`(→location), `intensity`(→severity),
  `onset`(→onset). Query-side coverage: character 39.9%, location 41.0% (severity/onset sparse).
- **Candidates**: NLICE diseases resolved to KG clusters (closed set, 29–43 diseases in KG).
- **Model (base, binary)**: attribute-blind phenotype-CUI overlap, IDF-weighted (our onlykg
  scorer). **Model (+attr)**: base + a discriminative attribute channel — for a matched finding,
  an *agreement bonus* when the patient attribute equals the KG edge's qualified attribute and a
  *contradiction penalty* for controlled-vocabulary mismatch (severity ordinal, onset binary);
  location matched by CUI/hierarchy; free-text character rewards agreement only. λ=μ=1 fixed
  (single global default, no per-benchmark tuning, §7). Code: `attr_score.py`,
  `attr_loader_nlice.py`, integrated in `onlykg_eval_unified2.py --attr`.

## Statistical design (randomization inference)

- **Leave-one-attribute-in**: measure ΔGTPA@1 of each attribute channel over base.
- **Permutation null** (permutation feature importance, Breiman 2001): shuffle the attribute's
  values across all KG edges that carry it (marginal preserved, disease↔attribute association
  destroyed), recompute Δ. K=200 reps → null distribution. Empirical p = P(null Δ ≥ observed);
  z = (obs − mean_null)/sd_null.
- **Bootstrap 95% CI** on the observed Δ by resampling patients (B=2000).
- **Credit rule**: a channel is credited iff p<0.05 **and** the bootstrap CI excludes 0.
- Code: `nlice_ablation.py` (self-contained; exact-CUI location match = conservative).

## Result (KG = kg_v2, dense-attribute IE; 29-way closed set, 6,450 patients)

| model | GTPA@1 | MRR |
|---|---|---|
| base (binary) | 8.20 | 0.213 |
| **+location** | **11.84** | 0.237 |
| +character | 8.20 | 0.213 |
| +onset | 8.20 | 0.213 |

| attribute | ΔGTPA@1 (pp) | null mean±sd | z | p (perm) | bootstrap 95% CI | credited |
|---|---|---|---|---|---|---|
| **location** (→UMLS CUI) | **+3.64** | −0.02 ± 0.39 | **9.35** | **0.005** | **[+3.19, +4.11]** | ✅ |
| character (free-text) | 0.00 | 0.00 | — | 1.000 | [0, 0] | ✗ null |
| onset (controlled) | 0.00 | 0.62 ± 1.01 | −0.61 | 1.000 | [0, 0] | ✗ null |

Under the full diagnosis pipeline (hierarchy + token-level location matching) the location gain
is larger: base @1 4.65 → +location 12.79 (**+8.1pp**), @5 27.1 → 33.0.

## Interpretation (Method claims)

1. **A normalized attribute causally improves zero-shot KG diagnosis.** Location (normalized to
   UMLS CUI) yields a permutation-significant gain (z=9.35, p=0.005; CI excludes 0), replicating
   the NLICE binary→attribute effect *without any supervised training on disease–symptom data*.
   This is direct evidence for contribution axis 2: attributes act as a cross-benchmark
   interlingua when normalized to a shared vocabulary.
2. **Only normalizable attributes transfer.** Character is null — its free-text surface forms
   (patient "throbbing" vs IE-extracted "sharp") do not align across sources, exactly as the
   schema anticipated (character lacks a controlled standard vocabulary). Onset is null here by
   sparsity (query-side 0.7%). This delineates *which* Bates attributes are KG-interlingua-ready
   (location, and by construction severity/onset where populated) versus KG+LLM-only (character).
3. **The signal requires dense attribute annotation on the KG side.** On the sparse KG
   (`kg.json`, edge-location 27%) the location gain is +1.44pp (p≈0.02); on the exhaustively
   re-extracted KG (`kg_v2.json`, edge-location 38%) it is +3.64pp (p=0.005). This is the
   diagnostic value of denser attribute IE — attribute-blind accuracy is unaffected/slightly
   lower on the dense KG, but the *attribute channel* it enables is what the density is for.

## Honest scope

- Closed 29-way NLICE set; absolute GTPA@1 is low (sparse zero-shot KG profiles) — the **Δ**, not
  the level, is the object of inference, and it is isolated by the permutation null (robust to
  the low base).
- character/onset/aggravating/relieving remain extrinsically unvalidated (free-text or sparse);
  reported as a limitation. The attribute story is thus: **location is a validated normalized
  interlingua attribute; the rest are candidates pending controlled-vocabulary alignment or a
  KG+LLM consumer.**
