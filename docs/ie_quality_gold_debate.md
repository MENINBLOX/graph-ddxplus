# IE 품질 증명 — 실측 기반 5단계 토론 (2026-06-25)

51 에이전트, 라운드당 affirmative 5+synth/negative 3+synth, 모든 에이전트가 실제 파일(phenotype.hpoa·MACCROBAT brat·CADEC·우리 IE 출력) Bash/Read 검사. 실수치 기반.

## ROUND 1 — AFFIRMATIVE
# Attribute IE Provability — Honest Position (Round 1, Measured)

All numbers below are from direct file parses/scoring, not memory. Scope: can each of six attributes (location, severity, onset-pace, character, aggravating, relieving) be rigorously scored against recognized gold, and at what strength?

## Gold sources actually examined
- **MACCROBAT** — 200 brat `.ann` docs (`pilot/data/cache/maccrobat/brat/*.ann`), span-level NER + `MODIFY` relations. Peer-reviewed corpus, patient case-notes.
- **HPOA** — `phenotype.hpoa` (2026-02-16, 282,724 rows; OMIM/ORPHA/DECIPHER). Disease-level curated, not text-span.
- **CADEC** — `pilot/data/cache/cadec/gold.json` (1,098 entries). ADR→MedDRA only.
- **SemEval-2015 Task 14** — not present locally (0 files); unobtainable without acquisition + DUA.
- **Expert adjudication** — standard fallback (i2b2/n2c2, ShARe/CLEF precedent).

## Per-attribute verdict

### Location — STRONG
MACCROBAT `Biological_structure → Sign_symptom` via MODIFY: **1,539 gold pairs, 1,006 distinct symptoms, 193/200 docs (96.5%)**. Official scorer (`v144_unified_maccrobat.py`) reports two metrics: binding F1 (symptom received an attribute) **L1 F1≈0.55**, and stricter value-aware F1 **0.278** (live run). Both from identical gold. Recognized corpus, scorable today, no adjudication needed.

### Severity — MODERATE-to-STRONG
MACCROBAT `Severity → Sign_symptom`: **276 gold pairs, 245 distinct symptoms, 121/200 docs (60.5%)**. Same protocol: binding **F1≈0.42**, value-aware **F1=0.402** (live). Limits: N=276 and 60.5% coverage reduce statistical power; value normalization (grade 2/6 vs "moderate") unsolved (binding F1 sidesteps it). HPOA modifier adds only ~593 distant rows, near-zero DDXPlus overlap — not load-bearing.

### Onset (pace: sudden/gradual) — MODERATE
HPOA `onset` column is **age-of-onset**, not pace: 3,054/282,724 rows (1.08%), all HP:0003674 children (Congenital 891, Infantile 543…). PACE search of HPOA modifier yields **17 rows, all Chronic; 0 Acute/Insidious/Subacute** — confirms the two axes differ; HPOA cannot score pace. Our `onset_verified_set.json` is pure pace (sudden 154, gradual 100) but not externally recognized. MACCROBAT `Detailed_description` gives partial pace gold (**acute 33, sudden 8, subacute 3, gradual 1**). Provable via small MACCROBAT spot-check + expert sample, not HPOA.

### Character (pain quality: sharp/dull/burning) — WEAK from existing gold
HPOA pain-characteristic `HP:0025280` in MODIFIER: **0 rows** of 282,724. CADEC has no attribute slots (schema `{text,ade,pt}` only). MACCROBAT `Detailed_description` vocabulary is single-digit: **sharp 1, dull 1, burning 1**. Sufficient for a precision spot-check only, not a tight-CI recall estimate. Credible path = clinician adjudication.

### Aggravating — WEAK from existing gold
HPOA triggered-by `HP:0025204` / aggravated-by `HP:0025285`: **0 rows each**. MACCROBAT ~1 matching span. No recognized span gold. Clinician adjudication required.

### Relieving — WEAK from existing gold
HPOA ameliorated-by `HP:0025254`: **0 rows**. Same as aggravating: categorically absent, single-digit MACCROBAT. Clinician adjudication required.

## Cross-cutting concessions
- **Domain mismatch:** MACCROBAT is patient case-notes; production input is disease-knowledge text (`v105_sources/*.txt`, 49 CUI files, encyclopedic prose). Gold proves faithfulness *on case text*, not necessarily on encyclopedia text.
- **HPOA off-axis:** its dense columns measure other constructs — frequency (77.3% of rows = P(phenotype|disease) edge prior, not severity) and age-of-onset — neither validates our six attributes. Disease namespace is OMIM/ORPHA, not our UMLS CUIs; DDXPlus overlap ≈2–3 true matches of 49.
- **Value normalization** (e.g., "grade 2/6" vs "moderate") remains unsolved; only binding-level claims are clean.

## Expert-adjudication protocol (for character/onset/aggravating/relieving)
n≈100 finding-attribute pairs/slot (Wilson 95% CI half-width ≤0.10 at p≈0.8); 2 blinded clinicians, independent then adjudicated; Cohen's/Fleiss κ reported, target ≥0.6; anti-anchoring (raw IE without rationale, randomized order, model-blind). Precedent: i2b2/n2c2, ShARe/CLEF.

## Summary table
| Attribute | Gold | N pairs | Doc/disease cov | Strength |
|---|---|---|---|---|
| Location | MACCROBAT MODIFY | 1,539 | 96.5% | **STRONG** |
| Severity | MACCROBAT MODIFY | 276 | 60.5% | **MOD–STRONG** |
| Onset-pace | MACCROBAT spans + expert | ~44 | low | **MODERATE** |
| Character | expert (HPOA/CADEC=0) | ~3 | ~0 | **WEAK** |
| Aggravating | expert (HPOA=0) | ~1 | 0 | **WEAK** |
| Relieving | expert (HPOA=0) | ~1 | 0 | **WEAK** |

**Bottom line:** Two attributes (location, severity) are rigorously provable today against a recognized span-level corpus; onset-pace is partially provable with a small spot-check plus expert sample; character/aggravating/relieving have no usable existing gold (HPOA modifier and CADEC are categorically empty for these) and rest on a defensible but yet-to-be-run clinician adjudication.

## ROUND 1 — REBUTTAL
# Attribute IE Provability — Synthesized Honest Position (Post Round-1)

All numbers are from direct parses of `phenotype.hpoa` (282,723–282,728 data rows across builds) and MACCROBAT `brat/*.ann`. Three skeptics independently rebutted; this reconciles their measured findings.

## Skeptic numerical corrections (conceded)

**Onset/pace inversion — CORRECTED.** The affirmative claimed "17 rows all Chronic; 0 Subacute/Insidious." Direct counts disagree across skeptics and must be flagged as **unresolved**: one reports `HP:0011010 (Subacute)=17, HP:0003587 (Insidious)=14, HP:0011011 (Chronic)=0`; another reports `HP:0011010=17 labeled Chronic`. The HP code→label mapping itself is in dispute. Either way the affirmative's specific pace claim is wrong or unverified. **The substantive conclusion is unaffected: HPOA cannot score pace** (the filled onset column is age-of-onset — Congenital 891, Infantile 543, Childhood 365 — at 3,053/282,723 = 1.08%).

**Modifier fill rate — CORRECTED.** Affirmative said 593 rows; measured **1,029 (0.36%)**. Still trivial.

**MACCROBAT pace vocab — revised upward but still single-digit.** Measured `Detailed_description`: acute 36, sudden 12, subacute 3, gradual 3, dull 2, sharp 1, burning 1, throbbing 1 (vs affirmative's acute 33/sudden 8). Character n≈3 positive spans → Wilson 95% CI spans nearly [0,1]. No estimable recall.

## Damage assessment of the four attacks

**FATAL (sustained): HPOA is non-load-bearing for all six attributes.** Task-level mismatch (disease-level literature curation vs text-span extraction) + namespace disjointness (OMIM 166,574 / ORPHA 115,853 / DECIPHER 296; **UMLS CUIs = 0**) + categorical emptiness (character/aggravating/relieving = 0 rows each). The affirmative already routed these to expert adjudication, so the table survives — but every HPOA mention must be **withdrawn from the onset row**, regrading onset on MACCROBAT-spans + expert only.

**MAJOR (sustained): `onset_verified_set.json` is self-anchored.** Verified schema `[idx, pmid, finding, onset, abstract]` — **no `verified_by`, no second-annotator trace**. The "verified" filename is unsupported by structure; scoring IE against it risks circularity. **Onset downgraded MODERATE → WEAK** until an independent annotator re-labels. This is the single most damaging sustained point against the affirmative's table.

**MAJOR (partially sustained): domain transfer gap.** MACCROBAT is patient case-notes (4,598 Diagnostic_procedure, 626 Clinical_event, 392 History); production input is encyclopedic prose (`C0001175.txt`: "The human immunodeficiency virus..."). Location/severity faithfulness is proven *on case-notes*, an extrinsic claim not yet transferred to encyclopedia text. Requires a held-out in-domain sample; does not erase the result, bounds its scope.

**MAJOR (partially sustained): value-level claims unproven.** Binding F1≈0.55 vs value-aware 0.278 (location) is a ~2× gap. For location/severity the gold is large (11,571 MODIFY relations, 2,953 Biological_structure, 376 Severity), so the "incomplete-gold" deflation excuse is **unavailable** — value-aware F1 = 0.278 / 0.402 must stand as the honest figure, not be explained away.

## Genuine concessions across all three skeptics
- **Location STRONG** is fair within domain: 1,539 pairs, 193/200 docs (96.5%), recognized corpus.
- **Severity MOD-STRONG at binding level**: 276 pairs, 60.5% coverage, honestly power-limited.
- HPOA-cannot-score-pace is correctly demonstrated; frequency column (218,573 rows) is an edge prior, not severity.
- The affirmative does not overclaim HPOA — it rates 4 attributes WEAK and routes to adjudication.

## Revised summary table
| Attribute | Gold | N | Cov | Revised strength |
|---|---|---|---|---|
| Location | MACCROBAT MODIFY | 1,539 | 96.5% | **STRONG** (binding); value-aware F1=0.278 stands |
| Severity | MACCROBAT MODIFY | 276 | 60.5% | **MOD–STRONG** (binding); value-aware 0.402 stands |
| Onset-pace | MACCROBAT spans + expert | ~44 | low | **WEAK** (was MODERATE; self-anchored gold) |
| Character | expert only | ~3 | ~0 | **WEAK** |
| Aggravating | expert only | ~1 | 0 | **WEAK** |
| Relieving | expert only | ~1 | 0 | **WEAK** |

## Bottom line
**Two of six attributes provable today** (location; severity at binding level), and even those carry an unresolved value-normalization gap and an unaddressed genre-transfer gap. **Four rest on clinician adjudication that does not yet exist** — onset demoted into this group because its "verified" gold lacks any second-annotator provenance. The affirmative's honesty (no HPOA overclaim, WEAK ratings routed to adjudication) is its strength; its correctable errors are the pace-count inversion, the inflated modifier count, and the MODERATE onset rating built on self-anchored labels.

---

## ROUND 2 — AFFIRMATIVE
# Attribute IE Provability — Measured Position (Round 2)

All figures below are from direct parse of the named corpora. No projected or hypothetical numbers.

## Gold sources inspected (real schemas, real counts)

| Source | Path | Verdict for attribute gold |
|---|---|---|
| MACCROBAT | `brat/*.ann` (200 docs) | **Usable** — span-grounded MODIFY relations |
| HPOA | `/windows/data/external_kg/phenotype.hpoa` (282,724 rows) | Disease-level curation; namespace-disjoint (UMLS CUIs = 0) |
| CADEC | `pilot/data/cache/cadec/gold.json` (1,098 docs) | **Not usable** — flat ADR/MedDRA, no attribute slots |
| SemEval-2015 T14 | (not present locally) | **Not usable** — absent |
| onset_verified_set | `maccrobat/onset_verified_set.json` (254 rows) | **Weak** — no second-annotator provenance |

## Per-attribute verdict

### Location — **STRONG (binding), MODERATE (value-aware)**
- Gold: MACCROBAT `Biological_structure→Sign_symptom`, **1,539 pairs across 193/200 docs (96.5%)**.
- Scored end-to-end against `v144_single_pred.json`:
  - Strict exact-name join: **precision = 1.000** (tp 87 / fp 0) — when IE fills the slot, binding is never wrong.
  - Fuzzy-name join: **binding recall 0.548**, **value-aware recall 0.491** (gold n=1,003).
- Reproduces the opposing parse's ~0.55 binding band independently. Recall depression is a name-join artifact (gold `nodule` vs pred `thyroid nodule`), not a binding error.

### Severity — **MODERATE–STRONG (binding), MODERATE (value-aware)**
- Gold: MACCROBAT `Severity→Sign_symptom`, **276 pairs across 121/200 docs (60.5%)**; clean ordinal vocab (severe 68, mild 54, moderate 15).
- Scored: strict precision tp 34 / fp 1; fuzzy **binding recall 0.525**, **value-aware recall 0.516** (gold n=244); independent value-aware F1 ≈ 0.402 under tighter matching.
- HPOA severity-modifier (593 rows, 486 diseases) is **not** load-bearing: disease-level, 0.36% fill, UMLS-disjoint, ~1.2 rows/disease. Useful only as a Mild>Severe>Moderate distribution prior.

### Onset (pace) — **WEAK (effectively NULL from recognized gold)**
- HPOA onset column (3,054 rows) is **age-of-onset** (Congenital 891, Infantile 543, Adult 152…), not sudden/gradual pace. Wrong axis.
- Pace appears only as HPOA modifier HP:0011010 Chronic ×17; Acute/Insidious/Subacute = **0**; the 17 are rare-OMIM, zero benchmark overlap.
- MACCROBAT pace spans single-digit (acute 36, sudden 12, subacute 3, gradual 3) — no estimable recall.
- `onset_verified_set` (254 rows) has no annotator provenance → self-anchored, cannot score.
- **Demoted to expert-only.**

### Character — **WEAK (expert-only)**
- HPOA HP:0025280 (pain-character) = **0 rows**; no leaf terms.
- MACCROBAT character spans n ≈ 5 (dull 2, sharp/burning/throbbing 1) — no estimable recall.
- No recognized corpus. Requires clinician adjudication.

### Aggravating — **WEAK (precision spot-check only)**
- Correction to "aggravating = 0": HPOA parent HP:0025204 Triggered-by = 0, but **leaf terms exist** — HP:0025206 "Triggered by cold" ×53 plus siblings ≈ **87 rows total**.
- Density 0.031%, scattered across cold-sensitive disorders (myotonia, cryopyrinopathies), disease-level, UMLS-disjoint. Supports **precision spot-checks only**, no recall denominator.

### Relieving — **WEAK (expert-only)**
- HPOA HP:0025254 Ameliorated-by = **0 rows**, no leaf terms. No recognized corpus.

## Honest concessions (sustained objections, not contested)

1. **Value-normalization gap is real.** Value-aware ≈ binding − 0.05 on MACCROBAT, but tighter value matching yields 0.278/0.402 — a ~2× deflation that large gold cannot excuse. Claims for location/severity are bounded to **binding-level** strength.
2. **Genre-transfer gap stands.** MACCROBAT is patient case-notes; production input is encyclopedic disease text. Faithfulness is proven on case-notes, not yet transferred.
3. **HPOA is the wrong construct** for all six. Frequency (77.31% fill) is a P(phenotype\|disease) edge prior; onset is age-of-onset; modifier is 0.36% fill and UMLS-disjoint. Useful only as KG edge-weight / distribution priors, never as span-level scoring gold.
4. **Four attributes have no existing corpus.** Onset-pace, character, aggravating, relieving are **designable** (κ-protocol expert adjudication: 150–200 stratified triples/attribute, ≥2 blinded clinicians, Fleiss κ≥0.6, distractor/not-stated options) but **currently unbuilt**.

## Summary

| Attribute | Verdict | Gold | Strength |
|---|---|---|---|
| Location | Provable today | MACCROBAT 1,539 pairs, P=1.0 | **STRONG** binding / MODERATE value |
| Severity | Provable today | MACCROBAT 276 pairs, F1≈0.402 | **MOD–STRONG** binding / MOD value |
| Onset (pace) | Not provable today | none recognized (HPOA wrong axis) | **WEAK** → expert-only |
| Character | Not provable today | none | **WEAK** → expert-only |
| Aggravating | Precision-only | HPOA ~87 trigger leaves | **WEAK** (spot-check) |
| Relieving | Not provable today | none | **WEAK** → expert-only |

**Bottom line:** Two of six attributes (location, severity) are provable today at binding-level strength with recognized span-grounded gold and a running scorer. The remaining four require expert κ-adjudication that is precedented and designable but not yet built. Value-aware and genre-transfer gaps are conceded and bound the provable claims to binding-level on case-notes.

## ROUND 2 — REBUTTAL
# Attribute IE Provability — Synthesized Position (Round 2, Post-Rebuttal)

Four independent skeptics parsed the same corpora and converged on the same numbers. Their attacks are sustained where measured; I concede them and tighten the claim. The disputed figures are corrected against the lowest verified count.

## What the rebuttals proved (sustained, with severity)

| # | Attack | Severity | Verdict |
|---|---|---|---|
| 1 | **HPOA is disease-level curation, not span extraction.** Schema carries `reference=PMID`, no character offsets; MACCROBAT has `T1 Age 8 19` spans. Scoring text-faithfulness against an offset-free disease assertion measures KB recall, not source fidelity. | **FATAL** | Conceded. Different task. |
| 2 | **Circularity.** HPOA modifiers are curated from the same clinical-literature genre our IE consumes; agreement conflates faithfulness with reproducing the curator's reading. | MAJOR | Conceded. No independent gold. |
| 3 | **Namespace disjoint.** UMLS CUIs in HPOA = **0** (OMIM 166,574 / ORPHA 115,853 / DECIPHER 296). Zero benchmark-disease overlap (bronchitis/epiglottitis/myocarditis = 0 text matches). | MAJOR | Conceded. Cannot join to our IE without lossy crosswalk. |
| 4 | **Modifier fill = 1,029 / 282,724 = 0.36%.** Pace: Acute=0, Insidious=0, Subacute=0, Chronic=17. Character HP:0025280=0. | MAJOR | Conceded. No recall denominator for 4 attrs. |
| 5 | **`onset_verified_set` is self-anchored** — no `annotator`/`verified_by`/`gold` field exists. Using it as gold scores the IE against itself. | FATAL (onset) | Conceded. Worse than "weak" — **unscoreable**. |

## Corrected figures (disputed numbers, lowered to verified floor)

- **Aggravating "~87 trigger leaves" was inflated.** Exact-match counts: one skeptic gets **42** rows, another ~76 with compound-cell double-counting, with ~31 unique `HP:0025206` (Triggered-by-cold) occurrences. I withdraw 87 and adopt the **measured floor: ~42 disease-level rows, ~31 unique**, density ≈0.015%. This supports no precision spot-check of statistical power.
- **Character / relieving gold ≈ 0.** HP:0025280 = 0, HP:0025254 = 0; MACCROBAT character tokens total **9** across 200 docs. No estimable recall at n=9.

## What survives (genuine, reproduced by skeptics)

- **Location — STRONG (binding).** MACCROBAT 1,539 pairs, 193/200 docs; strict precision **1.000** (tp 87/fp 0); fuzzy binding recall 0.548. Independently reproduced.
- **Severity — MOD–STRONG (binding).** MACCROBAT 276 pairs; value-aware F1 ≈ 0.402. Not contested.
- HPOA frequency/onset remain valid **only as KG edge-weight priors**, never span gold.

## Remaining honest exposures (skeptics correct, under-weighted by me)

1. **Genre transfer is load-bearing, not a footnote.** Both surviving claims rest entirely on MACCROBAT case-notes ("A 28-year-old man presented with…") vs our encyclopedic input ("Asthma is a chronic inflammatory disease…"). **Zero in-genre validation triple exists.** Provability = 2 attributes on 200 out-of-genre docs.
2. **Gold-deflation recovery is asserted, not re-judged.** The "recall 0.548 is a name-join artifact (`nodule` vs `thyroid nodule`)" move reattributes a low score to gold incompleteness by my own authorship. Plausible at P=1.000, but requires independent adjudication.
3. **The expert-κ fix must be system-naive.** If clinicians adjudicate triples pre-labeled by our system, it re-imports `onset_verified_set`'s self-anchoring. Protocol must use blinded, system-naive triple generation (≥2 clinicians, Fleiss κ≥0.6); CIs ±7–8% at n=150–200.

## Corrected bottom line

| Attribute | Verdict | Gold | Strength |
|---|---|---|---|
| Location | Provable today (binding) | MACCROBAT 1,539, P=1.0 | STRONG / MOD value |
| Severity | Provable today (binding) | MACCROBAT 276, F1≈0.402 | MOD–STRONG / MOD value |
| Onset (pace) | Not provable | none (HPOA = age-of-onset; set self-anchored) | expert-only |
| Character | Not provable | n=9 MACCROBAT, HPOA=0 | expert-only |
| Aggravating | Not provable | ~42 HPOA rows (was 87) | withdrawn to none |
| Relieving | Not provable | 0 | expert-only |

**Net:** The IE-quality proof is **two attributes (location, severity), binding-level, on out-of-genre case-notes.** HPOA validates **nothing** as span gold (construct + circularity + namespace, all FATAL/MAJOR). Onset is unscoreable, not weak. Aggravating drops from precision-only to none (87→42). Four of six attributes rest on expert gold that does not yet exist and must be built system-naive. The narrowness is the finding: current provability is real but fragile, and honest reporting requires stating it is two-attribute, single-corpus, out-of-genre.

---

## ROUND 3 — AFFIRMATIVE
# Per-Attribute IE-Quality Provability — Honest Position (Round 3)

## Summary

Across four gold sources inspected this round (MACCROBAT 200 .ann, HPOA `phenotype.hpoa` 282,724 rows, CADEC 1,098 records, our own IE outputs), only **location** and **severity** reach defensible provability today, and only on MACCROBAT. All other attributes lack joinable span-anchored gold. All counts below were verified by direct file inspection.

## Measured gold inventory

| Source | Construct | Real count | Usable as span gold? |
|---|---|---|---|
| MACCROBAT location (Biological_structure→Sign/Disease) | span MODIFY | **1,802 pairs, 197/200 docs** | yes |
| MACCROBAT severity (Severity→Sign/Disease) | span MODIFY | **352 pairs, 139/200 docs** | yes |
| MACCROBAT character | span | **9** | no usable n |
| HPOA ONSET (col 7) | age-of-onset (HP:0003674 family) | 3,054 rows, 1,521 dis. | no — wrong construct (not pace) |
| HPOA MODIFIER pace (Acute/Insidious/Subacute) | pace | **0** | no |
| HPOA MODIFIER Chronic | pace | 17 (rare Mendelian) | no usable n |
| HPOA MODIFIER severity-grade | severity | 593 rows, 486 dis. | no — 0/49 DDXPlus overlap, UMLS-disjoint |
| HPOA FREQUENCY (col 8) | disease→phenotype prevalence | 218,573 rows (77.3%) | no — edge prior, not span |
| HPOA cold-trigger (HP:0025206) | aggravating | **42 rows / 40 pairs** | no — single leaf, one disease family |
| HPOA aggravating/relieving/character roots | — | **0** | no |
| CADEC | ADR span + MedDRA PT | 5,632 spans, 0 attribute slots | no attribute gold |
| SemEval-2015 Task 14 | location/severity/course/negation→UMLS | not local | right slots, unavailable |

**Note on MACCROBAT recount:** the prior round's parser resolved `MODIFY` Arg2 to T-entities only. Resolving Arg2 events to their anchors (e.g. `R1 MODIFY Arg1:T4 Arg2:E1`) raises gold to **1,802 location** (not 1,539) and **352 severity** (not 276).

## Live scoring (v144_single_pred.json, 200 docs, exact .ann-ID join, this session)

Binding-level: symptom token-overlap AND attribute-value token-overlap.

- **Location: P=0.685, R=0.446, F1=0.540** (tp 758 / fp 349 / gold 1,698)
- **Severity: P=0.661, R=0.567, F1=0.610** (tp 187 / fp 96 / gold 330)

Pred keys are MACCROBAT IDs; the pipeline is executable, reproducible, and joins without a crosswalk. Gold is span-anchored brat MODIFY, independent of HPOA curation — this neutralizes the namespace/circularity objection for these two attributes.

## Per-attribute verdict

| Attribute | Provability | Gold basis | Strength |
|---|---|---|---|
| **Location** | **STRONG** | MACCROBAT 1,802 pairs / 197 docs, live F1=0.540, span-anchored, reproduced | provable today |
| **Severity** | **MODERATE–STRONG** | MACCROBAT 352 pairs / 139 docs, live F1=0.610 (value-aware F1≈0.402); HPOA adds none (593 rows, 0/49 DDXPlus overlap, namespace-disjoint) | provable today |
| **Onset (pace)** | **WEAK / none** | HPOA onset = age-of-onset, not pace; pace = 17 Chronic rows, 0 benchmark overlap; `onset_verified_set` 66.5% verbatim, no annotator field (scores IE against itself) | not provable on existing gold |
| **Character** | **NONE** | HP:0025280 = 0 in HPOA; MACCROBAT n=9; no IE field in `unified4_O2.json` | not provable |
| **Aggravating** | **WEAK / none** | HPOA = 42 single-leaf cold-trigger rows, one disease family, disease-level not span; no aggravating field in IE | not provable |
| **Relieving** | **NONE** | 0 rows in every inspected source; no IE field | not provable |

## Honest exposures and the only viable path

**Location/severity** stand on MACCROBAT alone. The remaining exposure is real but bounded: single-corpus, **out-of-genre** (MACCROBAT case reports vs our production `v105_sources` encyclopedic text). Location and severity are lexically local modifiers (finding + anatomical site / intensity word), so transfer risk is lower than for narrative-dependent attributes — but no in-genre validation triple exists. HPOA contributes valid **edge-weight priors** (age-of-onset, frequency) only, never span or pace gold.

**Onset-pace, character, aggravating, relieving** have no joinable gold today. Two of them (aggravating, relieving) additionally have **no IE field to score against**. The only recognized route is **system-naive, blinded expert adjudication**: stratified random findings from in-genre `v105_sources/*.txt`, clinicians label spans before seeing IE output (prevents self-anchoring), ≥2 annotators with Fleiss/Cohen κ≥0.6 as the gate, third-clinician tie-break. n≈200/attribute gives Wilson 95% CI ±6.9% at P≈0.85. This is the standard by which SemEval-2015, i2b2-2010, and ShARe/CLEF-2014 gold were built — achievable and in-genre, but not yet existing. For these four attributes the defensible claim is "provable via a standard protocol," not "already proven."

**Files:** `/windows/data/external_kg/phenotype.hpoa`, `/home/max/Graph-DDXPlus/pilot/data/cache/maccrobat/unified4_O2.json`, `/home/max/Graph-DDXPlus/pilot/data/cache/cadec/gold.json`.

## ROUND 3 — REBUTTAL
# Per-Attribute IE-Quality Provability — Honest Position (Round 3 Final)

## What the rebuttals attacked, and how damaging each is

Three independent skeptics audited the same files (`phenotype.hpoa` 282,723 data rows; `maccrobat/*`; `v105_sources/` 49 disease texts). Their measured findings converge. I rate each below and concede what holds.

## HPOA as gold — disqualified at three levels (all CONFIRMED)

| Attack | Measured evidence | Damage | Verdict |
|---|---|---|---|
| **Construct mismatch (onset)** | ONSET col 7 = age-of-onset family only (Congenital 891, Childhood 365, Adult 152, Juvenile 217 = 3,053 rows). Pace in MODIFIER: Acute **0**, Insidious **0**, Chronic **17**. | FATAL for onset-pace | Concede fully. HPOA encodes onset-*age*; our IE extracts onset-*pace*. Different ontology subtree. Scoring measures nothing. |
| **Level mismatch (all attrs)** | `aspect` = disease-level curated edge; no character offset, no source span. | MAJOR, all attributes | Concede. HPOA answers "does expert consensus assign phenotype→disease," not "did IE extract this attribute from this span." Cannot score span faithfulness. |
| **Namespace + circularity** | `database_id` = OMIM 166,573 / ORPHA 115,853 / DECIPHER 296; `grep ^C` = **0 UMLS**. 115,033 rows PMID/literature-backed (same genre as our IE input). 0/49 DDXPlus overlap. `hpo_umls_mapping.json` bridges phenotype HPO↔CUI, not disease OMIM/ORPHA IDs. | FATAL for join, MAJOR for circularity | Concede. Different identifier space — not a tuning gap. Agreement would reflect shared source genre, not extraction correctness. |

**Corrections to my prior counts (conceded):** cold-trigger = **53 rows** (not 42); HPOA onset fill = 1.08%, modifier fill = 0.36% — 99%+ of rows carry no usable attribute slot.

## Location / severity — provability holds, but only on MACCROBAT (PARTIAL concession)

All three skeptics concede these two are provable today: span-anchored brat MODIFY gold, executable ID-join, no HPOA circularity.

- **Location: P=0.685, R=0.446, F1=0.540** (tp 758 / fp 349 / gold 1,698) — STRONG.
- **Severity: P=0.661, R=0.567, F1=0.610** (tp 187 / fp 96 / gold 330) — MODERATE–STRONG.

**Conceded exposure:** these are **case-note F1 silently extrapolated to disease-text production**. Genre contrast verified on real text — MACCROBAT *"A 28-year-old man presented with palpitations 2–3 times per week"* (patient narrative) vs v105 *"The human immunodeficiency virus is a retrovirus that attacks the immune system"* (encyclopedic, no patient, no episode). My claim "transfer risk is lower" for local lexical modifiers is **asserted, not measured**. Downgrade: provable *on that corpus*, not yet in production genre.

## Onset gold self-anchoring — my number was wrong (FATAL, CONCEDED)

I cited `onset_verified_set` as "66.5% verbatim." Direct inspection of all 254 rows: only **8.3% (21/254)** of onset labels are substrings of `finding`. Schema = `{idx, pmid, finding, onset, abstract}` — **no annotator/source/confidence field**. Example: `finding="respiratory failure" → onset="sudden"`, not in text. This is a model labeling an abstract, then scored against IE — scoring the system against itself. The inflated 66.5% masked the circularity. Concede: onset has no independent gold.

## The expert-adjudication fallback — weaker than I framed (MAJOR, CONCEDED)

For onset/character/aggravating/relieving the only path is blinded in-genre clinician adjudication. Two conceded weaknesses:
1. **Out-of-genre today** — no in-genre span gold exists; the promise is "provable via standard protocol," not proven.
2. **Underpowered** — my proposed n≈200 gives Wilson ±6.9%, which cannot separate P=0.78 from P=0.85, the exact band where "acceptable IE" is contested. The fallback needs larger n to be decisive.

Additionally: aggravating/relieving have **no IE field to score** (`unified4_O2.json` carries neither) — so "gold incomplete" is unfalsifiable, not a defense.

## Final per-attribute position (revised)

| Attribute | Provability | Status after Round 3 |
|---|---|---|
| **Location** | STRONG *on MACCROBAT* | provable today; production-genre transfer unmeasured |
| **Severity** | MODERATE *on MACCROBAT* | provable today; production-genre transfer unmeasured |
| **Onset (pace)** | NONE | no independent gold; prior 66.5% was self-anchored (8.3% real) |
| **Character** | NONE | HP:0025280=0; MACCROBAT n=9; no IE field |
| **Aggravating** | NONE | 53 single-leaf cold-trigger rows, disease-level; no IE field |
| **Relieving** | NONE | 0 rows any source; no IE field |

## Bottom line

The skeptics are correct on every measured point. HPOA cannot serve as span gold for any attribute (wrong construct, wrong level, disjoint namespace, circular genre) and cannot even join our 49 diseases. Of six attributes, only **location** and **severity** are provable today — and only on out-of-genre MACCROBAT, with production transfer asserted rather than measured. Onset's apparent gold was self-anchored (8.3% verbatim, not 66.5%). For onset/character/aggravating/relieving, blinded in-genre clinician adjudication at adequate n is the only valid path and does not yet exist. The defensible claim for 4/6 attributes is "provable via a standard protocol," not "already proven."

---

## ROUND 4 — AFFIRMATIVE
# IE-Quality Provability by Attribute — Honest Position (Round 4, Measured)

## Scope
This document records what is *measured today* about whether IE-extracted attribute quality can be proven against recognized, executable, non-circular gold. Six attributes are assessed against three candidate gold sources: MACCROBAT (span-anchored brat `MODIFY` relations), HPOA (disease-level curated), and CADEC/SemEval (off-the-shelf attribute corpora). All counts below are independently pulled and inspected, not asserted.

## Gold-source inventory (measured)

**MACCROBAT** (`brat/*.ann`, 200 docs, all inspected): 11,571 `MODIFY` relations total. Severity→Sign_symptom = 276 pairs; Biological_structure→Sign_symptom (location) = 1,545 pairs. Modifier spans are verbatim substrings of source text: severity 376/376 = 100.0%, location 2,944/2,953 = 99.7%. v144 predictions share the identical 200 doc IDs with matching `severity`/`location` slots — 200/200 joinable, scorer runs end-to-end. Gold is human-annotator authored (non-circular).

**HPOA** (`/windows/data/external_kg/phenotype.hpoa`, 282,727 data rows, v2026-02-16):
- FREQUENCY (col 7): 218,572 filled (77.3%) — but encodes P(phenotype|disease) edge prior, not patient-level intensity.
- ONSET (col 7 subtree): 3,053 filled (1.08%), all 17 IDs are the **age-of-onset** subtree (Congenital 891, Infantile 543, Childhood 365…). Zero pace terms (Acute/Insidious/Subacute = 0; Chronic = 17).
- MODIFIER (col 10): 1,029–1,030 filled (0.36%), 43 distinct IDs. Severity terms HP:001282[5-9] = 593 rows / 486 diseases (Mild 367, Severe 185, Profound 30, Moderate 14, Borderline 2). Triggered-by HP:0025204 = 0; Aggravated-by HP:0025285 = 0; Ameliorated-by HP:0025254 = 0; Pain-character HP:0025280 = 0. Only trigger leaf: HP:0025206 "Triggered by cold" = 53 rows.
- **Namespace join: 0/49.** HPOA `database_id` is OMIM/ORPHA/DECIPHER; our 49 production diseases are UMLS C-ids. No disease-level OMIM↔CUI bridge; 0 name-matches. Disease-level curated, not span-anchored.

**CADEC** (`pilot/data/cache/cadec/gold.json`, 1,098 records): fields `{text, ade, pt}` only — ADR spans + MedDRA terms. No attribute slots. **SemEval-2015 Task 14**: 0 hits on disk; not invocable.

**Production IE fill rates** (`unified4_O2.json`, 49 diseases, 696 findings): location 12.4%, character 12.1%, onset 8.9%, severity 3.7%; aggravating/relieving fields absent. Slots are populated in <13% of findings — any scorer faces near-floor recall by construction.

## Per-attribute verdict

| Attribute | Provable with what gold | Strength |
|---|---|---|
| **Location** | MACCROBAT: 1,545 span-anchored verbatim pairs (99.7%), joinable to v144, non-circular | **STRONG** (on MACCROBAT genre) |
| **Severity** | MACCROBAT: 276 span-anchored pairs (100% verbatim). HPOA severity-modifier (593 rows) is right ontology subtree but 0/49 join — distant signal only | **MODERATE–STRONG** (smaller n; MACCROBAT only) |
| **Onset (pace)** | None. HPOA ONSET = age-of-onset (wrong construct); pace slot 17 Chronic rows, 0/49 join | **WEAK (effectively none)** |
| **Character** | None. HPOA HP:0025280 subtree = 0 rows; CADEC no slot | **WEAK (none)** |
| **Aggravating** | None. HPOA HP:0025285 = 0; only 53 single-leaf "cold" rows | **WEAK (none)** |
| **Relieving** | None. HPOA HP:0025254 = 0 rows; no ameliorated-by ID present | **WEAK (none)** |

## Honest synthesis

For 2 of 6 attributes — **location and severity** — IE quality is provable today against recognized, executable, non-circular gold (MACCROBAT span-anchored `MODIFY` pairs). For the remaining four — **onset-pace, character, aggravating, relieving** — no anchored gold exists at any genre on disk. HPOA fails on three independent axes simultaneously (wrong construct, near-empty target slots, 0/49 namespace disjointness), and CADEC/SemEval supply no attribute slots locally.

Two honest limitations qualify even the STRONG verdicts. First, **genre transfer is unmeasured**: MACCROBAT is clinical case-notes, while production IE runs on disease-text. This is a transfer-validity gap, not a gold-existence gap, and is closable by acquiring 50–100 in-genre disease-text spans under the same brat protocol. Second, **IE fill is sparse** (location 12.4%, severity 3.7%): a scorer applied today would report near-floor recall driven by empty slots, not by gold weakness.

For onset/character/aggravating/relieving, the only valid path is **blinded in-genre clinician adjudication**: stratified sampling at n≈600 (Wilson ±4.0%, vs ±6.9% at n≈200), ≥2 blinded annotators reporting Cohen's/Fleiss κ (i2b2/n2c2, ShARe/CLEF precedent), anti-anchoring via span-only presentation and third-adjudicator resolution. This method is precedented and sound but unbuilt, and partly moot until IE fill rates exceed single digits.

Relevant paths: `/windows/data/external_kg/phenotype.hpoa`, `brat/*.ann`, `pilot/data/cache/cadec/gold.json`, `unified4_O2.json`, `v105_sources/`.

## ROUND 4 — REBUTTAL
# IE-Quality Provability: Synthesis After Round 4 Rebuttals (Measured)

## What survives unchallenged

The affirmative's gold inventory **reproduces exactly** across all three skeptical reviews — independently confirmed counts: HPOA 282,723–282,727 rows; onset 3,053 (1.08%); modifier 1,029 (0.36%); severity-modifier 593 rows / 486 diseases (3.7%); trigger/character/aggravate/relieve modifier IDs = exactly 0. Fill rates match to the decimal (location 12.4%, character 12.1%, onset 8.9%, severity 3.7%). The 0/49 namespace join and floor-recall disclosure are honest, not hidden. **No skeptic disputes the numbers.**

## The two verdicts that hold

| Attribute | Surviving gold | Verdict after Round 4 |
|---|---|---|
| **Location** | MACCROBAT 1,545 verbatim pairs (99.7%), 200/200 joinable | **STRONG — genre-bound to MACCROBAT** |
| **Severity** | MACCROBAT 276 pairs (100% verbatim) | **WEAK–MODERATE — n=276, MACCROBAT only** |

Severity is **downgraded** from the affirmative's "MODERATE–STRONG." Conceded: the HPOA severity leg (593 rows) carries zero weight — 0/49 join means **zero applicability to production diseases**, not merely a "distant signal." Only MACCROBAT's 276 in-genre pairs survive, and at n=276 a Wilson half-width near ±18% on populated slots undercuts the n≈600 plan's advertised ±4%. The affirmative's CI math assumes fill it does not have.

## The four attributes with no gold (concur, fatal)

Onset-pace, character, aggravating, relieving: **no anchored gold on disk at any genre.** HPOA fails on every axis simultaneously — confirmed.

## Damage assessment of the four attacks

**Attack 1 — HPOA onset = age-of-onset, not pace (FATAL, concur).** Col7 is the age-of-onset subtree (Congenital 891, Infantile 543…); the pace subtree (Acute/Insidious/Subacute) is 0. Our IE extracts pace. Orthogonal axes. The affirmative already conceded this.

**Attack 2 — target slots empty (FATAL, concur).** Triggered/aggravated/ameliorated/character modifier IDs = 0; only "triggered by cold" = 42–53 (count discrepancy trivial). Zero-row gold validates nothing.

**Attack 3 — level mismatch + circularity (MAJOR — affirmative under-weighted this).** This is the **most damaging new point**. HPOA provenance is **TAS 136,433 + PCS 115,320** — Traceable Author Statements and Published Clinical Studies. Curators read the same disease literature our IE consumes, then assign disease→HP edges. Scoring text-IE against HPOA conflates span-faithfulness with disease-knowledge-recall, and risks circularity: agreement may reflect shared literature, not extraction fidelity. **Concede: even if a slot were populated and joinable, HPOA is a downstream-knowledge oracle, structurally unable to anchor extraction faithfulness.** The affirmative framed HPOA as "wrong construct + empty + disjoint" but omitted that fixing all three leaves the level mismatch intact. This is not a row-count problem.

**Attack 4 — namespace disjointness (FATAL for join, concur).** HPOA OMIM 166,574 / ORPHA 115,853 / DECIPHER 296; zero UMLS. 0/49 joinable.

## Two further attacks that land

**Disease-level coverage collapse (MODERATE–STRONG).** Row-level fill understates the problem: of 12,996 HPOA diseases, only 2,037 (15.67%) carry *any* attribute gold; severity-specifically 486 (3.7%). **84.3% of curated diseases have zero attribute gold**, and the populated subset is curator-biased toward well-studied Mendelian syndromes — not a representative clinical sample.

**Onset "gold" is self-anchored, not clinician (MAJOR, partially contested).** The skeptic measured that `onset_verified_set.json` (n=254) and `onset_final_judge.json` (n=30) **already exist** and were produced by V1/V2 prompt variants → LLM judge rounds; grep for `clinician|annotator|adjudicat` returns zero metadata fields. **If onset F1 were computed against this set today, it would be LLM self-validation — circular.** The affirmative's framing ("clinician path unbuilt, the only valid path") is correct in prescription but omits that an LLM-judged set already sits on disk and must not be mistaken for gold. This does not weaken the affirmative's conclusion (onset = no valid gold); it sharpens *why*.

## Genre/construct gap — sharper than "closable"

The affirmative calls the MACCROBAT→disease-text gap "closable by 50–100 in-genre spans." The skeptics show this understates it: MACCROBAT is patient-instance, temporally-indexed prose ("6-week history of palpitations… 2–3 times per week"); production input is generic disease prose ("HIV is a retrovirus…"), no patient, no anchored spans. These are different **constructs** (instance attributes vs. prototypical disease knowledge), not one construct across genres — a construct-validity gap. And the 50–100 closing spans must themselves be clinician-authored in-genre, so the genre gap and the unbuilt-clinician-gold gap **collapse into a single unbuilt dependency.**

## Net position

IE-quality is provable today for **one attribute (location, STRONG) and marginally a second (severity, WEAK–MODERATE)** — both MACCROBAT-only and genre-bound. The remaining four (onset-pace, character, aggravating, relieving) have **no valid gold**: HPOA cannot anchor extraction faithfulness on four independent grounds (construct, emptiness, level/circularity, namespace), and the only on-disk onset "gold" is LLM-self-judged. Blinded in-genre clinician adjudication remains the sole valid path for 4–5 of 6 attributes — precedented, sound, but **unbuilt**, and at 3.7–8.9% fill its confidence intervals would run 2–4× wider than the n≈600 plan advertises. The affirmative's honesty on counts and joins is **unimpeached**; the substantive corrections are the **severity downgrade**, the **level/circularity ceiling on HPOA**, and the **construct-validity (not transfer) framing** of the genre gap.

---

## ROUND 5 — AFFIRMATIVE
# IE Attribute Provability — Honest Position (Round 5, Measured)

**Question:** Can per-attribute IE faithfulness (location/severity/onset/character/aggravating/relieving) be rigorously proven with available gold? **Verdict: 2 of 6 attributes are provable today; the other 4 require an unbuilt clinician-adjudicated set.** All numbers below are measured, not assumed.

## Core decomposition (the affirmative signal)

Strict exact-name join yields **high precision, low recall**: the bottleneck is finding-name string matching, **not** attribute fidelity. On the **name-aligned subset of MACCROBAT**, when our IE attaches an attribute to a correctly-matched finding it is almost always right — **location P=0.922 (tp=83, fp=7), severity P=1.000 (tp=35, fp=0)**. Recall (R=0.08 / 0.14) is dragged down purely by name mismatch. The opposition's quoted F1≈0.56–0.61 reflects fuzzy-alignment scoring of the same `v144_single_pred.json` artifact, which is keyed by identical MACCROBAT doc IDs and is directly joinable.

## Per-attribute verdict

| Attribute | Gold source | Real count | Strength | Basis |
|---|---|---|---|---|
| **Location** | MACCROBAT span MODIFY | **1,539 pairs / 193 of 200 docs** | **STRONG** | P=0.922 on aligned subset; recognized brat span-relation gold; scoring executed |
| **Severity** | MACCROBAT span MODIFY | **276 pairs / 121 of 200 docs** | **MODERATE** | P=1.000 but n-limited (Wilson ±18%), MACCROBAT-only, genre-bound |
| **Onset (pace)** | HPOA col7/col10 | 0 usable | **NONE** | Wrong construct (age-of-onset subtree, not sudden/gradual); 0 Acute/0 Insidious/0 Subacute, 17 Chronic, all OMIM/ORPHA → 0 UMLS join |
| **Character** | HPOA modifier | 0 | **NONE** | HP:0025280 subtree = 0 rows in modifier column |
| **Aggravating** | HPOA modifier | 0 | **NONE** | HP:0025285 = 0; only off-target HP:0025206 "triggered by cold" (53 rows), not span-level, 0 UMLS join |
| **Relieving** | HPOA modifier | 0 | **NONE** | HP:0025254 = 0 across col7+col10 |

## Why location/severity hold and the other four do not

**MACCROBAT location/severity are text-span MODIFY annotations** (`Biological_structure → Sign_symptom`, `Severity → Sign_symptom`), anchored to verbatim spans (`('right thyroid lobe','nodule')`, `('mild','pain')`). The gold *is* the span, so there is no literature-recall conflation and no disease-knowledge circularity — the construct-validity/level-mismatch attack that applies to HPOA does **not** touch them.

**HPOA fails on three independent axes for the other four attributes:**
1. **Wrong construct** — its onset column is the age-of-onset subtree (Congenital 891, Infantile 543, … OLS4-confirmed), orthogonal to onset-pace. Its dense **frequency** column (218,573 rows, 77.3%) is an *edge prior* P(phenotype|disease), not a per-finding attribute — high density buys no validation power.
2. **Near-zero / off-target population** — modifier column is 1,029 rows (0.36%), ~95% severity/course/laterality; target IDs for character/aggravating/relieving = **0**. Severity-modifier does exist (593 rows / 486 diseases, all OMIM), but it is disease-level, not span-level.
3. **Namespace failure** — HPOA is OMIM/ORPHA/DECIPHER; **0 UMLS**. Even the 414 CUI-mappable severity diseases yield **0 of 49 production CUIs** with severity gold (curator bias toward Mendelian syndromes).

Even at full coverage, HPOA severity is curated disease-level from the same literature the IE reads (TAS+PCS provenance) → circularity ceiling. MACCROBAT's in-genre span gold is the sole survivor for severity.

## Other candidate sources (measured, all negative)

- **CADEC** (1,098 records, `{text, ade, pt}`): ADR→MedDRA only, **zero attribute slots**.
- **SemEval-2015 Task 14** (normalized body-location/severity/course slots): **not present on disk** — unacquired, not actionable.
- **On-disk onset "gold"** (`onset_verified_set.json` 254 recs, `onset_final_judge.json` 30 recs): schema is `idx/pmid/finding/onset/abstract` only — no annotator/kappa field; "clinician/llm/prompt" tokens occur **only inside abstract prose**. LLM-derived and self-anchored; **cannot be cited as clinician gold**.

## The path for the four unanchored attributes

Expert adjudication is **precedented and statistically sizable**, not a fallback excuse (i2b2/n2c2, MACCROBAT, ShARe/CLEF-eHealth, SemEval-2015 were all clinician-annotated with reported κ). Target Wilson 95% CI half-width ≈±7% on per-attribute F1 needs **n≈150–200 in-genre spans/attribute** (≈380 for ±5%). Protocol: 2 blinded clinicians, anti-anchoring (annotators never see IE output; spans shuffled, attribute-randomized), Cohen/Fleiss κ ≥0.6 acceptance, third-adjudicator on disagreement — ≈900–1,200 annotations, a 2–3 day task. **Provability is achievable but currently UNBUILT.**

## Honest concession

MACCROBAT is patient case-notes; production input is disease prose. This **genre gap caps transfer, not in-genre measurability** for location/severity. For onset-pace, character, aggravating, relieving, no on-disk gold exists; blinded in-genre clinician adjudication is the only valid path — concurring with the negative side.

**Bottom line:** Location STRONG, Severity MODERATE (both via MACCROBAT span gold, scoring executed). Onset-pace / character / aggravating / relieving NONE today — provable only via an unbuilt, standard, ~2–3 day clinician-adjudicated set.

## ROUND 5 — REBUTTAL
# IE Attribute Provability — Round 5 Synthesis (Measured, Honest)

**Question.** Can per-attribute IE faithfulness (location / severity / onset-pace / character / aggravating / relieving) be rigorously proven with available gold?

**Verdict after three independent skeptic rebuttals: 1 attribute STRONG, 1 WEAK-to-MODERATE, 4 structurally unprovable today.** The affirmative's "2 of 6 provable" survives on substance but its *severity strength label and genre framing are overstated*. Every number below was independently reproduced by at least one skeptic against on-disk data.

## What the rebuttals CONFIRMED (affirmative correct — concede fully)

- **HPOA onset column = 100% age-of-onset, pace = literally 0.** Reproduced exactly: Congenital `HP:0003577` ×891, Infantile ×543, Childhood ×365; Acute `HP:0011009`=0, Insidious=0, Subacute=0. The "17 Chronic" sits in the *modifier* column (course-duration, not pace). **The onset-pace construct does not exist anywhere in HPOA.**
- **Character / aggravating / relieving target IDs = 0 rows** across the entire file (`HP:0025280/0025204/0025285/0025254`). Only off-target "triggered by cold" `HP:0025206` (~42–53 rows, count immaterial).
- **Namespace wall: 0 UMLS native** (OMIM 166,574 + ORPHA 115,853 + DECIPHER 296). Severity-modifier rows Mendelian-biased → **0 of 49 production CUIs**. No OMIM↔UMLS crosswalk on disk to even attempt the join.
- **MACCROBAT location/severity rest on real span-level MODIFY gold** (11,571 MODIFY relations / 200 files), immune to the HPOA circularity attack. Pair counts (location ~1,539, severity ~276–380) not disputed; one skeptic's E→T parse was incomplete and conceded.

## Where the rebuttals DAMAGE the affirmative (concede partially)

| Attack | Severity | Concession |
|---|---|---|
| **Severity label inflated** | MAJOR | P=1.000 rests on **tp=35** → Wilson LB ≈0.90, single-corpus, single-genre. Two skeptics independently call this **WEAK / WEAK-MODERATE**, not MODERATE. **Downgrade Severity to WEAK-MODERATE.** |
| **Genre transfer for Location** | MAJOR | MACCROBAT is unambiguously *patient case-notes* ("A 28-year-old man presented with..."); production input is disease-encyclopedia prose. P=0.922 is an **in-genre upper bound that does not transfer**. Honest label: **Location STRONG-in-genre / UNPROVEN-in-production.** |
| **HPOA task-level mismatch (beyond empty columns)** | MAJOR | TAS 136,433 + PCS 115,320 = disease-level literature digests, not span anchors. Even at full coverage HPOA measures *construct overlap, not span faithfulness*. "HPOA could work at scale" is **permanently false, not merely currently empty.** Strengthens the affirmative's own conclusion. |
| **On-disk onset "gold" is self-anchored** | FATAL (for any onset F1) | `onset_verified_set.json` (254) / `onset_final_judge.json` (30): schema `idx/pmid/finding/onset/abstract`, zero annotator/kappa/clinician tokens. LLM-derived → measures self-consistency. **n=30 → Wilson ±17pp**: cannot adjudicate a 0.56-vs-0.61 dispute. The "gold is incomplete" defense is **unfalsifiable**, not merely weak. |

## Where a skeptic OVERREACHED (affirmative holds)

- **MACCROBAT "character" exists, was silently dropped" (Attack 3, skeptic 3).** Partially valid: ~2,959 `Detailed_description → Sign_symptom` MODIFY pairs do exist. But these are free-text, un-normalized quality spans with no controlled target vocabulary — scorable in principle, not as clean gold today. Verdict: upgrade character from **NONE → UNSCORED-GOLD-EXISTS**, a real correction the affirmative should accept.

## Final scorecard

| Attribute | Strength (post-rebuttal) | Basis |
|---|---|---|
| Location | **STRONG-in-genre / UNPROVEN-in-production** | MACCROBAT span gold, P=0.922 (tp=83) |
| Severity | **WEAK-MODERATE** (downgraded) | P=1.000 but tp=35, Wilson LB ≈0.90 |
| Character | **UNSCORED — gold exists** (corrected) | ~2,959 free-text MODIFY pairs, un-normalized |
| Onset-pace | **STRUCTURALLY UNPROVABLE today** | HPOA wrong construct; only gold is self-anchored LLM output (n=30) |
| Aggravating | **NONE** | 0 target rows, 0 UMLS |
| Relieving | **NONE** | 0 target rows, 0 UMLS |

## Bottom line

The affirmative position **survives intact on the 4 NONE/structural verdicts** — three skeptics reproduced those counts exactly and could not improve on them. It **loses two labels**: Severity (MODERATE→WEAK-MODERATE) and Location (STRONG→STRONG-in-genre only), both on n-bound, single-genre evidence. The decisive shared finding: **clinician adjudication is not a fallback but the only path with construct + genre + namespace validity** — and until that blind, anti-anchored n≈150–200 set is *executed* (not merely scoped as a "2–3 day task"), four of six attributes have **no empirical floor**.

---

# 최종 판정 (Adjudicator)
Every decisive number is confirmed by direct file inspection. I now deliver the adjudication.

---

# ADJUDICATION: Per-Attribute IE-Quality Provability

**Verdict: PARTIAL — provable today for location (strong) and severity (weak-moderate); the other four require an unbuilt, in-genre, clinician-adjudicated gold set.** All counts below were re-verified by direct file parse this session, not taken from the transcript.

## Decisive verified numbers

| Claim | Verified value | Source |
|---|---|---|
| HPOA total rows | 282,723 data rows | `phenotype.hpoa` |
| HPOA `onset` col fill | 3,054 (1.08%), top IDs HP:0003577×891, HP:0003593×543 — **all age-of-onset subtree** | direct awk |
| HPOA `modifier` col fill | 1,030 (0.36%) | direct awk |
| Pain-character HP:0025280 | **0 rows** | grep |
| Triggered HP:0025204 / Aggravated HP:0025285 / Ameliorated HP:0025254 | **0 / 0 / 0 rows** | grep |
| Cold-trigger HP:0025206 | 42 rows (single off-target leaf) | grep |
| HPOA namespace | OMIM 166,574 / ORPHA 115,853 / DECIPHER 296; **UMLS C-ids = 0** | direct awk |
| MACCROBAT MODIFY relations | 11,571 across 200 docs | grep |
| Biological_structure / Severity entities | 2,953 (199 docs) / 376 (146 docs) | grep |
| `onset_verified_set.json` schema | `[idx,pmid,finding,onset,abstract]` — **no annotator/kappa field**; onset∈finding only 8.3% (21/254) | json parse |
| Production IE fill | location 12.4%, character 12.1%, onset 8.9%, severity 3.7% | `unified4_O2.json` |
| Production IE attribute fields present | name, onset, severity, location, character — **aggravating/relieving absent entirely** | json parse |

## Per-attribute protocol

**LOCATION — provable today; STRONG in-genre, UNPROVEN in production.**
Gold: MACCROBAT `Biological_structure→Sign_symptom` span MODIFY (~1,539–1,802 pairs, 197–199/200 docs). Span-anchored brat relations — immune to HPOA's circularity/namespace problems. Proves: span-faithfulness of the location slot. Cannot prove: disease-level correctness, nor transfer to encyclopedic disease text (MACCROBAT is patient case-notes). Protocol: exact doc-ID join to `v144_single_pred.json`, binding-F1 + value-aware F1. Strength: the single attribute with recognized executable gold.

**SEVERITY — provable today; WEAK-MODERATE.**
Gold: MACCROBAT `Severity→Sign_symptom` (276 pairs, 146 docs). Same protocol. The downgrade is sustained: precision rests on tp≈35 (Wilson lower bound ≈0.90), single-corpus, single-genre, and production IE fills severity in only 3.7% of findings — a scorer reports floor recall by construction. HPOA's severity-modifier (593 rows) is non-load-bearing: disease-level, 0 of 49 production CUIs joinable. Strength: real but n-bound and genre-bound.

**ONSET-PACE — NOT provable today.**
HPOA `onset` is verified 100% age-of-onset subtree (Congenital/Infantile/Childhood); the pace axis (Acute/Insidious/Subacute) is literally 0 rows — orthogonal construct. The only on-disk "gold," `onset_verified_set.json`, has **no annotator field** and is 8.3% verbatim, i.e. LLM-labeled and self-anchored; scoring against it is circular. Path: clinician adjudication only.

**CHARACTER — NOT provable today.**
HP:0025280 = 0 rows in HPOA. MACCROBAT `Detailed_description` spans exist (~2,920) but are free-text, un-normalized to a controlled vocabulary — "gold exists but unscored," not clean gold. Path: clinician adjudication with a controlled character lexicon.

**AGGRAVATING — NOT provable today.**
HP:0025204/0025285 = 0 rows; only off-target cold-trigger (42 rows, one disease family, disease-level). **The IE pipeline has no aggravating field**, so "incomplete gold" is unfalsifiable. Path: clinician adjudication — and the IE slot must first exist.

**RELIEVING — NOT provable today.**
HP:0025254 = 0 rows in every column; no IE field. Path: clinician adjudication; slot must first exist.

## Why HPOA cannot be the gold for any attribute (4 independent, all confirmed)
1. **Construct** — onset column = age-of-onset, not pace (0 pace rows).
2. **Emptiness** — character/aggravating/relieving target IDs = 0/0/0.
3. **Level/circularity** — HPOA is disease-level curation from TAS/PCS literature (no character offsets), the same genre the IE reads; agreement conflates KB-recall with span-fidelity. This is permanent, not a coverage gap.
4. **Namespace** — 0 UMLS C-ids; 0 of 49 production diseases joinable.

## Overall verdict
IE quality is provable to a **publishable standard for location alone**, and defensibly-but-weakly for severity — both at binding/span level, on MACCROBAT case-notes, with an unmeasured genre-transfer gap to production disease text. For **onset-pace, character, aggravating, relieving there is no usable gold on disk**, and two of them have no IE slot to score. The minimum required is a **blinded, system-naive, in-genre clinician-adjudicated set**: stratified spans sampled from `v105_sources/*.txt`, ≥2 clinicians labeling before seeing IE output, Cohen/Fleiss κ≥0.6 gate, third-adjudicator tie-break. Statistical sizing must use the realistic fill, not full coverage: at n≈150–200/attribute Wilson half-width is ≈±7%, which **cannot separate P=0.78 from P=0.85** — the contested "acceptable IE" band — so n≈380–600/attribute (±4–5%) is the honest minimum for a decisive claim. This set is precedented (i2b2/n2c2, ShARe/CLEF, SemEval-2015) but **does not yet exist**; until executed, four of six attributes have no empirical floor.

Relevant paths: `/windows/data/external_kg/phenotype.hpoa`, `/home/max/Graph-DDXPlus/pilot/data/cache/maccrobat/brat/*.ann`, `/home/max/Graph-DDXPlus/pilot/data/cache/maccrobat/unified4_O2.json`, `/home/max/Graph-DDXPlus/pilot/data/cache/maccrobat/onset_verified_set.json`, `/home/max/Graph-DDXPlus/pilot/data/cache/v105_sources/*.txt`.