# Paper 1 (속성 IE 방법론/데이터셋) 독립 필요성 — 5차 적대적 공방 (2026-06-25)

51 에이전트(라운드당 affirmative 5+synth / negative 3+synth, 누적 5라운드 → adjudicator), 웹 문헌 기반. 사용자 지시: 정직·학술, 분리 안함 결론도 수용.

## ROUND 1 — AFFIRMATIVE
# Necessity of a Standalone Paper 1 (Attribute-IE Method + Resource) — Round 1 Affirmative Synthesis

**Scope.** Whether a standalone Paper 1 — a benchmark-blind, source-grounded, ontology-normalized clinical *attribute*-IE method plus an open disease→phenotype resource — is necessary independent of Paper 2's diagnosis system. Five angles were examined: methodological gap, evaluation-methodology gap, resource value, cost/local-LLM, and publication precedent.

## Consolidated Verdict by Angle

| Angle | Rating | Load-bearing basis |
|---|---|---|
| Methodological gap | Moderate (leaning strong) | Attribute/assertion IE benchmarks are near-universally patient-note-scoped; knowledge-text source is unaddressed |
| Evaluation-methodology gap | Moderate | Bundled eval (partial-gold + NLI faithfulness + gold-deflation + applicability) is integrative, not primitive |
| Resource value | Moderate-to-strong | <1% of HPO terms carry severity/time-course metadata; dominant KGs use bare triples |
| Cost/local-LLM | Moderate | Privacy/cost case is established (thus not novel); attribute-specific local IE is the narrow differentiator |
| Publication precedent | Strong (general); moderate (attribute niche) | Resource-then-application split is venue-endorsed and routine |

## Resolved Position

The five angles converge on a consistent picture rather than contradicting one another. Every angle independently concludes that **no single ingredient is novel**, yet **the specific combination is unaddressed**. The recurring differentiator across all angles is the same: the *source type* (disease-knowledge text, not patient notes) paired with the *target representation* (ontology-normalized qualified edges with per-finding location/severity/onset/character).

A surface tension exists between the evaluation-gap angle ("gold *does* exist for severity/course/location," weakening the no-benchmark premise) and the resource angle ("<1% HPO coverage = real gap"). These are reconciled by distinguishing **clinical-note attribute gold** (which exists: SemEval-2015 Task 14, n2c2 2018, CEGS N-GRID) from **disease-knowledge-text attribute resources** (which are absent, and where HPO's <1% modifier coverage is the documented hole). The gap is one of *domain coverage and representation*, not a missing primitive.

The cost/local-LLM angle is the weakest standalone pillar precisely because it is well-established; it contributes only when fused with the attribute-IE method. The publication-precedent angle is the strongest *enabling* condition (the split is permitted and routine: NeurIPS D&B, LREC, CADEC, i2b2/n2c2, MACCROBAT) but, by its own admission, demonstrates that the split is *permitted*, not that this specific paper is *required*.

## Critical Dependencies (honest caveats, agreed across angles)

1. **Contribution is combinatorial, not fundamental.** Each component (attribute IE, zero-shot LLM IE, HPO modifiers, NLI faithfulness, gold-deflation) has strong prior art.
2. **Auto-generated resource → quality scrutiny.** An 8B-LLM-built KG demands rigorous QC; intrinsic validation (MACCROBAT F1 ≈0.53–0.61, NLI faithfulness ≈83%, gold-deflation analysis) is necessary but below curated-gold standards.
3. **Unverified contingencies.** No published external uptake of the proposed resource yet (potential, not realized); no head-to-head commercial-model (GPT-4/Claude) comparison; attribute-IE-as-standalone-resource has no exact published precedent; SNOMED CT post-coordination overlap must be addressed.

## Overall Round-1 Necessity Verdict

**MODERATE, leaning toward established.** Necessity is supported but not decisively proven. The convergent evidence shows the task is genuinely under-addressed (strong on source/representation novelty), the resource gap is real and citable (<1% HPO coverage), and the resource-paper split is venue-endorsed (strong). Necessity rests on the *combination* and the *knowledge-text domain shift* — not on any new primitive — and remains contingent on demonstrated QC rigor, quantified coverage gain over HPOA, and (ideally) a commercial-model comparison. On the affirmative side for this round, necessity is **established at a moderate confidence level**, with publication precedent and the documented HPO coverage gap as its firmest supports, and the cost/privacy angle as its weakest.

## ROUND 1 — REBUTTAL
# Round 1 Skeptical Rebuttal — Standalone Paper 1 Is Not Necessary

**Scope.** This rebuttal contests the affirmative's "MODERATE, leaning toward established" verdict on the necessity of a standalone Paper 1. Three independent lines of attack — salami/LPU, novelty insufficiency, and resource redundancy plus validation weakness — converge on the same conclusion: necessity is **WEAK**, and the affirmative's firmest pillars are either pre-empted by prior art or structurally dependent on Paper 2.

## Attack 1 — The central combinatorial claim is occupied prior art (FATAL to novelty)
The affirmative concedes "no single ingredient is novel" and rests everything on the *combination*: knowledge-text source + ontology-normalized qualified attribute edges. That exact combination is published. Tu et al. used GPT-4 to annotate severity/onset/frequency for **17,502 of 17,548 HPO terms (99.7%) directly from the HPO knowledge base — not patient notes — released as open code+CSV** ([Frontiers Digit. Health 2026](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1794934/full)). This single precedent refutes three affirmative pillars at once: the "knowledge-text source is unaddressed" methodological gap, the "<1% HPO modifier coverage" resource gap, and the self-flagged "no exact published precedent" caveat. Ontology-normalized LLM extraction of location/severity with controlled-vocabulary prompting is also the field's current baseline, not a contribution ([CLINES medRxiv 2025](https://www.medrxiv.org/content/10.64898/2025.12.01.25341355v1.full.pdf); [PheNormGPT, PMC11498178](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11498178/); [RELATE, arXiv 2509.19057](https://arxiv.org/html/2509.19057v1)).

## Attack 2 — The "<1% HPO coverage" pillar is misleading (MAJOR)
The affirmative's firmest resource support is misstated against bare HPO. HPOA carries >115,000 disease-phenotype annotations *with age-of-onset and frequency*, and the Clinical Modifier subontology encodes severity, positionality, and aggravating factors ([HPO 2021, PMC7778952](https://pmc.ncbi.nlm.nih.gov/articles/PMC7778952/)). Orphanet HOOM qualifies every disease-phenotype link by frequency plus diagnostic-criterion flags as queryable OWL/SPARQL ([HOOM](https://sciences.orphadata.com/hoom/)). SNOMED CT models Finding Site + Severity via post-coordination ([SNOMED Editorial Guide](https://confluence.ihtsdotools.org/display/DOCEG/Clinical+Finding+Defining+Attributes)). The honest residual gap is narrow — *per-finding location/character on the disease→phenotype edge from knowledge text* — not "<1% coverage." The affirmative must re-baseline against HPOA + HOOM + the GPT-4-severity resource.

## Attack 3 — The validation firewall makes Paper 1 unable to prove fitness-for-purpose (MAJOR — the core structural problem)
The affirmative's anti-circularity boundary forbids Paper 1 from citing Paper 2's diagnostic ablation. That firewall is self-defeating. Stripped of downstream payoff, Paper 1's standalone evidence is span/relation F1 **0.53–0.61** on MACCROBAT (patient-note gold — not even the claimed knowledge-text domain) and ~83% NLI faithfulness, meaning **~1 in 6 attribute statements is unsupported** from an 8B model. "Gold-deflation" excuses the low F1 by claiming the metric is wrong rather than measuring the resource — a non-falsifiable defense. The proof that the attributes are useful in aggregate lives in Paper 2. Under COPE's three-part test (same population/methods/question), the two papers share corpus, IE pipeline, and overarching question; this is the canonical merge trigger ([COPE](https://publicationethics.org/guidance/cope-position/handling-duplicated-or-redundant-content-salami-slicing); [ICMJE](https://www.icmje.org/recommendations/browse/publishing-and-editorial-issues/overlapping-publications.html)).

## Genuine Affirmative Points Conceded
- **Source-type shift is real.** HPOA/HOOM are human-curated, not text-IE; LLM IE *from disease-knowledge text* with per-finding location/character is a genuine, if narrow, delta beyond Tu et al.'s HPO-only severity scope.
- **The resource→application split is venue-endorsed** (NeurIPS D&B, LREC). The affirmative correctly frames this as *permitting*, not *necessitating*.
- **Gold-deflation quantification is more rigorous than typical resource papers** — a modest but real methodological contribution.
- **Not misconduct.** COPE prescribes *proportionate* responses; transparent cross-citation and companion submission neutralize the ethics charge. The risk is "merge" reviews, not desk rejection.

## Overall Severity: MAJOR (necessity downgraded from MODERATE to WEAK)
The affirmative's verdict overstates a position resting on an unrealized contingency. Two of its three firmest pillars — the knowledge-text novelty and the "<1% coverage" gap — are pre-empted by Tu et al., HPOA, and HOOM. The third weakness is structural: the validation firewall the affirmative correctly imposes leaves Paper 1 unable to demonstrate fitness-for-purpose on its own. The charge is **curable** — via demonstrated independent external uptake, a quantified non-diagnostic coverage gain over HPOA/HOOM, and commercial-model comparison — but each is unproven today. **Recommended outcome: merge with Paper 2, or release Paper 1 as a dataset-only companion**, not a standalone research paper.

---

## ROUND 2 — AFFIRMATIVE
# Affirmative Synthesis — Round 2: Necessity of a Standalone Paper 1

**Overall verdict for this round: MODERATE-to-STRONG.** Across five independent angles, the necessity of Paper 1 as a standalone contribution survives the opposition's strongest attacks. The methodological/precedent pillars are strong; the resource and reproducibility pillars are moderate. The consistent residual weakness is that necessity rests on *recombination and prospective uptake*, not a new algorithmic primitive.

## 1. The methodological gap is concrete and unoccupied (STRONG)

Every major clinical attribute/assertion IE benchmark targets **patient notes**, not disease-knowledge text: i2b2 2010 (Uzuner et al., *JAMIA* 2011), SemEval-2015 Task 14 (Elhadad et al., 2015), n2c2 2018/2019 (Henry et al., *JAMIA* 2020), and CLAMP/cTAKES (Soysal et al., 2018; Savova et al., 2010). The source-type shift from third-person knowledge text to *population-level* qualified edges is a genuine task delta, not cosmetic. The nearest true analogue, AutoRD (Lu et al., *JMIR* 2024, PMC11683654) — local-LLM IE from disease-knowledge text into an ontology-grounded rare-disease KG — restricts relations to {produces, is_a, increases_risk_of, synonym, anaphora} with **explicitly no qualifiers**. A pipeline this close that omits attributes entirely is direct evidence the qualified-edge-from-knowledge-text task is unoccupied.

## 2. The opposition's "fatal" precedent is mis-scoped (STRONG)

Tu et al. (Murphy, Schilder, Skene, *Front. Digit. Health* 2026) is cited by the opposition as pre-emptive prior art. Verified against source, it shares **none** of Paper 1's defining axes: input is **structured HPO term IDs** (not free text), **severity only** (9 indicators, no location/character/onset/aggravating), annotation at **HPO-term level** (not disease→phenotype edge), and **cloud GPT-4** (not local 8B). The opposition itself conceded the source-type shift is "genuine." A precedent that does a strict subset cannot occupy the superset. Notably, Tu et al. is *also* a standalone LLM-annotation resource paper with no downstream system — confirming the split is editorially accepted (Claim refuting the salami charge).

## 3. Resource and edge-attribute novelty are real but narrow (MODERATE)

No existing edge-attributed resource carries location/character on disease→phenotype links: HOOM/Orphanet qualifies only by frequency/diagnostic-criterion flags; PrimeKG (Chandak et al., *Sci. Data* 2023) edges carry only frequency weights; SemMedDB uses UMLS predicates (deprecated Dec 2024). The delta — per-finding location + character, source-grounded from text, edge-level — is genuine and venue-viable (*Sci. Data*, *Database*), but narrow. The 8B faithfulness ceiling (~83% NLI; F1 0.53–0.61) tempers standalone reference-dataset value.

## 4. The evaluation framework defeats the "non-falsifiable" charge (MODERATE)

Gold incompleteness systematically deflates exact-match F1 — a recognized field problem (AdaK-NER, arXiv 2109.05233; JMIR 2025, e78432), not a self-serving excuse. For attributes lacking gold (onset/character), no settled metric exists (Laskar et al., ACL 2025), and NLI-based faithfulness is the recognized substitute (NAACL 2025 findings.433). This refutes Attack 3's "non-falsifiable defense" framing. Weakness: the four axes are an *assembly* of recognized components, not a named protocol — rigor, not necessity.

## 5. Reproducibility/privacy supply a downstream-free fitness criterion (MODERATE-to-STRONG)

API-based IE has structural reproducibility failures (arXiv 2512.00651; 2412.12004); a frozen, version-pinned local-8B artifact is the recommended remedy and addresses privacy demands the opposition's GPT-4 precedents do not (JAMIA Open 2025). This supplies a fitness-for-purpose axis that does **not** require Paper 2. Caveat: zero-shot 8B still trails on complex multi-relation IE (arXiv 2601.09367), so faithful multi-attribute extraction is non-trivial — hence publishable.

## Honest resolution of contradictions

The angles agree the gap is **real and unoccupied**; they disagree only on **strength of necessity**. Methodological/precedent/reproducibility arguments compel publishability; resource/evaluation arguments establish rigor but not unavoidability. No angle claims a new primitive — necessity is recombinative (benchmark-blind + source-grounded + local-8B + edge-level + multi-attribute) and contingent on **prospective independent uptake**, which precedent (CADEC ~268 citations; i2b2) shows is permitted and routinely cited but cannot yet guarantee for this silver (LLM-generated) artifact. **Necessity is established at MODERATE-to-STRONG: a standalone Paper 1 is justified and defensible, though not logically forced.**

## ROUND 2 — REBUTTAL
# Skeptical Rebuttal — Round 2 (Consolidated): Paper 1 Does Not Clear the Standalone Bar

**Overall severity: MAJOR (not fatal).** The affirmative's MODERATE-to-STRONG verdict overstates necessity. Three independent rebuttal lines — editorial/salami policy, novelty insufficiency, and resource redundancy — converge on the same correction: Paper 1 is justified as a **cross-referenced dataset/resource companion** to Paper 2, not as a co-equal standalone *methodology* paper. The affirmative's own framing concedes the load-bearing weakness: necessity is "recombinative" and "contingent on prospective independent uptake."

## 1. Editorial / salami-slicing exposure (MAJOR)
COPE defines salami slicing by shared **population + methods + question** ([COPE](https://publicationethics.org/guidance/cope-position/handling-duplicated-or-redundant-content-salami-slicing)). Papers 1 and 2 share corpus, the local-8B IE pipeline, and one research program; only the *evaluation lens* differs. Legitimate splits require each slice to answer a "distinct and important research question" and "stand alone" ([Jackson 2014, JCN](https://onlinelibrary.wiley.com/doi/10.1111/jocn.12439); [Smart 2017, DMCN](https://pubmed.ncbi.nlm.nih.gov/28691264/)) — precisely the test Paper 1 strains, since its data/method/corpus are not distinct. The affirmative's admission that Paper 1's value is "prospective and future-tense" matches ICMJE's "minor increments as distinct contributions" warning ([ICMJE](https://www.icmje.org/recommendations/browse/publishing-and-editorial-issues/overlapping-publications.html)); editors judge LPU on demonstrated, not hoped-for, contribution ([UH Research](https://research.uh.edu/the-big-idea/university-research-explained/salami-slicing-a-recipe-for-research-misconduct/)). The credible review outcome is "merge" or "release as dataset companion."

## 2. Novelty insufficiency (MAJOR)
Every primitive is established 2024–2025 prior art. Singh et al. ([arXiv 2507.01810](https://arxiv.org/pdf/2507.01810)) already combine controlled vocabulary + structured output + span-grounding + normalization with **small** models on clinical attribute-value extraction — Paper 1's exact stack, differing only in source *type*. AutoRD ([JMIR 2024](https://medinform.jmir.org/2024/1/e60665)) already does local-LLM ontology-grounded IE from disease-knowledge text; the residual delta is "add 5 qualifier slots," which is incremental engineering. Prompt-tuning wins (onset 44→90%) are artifact-tuning, not generalizable findings ([ChatIE](https://arxiv.org/pdf/2302.10205); [OEMA](https://arxiv.org/pdf/2511.15211)).

## 3. Resource redundancy (FATAL to "unoccupied resource"; MAJOR overall)
**PhenoSSU** ([JMIR 2021](https://www.jmir.org/2021/6/e26892)) already extracts a 12-attribute model — including **location** and **character** — from disease-knowledge text, normalized to SNOMED-CT/UMLS, released as a public edge-attributed phenotype KG, validated at κ=0.861. This occupies Paper 1's claimed superset; the affirmative's resource comparison (§3) omits it. Separately, `phenotype.hpoa` already carries Onset/Frequency/Severity across >8,600 diseases ([HPO 2021](https://academic.oup.com/nar/article/49/D1/D1207/6017351)), and severity was already GPT-4-annotated HPO-wide ([medRxiv 2024](https://www.medrxiv.org/content/10.1101/2024.06.10.24308475v1)). Paper 1's defensible novelty narrows to **edge-level location + character from a local 8B** — two attributes.

## 4. Validation cannot demonstrate fitness without Paper 2 (MAJOR)
Comparators clear a far higher intrinsic bar with no downstream system (PhenoSSU κ=0.861; HPO-severity 96% expert recall). Paper 1 offers F1 0.53–0.61, NLI ~83%, and **no expert gold for onset/character**, leaning on self-declared gold-deflation. Demonstrating the attributes "do something" requires Paper 2's ablation — which restates the salami charge rather than refuting it ([DataRubrics](https://arxiv.org/pdf/2506.01789); [dual gold+silver norm](https://arxiv.org/pdf/2509.16722)).

## Genuine concessions to the affirmative
- **Source-type / patient-notes vs. knowledge-text delta is real** (affirmative §1). Not disputed under any policy.
- **The attribute-on-knowledge-text combination minus PhenoSSU's SNOMED targeting is partly unoccupied**: edge-level location+character via local-8B targeting HPO is genuinely uncrowded — but narrow.
- **Gold-deflation is a real, cited phenomenon**, and the NLI-faithfulness substitute is methodologically respectable for ungolden axes.
- **The split is rescuable, not misconduct**: transparent cross-citation + companion disclosure materially lowers desk-reject risk ([COPE](https://publicationethics.org/guidance/cope-position/handling-duplicated-or-redundant-content-salami-slicing); [Editage](https://www.editage.com/insights/if-i-develop-and-submit-two-papers-for-from-a-questionnaire-survey-would-it-be-considered-salami-slicing)).
- **Reproducibility/privacy is a legitimate downstream-free fitness axis** (affirmative §5) — but supports a *resource* artifact, not a methods paper.

## Verdict
The affirmative establishes that *something* publishable exists; it does **not** establish a standalone co-equal methodology paper. Necessity collapses from MODERATE-to-STRONG to **MAJOR-weakened**: the honest residual is a narrow, cross-referenced **dataset/resource companion** (edge-level location+character, local-8B, HPO-targeted), explicitly disclosing Paper 2. The Tu et al. precedent supports exactly this resource framing — not a co-equal methods split.

---

## ROUND 3 — AFFIRMATIVE
## Round 3 Synthesis — Is Paper 1's Standalone Necessity Established?

**Scope.** Five affirmative angles were argued for the necessity of a standalone Paper 1 (benchmark-blind, zero-shot, source-grounded extraction of clinical qualifier attributes from *disease-knowledge text* into *HPO-modifier-normalized qualified edges* via a *local 8B* model). This document reconciles them and renders one honest verdict for this round.

### Angle-by-angle position

**1. Methodological gap — MODERATE.** No surveyed clinical attribute/assertion IE system operates on disease-knowledge text producing ontology-normalized qualified edges under zero-shot benchmark-blind constraints. Prior art targets *patient notes* (i2b2/VA, Uzuner 2011; SemEval-2015 Task 14, Elhadad; n2c2, Henry 2020; CLAMP/cTAKES, Soysal 2018, Savova 2010). Two of three opposing "occupancy" citations do not survive inspection: **PhenoSSU** is manual annotation + SVM/BiLSTM *classification* of pre-recognized entities to SNOMED (Deng 2021), not zero-shot generation; **Neveditsin et al. 2507.01810** concerns output-format parseability over clinical notes, not knowledge-text IE. **AutoRD** (Lu 2024) is the genuine adjacency but extracts disease↔feature *relations* with GPT-4 and has *no qualifier layer*. Honest ceiling: novelty is **recombinative** (register × normalization target × supervision regime), not a new primitive — publishable, but weaker than STRONG.

**2. Evaluation-methodology gap — WEAK-to-MODERATE.** Attribute axes (onset/character) lack released gold corpora; faithfulness/NLI substitution and gold-deflation are legitimate, citable primitives (Groza 2023; arXiv 2410.12222; Rebholz-Schuhmann PMC4301805). This defeats the skeptic's "validation needs Paper 2" claim. **But** Mahbub et al. (arXiv 2604.06028, 2026) already publish a six-stage weak-supervision validation framework as a standalone contribution, of which the proposed protocol is largely a subset. Defensible delta is narrow (per-evidence applicability + attribute-specific deflation), best as a methods section, not a co-equal framework paper.

**3. Resource value — MODERATE.** HPOA carries onset/frequency/severity but **no location or character axis** (Köhler 2021); PrimeKG/Hetionet edges are binary (Chandak 2023). PhenoSSU is real prior art but bounded: 193 infectious diseases, SNOMED-normalized, rule/manual (Deng 2021). The genuinely uncrowded cell — **HPO-normalized, broad-disease, local-8B-reproducible, span-grounded location+character edge attributes** — supports a standalone *data descriptor*, not a co-equal methods paper.

**4. Cost/local-LLM fitness — MODERATE.** "Can a privately-deployable small model do task X faithfully?" is independently publishable in top venues with no downstream system (Wiest, npj Digital Medicine 2024; npj 2025; NEJM AI 2025; JAMIA Open 2025), and open models still lag GPT-4 on fine-grained attributes (PMC12099357), so qualifier-level fitness on knowledge-text is genuinely open. But these precedents publish as *resources/pipelines* — strengthening the standalone case without reaching co-equal methodology.

**5. Publication precedent — MODERATE.** Standalone IE method/resource papers routinely out-cite their originating application (scispaCy, Neumann 2019; MetaMap, Aronson 2010; MACCROBAT, Caufield 2019), and *Scientific Data*/LREC sanction resources as primary contributions. This establishes the *form* is non-salami; it does not prove *this* artifact will earn comparable uptake.

### Resolving the contradictions

The angles do not conflict on facts; they converge on a consistent shape once two over-claims are dropped. (i) The strongest *necessity* claim — a co-equal **methods** paper resting on a novel primitive — is **not** supported: novelty is recombinative (Angle 1) and the validation framework is partly pre-occupied (Angle 2). (ii) The convergent, well-supported claim across Angles 3–5 is a standalone **resource/data-descriptor** (HPO-normalized location+character qualified edges, local-8B-reproducible), which the skeptic's own Round-2 concession endorses as legitimate and non-salami.

### Overall honest verdict for Round 3

**Necessity is established at MODERATE strength — but only for a standalone resource/data-descriptor contribution, not for a co-equal methods paper.** Across five independent angles the convergent finding is consistent: the salami charge is **rebutted** (separate publication is sanctioned; fitness can be demonstrated without Paper 2), yet every angle independently tops out at MODERATE because the surviving novelty is narrow (location + character; recombinative; partly pre-occupied evaluation). No angle reaches STRONG, and none collapses to WEAK except the evaluation-framework framing. The defensible position: **Paper 1 should stand as a reproducible HPO-normalized attribute-edge resource paper with an internal validation protocol — not as a heavyweight standalone methodology.**

**Round-3 necessity rating: MODERATE (resource framing) / NOT ESTABLISHED (co-equal methods framing).**

## ROUND 3 — REBUTTAL
# Round 3 Skeptic Rebuttal — Consolidated

**Bottom line: The rebuttal is moderately-to-strongly damaging. It does not overturn the affirmative's central concession-driven verdict, but it dismantles the higher of the two ratings (co-equal methods paper) and pushes the surviving claim down toward "dataset companion to Paper 2" rather than a freestanding MODERATE resource.**

---

## 1. Salami / editorial-policy attack — the affirmative defends on the wrong axis

The affirmative's novelty defense ("register × normalization target × supervision regime") is a defense by *method/outcome differences on a single underlying study*. COPE's redundant-content test turns on shared **population, methods, and question/hypothesis**, and explicitly rejects "splitting up papers by outcomes"; legitimacy requires a *distinctly separate question* ([COPE](https://publicationethics.org/guidance/cope-position/handling-duplicated-or-redundant-content-salami-slicing); [Biochemia Medica 2013](https://www.biochemia-medica.com/en/journal/23/3/10.11613/BM.2013.030/fullArticle)). Paper 1 and Paper 2 share one dataset (the IE outputs), one pipeline, and one overarching question. The editor heuristic is *purpose*: an artifact "generated by and for" the companion paper reads as the textbook smallest publishable unit ([IJE 2020](https://academic.oup.com/ije/article/49/1/281/5570871); [ORI/Roig](https://ori.hhs.gov/education/products/roig_st_johns/Salami%20slicing.html)). **Severity: major.** The affirmative answered the *form* question (is separate publication ever allowed?) but not the *content* question (is this artifact separable?).

## 2. Novelty — every primitive is pre-occupied

CLINES (Dec 2025) already performs reasoning-LLM extraction capturing **anatomical location, assertion status, numerical values/units**, normalized to UMLS via SapBERT, schema-reconciled and "ontology-grounded, auditable" ([medRxiv 2025.12.01.25341355](https://www.medrxiv.org/content/10.64898/2025.12.01.25341355v1)). RAG-with-UMLS NER, CoT/heuristic zero-shot prompting, and iterative error-driven prompt refinement are all commodity 2023–2025 technique ([JMIR Med Inform 2024 e55318](https://medinform.jmir.org/2024/1/e55318); [arXiv 2312.02296](https://arxiv.org/html/2312.02296v1)). AutoRD + CLINES as a union covers the proposed pipeline; the "uncrowded cell" is an *intersection gap*, not a *capability gap*. **Severity: major.** This directly corroborates — and hardens — the affirmative's own "recombinative, not new-primitive" admission.

## 3. Resource redundancy + validation deadlock

The supposedly empty "HPO-normalized location" cell is occupied at the ontology level: HPO's **Clinical modifier subontology (HP:0012823)** standardizes Position/Laterality/Spatial pattern/Severity ([JAX](https://www.informatics.jax.org/vocab/hp_ontology/HP:0012823)), and **SNOMED CT** disorders carry Finding Site, Severity, and Laterality as defining attributes ([SNOMED Editorial Guide](https://confluence.ihtsdotools.org/display/DOCEG/Clinical+Finding+Defining+Attributes)). A released **GPT-4 HPO-severity resource annotating 99.7% of phenotypes** without expert gold ([Frontiers Digital Health 2026](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1794934/full)) both pre-occupies the LLM-attribute-resource cell and undercuts the affirmative's "no gold is fine" excuse — that precedent is itself a *competing resource*. The surviving differentiators collapse to "local 8B" + "span-grounding" + free-text "character" — and un-normalized free text is the *weakest* resource form. Compounding this: the affirmative's own Round-3 boundary forbids citing Paper 2's diagnostic ablation, leaving Paper 1 with **no axis to show the edges are diagnostically meaningful rather than faithfully copied**. **Severity: major.**

---

## Genuine affirmative points conceded (honest)

- **Form is non-salami.** IE resource papers legitimately out-cite parents (scispaCy, MetaMap, MACCROBAT); *Scientific Data*/LREC sanction resources as primary contributions. The pure salami-by-form charge fails.
- **"Validation needs Paper 2" is defeated.** Intrinsic faithfulness/NLI and gold-deflation are legitimate, citable primitives. The skeptic withdraws that line.
- **The recombination is genuinely unoccupied as a tuple.** No single surveyed system does qualifiers on disease-knowledge text under benchmark-blind zero-shot with a local 8B. The affirmative's "recombinative" self-assessment is correct.
- **AutoRD genuinely lacks a qualifier layer**; **character** and **HP:0012823/0012824-normalized** edges appear unoccupied [unverified — no negative-evidence search confirms absence]. This is the strongest residual for the affirmative.
- **Local-8B fitness** is independently publishable in principle.

---

## Overall damage rating

**MODERATELY-TO-STRONGLY DAMAGING.** The rebuttal concedes the affirmative's strongest planks (form-legitimacy, intrinsic validation) honestly, so it cannot claim a knockout. But on the two ratings the affirmative actually issued, it lands hard on the higher one: the **co-equal methods framing is effectively destroyed** (novelty fully pre-occupied per CLINES/AutoRD/RAG-UMLS). The **MODERATE resource framing is wounded, not killed** — Attacks on content-redundancy (HPO/SNOMED/Frontiers) plus the validation deadlock push it below a freestanding MODERATE toward "**versioned data-descriptor companion to Paper 2.**" Net: the affirmative's NOT-ESTABLISHED (methods) verdict is *confirmed and strengthened*; its MODERATE (resource) verdict is *downgraded but not eliminated*. The one unverified hinge (HP:0012823/0012824 + character truly unoccupied) is where the affirmative retains its only path back to MODERATE.

---

## ROUND 4 — AFFIRMATIVE
# Round 4 Synthesis — Necessity of a Standalone Paper 1

**Scope.** Five affirmative angles tested whether Paper 1 (benchmark-blind, source-grounded, zero-shot extraction of clinical *attributes/qualifiers* — location, severity, onset, character — onto disease→phenotype edges, HPO/UMLS-normalized) is a *necessary, freestanding* contribution, independent of the downstream diagnosis system (Paper 2). All five converge on the same verdict: **MODERATE**.

## The convergent affirmative case

**1. A register-defined empty cell exists (methodological gap).** Prior attribute/qualifier IE operates on *patient clinical notes*: i2b2 2010 ([Uzuner et al., *JAMIA* 2011](https://academic.oup.com/jamia/article-abstract/18/5/552/830538)), SemEval-2015 Task 14 on ShARe ([Elhadad et al.](https://alt.qcri.org/semeval2015/task14/)), n2c2 2018/2019 on MIMIC ([Henry et al., *JAMIA* 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7489085/)), CLINES ([medRxiv 2025](https://www.medrxiv.org/content/10.64898/2025.12.01.25341355v1)). The disease-knowledge-text branch carries relations *without* qualifiers: AutoRD ([Lu et al., *JMIR* 2024](https://medinform.jmir.org/2024/1/e60665)) and RareDis ([Martínez-deMiguel et al., *JBI* 2021](https://www.sciencedirect.com/science/article/pii/S1532046421002902)) annotate `produces/is_a` only — no attribute layer. The cell "SemEval-2015-style qualifiers on disease-knowledge edges" is genuinely unoccupied.

**2. A reusable evaluation protocol for *ungolded* attributes.** Severity/onset are the worst-resourced (F1≈0.37 vs trigger 0.97, [Lybarger et al. 2020](https://arxiv.org/pdf/2012.00974)); clinical factuality benchmarks were absent until FactEHR ([Munnangi et al. 2024](https://arxiv.org/pdf/2412.12422)). Gold-deflation is peer-reviewed ([Li et al., ICLR 2021](https://arxiv.org/abs/2012.05426)), making explicit deflation-bounding a contribution; NLI faithfulness is the accepted no-gold surrogate ([NLI4CT, SemEval-2024](https://arxiv.org/pdf/2408.03127)).

**3. A genuine resource gap.** No surveyed KG carries source-grounded per-edge qualifiers: SemMedDB ([Kilicoglu et al. 2012](https://academic.oup.com/bioinformatics/article/28/23/3158/195282)), DisGeNET ([Piñero et al. 2020](https://academic.oup.com/nar/article/48/D1/D845/5611674)), PrimeKG/Hetionet ([Chandak et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9893183/)). HPOA partially occupies it but at disease-level, expert-curated, non-span ([Köhler et al. 2021](https://academic.oup.com/nar/article/49/D1/D1207/6017351)).

**4. Reproducibility as an independent property.** Local open-weight 8B IE is an active agenda ([Lentzen et al., *JAMIA Open* 2025](https://academic.oup.com/jamiaopen/article/8/5/ooaf109/8270821)); commercial-API resources are non-reproducible even at temp 0 ([Staudinger et al. 2025](https://arxiv.org/html/2510.25506v3)), so a frozen 8B pipeline is auditable where a GPT-4 resource is not. 8B-class parity is demonstrable ([PMC12894992](https://pmc.ncbi.nlm.nih.gov/articles/PMC12894992/)).

**5. Editorial precedent for resource-then-application splitting.** scispaCy, MetaMap, CADEC ([Karimi et al. 2015](https://pubmed.ncbi.nlm.nih.gov/25817970/)), MACCROBAT ([Caufield et al. 2019](https://academic.oup.com/database/article/doi/10.1093/database/bay143/5290151)), HPO — all published apart from their applications with independent uptake, satisfying COPE's distinct-question test.

## Resolving the tensions

The angles agree more than they conflict. The single recurring **honest weakness** across all five: the contribution is **recombinative, not a new primitive** — a register transposition + qualifier-layer addition, with severity *partially* pre-occupied (Frontiers GPT-4 HPO-severity, term-level), and "character" un-normalized and weakest. The Frontiers resource is consistently neutralized the same way: it is *term-level, single-modifier, non-reproducible* — different granularity, different unit, foreclosing neither the resource nor the method. Three items remain **[unverified]**: absolute absence of "character" qualifiers, per-evidence applicability as a named construct, and independent reuse precedent for this *specific narrow* artifact (general-purpose release form is the precondition).

## Overall verdict for Round 4

**Necessity is ESTABLISHED at MODERATE strength.** The empty register cell is confirmed by the canonical disease-text corpus (RareDis) lacking the qualifier layer, the resource gap holds across six KGs, and the reproducibility/precedent planks defeat the skeptic's pre-occupation and salami attacks. This supports a freestanding **methods-plus-resource paper**, intrinsically justifiable without Paper 2 — but **not** a co-equal *novel-task-family* claim: no new primitive, partial severity pre-occupation, and independent demand contingent on general-purpose (MACCROBAT-like) release. Conditional on that release form, Paper 1 is necessary and defensible; absent it, the case weakens toward salami.

## ROUND 4 — REBUTTAL
# Round 4 Skeptic Rebuttal — Synthesis Against Paper 1's "MODERATE Necessity"

The affirmative establishes Paper 1's necessity at **MODERATE**, resting on five planks: a register-defined empty cell, an ungolded-attribute evaluation protocol, a resource gap, reproducibility, and editorial precedent. Three skeptic lines — salami-slicing, novelty insufficiency, and resource redundancy — converge on a narrower verdict: **the contribution survives only as a dataset/companion release, not a co-equal standalone methods paper.**

## Attack 1 — Novelty is recombinative, and every primitive is pre-occupied [MAJOR]

The affirmative concedes the contribution is "recombinative, not a new primitive." Each component is established prior art. Source-grounded, hallucination-controlled structured IE is the dominant 2024–25 clinical paradigm ([mCODEGPT, *Nature Comms Med* 2025](https://www.nature.com/articles/s43856-025-01116-x); [BoostCD 2025](https://arxiv.org/pdf/2506.14901)). Controlled-vocabulary/ontology-grounded zero-shot extraction is occupied ([OEMA, arXiv 2511.15211](https://arxiv.org/pdf/2511.15211); [Entity-Decomposition 2407.04629](https://arxiv.org/pdf/2407.04629)). scispaCy/HPO normalization is commodity tooling ([PhenoGPT, PMC10801236](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801236/); [retriever-augmented HPO 90.3% Top-1, 2409.13746](https://arxiv.org/html/2409.13746v1/)). Severity/onset *modifier* extraction is pre-occupied at term level (GPT-4, ~97% recall on all HPO terms, [Frontiers 2026](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1794934/full)). Prompt-iteration recall lifts (onset 44→90%) are the textbook iterative protocol ([JMIR Med Inform 2024](https://medinform.jmir.org/2024/1/e55318)), not findings. The novelty collapses to *register transposition + edge-attachment* — a transposition, not a method.

## Attack 2 — The resource is substantially pre-occupied per-edge [MAJOR]

The affirmative concedes only *partial* severity pre-occupation. The redundancy is broader: HPOA + Orphanet already carry, per disease→phenotype edge, frequency, onset/clinical course, and clinical modifiers including severity, positionality/location, and triggers ([Köhler 2021](https://academic.oup.com/nar/article/49/D1/D1207/6017351); [HPO 2019](https://academic.oup.com/nar/article/47/D1/D1018/5198478)). Four of five claimed attributes (location, severity, onset, aggravating) already exist as standardized per-edge qualifiers in the canonical resource. The contribution reduces to *adding text-span provenance* to a substantively existing layer — incremental, not an empty cell. "Character" — the one genuinely un-pre-occupied attribute — is exactly the cell with no gold, no normalization, and **[unverified]** novelty: the thinnest possible standalone hook.

## Attack 3 — Validation that proves value is structurally exiled to Paper 2 [MAJOR, structural]

A resource paper's central claim is fitness-for-use. Intrinsic numbers cannot establish it: F1≈0.53–0.61 on MACCROBAT sits at or below reported severity ceilings (0.37–0.61, [Lybarger 2020](https://arxiv.org/pdf/2012.00974)) — field-mediocre by the affirmative's own weak standard. Gold-deflation explains *why* the number is low but converts a weak score into an unfalsifiable one. No expert gold exists for onset/character; faithfulness rests on an NLI surrogate (~83%) that is contested in clinical settings (claim-verification F1 0.44, [healthcare NLI 2025](https://arxiv.org/html/2512.16189v3)) and cannot localize errors. Fitness is demonstrable **only** via downstream diagnostic ablation — which lives in Paper 2. This is the salami signature: the validation proving value sits in the other paper. Per COPE/ICMJE, where the later paper is unpublished, the prescribed response is "invited to expand and rewrite... citing the earlier publication" — i.e., **merge** ([ICMJE Overlapping Publications](https://www.icmje.org/recommendations/browse/publishing-and-editorial-issues/overlapping-publications.html)).

## Conceded affirmative points (genuinely solid)

- **Register transposition is real.** RareDis/AutoRD annotate `produces/is_a` only; the canonical disease-text corpus lacks the qualifier layer. The methodological gap exists.
- **Reproducibility is a legitimate, citable advantage.** A frozen local-8B pipeline is auditable where a GPT-4 resource is not ([Staudinger 2025](https://arxiv.org/html/2510.25506v3)).
- **Resource-then-application editorial precedent holds** (CADEC, MACCROBAT, scispaCy published apart with independent uptake) — COPE's distinct-question escape hatch *if* released MACCROBAT-style.
- The salami charge is **not** triggered by shared data/pipeline alone — "lack of transparency does" ([PMC 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8880828/)). Transparent cross-referencing defeats the naive charge.

## Overall verdict

**Damage rating: MODERATE-to-MAJOR — narrowing, not overturning.** The skeptic case does not reduce Paper 1 to zero; the register cell, reproducibility, and precedent planks survive intact. But it forecloses the affirmative's strongest reading: novelty is recombinative with every primitive occupied, four of five attributes are pre-occupied per-edge in HPOA/Orphanet, and the fitness-for-use proof is structurally exiled to Paper 2. The affirmative's own MODERATE collapses to its lower bound: **necessary and defensible only as a methods-plus-resource companion conditioned on a general-purpose MACCROBAT-style release decoupled from DDXPlus/SymCat vocabulary, with explicit cross-referencing.** Absent that release form, the predicted editorial response is **merge or desk-reject**, exactly as the affirmative concedes its case "weakens toward salami."

---

## ROUND 5 — AFFIRMATIVE
# Round 5 Synthesis — Necessity of a Standalone Paper 1

**Scope.** Five independent affirmative angles tested whether an attribute-normalized, source-grounded disease→phenotype IE resource/method merits standalone publication, separate from the downstream diagnosis system (Paper 2). Verdict reported per angle, then reconciled.

## Per-angle verdicts

| Angle | Rating | Load-bearing evidence |
|---|---|---|
| Methodological gap (register + qualifier layer) | **Moderate** | No prior attribute-IE benchmark targets disease-knowledge text; all use patient notes (i2b2 2010, SemEval-2015 Task 14, n2c2 2018/19 — Uzuner *JAMIA* 2011; Elhadad *SemEval* 2015). RareDis (NORD text) annotates entities/relations but no qualifier layer (Martínez-deMiguel *JBI* 2021). OntoGPT/SPIRES has no clinical-qualifier slots (Caufield *Bioinformatics* 2024). |
| Evaluation protocol | **Moderate** | Field-wide clinical-NLP eval gap documented (PMC8487512; PMC10014687). Gold-deflation is quantified, not an excuse (17.3% passage-supported FPs, arXiv 2601.16711; 51.67%, arXiv 2305.05003). Single-axis NLI insufficient → bundle justified (PMC11441639). |
| Resource value | **Moderate (strengthened)** | HPOA modifier slots <1% populated and **term-level, not edge-level** (Frontiers Digital Health 2026; HPO NAR 2024). PrimeKG/Hetionet edges typed but unqualified; SemMedDB deprecated 2024-12. |
| Cost / reproducibility / local-LLM | **Moderate** | API drift up to 43%, temp-0 insufficient (NEJM AI 2024); PHI requires on-prem (JAMIA Open ooaf109 2025). But 8B-parity claim survives only as scoped existence proof — prevailing literature shows 8B trails GPT-4 zero-shot (PMC11221943). |
| Publication precedent | **Strong (1 caveat)** | ShARe/CLEF 2014 Task 2 = near-exact attribute set (Severity/Course/Body Location/Conditional), released standalone on PhysioNet. CADEC ~268 cites (reuse-dominant), scispaCy/MACCROBAT/n2c2 reused independently. HPO maintainers: "<1% of terms contain metadata such as time course and severity" (Köhler *NAR* 2019). |

## Resolving contradictions

- **HPOA redundancy.** The skeptic's "4/5 attributes pre-occupied per-edge" is empirically refuted, not merely unproven. Three independent sources converge: HPO maintainers' "<1% of terms" (Köhler 2019), Frontiers 2026 ("less than 1%"), and the format spec marking `Modifier` optional. The earlier "[unverified] Modifier rate" boundary (methodological-gap angle) is **resolved**: the rate is <1% and, critically, **term-level not edge-level**. Schema capacity ≠ coverage. This converts redundancy from a major to a minor objection.
- **8B-parity.** Honestly downgraded. "Small local matches commercial" is contradicted by 8B-specific zero-shot evidence and survives only as a narrow existence proof for source-grounded structured-attribute IE. It is **not** a load-bearing necessity claim; reproducibility and privacy (Claims 1–2) carry this angle independently.
- **Novelty vs. transposition.** Conceded consistently: every primitive (scispaCy, HPO modifiers, NLI, deflation, applicability, source-grounding) is prior art. Necessity rests on *recombination + register transposition*, not a new primitive — bounded but real.

## Overall honest verdict — THIS round

**Necessity is established at MODERATE-to-STRONG strength, conditional on release design.**

Four angles independently land at *moderate*; precedent lands at *strong*. They converge rather than conflict: the contribution is a **populated, edge-qualified, register-transposed resource + reusable evaluation protocol**, filling an empirically demonstrated gap (HPOA <1% populated, no disease-text qualifier benchmark, no qualified-edge KG). The pattern is editor-accepted and non-salami when cross-referenced (PMC8880828).

Necessity does **not** rest on novelty of any primitive, nor on the 8B-parity claim. It holds **conditional on** three release requirements all five angles agree on:
1. **Register-general, vocabulary-decoupled** release (NORD/StatPearls/MACCROBAT-style; not bound to DDXPlus/SymCat).
2. **A small expert-gold attribute layer** so fitness-for-extraction is shown intrinsically (not borrowed from Paper 2).
3. **A falsifiable deflation-adjudication procedure** (expert review of a FP sample) so the evaluation protocol is not "unfalsifiable."

If those conditions are unmet, the angle collapses toward the skeptic's lower bound (dataset-only, fitness exiled to Paper 2). If met, Paper 1 stands alone as a resource+methods contribution — its necessity does not depend on Paper 2's diagnostic results.

## ROUND 5 — REBUTTAL
# Round 5 Rebuttal — Consolidated Skeptic Case Against a Standalone Paper 1

**Thesis.** The affirmative's own Round 5 synthesis concedes the decisive ground: necessity rests not on novelty of any primitive but on *recombination + register transposition*, and holds only "conditional on" three unmet release requirements. Four independent attack lines converge to show that, as the work currently stands, those conditions are unmet and the contribution collapses toward the skeptic's lower bound: a **dataset/companion release**, not a co-equal methods paper.

## 1. Salami / editorial policy — aggregate MAJOR
The COPE "same population, same methods, same question" test ([COPE](https://publicationethics.org/guidance/cope-position/handling-duplicated-or-redundant-content-salami-slicing)) cuts *against* Paper 1: the IE and KG come from one pipeline, one LLM, one corpus, one team, one continuous study whose telos is diagnosis — the affirmative concedes the IE was *built for* Paper 2. The affirmative's load-bearing cite, PMC8880828, is **conditional, not blanket cover**: it permits same-dataset splitting only when the project is *genuinely too large for one paper* with *transparent justification* ([PMC8880828](https://pmc.ncbi.nlm.nih.gov/articles/PMC8880828/)). An attribute-IE method + small gold layer is a normal-sized methods paper and **fails the "too large" prong on its face**. The "released standalone" precedent (ShARe/CLEF, CADEC, MACCROBAT) supports a **dataset release** — not a second full paper alongside the application ([*Int J Epidemiol* 2020](https://academic.oup.com/ije/article/49/1/281/5570871)). Concurrent submission with shared corpus/model and no cross-citation is a concrete desk-reject trigger.

## 2. Novelty-as-method insufficient — MAJOR
The claimed recombination (zero-shot + schema-defined attributes + ontology-grounding) is **already instantiated**. SPIRES/OntoGPT is zero-shot, schema-driven (LinkML slots = attribute set), and deterministically grounds to external vocabularies ([SPIRES btae104](https://academic.oup.com/bioinformatics/article/40/3/btae104/7612230)); adding Severity/Location slots is configuration, not contribution. CLINES does LLM attribute assignment + UMLS normalization + zero-shot ([CLINES](https://www.medrxiv.org/content/10.64898/2025.12.01.25341355v1)). Guideline-in-prompt iteration is a named, published technique ([Hu, arXiv 2303.16416](https://arxiv.org/pdf/2303.16416); [annotation-guideline NER](https://www.sciencedirect.com/science/article/pii/S1386505625004472)); the "onset 44→90%" figure is a *result*, not a method. This forces the affirmative off "novel method" onto "novel populated resource + protocol."

## 3. Validation weakness — MAJOR ×3 (the central blow)
- **Intrinsic F1 trails prior art.** Body-location and severity modifiers are already extracted at F1 0.74–0.93 ([PMC3994852](https://pmc.ncbi.nlm.nih.gov/articles/PMC3994852); [PMC7647140](https://pmc.ncbi.nlm.nih.gov/articles/PMC7647140/)). A flagship 0.53–0.61 on the *two easiest* attributes cannot claim intrinsic methodological merit. Register transposition relocates the number; it does not rescue it.
- **Deflation is currently unfalsifiable.** The affirmative *lists* the FP-adjudication procedure as an unmet condition; until delivered, "gold is incomplete, our true F1 is higher" is an excuse, and the 17.3%/51.67% figures are borrowed from other corpora.
- **3/5 attributes have zero human gold.** Onset/character are self-evaluated against the model's own scheme. "Fitness shown intrinsically" is false for 60% of the attribute set today.

## Genuine concessions (solid affirmative points)
1. **Distinct research question** (IE methodology ≠ diagnosis) is real — COPE leaves the call to editors case-by-case, so salami is *not fatal*.
2. **The edge-qualified, register-transposed resource niche is genuinely open.** HPOA modifiers are <1% populated and term-level; PrimeKG/Hetionet/RTX-KG2 edges are unqualified ([PrimeKG](https://www.nature.com/articles/s41597-023-01960-3); [RTX-KG2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9520835/)). Redundancy is correctly **minor/conceded**.
3. The deflation protocol + expert-gold layer, **if delivered**, are genuine standalone substance.

## Overall damage rating: MAJOR (not fatal)
No single attack is a kill-shot, and the affirmative's resource/niche argument survives intact. But three independent lines (intent/salami, method novelty, intrinsic validation) converge on one verdict: **"is the resource good?" is answerable today only via Paper 2's downstream result.** The affirmative's case stands *only at its own conditional floor* — and all three release conditions (register-general release, expert gold, falsifiable deflation) are presently unmet. As described — pipeline-coupled, fitness-deferred, sub-prior-art F1 — Paper 1 should ship as a **dataset/companion release**, not a co-equal standalone methods paper. The affirmative wins the *potential*; the skeptic wins the *present state*.

---

# 최종 판정 (Adjudicator)
# Final Verdict: Standalone Necessity of Paper 1 (Attribute-IE Method/Resource)

## Recommendation: One paper + dataset/resource companion — do NOT publish Paper 1 as a co-equal standalone *methods* paper

The evidence across five rounds converges, and the affirmative itself migrated to this position. By Round 3 the affirmative explicitly conceded the **co-equal methods framing is NOT ESTABLISHED**, retreating to a resource/data-descriptor claim. The skeptic then wounded even that, leaving a contribution whose present-state fitness-for-use is provable only through Paper 2.

## (1) Surviving Affirmative Necessities

| Necessity | Strength | Status after rebuttal |
|---|---|---|
| **Register transposition is a real empty cell** — qualifier-layer IE on disease-knowledge text (vs. patient notes); canonical disease-text corpora (RareDis, AutoRD) annotate `produces/is_a` only, no qualifier layer | **Moderate** | SURVIVED — conceded by skeptic in every round |
| **Edge-qualified resource niche is genuinely open** — HPOA modifiers <1% populated AND term-level not edge-level; PrimeKG/Hetionet/RTX-KG2 edges unqualified | **Moderate** | SURVIVED — skeptic downgraded redundancy to "minor/conceded" in R5 |
| **Reproducibility/privacy of a frozen local-8B pipeline** vs. non-reproducible GPT-4 resources (API drift, PHI/on-prem) | **Moderate** | SURVIVED — but supports a *resource*, not a *method* |
| **Resource-then-application editorial form is non-salami** (CADEC, scispaCy, MACCROBAT, ShARe/CLEF) | **Strong (form only)** | SURVIVED — but proves the split is *permitted*, not *required*, and supports a **dataset release**, not a second full paper |

## (2) Surviving Fatal/Major Objections

1. **Novelty is recombinative, every primitive pre-occupied (MAJOR, decisive against methods framing).** Affirmative conceded this in all five rounds. SPIRES/OntoGPT (zero-shot, schema/LinkML slots, deterministic ontology grounding) and CLINES (LLM attribute assignment + UMLS normalization + zero-shot) instantiate the stack; adding Severity/Location slots is configuration. Prompt-iteration "onset 44→90%" is a *result*, not a method.

2. **Intrinsic validation trails prior art (MAJOR, structural).** Body-location/severity modifiers are already extracted at F1 0.74–0.93 (PMC3994852, PMC7647140); the flagship 0.53–0.61 sits on the *two easiest* attributes and at/below severity ceilings (Lybarger 0.37–0.61). Register transposition relocates the number, it does not rescue it.

3. **Fitness-for-use is structurally exiled to Paper 2 (MAJOR, the central blow).** A resource paper's core claim is fitness-for-purpose. Gold-deflation explains a low F1 but renders it unfalsifiable until the FP-adjudication procedure is delivered. **3 of 5 attributes (onset, character) have zero human gold** — self-evaluated against the model's own scheme. The proof the attributes are diagnostically meaningful (not merely faithfully copied) lives in Paper 2's ablation — the salami signature.

4. **COPE "too large" prong fails on its face (MAJOR).** Same corpus, same pipeline, same LLM, one continuous study whose telos is diagnosis. PMC8880828 (affirmative's own cite) permits same-dataset splitting only when *genuinely too large for one paper*. An attribute-IE method + small gold layer is a normal-sized methods section.

## (3) Why not the other two options

- **Two separate papers — REJECTED.** Requires a co-equal methods contribution the affirmative itself abandoned by Round 3; novelty fully pre-occupied; fitness unprovable without Paper 2.
- **One fully combined paper — viable but not optimal.** The register-transposed, edge-qualified, locally-reproducible *artifact* has genuine independent reuse value (the niche survived). Burying it entirely forfeits citable resource uptake (CADEC ~268 cites precedent).

## (4) The exact reframing required (companion release, not split)

Release Paper 1's output as a **versioned, cross-referenced dataset/resource companion** (data-descriptor or repository release, e.g. *Scientific Data*-style), NOT a co-equal methods paper. Conditions all five rounds agreed on — without these it collapses to "dataset-only with fitness exiled to Paper 2":

1. **Register-general, vocabulary-decoupled release** (NORD/StatPearls/MACCROBAT-style; not bound to DDXPlus/SymCat vocabulary, or the salami/intent charge triggers).
2. **A small expert-gold attribute layer** so extraction fitness is shown intrinsically — critically covering **onset and character** (the 3/5 currently un-golded), not just the easy axes.
3. **A falsifiable deflation-adjudication procedure** (expert review of a FP sample) so the evaluation protocol is not an unfalsifiable excuse.
4. **Transparent cross-citation** of the diagnosis work (defeats the naive salami charge per PMC8880828).

## Decisive points

- The affirmative's own R3/R5 concessions ("recombinative, not a new primitive"; "conditional on" three *unmet* release requirements) are dispositive — the case never recovered to a methods-paper rating.
- The validation firewall is genuinely self-defeating: forbidding Paper 2's ablation while having no expert gold for 3/5 attributes leaves no axis to demonstrate the attributes are diagnostically meaningful today.
- The resource niche (edge-level location+character, HPO-modifier-normalized, local-8B-reproducible) is the one plank that survived all five rebuttals intact — which is precisely why the answer is **companion resource, not a second paper**.

**Bottom line:** The affirmative wins the *potential* (a reusable artifact exists); the skeptic wins the *present state* (its value is currently demonstrable only through Paper 2). Ship the IE as a transparently cross-referenced dataset/resource companion under the conditions above — not as a standalone co-equal methods paper.