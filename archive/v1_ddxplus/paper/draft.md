# Traceable automatic differential diagnosis via parameter-free knowledge graph exploration

## Title Page

**Title**: Traceable automatic differential diagnosis via parameter-free knowledge graph exploration

**Running Head**: Traceable KG Differential Diagnosis

**Author**: [Author Name]<sup>1</sup>

**Affiliation**:
<sup>1</sup>[Institution]

**Corresponding Author**:
- Name: [Corresponding Author Name]
- E-mail: [Email]

---

## Abstract

**Objective**: Automatic differential diagnosis systems based on reinforcement learning or large language models depend on training data and produce decision processes that are not directly traceable to explicit medical knowledge. We investigated whether parameter-free 2-hop knowledge graph exploration can achieve competitive diagnostic performance and identified which design factors drive that performance.

**Materials and Methods**: We mapped 49 diseases and 223 evidences from the DDXPlus benchmark to Unified Medical Language System concepts and built a bipartite knowledge graph in Neo4j. The system iteratively inquires symptoms and ranks candidate diseases via 2-hop traversal without parameter learning. We evaluated 200 configurations (inquiry strategies × stopping criteria × scoring functions) on 134,529 DDXPlus test cases, using analysis of variance on a validation subset (n=1,000) to identify dominant factors.

**Results**: Denied symptom-based hypothesis elimination was the dominant factor, explaining 94.67% of variance in inquiry efficiency (F=788.4, p<.001, η²=0.947); the remaining three factors contributed a combined η²=0.024. The best configuration (denied threshold=6, Top-3 Stability stopping, evidence ratio scoring) reached ground truth pathology accuracy at Top-1 of 91.05% with 23.1 questions on average, versus 75.39% for the AARLC baseline. In a controlled comparison with MEDDxAgent (GPT-4o) on the same 100 cases, MEDDxAgent reached 86% versus 81% at inquiry length=15, showing that a large language model with pre-trained medical knowledge retains an accuracy advantage. Every diagnostic path was recoverable as a graph trajectory.

**Conclusion**: Parameter-free 2-hop knowledge graph exploration achieves diagnostic performance comparable to learning-based approaches on DDXPlus without model training or benchmark-specific data. The principal contribution is identifying denied symptom-based hypothesis elimination as the dominant design factor, which reframes the optimization target for future knowledge graph symptom checkers. External validation on real clinical data is required before clinical application.

**Keywords**: Differential Diagnosis, Unified Medical Language System, Biological Ontologies, Knowledge Bases, Algorithms

---

## 1. Introduction

Automatic differential diagnosis systems interact with patients to collect symptoms and progressively refine a ranked list of candidate diseases [1]. The goal is to automate the iterative clinical process of hypothesis generation, evidence gathering, and hypothesis testing [2].

On the DDXPlus benchmark, reinforcement learning approaches achieved GTPA@1 of 75.39% [1] but depend on training data and produce decision processes that are not directly traceable to graph structure. Large language model (LLM) approaches reached higher accuracy. MEDDxAgent, a GPT-4o-based modular agent, reported GTPA@1 of 86% (weak matching) [3], while also carrying risks of hallucination and inaccurate diagnoses acknowledged by the original authors, together with API costs, latency, and reproducibility constraints [3]. More fundamentally, learning-based models (whether reinforcement learning or LLM) embed clinical reasoning within model parameters, which limits direct traceability of individual diagnostic decisions to explicit medical knowledge. This makes such models difficult to meet the interpretability requirements for medical artificial intelligence adoption [2, 3] and the explainability mandates of the European Union Artificial Intelligence Act (2024) [4].

Knowledge graphs (KGs) offer an alternative to these black-box models. A KG stores symptom–disease relationships in an explicit, symbolic form. UMLS integrates over 200 medical vocabularies and assigns unique concept identifiers (CUIs) to more than 4 million medical concepts, providing standardized medical knowledge [5]. KG-based reasoning can present diagnostic evidence directly as graph traversal paths, offering interpretability, reproducibility, and verifiability [6, 7]. KGs have been applied to medical domains including disease prediction using UMLS clinical concept relations [8] and prescription recommendation through knowledge graph construction [9].

Research on automatic differential diagnosis falls into three categories. In learning-based approaches, AARLC, a reinforcement learning agent, achieved GTPA@1 of 75.39%, while BASD, a supervised learning method, reported 67.71% on DDXPlus [1]. These approaches depend on training data and embed decision processes within model internals, limiting interpretability [10]. Bayesian network-based systems provide transparency through probabilistic reasoning but require expert knowledge or training data for constructing conditional probability tables [11]. In LLM-based approaches, MEDDxAgent, a GPT-4o-based modular agent framework, reported GTPA@1 of 86% on DDXPlus [3], but faces hallucination risks [12], API costs, and reproducibility constraints [3]. In KG-based approaches, symptom–disease relationships are represented explicitly, providing interpretability and reproducibility [6, 7], but existing studies have focused primarily on diagnostic inference; no prior work has combined active symptom inquiry with systematic stopping criteria comparison within a unified framework.

We propose GraphTrace, a method that performs automatic differential diagnosis using 2-hop knowledge graph exploration over symptom–disease relationships, without any model parameter learning. It differs from prior work by constructing an end-to-end pipeline from symptom inquiry to final diagnosis using only UMLS-based graph traversal, and by conducting a systematic factor decomposition of 200 design combinations to isolate the mechanism driving performance. Our research questions are: (1) Can parameter-free 2-hop knowledge graph exploration achieve diagnostic performance in the range of existing learning-based approaches on the DDXPlus benchmark? (2) Which design factors in the symptom inquiry strategy most strongly determine exploration performance? (3) What combination of stopping criteria and scoring functions balances accuracy against inquiry length?

---

## 2. Methods

### 2.1 System Overview

The system consists of three components: (1) symptom inquiry, which selects the next symptom to ask; (2) stopping criteria, which decide when to stop asking; and (3) diagnostic scoring, which ranks diseases from collected symptoms. The system operates without a training phase; all knowledge comes from the structural information of the UMLS knowledge graph (Figure 1).

![fig](fig1.png)

**Figure 1.** Overview of the 2-hop knowledge graph exploration system. In the inquiry phase, confirmed symptoms retrieve candidate diseases from the knowledge graph (1st hop), and these candidate diseases generate the next symptom to ask the patient (2nd hop). A Yes (+) response adds a confirmed symptom; a No (-) response adds a denied symptom. This cycle repeats (dashed line) until the stopping criteria are met (the top-3 ranked diagnoses remain unchanged for n consecutive turns). In the decision phase, the evidence ratio scoring function ranks all candidate diseases based on accumulated confirmed and denied symptoms, producing the final ranked differential diagnosis.

### 2.2 Knowledge Graph Construction

We built a knowledge graph from UMLS symptom–disease relationship data. The graph consists of symptom nodes and disease nodes connected by INDICATES (symptom indicates disease) edges. Each node carries a UMLS CUI, and the graph is stored in Neo4j.

For evaluation on DDXPlus, we mapped 49 diseases and 223 evidences to UMLS CUIs. Throughout this paper, "parameter-free" denotes a precise operational definition: no model parameters are learned from DDXPlus training, validation, or test data. The UMLS concept mapping is a one-time knowledge base construction step performed prior to any experiment and is analogous to ontology preparation in classical knowledge-based systems. We explicitly acknowledge that this step involves clinician judgment (see below) and therefore constitutes human-curated domain knowledge injection, not statistical learning.

Diseases were mapped via ICD-10 codes provided by DDXPlus (49/49, 100%). For evidences, we extracted medical terms from the DDXPlus English question fields and queried the UMLS Metathesaurus API. When the API returned multiple candidate CUIs, a reviewer with clinical training selected the most semantically appropriate CUI. Of 223 evidences, 209 (93.7%) were mapped successfully; the 19 unmapped items (8.5%) comprise 14 multi-value sub-attributes (pain location, pain character, skin lesion color) that exist as qualifiers of parent symptoms without independent UMLS concepts, and 5 comorbidity items (diabetes, chronic obstructive pulmonary disease, metastatic cancer, pneumonia history, asthma family history) that possess CUIs but could not be represented in a bipartite symptom–disease graph structure. Because the mapping was performed by reviewers who were aware of the DDXPlus schema, we cannot exclude the possibility that this process introduced information from the benchmark into the knowledge graph; this is discussed as a limitation in Section 4.5.

### 2.3 Symptom Inquiry: 2-Hop Algorithm

The algorithm starts from the patient's initial symptom and explores the knowledge graph in two steps. The first hop retrieves all diseases linked to confirmed symptoms, forming a candidate disease set. The second hop extracts symptoms linked to these candidates, generating next-question candidates. This design aligns with the hypothetico-deductive model of clinical reasoning [12, 13].

#### 2.3.1 Candidate Generation

We compared three design factors in a full factorial design: (1) Co-occurrence (Yes/No), ranking by co-occurrence frequency with confirmed symptoms within the same disease; (2) Denied threshold (0–10, 11 levels), excluding diseases with N or more denied symptoms from candidate symptom generation; (3) Antecedent priority (Yes/No), exploring current symptoms before medical history.

Co-occurrence ranking is supported by symptom cluster research: symptoms of the same disease share molecular pathways [14], and the presence of one symptom raises the probability of others in the same cluster [15].

#### 2.3.2 Selection Strategies

We compared six strategies for selecting the next question from generated candidates.

**(A) Greedy (baseline)**: Selects the top-ranked candidate.

**(B) Information-theoretic**: Computes the expected entropy reduction by simulating yes/no responses for each candidate and selects the one that maximizes expected information gain [6]. The prior P(yes) is estimated from KG structure (the ratio of diseases linked to that symptom among current candidates), not from training data. Variants include maximum information gain and equal binary split.

**(C) Game-theoretic**: Simulates both responses for each candidate and selects the symptom that maximizes the worst-case (minimax) diagnostic score.

#### 2.3.3 Evaluation Metric: Hit Rate

To evaluate inquiry strategies independently of diagnostic performance, we used hit rate: the fraction of asked symptoms to which the patient responds "yes." Each run continues without stopping criteria until all candidate symptoms are exhausted (maximum 223 questions), isolating the strategy's efficiency. This separates the information gathering phase from the inference phase [16].

### 2.4 Stopping Criteria

We compared five stopping criteria: two scoring-independent and three scoring-dependent. Preliminary experiments excluded Confidence ≥ θ and Entropy < θ (74–80 questions on average) and Consecutive Miss and Marginal Hit Rate (accuracy of only 62–63%).

**(A) Cumulative Confirmed ≥ N**: Stops when confirmed symptom count reaches N. Based on sequential hypothesis testing, where a decision is finalized upon reaching an evidence threshold [7].

**(B) Hit Rate Plateau**: Stops when the slope of the cumulative confirmed symptom curve converges to zero [17].

**(C) Top-1 Stability**: Stops when the top-ranked diagnosis remains unchanged for n consecutive turns [13].

**(D) Top-3 Stability**: Stops when the top-3 diagnoses remain unchanged for n consecutive turns. The hypothetico-deductive model holds that clinicians finalize a diagnosis when a limited set of hypotheses stabilizes [2], and given working memory capacity constraints (~4 items) [18], Top-3 represents a lower bound. This criterion structurally prevents premature closure (early fixation on a single hypothesis) by requiring multiple competing hypotheses to stabilize [19].

**(E) Confidence Gap ≥ δ**: Stops when the score gap between the first- and second-ranked diagnoses exceeds δ [20].

All stopping criteria had a maximum question limit of 223 (the total number of DDXPlus evidences).

### 2.5 Diagnostic Scoring

We compared five scoring functions from three theoretical categories. In each, *c* denotes confirmed symptom count, *d* denied count, and *t* total symptom count for a disease. Preliminary experiments excluded Naive Bayes, BM25, and log-likelihood ratio (GTPA@1 below 13%; see Section 4.3).

Functions (A)–(C) share a common structure: a ratio term (precision proxy) multiplied by *c*. The ratio captures the match between disease and symptoms; the *c* multiplier reflects absolute evidence quantity. Between two diseases with equal confirmation rates, the one with more evidence scores higher.

**(A) Evidence Ratio**: $\frac{c}{c+d+1} \times c$ — confirmed fraction among questioned symptoms × confirmed count.

**(B) Coverage**: $\frac{c}{t+1} \times c$ — confirmed fraction of total disease symptoms × confirmed count. *t* is the number of symptoms linked to the disease in the UMLS KG.

**(C) Jaccard**: $\frac{c}{c+d+u} \times c$ (*u* = unasked symptom count). Set similarity [21] × confirmed count.

**(D) IDF-weighted**: Sums the inverse disease frequency of each confirmed symptom (rarer symptoms receive higher weight). Based on the information retrieval principle of valuing rare matches [22].

**(E) Cosine**: Cosine similarity between the confirmed symptom vector and the disease symptom vector [23].

### 2.6 Dataset and Evaluation

We evaluated on DDXPlus [1], which contains 49 diseases and 223 evidences with training, validation, and test splits. Each patient case consists of:

- Patient profile: Age, sex
- Initial evidence: One chief complaint (the system's starting point)
- Evidences: Full list of the patient's symptoms and history (mean 20.1 per patient)
- Pathology: Ground truth disease
- Differential diagnosis: Ground truth list with probabilities

The system starts from the initial evidence, explores symptoms in the evidences list, and predicts the pathology. GraphTrace's knowledge graph contains all UMLS symptom–disease relationships; for DDXPlus evaluation, the 49 diseases and 223 evidences were mapped to UMLS CUIs.

Metrics: GTPA@k (fraction of cases with the ground truth in the top-k predictions) and average IL (mean questions per diagnosis).

### 2.7 Experimental Design

The study has three stages. Stage 1 (symptom inquiry strategy comparison) used 1,000 stratified-sampled cases (seed=42) from the validation set, reflecting the distribution of the 49 diseases, because exhaustive exploration without stopping criteria on the full validation set was computationally prohibitive. Stage 2 (final diagnostic performance) fixed three factors based on the Stage 1 analysis of variance and evaluated 200 configurations (8 denied threshold levels × 5 stopping criteria × 5 scoring functions) on all 134,529 test cases. Threshold 1–8 was evaluated on the full validation set (132,448 cases) using Top-3 Stability stopping and evidence ratio scoring. Threshold=6 achieved the highest GTPA@1 (90.60%) on the validation set and was selected as the fixed value for test-set evaluation.

Stage 3 (prior work comparison) evaluated the best Stage 2 configuration under MEDDxAgent's conditions [3]: the same 100 cases (top 100 after seed=42 shuffle) and the same question limits (IL=5, 10, 15), eliminating sample and budget differences.

---

## 3. Results

### 3.1 Symptom Inquiry Strategy Comparison

All 264 combinations of four design factors (co-occurrence 2 levels, denied threshold 11, antecedent 2, selection strategy 6) were evaluated on 1,000 validation cases (stratified, seed=42). Each ran until candidates were exhausted (maximum 223 questions). Table 1 reports ANOVA results with hit rate as the dependent variable. Hit rate served as a proxy for comparing relative effect sizes; final diagnostic performance was validated with GTPA@1 in Stage 2 (Section 3.2).

**Table 1. Exploration factor ANOVA**

| Source | df | F | p | η² |
|--------|:--:|------:|:----:|:------:|
| **Denied threshold** | **10** | **788.4** | **<.001** | **0.9467** |
| Antecedent | 1 | 111.9 | <.001 | 0.0134 |
| Selection strategy | 5 | 17.2 | <.001 | 0.0103 |
| Co-occurrence | 1 | 0.0 | .998 | 0.0000 |
| Residual | 246 | — | — | — |

> Table 1: ANOVA (analysis of variance) on hit rate across 264 combinations (2×11×2×6). 1,000 validation cases (stratified, seed=42). Hit rate = confirmed symptoms / total questions. df, degrees of freedom; F, F-statistic; p, p-value; η², eta-squared (proportion of variance explained). Bold = largest η².

Denied threshold explained 94.67% of hit rate variance (F=788.4, p<.001, η²=0.9467; large effect at η²>0.14 [24]). The remaining three factors contributed a combined η²=0.0237 (2.37%). Antecedent (η²=0.0134) and selection strategy (η²=0.0103) reached significance (p<.001) but had limited practical effect. Co-occurrence had no effect (F=0.0, p=.998).

Hit rate decreased monotonically with denied threshold (threshold=1: 29.22%, threshold=10: 5.04%). A higher hit rate does not necessarily favor diagnosis. As threshold increases, hit rate drops but confirmed symptom count rises (threshold=1: 3.66, threshold=5: 8.07), creating a trade-off. Confirmed count plateaued at 8.27–8.36 for threshold ≥ 6.

Based on these results, we fixed co-occurrence=Yes, selection=greedy, antecedent=No. Confirmatory tests at the best combination (Threshold=6, Top-3 Stability, Evidence Ratio) verified these fixations hold for GTPA@1 (Table 2).

**Table 2. Confirmatory test for fixed factors**

| Fixed factor | Baseline | Alternative | GTPA@1 (base) | GTPA@1 (alt) | Δ |
|:-------------|:---------|:------------|:-------------:|:------------:|:---:|
| Antecedent | No | Yes | 91.05% | 74.12% | -16.93%p |
| Selection | Greedy | Binary split | 91.05% | 79.36% | -11.69%p |
| Co-occurrence | Yes | No | 91.05% | 69.56% | -21.49%p |

> Table 2: Confirmatory tests at the best combination (Threshold=6, Top-3 Stability, Evidence Ratio). 134,529 test cases. Δ, difference; %p, percentage points. GTPA@1, Ground Truth Pathology Accuracy at Top-1.

All baselines outperformed their alternatives. Co-occurrence had no effect on hit rate (η²=0.00) but produced a 21.49%p GTPA@1 gap, showing that inquiry efficiency and diagnostic accuracy do not share the same factor structure. For the final experiments, denied threshold was varied across 1–8 as the sole dominant factor. Thresholds 6–8 were included to test accuracy changes in the confirmed symptom saturation zone.

### 3.2 Final Diagnosis: Stopping Criteria × Scoring Comparison

We evaluated 200 combinations (8 threshold levels × 5 stopping criteria × 5 scoring functions) on all 134,529 test cases (Table 3).

**Table 3. Final diagnostic performance**

**(A) Denied threshold** (Stopping: Top-3 Stability, Scoring: Evidence Ratio)

| Denied threshold | GTPA@1 | Avg IL |
|:----------------:|:------:|:------:|
| 1 | 57.63% | **10.0** |
| 2 | 76.99% | 17.7 |
| 3 | 86.31% | 22.1 |
| 4 | 89.80% | 23.6 |
| 5 | 90.29% | 23.2 |
| 6 | **91.05%** | 23.1 |
| 7 | 89.90% | 23.0 |
| 8 | 89.82% | 23.3 |

**(B) Stopping criteria** (Threshold=6, Scoring: Evidence Ratio)

| Stopping criteria | GTPA@1 | Avg IL |
|----------|:------:|:------:|
| Confidence Gap ≥ 0.05 | **98.20%** | 48.3 |
| Top-3 Stability (n=5) | 91.05% | 23.1 |
| Cumulative Confirmed ≥ 5 | 79.71% | 18.5 |
| Top-1 Stability (n=5) | 75.89% | **10.8** |
| Hit Rate Plateau | 75.39% | 16.8 |

**(C) Scoring functions** (Threshold=6, Stopping: Top-3 Stability)

| Scoring function | GTPA@1 | Avg IL |
|---------|:------:|:------:|
| Evidence Ratio | **91.05%** | 23.1 |
| Cosine | 86.65% | 23.0 |
| Coverage | 82.95% | 23.1 |
| Jaccard | 81.46% | **22.9** |
| TF-IDF | 78.26% | 23.3 |

> Table 3: 200 combinations (8×5×5) on 134,529 test cases. Three factors fixed from Table 1 ANOVA; denied threshold (A), stopping criteria (B), and scoring function (C) varied. Each sub-table fixes the other two at optimal values. Bold = best value per column. GTPA@1, Ground Truth Pathology Accuracy at Top-1; Avg IL, average Inquiry Length; TF-IDF, Term Frequency–Inverse Document Frequency.

Ground truth pathology accuracy at Top-1 reached 91.05% at denied threshold=6, with diminishing returns from threshold=5 (90.29%) to threshold=6 (91.05%), indicating saturation (Table 3(A)). Confidence Gap stopping reached 98.20% GTPA@1 but required an average of 48.3 questions; Top-3 Stability reached 91.05% in 23.1 questions (Table 3(B)). Evidence Ratio was the highest-performing scoring function at 91.05%, followed by Cosine at 86.65% (Table 3(C)). The majority of the 200 configurations exceeded the AARLC reinforcement learning baseline of 75.39%. The primary contribution of this experiment is not the identification of a single optimal configuration but the systematic mapping of how each design factor contributes to diagnostic performance.

### 3.3 Comparison with Prior Work

Table 4 compares GraphTrace with prior work on DDXPlus.

**Table 4. Comparison with prior work on DDXPlus**

**(A) Interactive Symptom Inquiry**

| Method | Training | GTPA@1 | GTPA | Avg IL |
|------|:--------:|:------:|:----:|:------:|
| Most frequent | No | 6.50% | — | 0 |
| Random inquiry | No | 62.26% | — | 15.0 |
| BASD [1] | Yes | 67.71% | 99.30% | **17.86** |
| AARLC [1] | Yes | 75.39% | 99.92% | 25.75 |
| GraphTrace | No | **91.05%** | **99.93%**a) | 23.1 |

**(B) Controlled comparison with MEDDxAgent [3]**

| IL | MEDDxAgent (GPT-4o) | MEDDxAgent (Llama-70B) | MEDDxAgent (Llama-8B) | GraphTrace |
|:--:|:-------------------:|:---------------------:|:--------------------:|:-----------------:|
| =5 | **74%** | 61% | 34% | 52% |
| =10 | **78%** | 71% | 56% | 74% |
| =15 | **86%** | 68% | 58% | 81% |

**(C) Complete profile setting**

| Method | Training | Accuracy |
|------|:--------:|:--------:|
| DDxT [25] | Yes | **99.98%** |
| LoRA-LLaMA [26] | Yes | 99.81% |
| BERT multi-label [27] | Yes | 97.44%b) |
| StreamBench [28] | Noc) | 92.01% |
| GraphTrace | No | 97.21% |

> Table 4: (A) Interactive: full test set (134,529 cases). AARLC and BASD are 3-run averages [1]. (B) Reproduction of MEDDxAgent conditions: same 100 cases (HuggingFace StreamBench DDXPlus test, seed=42), same IL limits. MEDDxAgent GTPA@1 uses weak matching [3]. (C) Complete profile: classification with all positive symptoms as input. a) Reported as GTPA@10. b) F1 score. c) LLM pre-training required but no DDXPlus-specific training. GTPA@1, Ground Truth Pathology Accuracy at Top-1; GTPA, Ground Truth Pathology Accuracy (any rank); Avg IL, average Inquiry Length; IL, Inquiry Length.

In the interactive symptom inquiry setting (Table 4(A)), GraphTrace achieved a GTPA@1 that was 15.66 percentage points above the AARLC reinforcement learning baseline (91.05% versus 75.39%), with a lower average inquiry length (23.1 versus 25.75 questions). AARLC was introduced as the accompanying baseline of the DDXPlus benchmark in 2022 and does not represent the current state of the art on this benchmark; more recent learning-based methods, including LLM-based approaches, report higher accuracy (Table 4).

Table 4(B) reports a controlled comparison with MEDDxAgent [3], reproducing the original evaluation protocol. We used the identical 100 test cases (top 100 after seed=42 shuffle of the 1,764 cases in the HuggingFace StreamBench DDXPlus test split) and matched question budgets (IL=5, 10, 15). Under this protocol, MEDDxAgent (GPT-4o) outperformed GraphTrace at IL=15 (86% vs 81% GTPA@1; MEDDxAgent accuracy uses weak matching as reported in [3]). MEDDxAgent's advantage is consistent with its use of LLM pre-trained medical knowledge, which is absent from the parameter-free knowledge graph approach. GraphTrace is therefore positioned as a complementary approach: it trades a small accuracy margin for full diagnostic path traceability and avoids both model training and inference-time API costs.

The complete profile setting (c) is a classification task without symptom collection and not directly comparable to interactive settings [1]. GraphTrace's KG scoring reached 97.21% without training, representing an upper bound for KG-based reasoning. The 6.16%p gap from the interactive result (91.05%) reflects information loss during exploration.

### 3.4 Traceability

Every diagnostic decision is traceable as a knowledge graph path [29]. We demonstrate with a test case (Table 5).

**Table 5. Diagnostic trace example**

| IL | Question (2nd hop) | Response | Top-3 diagnoses |
|:--:|---------------|:----:|----------|
| 0 | — (Initial: Angina) | + | — |
| 1 | **Exertion** | **+** | **Pneumothorax**, Unstable angina, Stable angina |
| 2 | **Dyspnea** | **+** | Unstable angina, **Pneumothorax**, Stable angina |
| 3 | **Travel History** | **+** | Unstable angina, **Pneumothorax**, Stable angina |
| 4 | **Pain** | **+** | Unstable angina, **Pneumothorax**, Stable angina |
| 5 | Tobacco Use Disorder | - | Unstable angina, **Pneumothorax**, Myocarditis |
| 6 | Cough | - | Unstable angina, **Pneumothorax**, Myocarditis |
| 7 | Nausea | - | **Pneumothorax**, Myocarditis, Pulmonary edema |
| 8 | Fatigue | - | **Pneumothorax**, Myocarditis, Pulmonary edema |
| 9 | Alcoholism | - | **Pneumothorax**, Myocarditis, Pulmonary edema |
| 10 | Fever | - | **Pneumothorax**, Myocarditis, Pulmonary edema |
| 11 | Obesity | - | **Pneumothorax**, Myocarditis, Pulmonary edema |
| | **Top-3 Stable → Stop** | | **Pneumothorax (correct)** |

> Table 5: Full trace for one test case. Configuration: Denied Threshold=6, Antecedent=No, Greedy, Top-3 Stability (n=5), Evidence Ratio. + = confirmed; - = denied. Bold = ground truth (Pneumothorax). IL, Inquiry Length.

The system diagnosed spontaneous pneumothorax in 11 questions. Pneumothorax initially ranked first from the chief complaint (Angina). At IL=4, after Pain was confirmed, Unstable angina took the top position (IL=4–6), as chest pain with dyspnea and exertional worsening appears in both cardiac and pulmonary conditions. Denied symptoms (Tobacco Use Disorder, Cough, Nausea) then eliminated competing hypotheses, and Pneumothorax recovered Top-1 at IL=7. Top-3 remained stable from IL=7 to 11, triggering the stopping criterion. At each step:

- Why this symptom was asked: Most connected to current candidate diseases in the 2-hop exploration (Greedy top-1)
- Why this disease ranks first: Confirmed symptoms (Angina, Exertion, Dyspnea, Pain) match Pneumothorax; denied symptoms eliminate competitors (COPD, Bronchitis, etc.)
- Why the system stopped: Top-3 unchanged for 5 consecutive turns (IL=7–11)

What GraphTrace provides is traceability, not interpretability [30]. All diagnostic paths can be followed as graph trajectories, but whether a path is clinically reasonable requires separate clinical evaluation. Top-1 diagnoses fluctuate in early stages (IL=1–3) when evidence is sparse; this is a structural property, and clinical interpretation of early-stage rankings is not appropriate. For traceability to become clinical interpretability, domain experts must evaluate path validity at later stages where sufficient evidence has accumulated.

---

## 4. Discussion

The principal finding of this study is that, across a systematic comparison of 200 design configurations, denied symptom-based hypothesis elimination emerged as the dominant performance factor in 2-hop knowledge graph differential diagnosis, explaining 94.67% of the variance in inquiry efficiency. This result reframes the design problem for knowledge graph-based symptom checkers: rather than optimizing which symptom to ask next, the larger performance gains lie in how aggressively candidate diseases are eliminated when symptoms are denied. Without any model parameter learning, the best configuration achieved GTPA@1 of 91.05% on 134,529 DDXPlus test cases with an average of 23.1 questions, placing parameter-free knowledge graph exploration in the same performance range as learning-based baselines for this benchmark.

### 4.1 Mechanism of Symptom Inquiry Strategies

Denied threshold was the sole dominant factor, explaining 94.67% of hit rate variance (Table 1). The other three factors contributed a combined η²=0.024. In KG-based diagnostic reasoning, the elimination process (narrowing hypotheses through denied symptoms) matters far more than which specific symptom to ask next.

Hit rate and final accuracy did not scale proportionally. Lower thresholds raised hit rate, but final diagnosis favored higher thresholds (Threshold=6: 91.05% vs Threshold=3: 86.31%, Table 3(A), where more confirmed symptoms accumulated.

### 4.2 Stopping Criteria and Scoring

Top-3 Stability balanced diagnostic accuracy (91.05%) and inquiry length (23.1 questions on average). Requiring multiple competing hypotheses to stabilize, rather than a single top candidate, may structurally reduce the risk of premature closure, a recognized source of diagnostic error [19]. Confidence Gap stopping reached higher accuracy (98.20%) but required 48.3 questions on average. As a point of reference only, studies using the Roter Interaction Analysis System have reported 20–33 history-taking questions in physician–patient encounters [31], and consultation time is typically constrained (Korean outpatient visits: 4.1–6.8 minutes [32]; United States primary care visits: 15–20 minutes [33]). Automated symptom inquiry is not equivalent to clinician-led history taking, and these numbers are cited only to contextualize the question budget range. Any clinical application would require prospective evaluation against clinician baselines.

Evidence ratio achieved the highest GTPA@1 among scoring functions. TF-IDF and other information retrieval methods performed poorly. Confirmed symptoms are sparse while denied symptoms are numerous in differential diagnosis, a pattern unlike standard text retrieval settings.

### 4.3 Disease-Level Performance

Three of 49 diseases (Acute dystonic reactions, Localized edema, Spontaneous rib fracture) reached 100%, while others showed lower performance (Table 6).

**Table 6. Disease-level GTPA@1**

| Disease | Cases | GTPA@1 | Avg IL |
|------|:--------:|:------:|:------:|
| Acute rhinosinusitis | 1,829 | 60.74% | 19.9 |
| Pericarditis | 3,095 | 60.94% | 24.5 |
| Acute pulmonary edema | 2,598 | 71.40% | 20.4 |
| Unstable angina | 2,880 | 71.70% | 21.1 |
| Bronchitis | 3,594 | 77.05% | 24.8 |
| … | | | |
| Allergic sinusitis | 2,411 | 99.54% | 22.8 |
| Acute COPD exacerbation | 2,153 | 99.63% | 18.3 |
| Acute dystonic reactions | 3,302 | 100.00% | 24.8 |
| Localized edema | 3,734 | 100.00% | 13.8 |
| Spontaneous rib fracture | 778 | 100.00% | 22.3 |

> Table 6: GraphTrace (Threshold=6, Antecedent=No, Greedy, Top-3 Stability, Evidence Ratio) on 134,529 test cases. Bottom 5 and top 5 of 49 diseases. GTPA@1, Ground Truth Pathology Accuracy at Top-1; Avg IL, Inquiry Length.

Low-performing diseases share a common pattern: confusion within symptomatically similar disease groups. Acute rhinosinusitis (60.74%) overlaps with Chronic rhinosinusitis (99.34%) and Allergic sinusitis (99.54%); Pericarditis (60.94%) is hard to separate from Unstable angina (71.70%) and Possible NSTEMI/STEMI (87.50%) in the cardiac cluster. The KG elimination strategy differentiates well between disease groups but struggles with fine distinctions within groups. Diseases with unique symptom signatures, such as Localized edema (100%, avg IL=13.8), are diagnosed accurately with few questions.

### 4.4 Potential Applications

The parameter-free KG method can be extended in two directions.

First, as a verification module in neuro-symbolic AI. LLM hallucination is a recognized concern in medicine [3], and GraphTrace's 2-hop paths could verify LLM outputs or constrain their search space in a hybrid architecture, serving as a symbolic guardrail checking LLM–KG consistency. This has not been validated here and requires future work.

Second, since GraphTrace needs no GPU or training data and all paths are explicit [29], it could serve as a pre-specialist triage aid in resource-limited settings.

### 4.5 Limitations

First, data and external validity. DDXPlus is a synthetic benchmark generated from an underlying symptom–disease model [1]; reported diagnostic accuracies on DDXPlus therefore reflect performance on a dataset whose generative structure is more consistent than real-world clinical presentations. A particular concern for knowledge graph-based methods is that the UMLS graph and the DDXPlus generative schema may share substantial structural overlap: mapping 49 diseases and 223 evidences into UMLS places the knowledge graph in close alignment with the same concepts that DDXPlus was built to cover. The complete profile accuracy of 97.21% achieved by the scoring function alone (Table 4(C)) is consistent with this interpretation and should be viewed as an upper bound on what knowledge graph alignment with the benchmark can achieve, rather than as a transferable ceiling. External validation on non-synthetic clinical data, including encounters with ambiguous presentations, comorbidities, and rare diseases, is required before any claim about real-world diagnostic utility can be made.

Second, methodological constraints of the experimental design. The denied threshold was independently validated on the full validation set (132,448 cases): threshold=6 achieved the highest GTPA@1 (90.60%) and was selected as the fixed value for test-set evaluation. Hit rate also does not capture the diagnostic value of denied symptoms. Higher denied thresholds reduced hit rate but raised final accuracy through denied symptom accumulation (Table 3(A)), so we treated hit rate as a proxy for relative factor comparison in Stage 1 and validated final performance with GTPA@1 in Stage 2. The limited effects of co-occurrence and selection strategy (combined η²=0.024) suggest that the 2-hop framework is approaching its structural ceiling; 3-hop exploration, spreading activation [34], and graph embedding methods warrant future investigation. The system also cannot actively propose history-taking questions beyond what the knowledge graph encodes. The mapping of human working memory capacity [18] to the Top-3 stopping criterion was a heuristic design choice rather than a cognitive-science-validated model and should be regarded as such.

Third, incomplete UMLS mapping. Of 223 evidences, 19 (8.5%) were unmapped. These include 14 multi-value attributes (pain location, characteristics, skin lesion color) that exist as qualifiers of parent symptoms without independent UMLS concepts, and 5 comorbidities (diabetes, COPD, metastatic cancer, pneumonia history, asthma family history) that have CUIs but are difficult to represent in a bipartite symptom–disease KG. The system cannot distinguish "chest pain" from "abdominal pain" and cannot use comorbidity information. Integrating multi-value attributes and adding comorbidity nodes are future extensions. Node degree imbalance in the UMLS KG may also bias some scoring functions.

In conclusion, a systematic factor decomposition of 2-hop knowledge graph exploration on the DDXPlus benchmark identified denied symptom-based hypothesis elimination as the dominant driver of diagnostic performance (η²=0.947). Under the best configuration, parameter-free exploration achieved GTPA@1 of 91.05% with an average of 23.1 questions, placing it in the performance range of learning-based baselines on this benchmark. The method provides full trajectory traceability through knowledge graph paths and requires no model training, benchmark-specific data, or API access. These findings should be interpreted within the limitations of synthetic benchmark evaluation; external validation on real clinical data is an essential next step before any application beyond algorithmic research is considered.

---

## 5. Acknowledgments

This work was supported by the Starting growth Technological R&D Program (TIPS Program, RS-2024-00511590) funded by the Ministry of SMEs and Startups (Mss, Korea) in 2024.

## Conflict of Interest

Juyoung Kim and Jonghyung Park are employed by Meninblox, Inc., which is developing knowledge graph-based diagnostic technologies. This study was supported by a Tech Incubator Program for Startup (TIPS) grant awarded to Meninblox, Inc. (RS-2024-00511590). Chang Seok Bang declares no competing financial interests related to this work. The authors declare that the funding body had no role in study design, data analysis, interpretation of results, or the decision to publish.

---

## References

1. Fansi Tchango A, Goel R, Wen Z, Martel J, Ghosn J. DDXPlus: a new dataset for automatic medical diagnosis. In: Proceedings of the 36th Conference on Neural Information Processing Systems (NeurIPS 2022); 2022. p. 31306-31318. doi:10.5555/3600270.3602540.

2. Elstein AS, Schwartz A. Clinical problem solving and diagnostic decision making: selective review of the cognitive literature. BMJ. 2002 Mar 23;324(7339):729-32. doi: 10.1136/bmj.324.7339.729. Erratum in: BMJ. 2006 Nov 4;333(7575):944. Schwarz, Alan [corrected to Schwartz, Alan]. PMID: 11909793; PMCID: PMC1122649.

3. Rose D, Hung CC, Lepri M, Alqassem I, Gashteovski K, Lawrence C. MEDDxAgent: A unified modular agent framework for explainable automatic differential diagnosis. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL); 2025. doi:10.18653/v1/2025.acl-long.677.

4. European Union. Regulation (EU) 2024/1689 of the European Parliament and of the Council (Artificial Intelligence Act) [Internet]. 2024 Jul [cited 2026 Mar 27]. Available from: https://eur-lex.europa.eu/eli/reg/2024/1689/oj

5. Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. Nucleic Acids Res. 2004 Jan 1;32(Database issue):D267-70. doi: 10.1093/nar/gkh061. PMID: 14681409; PMCID: PMC308795.

6. Guan H, Baral C. A Bayesian approach for medical inquiry and disease inference in automated differential diagnosis [Internet]. 2021 [cited 2026 Mar 27]. Available from: https://arxiv.org/abs/2110.08393

7. Wald A. Sequential tests of statistical hypotheses. Ann Math Stat. 1945;16(2):117-186. doi:10.1214/aoms/1177731118.

8. Jo SH, Lee KS. Disease Prediction By Learning Clinical Concept Relations. Journal of the Korea Information Processing Society. 2022;11(1):35-40.

9. Kim SK, Lee D, Kim A, Nam S. Construction of Korean Medicine Knowledge Graph and Development of Prescription Recommendation System based on RippleNet. J Korean Med. 2025;46(2):51-62. doi:10.13048/jkm.25017.

10. Tjoa E, Guan C. A Survey on Explainable Artificial Intelligence (XAI): Toward Medical XAI. IEEE Trans Neural Netw Learn Syst. 2021 Nov;32(11):4793-4813. doi: 10.1109/TNNLS.2020.3027314. Epub 2021 Oct 27. PMID: 33079674.

11. Polotskaya K, Muñoz-Valencia CS, Rabasa A, Quesada-Rico JA, Orozco-Beltrán D, Barber X. Bayesian Networks for the Diagnosis and Prognosis of Diseases: A Scoping Review. Machine Learning and Knowledge Extraction. 2024;6(2):1243-1262. doi:10.3390/make6020058.

12. Asgari E, Montaña-Brown N, Dubois M, Khalil S, Balloch J, Yeung JA, Pimenta D. A framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation. NPJ Digit Med. 2025 May 13;8(1):274. doi: 10.1038/s41746-025-01670-7. PMID: 40360677; PMCID: PMC12075489.

13. Bloodgood M, Vijay-Shanker K. A method for stopping active learning based on stabilizing predictions and the need for user-adjustable stopping. In: Proc 13th Conference on Computational Natural Language Learning (CoNLL-2009). Boulder, Colorado: Association for Computational Linguistics; 2009. p. 39-47.

14. Zhou X, Menche J, Barabási AL, Sharma A. Human symptoms-disease network. Nat Commun. 2014 Jun 26;5:4212. doi: 10.1038/ncomms5212. PMID: 24967666.

15. Miaskowski C, Barsevick A, Berger A, Casagrande R, Grady PA, Jacobsen P, Kutner J, Patrick D, Zimmerman L, Xiao C, Matocha M, Marden S. Advancing Symptom Science Through Symptom Cluster Research: Expert Panel Proceedings and Recommendations. J Natl Cancer Inst. 2017 Jan 24;109(4):djw253. doi: 10.1093/jnci/djw253. PMID: 28119347; PMCID: PMC5939621.

16. Kao H-C, Tang K-F, Chang E. Context-aware symptom checking for disease diagnosis using hierarchical reinforcement learning. In: Proceedings of the 32nd AAAI Conference on Artificial Intelligence (AAAI-18); 2018. doi:10.1609/aaai.v32i1.11902.

17. Ishibashi H, Hino H. Stopping criterion for active learning based on deterministic generalization bounds. In: Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR 108; 2020. p. 386-397.

18. Cowan N. The magical number 4 in short-term memory: a reconsideration of mental storage capacity. Behav Brain Sci. 2001 Feb;24(1):87-114; discussion 114-85. doi: 10.1017/s0140525x01003922. PMID: 11515286.

19. Croskerry P. Clinical cognition and diagnostic error: applications of a dual process model of reasoning. Adv Health Sci Educ Theory Pract. 2009 Sep;14 Suppl 1:27-35. doi: 10.1007/s10459-009-9182-2. Epub 2009 Aug 11. PMID: 19669918.

20. Settles B. Active Learning Literature Survey [Internet]. CS Tech Report 1648, University of Wisconsin-Madison. 2009 [cited 2026 Mar 27]. Available from: http://digital.library.wisc.edu/1793/60660

21. Jaccard P. Etude comparative de la distribution florale dans une portion des Alpes et du Jura. Lausanne: Impr. Corbaz; 1901.

22. Salton G, Buckley C. Term-weighting approaches in automatic text retrieval. Inf Process Manag. 1988;24(5):513-523. doi:10.1016/0306-4573(88)90021-0.

23. Salton G, Wong A, Yang CS. A vector space model for automatic indexing. Commun ACM. 1975;18(11):613-620. doi:10.1145/361219.361220.

24. Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Hillsdale (NJ): Lawrence Erlbaum Associates; 1988.

25. Alam MM, Raff E, Oates T, Matuszek C. DDxT: Deep generative transformer models for differential diagnosis. In: 1st Workshop on Deep Generative Models for Health (DGM4H) at NeurIPS; 2023. Available from: https://arxiv.org/abs/2312.01242

26. Kang L, Fu X, Ramos Terrades O, Vazquez-Corral J, Valveny E, Karatzas D. LLM-driven medical document analysis: Enhancing trustworthy pathology and differential diagnosis. In: Document Analysis and Recognition – ICDAR 2025. Lecture Notes in Computer Science, vol. 16025. Springer; 2026. p. 613-628. doi:10.1007/978-3-032-04624-6_36.

27. Sadi AA, Khan MA, Saber LB. Automatic differential diagnosis using transformer-based multi-label sequence classification [Internet]. 2024 [cited 2026 Mar 27]. Available from: https://arxiv.org/abs/2408.15827

28. Wu C-K, Tarn ZR, Lin C-Y, Chen Y-N, Lee H-Y. StreamBench: Towards benchmarking continuous improvement of language agents. In: Proceedings of the 38th International Conference on Neural Information Processing Systems (NeurIPS); 2024. p. 3398.

29. Benish WA. A Review of the Application of Information Theory to Clinical Diagnostic Testing. Entropy (Basel). 2020 Jan 14;22(1):97. doi: 10.3390/e22010097. PMID: 33285872; PMCID: PMC7516534.

30. Rudin C. Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead. Nat Mach Intell. 2019 May;1(5):206-215. doi: 10.1038/s42256-019-0048-x. Epub 2019 May 13. PMID: 35603010; PMCID: PMC9122117.

31. Roter DL, Hall JA. Doctors Talking with Patients/Patients Talking with Doctors: Improving Communication in Medical Visits. 2nd ed. Westport, CT: Praeger; 2006.

32. Lee M, Oh B, You M. Consultation time and communication patterns in outpatient care: an observational study in South Korea. BMC Health Serv Res. 2025 Aug 30;25(1):1159. doi: 10.1186/s12913-025-13431-z. PMID: 40886011; PMCID: PMC12398989.

33. Irving G, Neves AL, Dambha-Miller H, Oishi A, Tagashira H, Verho A, Holden J. International variations in primary care physician consultation time: a systematic review of 67 countries. BMJ Open. 2017 Nov 8;7(10):e017902. doi: 10.1136/bmjopen-2017-017902. PMID: 29118053; PMCID: PMC5695512.

34. Collins AM, Loftus EF. A spreading-activation theory of semantic processing. Psychol Rev. 1975;82(6):407-428. doi:10.1037/0033-295X.82.6.407.
