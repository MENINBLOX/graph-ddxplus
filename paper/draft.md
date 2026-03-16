# Gr-CoT: Graph-Refined Chain-of-Thought for Medical Diagnosis with Small Language Models

## Abstract

Large language models (LLMs) have shown promising results in medical diagnosis tasks, but they often suffer from hallucinations and lack of factual grounding. We propose **Gr-CoT (Graph-Refined Chain-of-Thought)**, a framework that combines small LLMs (4B parameters) with UMLS-based knowledge graphs for interactive medical diagnosis. Our approach divides responsibilities between the knowledge graph (candidate generation, diagnosis scoring, stopping criteria) and the LLM (symptom selection, final diagnosis refinement). On the DDXPlus benchmark, Gr-CoT achieves **86.0% GTPA@1 accuracy**, outperforming the AARLC baseline (75.39%) by **10.6 percentage points** while requiring **fewer interactions** (22.1 vs 25.75). Our results demonstrate that small LLMs augmented with structured medical knowledge can achieve superior diagnostic performance compared to reinforcement learning-based approaches.

---

## 1. Introduction

### 1.1 Motivation

Automated medical diagnosis systems have the potential to improve healthcare accessibility and reduce diagnostic errors. Recent advances in large language models (LLMs) have enabled natural language-based diagnostic reasoning. However, LLM-only approaches face critical challenges:

1. **Hallucination**: LLMs may generate plausible but incorrect medical information
2. **Inconsistency**: Same symptoms may lead to different diagnoses across sessions
3. **Lack of transparency**: Reasoning process is opaque to clinicians

Knowledge graphs (KGs) offer structured, verified medical relationships but lack flexible reasoning capabilities. We propose combining the strengths of both approaches.

### 1.2 Research Question

> Can small LLMs (4B parameters) combined with medical knowledge graphs outperform state-of-the-art reinforcement learning methods in interactive medical diagnosis?

### 1.3 Contributions

1. **Gr-CoT Framework**: A novel architecture that divides diagnostic tasks between KG and LLM components based on their strengths
2. **Improved Diagnosis Scoring**: A new formula that considers disease coverage, symptom coverage, and denied symptom penalties
3. **UMLS Integration**: Complete mapping between DDXPlus and UMLS without requiring embedding models
4. **Empirical Validation**: Demonstration that small LLMs with KG augmentation outperform AARLC on DDXPlus

---

## 2. Related Work

### 2.1 Automatic Medical Diagnosis

Traditional approaches to automated diagnosis include rule-based expert systems, Bayesian networks, and machine learning classifiers. Recent work has focused on deep learning and reinforcement learning methods.

**AARLC (Adaptive Automatic Diagnosis with Reinforcement Learning and Constraints)** [Fansi Tchango et al., 2022] achieved state-of-the-art results on DDXPlus with 75.39% GTPA@1 accuracy using reinforcement learning with action constraints.

### 2.2 LLMs in Healthcare

Large language models have been applied to various medical tasks including question answering, clinical note generation, and diagnosis prediction. However, concerns about factual accuracy have limited their deployment in clinical settings.

### 2.3 Knowledge Graph-Augmented LLMs

Recent work has explored combining LLMs with knowledge graphs for improved factuality. In the medical domain, knowledge graphs like UMLS, SNOMED-CT, and disease-specific ontologies provide structured medical knowledge.

---

## 3. Method

### 3.1 Problem Formulation

Given an initial symptom (chief complaint), the system must:
1. Ask clarifying questions about additional symptoms
2. Determine when sufficient information has been gathered
3. Provide a final diagnosis

This is formulated as an interactive diagnosis task where the system interacts with a patient simulator.

### 3.2 Gr-CoT Architecture

Our framework divides responsibilities between two components:

```
┌─────────────────────────────────────────────────────┐
│                  Gr-CoT Framework                    │
├─────────────────────────────────────────────────────┤
│  Knowledge Graph (Neo4j + UMLS)                      │
│  ├─ Candidate symptom generation (2-hop traversal)  │
│  ├─ Diagnosis scoring (coverage + penalty)          │
│  └─ Stopping criteria (confidence thresholds)       │
├─────────────────────────────────────────────────────┤
│  Small LLM (4B parameters)                          │
│  ├─ Symptom selection from KG candidates            │
│  └─ Final diagnosis selection from KG top-k         │
└─────────────────────────────────────────────────────┘
```

### 3.3 Knowledge Graph Construction

We construct a symptom-disease knowledge graph from patient data:

1. **Node Types**: Symptoms (209) and Diseases (49)
2. **Edge Type**: INDICATES (symptom indicates disease)
3. **Mapping**: All nodes mapped to UMLS Concept Unique Identifiers (CUIs)

The mapping is performed using UMLS Metathesaurus string matching without requiring embedding models, achieving 100% coverage.

### 3.4 Diagnosis Scoring Formula

We propose an improved scoring formula for ranking disease candidates:

```
score = dcov × scov × (1 - den_ratio)²
```

Where:
- **dcov** (disease coverage) = confirmed_count / total_symptoms
- **scov** (symptom coverage) = confirmed_count / (total_confirmed + ε)
- **den_ratio** (denied ratio) = denied_count / (total_symptoms + 1)

Key design choices:
- **dcov**: Measures how many of the disease's symptoms are confirmed
- **scov**: Measures how well the confirmed symptoms match this disease
- **Squared penalty**: Strongly penalizes diseases with denied symptoms

### 3.5 Stopping Criteria

The KG determines when to stop questioning based on:
1. Top-1 score ≥ 0.8 (high confidence)
2. Score gap between top-1 and top-2 ≥ 0.3 (clear differentiation)
3. Only one candidate disease remains
4. Maximum interaction limit reached (25 questions)

### 3.6 LLM Role

The small LLM (4B parameters) performs two tasks:

**Symptom Selection**: Given KG-generated candidate symptoms (top-10), the LLM selects the most clinically relevant symptom to ask about next.

**Final Diagnosis**: Given KG-generated diagnosis candidates (top-5), the LLM makes the final diagnosis selection, contributing approximately +7% accuracy improvement over using KG top-1 directly.

---

## 4. Experimental Setup

### 4.1 Dataset

**DDXPlus** [Fansi Tchango et al., 2022]: A large-scale medical diagnosis dataset containing:
- 1.3 million synthetic patient cases
- 49 pathologies
- 223 symptoms/evidences
- Ground truth differential diagnoses with probability distributions

### 4.2 Evaluation Metrics

- **GTPA@1** (Ground Truth Pathology Accuracy): Top-1 diagnosis accuracy
- **IL** (Interaction Length): Average number of questions asked
- **DDR** (Differential Diagnosis Recall): Recall of ground truth differential diagnoses
- **DDF1** (Differential Diagnosis F1): F1 score for differential diagnosis

### 4.3 Baselines

- **AARLC**: State-of-the-art reinforcement learning approach (75.39% GTPA@1)
- **KG-only**: Knowledge graph without LLM selection
- **All-Symptoms**: Upper bound using all patient symptoms

### 4.4 Implementation Details

- **LLM**: qwen3:4b-instruct-2507-fp16 (4B parameters)
- **Knowledge Graph**: Neo4j with patient-derived symptom-disease relations
- **Temperature**: 0.1 for consistent outputs
- **Maximum IL**: 25 (matching AARLC evaluation)

---

## 5. Results

### 5.1 Main Results

| Method | GTPA@1 | IL |
|--------|--------|-----|
| AARLC (baseline) | 75.39% | 25.75 |
| KG-only | 75.0% | 22.1 |
| **Gr-CoT (Ours)** | **86.0%** | **22.1** |

Our method achieves:
- **+10.6%** improvement in GTPA@1 over AARLC
- **-3.65** reduction in interaction length

### 5.2 Ablation Study: Scoring Formula

| Scoring Formula | GTPA@1 | IL |
|-----------------|--------|-----|
| Baseline (linear penalty) | 67.0% | 17.3 |
| Improved (squared penalty) | **86.0%** | 22.1 |

The improved scoring formula provides **+19%** absolute improvement.

### 5.3 Ablation Study: Symptom Selection

| Selection Strategy | GTPA@1 | IL |
|-------------------|--------|-----|
| Coverage-based | **86.0%** | 22.1 |
| Information Gain (Gini) | 82.0% | 22.0 |

Information Gain-based selection underperforms because it tends to select symptoms that elicit "NO" responses, which triggers the squared denied penalty.

### 5.4 Ablation Study: LLM Contribution

| Final Diagnosis | GTPA@1 |
|-----------------|--------|
| KG Top-1 only | 75.0% |
| LLM selection from Top-5 | **82.0%** |

The LLM contributes **+7%** by selecting from KG candidates, demonstrating the value of combining KG structure with LLM reasoning.

---

## 6. Analysis

### 6.1 Why Does Gr-CoT Work?

**Division of Labor**: The KG provides structured, reliable candidate generation and scoring, while the LLM provides flexible selection based on clinical reasoning. Neither component alone achieves optimal performance.

**Squared Penalty Effect**: The squared penalty for denied symptoms is crucial. It strongly down-weights diseases that don't match the patient's symptom profile, improving discrimination.

**Coverage-Based Selection**: Prioritizing symptoms connected to many diseases ensures broad exploration before narrowing down, which aligns well with the scoring formula.

### 6.2 Comparison with AARLC

AARLC uses reinforcement learning to learn optimal questioning policies. Our approach achieves better results through:
1. Explicit medical knowledge encoding (UMLS-based KG)
2. Interpretable scoring formula
3. Small LLM for clinical reasoning

### 6.3 Limitations

- **DDR/DDF1**: Our method optimizes for top-1 accuracy; differential diagnosis metrics require additional work
- **Dataset Scope**: Evaluated only on DDXPlus; generalization to other datasets needs verification
- **Real-world Gap**: Synthetic patient data may not fully capture clinical complexity

---

## 7. Discussion

### 7.1 Clinical Applicability

The Gr-CoT framework offers several advantages for clinical deployment:

1. **Interpretability**: The KG provides explicit reasoning traces
2. **Efficiency**: 4B parameter LLM runs on consumer hardware
3. **Reliability**: KG grounding reduces hallucination risk

### 7.2 UMLS Integration

Our complete UMLS mapping demonstrates that:
1. Standard medical terminologies can bridge research benchmarks and clinical systems
2. Embedding models are not always necessary for medical concept mapping
3. The same approach can generalize to other UMLS-coded datasets

### 7.3 Future Directions

1. **Multi-turn Reasoning**: Incorporate conversation history into LLM prompts
2. **Uncertainty Quantification**: Provide confidence intervals for diagnoses
3. **Real-world Validation**: Test on actual clinical data with physician oversight

---

## 8. Conclusion

We presented Gr-CoT, a framework combining small LLMs with UMLS-based knowledge graphs for interactive medical diagnosis. By dividing responsibilities between KG (candidate generation, scoring) and LLM (selection, refinement), we achieve **86.0% GTPA@1 accuracy** on DDXPlus, outperforming the AARLC baseline by **10.6 percentage points**. Our results demonstrate that small, efficient LLMs augmented with structured medical knowledge can achieve superior diagnostic performance while maintaining interpretability and deployability.

---

## References

[To be added based on final venue requirements]

1. Fansi Tchango, A., et al. (2022). DDXPlus: A New Dataset For Automatic Medical Diagnosis. NeurIPS Datasets and Benchmarks Track.

2. [UMLS Reference]

3. [Additional references to be added]

---

## Appendix

### A. UMLS Mapping Details

[Details on symptom and disease mapping methodology]

### B. Prompt Templates

[LLM prompt templates for symptom selection and diagnosis]

### C. Hyperparameter Sensitivity

[Additional ablation studies]
