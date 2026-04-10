# Introduction Section - Draft

## 1. Introduction

Differential diagnosis—the process of distinguishing between diseases with similar presentations—is a fundamental skill in clinical medicine. A physician must systematically gather symptoms through patient questioning, integrate this information with medical knowledge, and reason about the most likely diagnoses. Automating this process has been a long-standing goal in medical AI, with applications ranging from clinical decision support to telemedicine triage.

### 1.1 Challenges in Automatic Differential Diagnosis

Automatic differential diagnosis systems face several key challenges:

1. **Knowledge Integration**: Systems must encode comprehensive medical knowledge about disease-symptom relationships, disease prevalence, and clinical reasoning patterns.

2. **Efficient Information Gathering**: Unlike static classification tasks, diagnosis requires sequential decision-making—determining which symptoms to ask about to maximize diagnostic information while minimizing patient burden.

3. **Uncertainty Handling**: Medical diagnosis inherently involves uncertainty. Systems must reason about probabilities and provide calibrated confidence estimates.

4. **Computational Efficiency**: Clinical deployment requires real-time responses, ruling out approaches that require extensive computation per patient.

### 1.2 Existing Approaches and Their Limitations

**Reinforcement Learning Approaches**: Recent work has applied deep reinforcement learning to automatic diagnosis. AARLC (Fansi Tchango et al., 2022) achieves strong performance on the DDXPlus benchmark by learning a policy that maximizes diagnostic accuracy while minimizing interaction length. However, RL approaches require extensive environment interaction, reward engineering, and task-specific training.

**Large Language Model Approaches**: Large language models (LLMs) like GPT-4 and Claude have demonstrated impressive medical reasoning capabilities. The H-DDx benchmark (ACL 2025) showed that frontier LLMs can perform differential diagnosis directly from patient descriptions. However, these models are computationally expensive, have high API costs, and raise privacy concerns when patient data must be sent to external services.

**Knowledge Graph Approaches**: Medical knowledge graphs (KGs) encode structured relationships between diseases, symptoms, and other clinical concepts. Systems like UMLS provide comprehensive ontologies used in clinical NLP. However, KG-only approaches often lack the flexibility to handle natural language and contextual reasoning.

### 1.3 Our Approach: Small LLM + Knowledge Graph

We propose a hybrid approach that combines the strengths of small LLMs (3-8B parameters) with medical knowledge graphs. Our key insight is that knowledge retrieval and selection can be decoupled:

- **Knowledge Graph** handles the knowledge-intensive task of identifying relevant symptoms and scoring candidate diagnoses based on confirmed evidence.

- **Small LLM** handles the selection task of choosing among top-K candidates based on clinical context.

This division of labor enables several advantages:

1. **No Task-Specific Training**: Unlike RL approaches, our method requires no fine-tuning or environment interaction. Knowledge is provided by UMLS; reasoning is handled by pretrained LLM capabilities.

2. **Computational Efficiency**: Small LLMs (3-8B) are 10-100x cheaper to run than large LLMs, enabling deployment on commodity hardware.

3. **Privacy Preservation**: All inference runs locally, avoiding the need to send patient data to external APIs.

4. **Interpretability**: KG traversal paths provide natural explanations for diagnostic reasoning.

### 1.4 Contributions

Our main contributions are:

1. **A novel hybrid architecture** combining small LLMs with UMLS knowledge graph for automatic differential diagnosis, requiring no task-specific training.

2. **Comprehensive evaluation** on the DDXPlus benchmark, demonstrating 86.2% GTPA@1 accuracy—outperforming AARLC (75.4%) by 10.8 percentage points while using 26.7% fewer questions.

3. **Model-agnostic design** validated across 9 different LLMs (3-8B parameters), showing consistent performance regardless of model choice.

4. **Ablation studies** on key hyperparameters (candidate count K, scoring formula), providing empirical guidance for system design.

### 1.5 Paper Organization

The remainder of this paper is organized as follows:
- Section 2 reviews related work in automatic diagnosis, medical LLMs, and knowledge graphs.
- Section 3 describes our method, including the KG module and LLM selection module.
- Section 4 presents experimental setup and evaluation metrics.
- Section 5 reports results and ablation studies.
- Section 6 discusses implications, limitations, and future directions.
- Section 7 concludes the paper.

---

## Key Figures for Introduction

### Figure 1: System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input                                    │
│   Patient: 45M, Chief Complaint: Chest Pain                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   UMLS Knowledge Graph                          │
│                                                                 │
│   Chest Pain ──may_cause──▶ [MI, Angina, GERD, ...]            │
│                    │                                            │
│              for each disease                                   │
│                    │                                            │
│                    ▼                                            │
│   [Symptoms: Dyspnea, Sweating, Nausea, ...]                   │
│                                                                 │
│   Output: Top-3 Symptoms ranked by coverage                     │
│   1. Dyspnea (45%)                                             │
│   2. Sweating (32%)                                            │
│   3. Radiation to arm (23%)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Small LLM (4B)                               │
│                                                                 │
│   Prompt: "Select best symptom to ask..."                      │
│   Response: "1"                                                │
│                                                                 │
│   Selected: Dyspnea                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Patient Simulator                             │
│                                                                 │
│   Q: "Do you have shortness of breath?"                        │
│   A: "Yes"                                                      │
│                                                                 │
│   Update: confirmed_symptoms += [Dyspnea]                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                        Repeat until
                       stop condition
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Diagnosis                            │
│                                                                 │
│   1. Myocardial Infarction (52%)                               │
│   2. Unstable Angina (31%)                                     │
│   3. GERD (17%)                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Figure 2: Performance Comparison

```
GTPA@1 Accuracy (%)
│
90┤                          ┌──────┐
  │                          │ 86.2 │ Ours
85┤                          │      │
  │                          │      │
80┤                          │      │
  │       ┌──────┐           │      │
75┤       │ 75.4 │           │      │
  │       │AARLC │           │      │
70┤       │      │           │      │
  │       └──────┘           └──────┘
  └────────────────────────────────────
         Baseline              Ours
```
