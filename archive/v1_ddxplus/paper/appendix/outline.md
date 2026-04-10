# Paper Outline: Small LLM + UMLS Knowledge Graph for Automatic Differential Diagnosis

## Title Options
1. "Efficient Differential Diagnosis with Small Language Models and Medical Knowledge Graphs"
2. "KG-Guided Small LLMs for Cost-Effective Automatic Differential Diagnosis"
3. "Bridging Knowledge Graphs and Small Language Models for Medical Diagnosis"

---

## Abstract (~150 words)

**Background**: Automatic differential diagnosis (DDx) systems require both medical knowledge and efficient reasoning. Large language models (LLMs) achieve strong performance but are computationally expensive. Reinforcement learning approaches like AARLC require extensive training.

**Method**: We propose a hybrid approach combining small LLMs (3-8B parameters) with UMLS knowledge graph traversal. The KG provides candidate symptoms and diagnoses ranked by relevance, while the LLM selects among top-K candidates.

**Results**: On DDXPlus dataset, our method achieves 86.2% GTPA@1 accuracy, outperforming AARLC (75.4%) by +10.8 percentage points while requiring 6.9 fewer questions on average (18.9 vs 25.8 IL).

**Conclusion**: Small LLMs combined with structured medical knowledge can achieve state-of-the-art diagnostic accuracy with significantly lower computational cost than large LLMs or RL-based approaches.

---

## 1. Introduction

### 1.1 Problem Statement
- Differential diagnosis is critical in clinical practice
- Existing approaches: RL-based (AARLC), large LLM-based (GPT-4, Claude)
- Trade-off between accuracy, cost, and efficiency

### 1.2 Motivation
- Small LLMs (3-8B) are 10-100x cheaper than large LLMs
- Medical knowledge graphs (UMLS) encode structured clinical knowledge
- Can we combine these for efficient, accurate diagnosis?

### 1.3 Contributions
1. Novel hybrid architecture: Small LLM + UMLS KG
2. KG-guided candidate selection for symptom inquiry and diagnosis
3. Empirical validation on DDXPlus: +10.8% GTPA@1 vs AARLC
4. Comprehensive ablation studies on hyperparameters

---

## 2. Related Work

### 2.1 Automatic Differential Diagnosis
- Rule-based systems (early approaches)
- Bayesian networks
- Deep learning approaches

### 2.2 Reinforcement Learning for Diagnosis
- AARLC (DDXPlus paper)
- Policy gradient methods
- Reward shaping for medical tasks

### 2.3 LLMs in Medical Diagnosis
- GPT-4, Claude for clinical reasoning
- H-DDx benchmark (hierarchical diagnosis)
- Cost and latency challenges

### 2.4 Knowledge Graphs in Healthcare
- UMLS Metathesaurus
- Disease-symptom relationships
- Graph neural networks for medical KGs

---

## 3. Method

### 3.1 Problem Formulation
- Input: Patient demographics, chief complaint
- Process: Sequential symptom inquiry
- Output: Ranked differential diagnosis list

### 3.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Patient Simulator                        │
│  (Demographics, Chief Complaint, Ground Truth Symptoms)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      UMLS Knowledge Graph                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Symptom    │───▶│   2-hop     │───▶│  Diagnosis  │     │
│  │  Expansion  │    │  Traversal  │    │  Scoring    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ Top-K Candidates
┌─────────────────────────────────────────────────────────────┐
│                    Small LLM (3-8B)                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Symptom   │    │    Stop     │    │  Diagnosis  │     │
│  │  Selection  │    │  Decision   │    │  Selection  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Knowledge Graph Module

#### 3.3.1 Symptom Candidate Generation
- 2-hop traversal from chief complaint
- Coverage-based scoring: symptoms that distinguish candidate diseases
- Return Top-K candidates to LLM

#### 3.3.2 Diagnosis Scoring
- v18_coverage formula: `confirmed / (total_symptoms + 1) × confirmed`
- Bayesian-inspired probability estimation
- Cumulative probability cutoff for differential diagnosis list

#### 3.3.3 Stop Decision
- Confidence threshold (Top-1 ≥ 80%)
- Gap threshold (Top-1 - Top-2 ≥ 30%)
- Minimum questions (≥ 3) for safety

### 3.4 LLM Selection Module

#### 3.4.1 Prompt Design
- Present Top-K candidates with probabilities
- Guidance: "Option 1 has highest probability. Select 1 unless specific clinical reason."
- Output: Single number selection

#### 3.4.2 Model Agnostic
- Works with any instruction-following LLM
- Tested: Llama, Qwen, Gemma, Phi, Ministral (3-8B)

### 3.5 Training-Free Approach
- No fine-tuning required
- KG provides domain knowledge
- LLM provides reasoning capability

---

## 4. Experimental Setup

### 4.1 Dataset: DDXPlus
- 49 diseases, 223 symptoms
- Train/Validate/Test splits
- Severity levels (1-5), we use severity=2 (moderate)

### 4.2 Evaluation Metrics
| Metric | Definition |
|--------|------------|
| GTPA@1 | Ground truth pathology at rank 1 |
| DDR | Differential diagnosis recall |
| DDP | Differential diagnosis precision |
| DDF1 | Harmonic mean of DDR and DDP |
| IL | Interaction length (# questions) |

### 4.3 Baselines
- AARLC (DDXPlus paper): RL-based, 75.4% GTPA@1
- H-DDx baselines: Claude Sonnet, GPT-4o

### 4.4 Implementation Details
- KG: Neo4j with UMLS subset
- LLM: vLLM for batch inference
- Hardware: Single GPU (RTX 4090 / A100)

---

## 5. Results

### 5.1 Main Results (Table 1)

| Method | GTPA@1 | DDR | DDF1 | IL |
|--------|--------|-----|------|-----|
| AARLC | 75.4% | 97.7% | 78.2% | 25.8 |
| **Ours (avg.)** | **86.2%** | 52.8% | 47.4% | **18.9** |

**Key findings:**
- +10.8%p improvement in GTPA@1
- 6.9 fewer questions on average
- Consistent across 9 different LLMs

### 5.2 Model Comparison (Table 2)
- All models achieve 85.6-86.4% GTPA@1
- Model choice has minimal impact (< 1% variance)
- Demonstrates model-agnostic nature

### 5.3 Ablation Studies

#### 5.3.1 Top-K Selection (Table 3)
- K ∈ {1,2,3}: equivalent accuracy (85.6-85.8%)
- K ≥ 5: IL degradation (+1.5 questions)
- Selected K=3 for clinical override capability

#### 5.3.2 Diagnosis Top-K (Table 4)
- Symptom K=3 fixed, vary diagnosis K
- Minimal variance (0.4%p) across K values
- K=3 selected for consistency

#### 5.3.3 Scoring Formula (Appendix)
- v18_coverage: 85.6% (selected)
- v15_ratio: 85.4%
- v7_additive: 75.0%

---

## 6. Discussion

### 6.1 Why Does Small LLM + KG Work?
- KG handles knowledge retrieval (what symptoms to ask)
- LLM handles selection (which one to pick)
- Division of labor reduces LLM burden

### 6.2 Trade-offs: GTPA@1 vs DDR
- Our method optimizes for final diagnosis (GTPA@1)
- AARLC optimizes for full differential list (DDR)
- Different clinical use cases

### 6.3 Limitations
- DDR lower than AARLC (52.8% vs 97.7%)
- Requires UMLS knowledge graph setup
- English-only evaluation

### 6.4 Clinical Implications
- Cost-effective deployment in resource-limited settings
- Real-time diagnosis support
- Explainable via KG traversal paths

---

## 7. Conclusion

We presented a hybrid approach combining small LLMs with UMLS knowledge graph for automatic differential diagnosis. Our method achieves 86.2% GTPA@1 accuracy on DDXPlus, outperforming the AARLC baseline by 10.8 percentage points while requiring 6.9 fewer questions. The approach is model-agnostic, training-free, and computationally efficient, making it suitable for practical clinical deployment.

### Future Work
- Multi-language support
- Fine-tuning for improved DDR
- Integration with EHR systems
- Clinical validation studies

---

## Appendix

### A. Hyperparameter Settings
### B. Full Model Results
### C. Error Analysis
### D. KG Statistics
### E. Prompt Templates

---

## References (Key Citations)

1. DDXPlus dataset and AARLC: [Fansi et al., 2022]
2. H-DDx benchmark: [ACL 2025]
3. UMLS Metathesaurus: [NLM]
4. vLLM: [Kwon et al., 2023]
5. Llama 3: [Meta AI, 2024]
6. Medical KG reasoning: [...]
