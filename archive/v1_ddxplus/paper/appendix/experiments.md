# Experiments Section - Detailed Draft

## 4. Experimental Setup

### 4.1 Dataset

We evaluate on **DDXPlus** (Fansi Tchango et al., 2022), a large-scale synthetic dataset for automatic medical diagnosis.

| Statistic | Value |
|-----------|-------|
| Diseases | 49 |
| Symptoms | 223 |
| Train patients | 1,012,155 |
| Validate patients | 42,255 |
| Test patients | 100,000+ |

**Severity Levels**:
- Level 1: Mild conditions
- Level 2: Moderate conditions (our focus)
- Level 3: Severe conditions
- Level 4: Emergency conditions
- Level 5: Critical conditions

We focus on severity level 2 (moderate) following prior work, which provides a balanced challenge between trivial and extremely difficult cases.

### 4.2 Evaluation Metrics

#### Primary Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **GTPA@1** | $\mathbb{1}[\hat{d}_1 = d^*]$ | Top-1 diagnostic accuracy |
| **IL** | $\frac{1}{N}\sum_i t_i$ | Average interaction length |

#### Secondary Metrics (Differential Diagnosis Quality)

| Metric | Formula | Description |
|--------|---------|-------------|
| **DDR** | $\frac{|\hat{\mathcal{D}} \cap \mathcal{D}^*|}{|\mathcal{D}^*|}$ | Recall of ground truth diseases |
| **DDP** | $\frac{|\hat{\mathcal{D}} \cap \mathcal{D}^*|}{|\hat{\mathcal{D}}|}$ | Precision of predicted diseases |
| **DDF1** | $\frac{2 \cdot DDR \cdot DDP}{DDR + DDP}$ | F1 score for differential diagnosis |

### 4.3 Baselines

#### AARLC (Fansi Tchango et al., 2022)
- Reinforcement learning with Advantage Actor-Critic
- Trained on DDXPlus environment
- State-of-the-art on DDXPlus benchmark

| Metric | AARLC |
|--------|-------|
| GTPA@1 | 75.39% |
| DDR | 97.73% |
| DDF1 | 78.24% |
| IL | 25.75 |

#### H-DDx Baselines (ACL 2025)
| Model | Top-5 | HDF1 |
|-------|-------|------|
| Claude Sonnet 4 | 83.9% | 36.7% |
| GPT-4o | 80.4% | 35.0% |

### 4.4 Models Evaluated

We evaluate 8 small LLMs spanning 3-8B parameters:

| Model | Parameters | Developer |
|-------|------------|-----------|
| Ministral-3B | 3B | Mistral AI |
| Qwen3-4B | 4B | Alibaba |
| gemma-3-4b | 4B | Google |
| medgemma-1.5-4b | 4B | Google (medical) |
| Phi-4-mini | 4B | Microsoft |
| Qwen3-8B | 8B | Alibaba |
| Ministral-8B | 8B | Mistral AI |
| Llama-3.1-8B | 8B | Meta |

### 4.5 Implementation Details

**Knowledge Graph**:
- Database: Neo4j 5.x
- UMLS version: 2024AA
- Relationships: may_cause, has_finding, associated_with
- Index: Full-text search on concept names

**LLM Inference**:
- Framework: vLLM 0.6.x
- Batch size: Dynamic (all active patients per round)
- Temperature: 0.1
- Max tokens: 128
- Stop tokens: newline, EOS

**Hardware**:
- GPU: NVIDIA RTX 4090 (24GB) or A100 (40GB)
- RAM: 64GB
- Storage: NVMe SSD

**Hyperparameters**:
| Parameter | Value | Selection |
|-----------|-------|-----------|
| Symptom K | 3 | Ablation study |
| Diagnosis K | 3 | Ablation study |
| Scoring | v18_coverage | Ablation study |
| Min questions | 3 | Safety constraint |
| Max questions | 50 | Upper bound |
| Confidence threshold | 0.8 | Empirical |
| Gap threshold | 0.3 | Empirical |

---

## 5. Results

### 5.1 Main Results

**Table 1: Comparison with AARLC baseline (n=10,000, severity=2)**

| Method | GTPA@1 | DDR | DDF1 | IL |
|--------|--------|-----|------|-----|
| AARLC | 75.4% | 97.7% | 78.2% | 25.8 |
| Ours (avg.) | **86.2%** | 52.8% | 47.4% | **18.9** |
| Δ | **+10.8%** | -44.9% | -30.8% | **-6.9** |

**Key observations**:
1. **GTPA@1 improvement**: Our method achieves 86.2% accuracy, outperforming AARLC by 10.8 percentage points
2. **Efficiency gain**: Average of 18.9 questions vs 25.8 for AARLC (26.7% reduction)
3. **DDR trade-off**: Lower DDR reflects our focus on top-1 accuracy rather than full differential list

### 5.2 Model Comparison

**Table 2: Results by model (n=10,000)**

| Model | GTPA@1 | IL |
|-------|--------|-----|
| Ministral-3B | 86.4% | 19.1 |
| Qwen3-4B | 86.0% | 19.1 |
| gemma-3-4b | 86.2% | 18.6 |
| medgemma-1.5-4b | 86.3% | 18.9 |
| Phi-4-mini | 86.3% | 18.6 |
| Qwen3-8B | 86.2% | 19.1 |
| Ministral-8B | 86.2% | 18.6 |
| Llama-3.1-8B | 85.6% | 19.2 |

**Observation**: Performance variance across models is minimal (<1%), demonstrating that our KG-guided approach is model-agnostic. The knowledge graph handles the heavy lifting of medical reasoning.

### 5.3 Ablation Studies

#### 5.3.1 Joint Top-K Ablation

**Table 3: Effect of candidate count K (n=500)**

| K | GTPA@1 | IL | Notes |
|---|--------|-----|-------|
| 1 | 85.8% | 19.1 | No LLM choice |
| 2 | 85.6% | 19.4 | |
| **3** | **85.6%** | **19.3** | **Selected** |
| 4 | 85.4% | 19.6 | |
| 5 | 85.2% | 20.8 | IL degradation |
| 6 | 85.0% | 21.3 | |
| 8 | 85.0% | 21.7 | |
| 10 | 85.2% | 21.8 | |

**Analysis**:
- K ∈ {1, 2, 3} achieve equivalent accuracy (within 0.2%)
- K ≥ 5 leads to increased IL without accuracy benefit
- We select K=3 to provide LLM with choice flexibility

#### 5.3.2 Diagnosis-Specific Ablation

**Table 4: Diagnosis K with symptom K=3 fixed (n=500)**

| $K_d$ | GTPA@1 | IL |
|-------|--------|-----|
| 1 | 85.6% | 19.4 |
| **3** | **85.2%** | **19.4** |
| 5 | 85.2% | 19.4 |
| 10 | 85.0% | 19.4 |

**Analysis**: Diagnosis K has minimal impact on performance. We use K=3 for consistency with symptom selection.

### 5.4 Efficiency Analysis

**Table 5: Throughput comparison**

| Method | Throughput | Cost per 1K |
|--------|------------|-------------|
| GPT-4o API | ~10 req/min | ~$5.00 |
| Claude Sonnet API | ~15 req/min | ~$3.00 |
| Ours (Qwen3-4B) | ~200 req/min | ~$0.05 |
| Ours (Llama-8B) | ~150 req/min | ~$0.10 |

Our method achieves 15-20x higher throughput and 30-100x lower cost compared to large LLM APIs.

---

## 6. Discussion

### 6.1 Why Does KG + Small LLM Work?

The success of our approach can be attributed to:

1. **Division of labor**: KG handles knowledge-intensive retrieval, LLM handles selection
2. **Reduced search space**: Top-K candidates focus LLM attention on relevant options
3. **Structured guidance**: Probability scores help LLM understand relative importance

### 6.2 GTPA@1 vs DDR Trade-off

Our method optimizes for final diagnosis accuracy (GTPA@1) at the expense of differential diagnosis recall (DDR). This reflects different clinical use cases:

- **GTPA@1 focus**: Point-of-care triage, initial diagnosis
- **DDR focus**: Comprehensive workup, specialist referral

AARLC's high DDR (97.7%) comes from its RL training objective that rewards the full differential list.

### 6.3 Limitations

1. **DDR lower than AARLC**: Our method produces fewer candidate diagnoses
2. **UMLS dependency**: Requires access to UMLS knowledge graph
3. **English only**: Current evaluation limited to English
4. **Synthetic data**: DDXPlus is synthetic; real clinical validation needed

### 6.4 Future Directions

1. **Hybrid training**: Fine-tune LLM on DDXPlus for improved DDR
2. **Multi-language**: Extend to other languages via UMLS translations
3. **EHR integration**: Connect to electronic health records
4. **Clinical validation**: Partner with hospitals for real-world evaluation
