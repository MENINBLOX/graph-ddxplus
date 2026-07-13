#!/usr/bin/env python3
"""IE 비교: Qwen3.5-9B 추가 (Gemma vs Gemini vs Qwen, v2 prompt, 20 papers).

이전 두 실험에서 사용한 동일 20개 papers (papers1 + papers2)에서 Qwen3.5-9B로 v2 IE 실행.
모든 결과를 blind judging으로 평가.
"""
from __future__ import annotations
import os, json, time, random, re
from pathlib import Path
from google import genai
from vllm import LLM, SamplingParams

with open('.env') as f:
    for line in f:
        if line.startswith('GEMINI_API_KEY='):
            os.environ['GEMINI_API_KEY'] = line.split('=', 1)[1].strip()


IE_PROMPT_V2 = """# Task: Clinical Phenotype Extraction

You are extracting diagnostic phenotypes (clinical features) for the disease "{disease}" from a PubMed abstract.

# Definitions (Entity Type Specification, HPO-aligned)

A "diagnostic phenotype" is a clinically observable abnormality that a physician would consider when diagnosing a patient. This includes:
- **Symptoms**: patient-reported (cough, pain, fatigue, dyspnea, fever, nausea)
- **Signs**: physician-observable (rash, jaundice, edema, tachycardia, hepatomegaly)
- **Lab findings**: measurable abnormalities (elevated CRP, hyperphenylalaninemia, anemia)
- **Imaging findings**: visible on imaging (hilar lymphadenopathy, pulmonary nodule, cardiomegaly)
- **Pathological findings**: tissue abnormalities (granuloma, necrosis, fibrosis)

# Annotation Guidelines

EXTRACT phenotypes that are:
- Explicitly stated in the abstract as observed in patients with {disease}
- Specific clinical features (not generic terms like "symptoms" or "abnormality")
- Discrete observations (not graded scores or composite measures)

DO NOT EXTRACT:
- The disease name itself ("{disease}", subtypes, or near-synonyms)
- Genes, proteins, or molecular targets (BRCA1, TNF-alpha, Tat protein)
- Molecular mechanisms (HIV attachment, autophagy, signaling pathways)
- Treatments or interventions (antibiotics, surgery, antiretrovirals)
- Measurement instruments or scores (SNOT-22, Lund-Mackay, scoring systems)
- Study design terms (cohort, randomized trial, mortality rate)
- Comparator diseases mentioned only for differential diagnosis
- Vague generic descriptors ("clinical course", "outcomes", "complications" without specifics)

# Precision Standard

When uncertain whether a term is a diagnostic phenotype, EXCLUDE it. False positives degrade KG quality more than false negatives.

# Output Format

For each extracted phenotype, write one line in this exact format:
PHENOTYPE: <standard medical term>

Use canonical UMLS-style medical terminology (e.g., "Dyspnea" not "shortness of breath", "Hemoptysis" not "coughing up blood"). If unsure, use the abstract's terminology.

# Abstract

{text}

# Extracted Phenotypes for {disease}"""


def parse_phenotypes(text):
    findings = []
    for line in text.split('\n'):
        line = line.strip()
        m = re.match(r'(?:PHENOTYPE\s*:\s*|[-•*]\s*|\d+\.\s*)(.+)', line)
        if m:
            f = m.group(1).strip().rstrip('.,;:')
            if f and len(f) > 2 and len(f) < 100:
                findings.append(f)
        elif line and not line.startswith('#') and len(line) > 2 and len(line) < 100:
            if not any(s in line.lower() for s in ['note:', 'task:', 'extract', 'definition', 'phenotype']):
                findings.append(line.rstrip('.,;:'))
    return findings


def main():
    # Load papers from prior experiments to ensure same 20 papers
    papers = []
    for path in ['pilot/results/ie_compare_v2.json', 'pilot/results/ie_compare_v2_papers2.json']:
        with open(path) as f:
            for case in json.load(f):
                papers.append({
                    'pmid': case['pmid'],
                    'seed_disease': case['disease'],
                    'text': case['abstract'],
                    'gemini_v2': case['gemini_v2'],
                    'gemma_v2': case['gemma_v2'],
                    'text_words': case['text_words'],
                })
    print(f"Loaded {len(papers)} papers from prior experiments")

    # Run Qwen3.5-9B
    print("="*80)
    print("Running Qwen3.5-9B with v2 prompt...")
    print("="*80)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="Qwen/Qwen3.5-9B", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=600)

    convs = [[{"role": "user", "content": IE_PROMPT_V2.format(disease=p['seed_disease'], text=p['text'])}]
             for p in papers]
    outs = llm.chat(convs, sampling)

    qwen_v2 = []
    for i, out in enumerate(outs):
        text = out.outputs[0].text.strip()
        findings = parse_phenotypes(text)
        qwen_v2.append(findings)
        print(f"\n[{i+1}] {papers[i]['seed_disease']} (Qwen v2): {len(findings)} findings")
        for f in findings: print(f"  - {f}")

    # Save combined
    comparison = []
    for i, p in enumerate(papers):
        comparison.append({
            "pmid": p['pmid'], "disease": p['seed_disease'],
            "text_words": p['text_words'], "abstract": p['text'],
            "gemini_v2": p['gemini_v2'], "gemma_v2": p['gemma_v2'],
            "qwen_v2": qwen_v2[i],
        })
    with open('pilot/results/ie_compare_three_models.json', 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Quantity
    print()
    print("="*80)
    print("Quantity summary (20 papers, v2 prompt, 3 models)")
    print("="*80)
    g_total = sum(len(c['gemini_v2']) for c in comparison)
    m_total = sum(len(c['gemma_v2']) for c in comparison)
    q_total = sum(len(c['qwen_v2']) for c in comparison)
    print(f"  Gemini v2: {g_total} total ({g_total/len(papers):.1f}/paper)")
    print(f"  Gemma v2:  {m_total} total ({m_total/len(papers):.1f}/paper)")
    print(f"  Qwen v2:   {q_total} total ({q_total/len(papers):.1f}/paper)")

    # Blind judging — anonymized 3-way
    print()
    print("="*80)
    print("Blind 3-way judging by Gemini-3-flash-preview")
    print("="*80)
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    g_total_score = 0; g_count = 0; g_useful = 0
    m_total_score = 0; m_count = 0; m_useful = 0
    q_total_score = 0; q_count = 0; q_useful = 0
    random.seed(42)

    for case in comparison:
        disease = case['disease']
        items = ([(f, 'G') for f in case['gemini_v2']]
                + [(f, 'M') for f in case['gemma_v2']]
                + [(f, 'Q') for f in case['qwen_v2']])
        if not items: continue
        random.shuffle(items)

        item_text = "\n".join(f"{i+1}. {f}" for i, (f, _) in enumerate(items))
        judge_prompt = f"""Score each item below for its usefulness as a diagnostic feature for "{disease}":

- 2 = Useful diagnostic feature (symptom/sign/lab/imaging finding a physician would consider)
- 1 = Weakly useful (too generic or vague)
- 0 = NOT useful for diagnosis (gene names, measurement scores, molecular mechanisms, treatments, the disease name itself, study methodology)

Items:
{item_text}

Output ONLY a numbered list of scores:
1. <0/1/2>
2. <0/1/2>
..."""
        resp = client.models.generate_content(model='gemini-3-flash-preview', contents=judge_prompt)
        scores = {}
        for line in resp.text.split('\n'):
            m = re.match(r'\s*(\d+)\s*\.?\s*(\d)', line.strip())
            if m: scores[int(m.group(1)) - 1] = int(m.group(2))

        g_scores = []; m_scores = []; q_scores = []
        for i, (f, src) in enumerate(items):
            s = scores.get(i, 0)
            if src == 'G':
                g_scores.append(s); g_total_score += s; g_count += 1
                if s == 2: g_useful += 1
            elif src == 'M':
                m_scores.append(s); m_total_score += s; m_count += 1
                if s == 2: m_useful += 1
            else:
                q_scores.append(s); q_total_score += s; q_count += 1
                if s == 2: q_useful += 1

        def fmt(scores_list):
            if not scores_list: return "n/a"
            return f"mean={sum(scores_list)/len(scores_list):.2f} u={sum(1 for s in scores_list if s==2)}/{len(scores_list)}"

        print(f"  [{disease[:25]:25s}] Gemini {fmt(g_scores)}  Gemma {fmt(m_scores)}  Qwen {fmt(q_scores)}")

    print()
    print("="*80)
    print("최종 비교 (20 papers, v2 prompt, 3 models, blind judging)")
    print("="*80)
    print(f"  Gemini v2: {g_count} ext, {g_useful} useful ({100*g_useful/max(g_count,1):.1f}%), mean={g_total_score/max(g_count,1):.2f}")
    print(f"  Gemma  v2: {m_count} ext, {m_useful} useful ({100*m_useful/max(m_count,1):.1f}%), mean={m_total_score/max(m_count,1):.2f}")
    print(f"  Qwen   v2: {q_count} ext, {q_useful} useful ({100*q_useful/max(q_count,1):.1f}%), mean={q_total_score/max(q_count,1):.2f}")


if __name__ == "__main__":
    main()
