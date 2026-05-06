#!/usr/bin/env python3
"""IE 비교 v2: 학술적으로 검증된 프롬프트 개선.

기존 v1 프롬프트의 문제:
- "rules" 만 나열하여 LLM이 noise (분자기전, 점수도구, 유전자) 추출 다수
- Entity type 정의 부재로 ambiguous cases (e.g., "BRCA1") 처리 불일치
- 추출 후 self-verification 단계 없음

학술 근거 적용:
1. GoLLIE (ICLR 2024): annotation guidelines block (+8-12%p F1)
2. GPT-RE (ACL 2023): entity-type definitions
3. Wadhwa 2023: precision-first explicit instructions
4. Tang 2023: structured CoT for IE
5. HPO SOP: phenotype definition criteria

비교: gemma-4-E4B vs gemini-3-flash-preview, 동일 v2 prompt, 동일 10 papers.
평가: blind judging (Gemini가 anonymized 평가).
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

random.seed(42)


# v2 프롬프트: 학술적 패턴 적용
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
    """Parse PHENOTYPE: lines from output, robust to common variations."""
    findings = []
    for line in text.split('\n'):
        line = line.strip()
        # Match "PHENOTYPE: X" or just bullet/numbered lines
        m = re.match(r'(?:PHENOTYPE\s*:\s*|[-•*]\s*|\d+\.\s*)(.+)', line)
        if m:
            f = m.group(1).strip().rstrip('.,;:')
            if f and len(f) > 2 and len(f) < 100:
                findings.append(f)
        elif line and not line.startswith('#') and len(line) > 2 and len(line) < 100:
            # Tolerate plain lines
            if not any(s in line.lower() for s in ['note:', 'task:', 'extract', 'definition']):
                findings.append(line.rstrip('.,;:'))
    return findings


def main():
    # Same 10 papers as v1
    with open('pilot/data/exp_documents.json') as f:
        data = json.load(f)
    docs = data['documents']

    by_disease = {}
    for d in docs:
        by_disease.setdefault(d['seed_disease'], []).append(d)

    diseases = sorted(by_disease.keys())
    random.shuffle(diseases)
    sample_papers = []
    for dn in diseases:
        papers = [p for p in by_disease[dn] if 200 <= p['text_words'] <= 500]
        if not papers: continue
        sample_papers.append(papers[0])
        if len(sample_papers) >= 10: break

    print(f"Selected {len(sample_papers)} papers (same as v1)")
    for i, p in enumerate(sample_papers):
        print(f"  {i+1}. PMID {p['pmid']} - {p['seed_disease']}")
    print()

    # ========== Gemini IE v2 ==========
    print("="*80)
    print("Running Gemini-3-Flash-Preview with v2 prompt...")
    print("="*80)
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    gemini_v2 = []
    for i, p in enumerate(sample_papers):
        prompt = IE_PROMPT_V2.format(disease=p['seed_disease'], text=p['text'])
        try:
            resp = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
            findings = parse_phenotypes(resp.text)
        except Exception as e:
            findings = [f"ERROR: {e}"]
        gemini_v2.append(findings)
        print(f"\n[{i+1}] {p['seed_disease']} (Gemini v2): {len(findings)} findings")
        for f in findings: print(f"  - {f}")

    # ========== Gemma IE v2 ==========
    print()
    print("="*80)
    print("Running Gemma-4-E4B-it with v2 prompt...")
    print("="*80)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=600)

    convs = []
    for p in sample_papers:
        prompt = IE_PROMPT_V2.format(disease=p['seed_disease'], text=p['text'])
        convs.append([{"role": "user", "content": prompt}])

    outs = llm.chat(convs, sampling)
    gemma_v2 = []
    for i, out in enumerate(outs):
        text = out.outputs[0].text.strip()
        findings = parse_phenotypes(text)
        gemma_v2.append(findings)
        print(f"\n[{i+1}] {sample_papers[i]['seed_disease']} (Gemma v2): {len(findings)} findings")
        for f in findings: print(f"  - {f}")

    # ========== Save ==========
    comparison = []
    for i, p in enumerate(sample_papers):
        comparison.append({
            "pmid": p['pmid'],
            "disease": p['seed_disease'],
            "text_words": p['text_words'],
            "abstract": p['text'],
            "gemini_v2": gemini_v2[i],
            "gemma_v2": gemma_v2[i],
        })
    with open('pilot/results/ie_compare_v2.json', 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print()
    print("="*80)
    print("Quantity summary (v2 prompt)")
    print("="*80)
    g_total = sum(len(r) for r in gemini_v2)
    m_total = sum(len(r) for r in gemma_v2)
    print(f"Gemini v2: {g_total} total ({g_total/len(sample_papers):.1f}/paper)")
    print(f"Gemma v2:  {m_total} total ({m_total/len(sample_papers):.1f}/paper)")

    # ========== Blind judging ==========
    print()
    print("="*80)
    print("Blind judging by Gemini-3-flash-preview (anonymized)")
    print("="*80)

    g_total_score = 0; g_count = 0; g_useful = 0
    m_total_score = 0; m_count = 0; m_useful = 0

    for case in comparison:
        disease = case['disease']
        items = [(f, 'G') for f in case['gemini_v2']] + [(f, 'M') for f in case['gemma_v2']]
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
            if m:
                scores[int(m.group(1)) - 1] = int(m.group(2))

        g_scores = []; m_scores = []
        for i, (f, src) in enumerate(items):
            s = scores.get(i, 0)
            if src == 'G':
                g_scores.append(s); g_total_score += s; g_count += 1
                if s == 2: g_useful += 1
            else:
                m_scores.append(s); m_total_score += s; m_count += 1
                if s == 2: m_useful += 1

        g_str = f"mean={sum(g_scores)/len(g_scores):.2f} u={sum(1 for s in g_scores if s==2)}/{len(g_scores)}" if g_scores else "n/a"
        m_str = f"mean={sum(m_scores)/len(m_scores):.2f} u={sum(1 for s in m_scores if s==2)}/{len(m_scores)}" if m_scores else "n/a"
        print(f"  [{disease[:30]:30s}] Gemini {g_str}  vs  Gemma {m_str}")

    print()
    print("="*80)
    print("최종 비교 (v2 prompt + blind judging)")
    print("="*80)
    print(f"  Gemini v2: {g_count} extractions, {g_useful} useful ({100*g_useful/max(g_count,1):.1f}%), mean={g_total_score/max(g_count,1):.2f}")
    print(f"  Gemma  v2: {m_count} extractions, {m_useful} useful ({100*m_useful/max(m_count,1):.1f}%), mean={m_total_score/max(m_count,1):.2f}")


if __name__ == "__main__":
    main()
