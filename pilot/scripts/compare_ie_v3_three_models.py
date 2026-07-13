#!/usr/bin/env python3
"""IE v3 3-way 비교: Gemma vs Gemini vs Qwen3.5-9B (예시 제거, 원칙만).

v2 프롬프트의 괄호 속 예시(cough, fever, BRCA1 등)를 모두 제거.
- Definitions: 카테고리만 추상적으로 서술
- DO NOT EXTRACT: 카테고리명만, 예시 없음
- Output Format: 형식 명세만, "e.g., Dyspnea not shortness of breath" 제거

20 papers (papers1 seed=42 + papers2 seed=99) 동일 사용.
Blind 3-way judging by Gemini-3-flash-preview.
"""
from __future__ import annotations
import os, json, random, re
from google import genai
from vllm import LLM, SamplingParams

with open('.env') as f:
    for line in f:
        if line.startswith('GEMINI_API_KEY='):
            os.environ['GEMINI_API_KEY'] = line.split('=', 1)[1].strip()


IE_PROMPT_V3 = """# Task: Clinical Phenotype Extraction

Extract diagnostic phenotypes (clinical features) for the disease "{disease}" from the following PubMed abstract.

# Definition

A diagnostic phenotype is a clinically observable abnormality of an individual patient that a clinician evaluates when establishing a diagnosis. It belongs to exactly one of the following observation modalities:
- A subjective sensation or complaint reported directly by the patient
- A physical finding objectively observed by a clinician during examination
- A quantitative biochemical, hematological, or immunological abnormality measured in body fluids or tissues
- An abnormal structural or functional finding detected by medical imaging
- A microscopic or gross abnormality identified in tissue specimens

# Inclusion Criteria

A candidate qualifies only if ALL of the following hold:
- It is explicitly stated in the abstract as occurring in patients with the target disease
- It denotes a specific, discrete clinical observation rather than a category, summary, or qualitative descriptor
- It is an attribute of the patient's body, not of an external entity, instrument, study, intervention, or biological mechanism
- It is identifiable as a single phenotypic concept rather than a compound clause

# Exclusion Criteria

A candidate MUST NOT be extracted if any of the following applies:
- It names the target disease itself, any of its subtypes, alternative designations, or closely synonymous terms
- It refers to a gene, gene product, protein, receptor, antigen, antibody, or any molecular entity
- It refers to a pathophysiological or molecular mechanism, pathway, or biological process
- It refers to any therapeutic, surgical, pharmacological, or supportive intervention
- It refers to a diagnostic instrument, scoring system, questionnaire, scale, or composite index
- It refers to study design, sample characteristics, prevalence, incidence, mortality, or epidemiological measurement
- It refers to a different disease mentioned only for differential diagnosis or comparison
- It is a generic narrative term denoting an outcome, course, severity grade, or clinical state without naming a specific abnormality
- It refers to a temporal pattern, demographic attribute, comorbidity status, or risk factor that is not itself a clinical observation

# Precision Standard

If a candidate's classification is ambiguous, it must be excluded. False positives degrade knowledge graph quality more than false negatives.

# Output Specification

Express each phenotype using canonical medical terminology consistent with controlled biomedical vocabularies. Normalize non-canonical phrasing to its canonical form when the canonical form is unambiguous; otherwise reuse the abstract's wording.

Output exactly one phenotype per line, each line in the form:
PHENOTYPE: <term>

The output must consist solely of such PHENOTYPE lines. Do not produce any analytical commentary, reasoning, headers, bullet markers, numbering, or surrounding text.

# Abstract

{text}

# Extracted Phenotypes for {disease}"""


def parse_phenotypes(text):
    findings = []
    for line in text.split('\n'):
        line = line.strip()
        # Strict: only accept "PHENOTYPE: X" lines to avoid Qwen-style reasoning leakage
        m = re.match(r'PHENOTYPE\s*:\s*(.+)', line, re.IGNORECASE)
        if m:
            f = m.group(1).strip().rstrip('.,;:')
            f = re.sub(r'^[\*\-•\d\.\s]+', '', f).strip()
            if f and 2 < len(f) < 100:
                findings.append(f)
    return findings


def main():
    # Load same 20 papers from prior experiments
    papers = []
    for path in ['pilot/results/ie_compare_v2.json', 'pilot/results/ie_compare_v2_papers2.json']:
        with open(path) as f:
            for case in json.load(f):
                papers.append({
                    'pmid': case['pmid'],
                    'disease': case['disease'],
                    'text': case['abstract'],
                    'text_words': case['text_words'],
                })
    print(f"Loaded {len(papers)} papers from prior experiments")

    # === Gemini v3 ===
    print("="*80); print("Gemini-3-Flash-Preview v3 (no examples)..."); print("="*80)
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    gemini_v3 = []
    for i, p in enumerate(papers):
        prompt = IE_PROMPT_V3.format(disease=p['disease'], text=p['text'])
        try:
            resp = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
            findings = parse_phenotypes(resp.text)
        except Exception as e:
            findings = []
            print(f"  ERROR: {e}")
        gemini_v3.append(findings)
        print(f"[{i+1:2d}] {p['disease'][:30]:30s} (Gemini v3): {len(findings)}")

    # === Qwen3.5-9B v3 ===
    print(); print("="*80); print("Qwen3.5-9B v3..."); print("="*80)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="Qwen/Qwen3.5-9B", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=2000)
    convs = [[{"role": "user", "content": IE_PROMPT_V3.format(disease=p['disease'], text=p['text'])}] for p in papers]
    outs = llm.chat(convs, sampling)
    qwen_v3 = []
    for i, out in enumerate(outs):
        findings = parse_phenotypes(out.outputs[0].text)
        qwen_v3.append(findings)
        print(f"[{i+1:2d}] {papers[i]['disease'][:30]:30s} (Qwen v3): {len(findings)}")
    del llm
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    # === Gemma v3 ===
    print(); print("="*80); print("Gemma-4-E4B-it v3..."); print("="*80)
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=2000)
    convs = [[{"role": "user", "content": IE_PROMPT_V3.format(disease=p['disease'], text=p['text'])}] for p in papers]
    outs = llm.chat(convs, sampling)
    gemma_v3 = []
    for i, out in enumerate(outs):
        findings = parse_phenotypes(out.outputs[0].text)
        gemma_v3.append(findings)
        print(f"[{i+1:2d}] {papers[i]['disease'][:30]:30s} (Gemma v3): {len(findings)}")

    # Save
    comparison = []
    for i, p in enumerate(papers):
        comparison.append({
            "pmid": p['pmid'], "disease": p['disease'],
            "text_words": p['text_words'], "abstract": p['text'],
            "gemini_v3": gemini_v3[i], "gemma_v3": gemma_v3[i], "qwen_v3": qwen_v3[i],
        })
    with open('pilot/results/ie_compare_v3_three_models.json', 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Quantity
    print(); print("="*80); print("Quantity (20 papers, v3 prompt = principles only)"); print("="*80)
    g_total = sum(len(c['gemini_v3']) for c in comparison)
    m_total = sum(len(c['gemma_v3']) for c in comparison)
    q_total = sum(len(c['qwen_v3']) for c in comparison)
    print(f"  Gemini v3: {g_total} ({g_total/20:.1f}/paper)")
    print(f"  Gemma v3:  {m_total} ({m_total/20:.1f}/paper)")
    print(f"  Qwen v3:   {q_total} ({q_total/20:.1f}/paper)")

    # Blind 3-way judging
    print(); print("="*80); print("Blind 3-way judging (Gemini)"); print("="*80)
    g_score = g_count = g_useful = 0
    m_score = m_count = m_useful = 0
    q_score = q_count = q_useful = 0
    random.seed(42)

    for case in comparison:
        disease = case['disease']
        items = ([(f, 'G') for f in case['gemini_v3']]
                + [(f, 'M') for f in case['gemma_v3']]
                + [(f, 'Q') for f in case['qwen_v3']])
        if not items: continue
        random.shuffle(items)
        item_text = "\n".join(f"{i+1}. {f}" for i, (f, _) in enumerate(items))
        # Judge prompt also principle-only (no examples)
        judge_prompt = f"""Score each candidate below as a diagnostic feature for the disease "{disease}".

Scoring scale:
- 2: A specific, useful diagnostic phenotype that a physician would consider when diagnosing this disease.
- 1: Weakly relevant — a real clinical concept but too generic, vague, or borderline to be diagnostically useful.
- 0: Not a useful diagnostic phenotype.

Candidates:
{item_text}

Output ONLY a numbered list of integer scores in the order given, one per line:
1. <0|1|2>
2. <0|1|2>
..."""
        resp = client.models.generate_content(model='gemini-3-flash-preview', contents=judge_prompt)
        scores = {}
        for line in resp.text.split('\n'):
            m = re.match(r'\s*(\d+)\s*\.?\s*(\d)', line.strip())
            if m: scores[int(m.group(1)) - 1] = int(m.group(2))

        gs = []; ms = []; qs = []
        for i, (f, src) in enumerate(items):
            s = scores.get(i, 0)
            if src == 'G':
                gs.append(s); g_score += s; g_count += 1
                if s == 2: g_useful += 1
            elif src == 'M':
                ms.append(s); m_score += s; m_count += 1
                if s == 2: m_useful += 1
            else:
                qs.append(s); q_score += s; q_count += 1
                if s == 2: q_useful += 1

        def fmt(L):
            if not L: return "n/a"
            return f"mean={sum(L)/len(L):.2f} u={sum(1 for s in L if s==2)}/{len(L)}"
        print(f"  [{disease[:25]:25s}] G {fmt(gs)}  M {fmt(ms)}  Q {fmt(qs)}")

    print(); print("="*80); print("FINAL (v3 prompt, principles only, 20 papers)"); print("="*80)
    print(f"  Gemini v3: {g_count} ext, {g_useful} useful ({100*g_useful/max(g_count,1):.1f}%), mean={g_score/max(g_count,1):.2f}")
    print(f"  Gemma  v3: {m_count} ext, {m_useful} useful ({100*m_useful/max(m_count,1):.1f}%), mean={m_score/max(m_count,1):.2f}")
    print(f"  Qwen   v3: {q_count} ext, {q_useful} useful ({100*q_useful/max(q_count,1):.1f}%), mean={q_score/max(q_count,1):.2f}")


if __name__ == "__main__":
    main()
