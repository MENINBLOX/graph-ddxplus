#!/usr/bin/env python3
"""IE 품질 비교: gemini-3-flash-preview vs gemma-4-E4B on 10 PubMed papers.

같은 PubMed abstract에서 같은 IE 프롬프트로 disease symptoms를 추출.
모델 capability가 IE 품질에 미치는 영향 직접 측정.
"""
from __future__ import annotations
import os, json, time, random
from pathlib import Path
from google import genai
from vllm import LLM, SamplingParams

# Load API key
with open('.env') as f:
    for line in f:
        if line.startswith('GEMINI_API_KEY='):
            os.environ['GEMINI_API_KEY'] = line.split('=', 1)[1].strip()

random.seed(42)


IE_PROMPT_TEMPLATE = """You are extracting clinical information from a PubMed abstract about "{disease}".

Abstract:
{text}

Task: Extract the SYMPTOMS, SIGNS, and CLINICAL FINDINGS that are described in this abstract as being associated with {disease}.

Rules:
- Include only findings explicitly mentioned in the text
- Use standard medical terminology (e.g., "dyspnea" not "shortness of breath" if abstract uses dyspnea)
- Exclude: treatments, drugs, study methodology, statistics, demographics
- Exclude: other diseases mentioned only as comparisons or differentials
- Output one finding per line, no numbering, no bullet points, no commentary

Findings:"""


def main():
    # Load 10 diverse papers (one per disease)
    with open('pilot/data/exp_documents.json') as f:
        data = json.load(f)
    docs = data['documents']

    # Group by disease, pick 10 different diseases
    by_disease = {}
    for d in docs:
        by_disease.setdefault(d['seed_disease'], []).append(d)

    # Pick 10 diseases with substantial abstracts
    diseases = sorted(by_disease.keys())
    random.shuffle(diseases)
    sample_papers = []
    for dn in diseases:
        papers = [p for p in by_disease[dn] if 200 <= p['text_words'] <= 500]
        if not papers: continue
        sample_papers.append(papers[0])
        if len(sample_papers) >= 10: break

    print(f"Selected {len(sample_papers)} papers")
    print()
    for i, p in enumerate(sample_papers):
        print(f"{i+1}. PMID {p['pmid']} - {p['seed_disease']} ({p['text_words']} words)")
    print()

    # ========== Gemini IE ==========
    print("="*80)
    print("Running Gemini-3-Flash-Preview...")
    print("="*80)
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    gemini_results = []
    for i, p in enumerate(sample_papers):
        prompt = IE_PROMPT_TEMPLATE.format(disease=p['seed_disease'], text=p['text'])
        try:
            resp = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
            findings = [l.strip().lstrip('-•*0123456789. ').strip() for l in resp.text.split('\n') if l.strip()]
            findings = [f for f in findings if f and len(f) > 2 and len(f) < 100]
        except Exception as e:
            findings = [f"ERROR: {e}"]
        gemini_results.append(findings)
        print(f"\n[{i+1}] {p['seed_disease']} (Gemini): {len(findings)} findings")
        for f in findings: print(f"  - {f}")

    # ========== Gemma IE ==========
    print()
    print("="*80)
    print("Running Gemma-4-E4B-it...")
    print("="*80)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=500)

    convs = []
    for p in sample_papers:
        prompt = IE_PROMPT_TEMPLATE.format(disease=p['seed_disease'], text=p['text'])
        convs.append([{"role": "user", "content": prompt}])

    outs = llm.chat(convs, sampling)
    gemma_results = []
    for i, out in enumerate(outs):
        text = out.outputs[0].text.strip()
        findings = [l.strip().lstrip('-•*0123456789. ').strip() for l in text.split('\n') if l.strip()]
        findings = [f for f in findings if f and len(f) > 2 and len(f) < 100]
        gemma_results.append(findings)
        print(f"\n[{i+1}] {sample_papers[i]['seed_disease']} (Gemma): {len(findings)} findings")
        for f in findings: print(f"  - {f}")

    # Save side-by-side
    comparison = []
    for i, p in enumerate(sample_papers):
        comparison.append({
            "pmid": p['pmid'],
            "disease": p['seed_disease'],
            "text_words": p['text_words'],
            "abstract": p['text'][:500] + "..." if len(p['text']) > 500 else p['text'],
            "full_text": p['text'],
            "gemini_findings": gemini_results[i],
            "gemma_findings": gemma_results[i],
        })

    with open('pilot/results/ie_compare_gemini_gemma.json', 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print()
    print("="*80)
    print("Summary statistics:")
    print("="*80)
    g_total = sum(len(r) for r in gemini_results)
    m_total = sum(len(r) for r in gemma_results)
    print(f"Gemini:  total findings = {g_total}, avg per paper = {g_total/len(sample_papers):.1f}")
    print(f"Gemma:   total findings = {m_total}, avg per paper = {m_total/len(sample_papers):.1f}")

    # Set overlap
    print(f"\nPer-paper Jaccard overlap (lowercase string match):")
    overlaps = []
    for i, p in enumerate(sample_papers):
        g_set = {f.lower().strip() for f in gemini_results[i]}
        m_set = {f.lower().strip() for f in gemma_results[i]}
        inter = len(g_set & m_set)
        union = len(g_set | m_set)
        jacc = inter / union if union else 0
        overlaps.append(jacc)
        print(f"  [{i+1}] {p['seed_disease'][:30]:30s} G={len(g_set):2d} M={len(m_set):2d} ∩={inter:2d} J={jacc:.2f}")
    print(f"\nMean Jaccard: {sum(overlaps)/len(overlaps):.2f}")
    print(f"\nResults saved to pilot/results/ie_compare_gemini_gemma.json")


if __name__ == "__main__":
    main()
