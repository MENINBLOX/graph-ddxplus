#!/usr/bin/env python3
"""Multi-source LLM IE: extract disease–phenotype edges from sections.jsonl.

Uses gemma-4-E4B-it via vLLM. Same v3 principle-only prompt across all sources.
Each section becomes one IE call; output records (disease, phenotype, source, source_id, section).
"""
from __future__ import annotations
import os, json, re, time, sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

ROOT = Path("/home/max/Graph-DDXPlus/data/medkg")
SEC_PATH = ROOT / "processed" / "sections.jsonl"
OUT_PATH = ROOT / "processed" / "edges_ie.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# v3 principles-only prompt (refined for textbook content)
IE_PROMPT = """# Task: Clinical Phenotype Extraction

Extract diagnostic phenotypes (clinical features) for the disease "{disease}" from the following medical text.

# Definition

A diagnostic phenotype is a clinically observable abnormality of an individual patient that a clinician evaluates when establishing a diagnosis. It belongs to exactly one of the following observation modalities:
- A subjective sensation or complaint reported directly by the patient
- A physical finding objectively observed by a clinician during examination
- A quantitative biochemical, hematological, or immunological abnormality measured in body fluids or tissues
- An abnormal structural or functional finding detected by medical imaging
- A microscopic or gross abnormality identified in tissue specimens

# Inclusion Criteria

A candidate qualifies only if ALL of the following hold:
- It is explicitly stated in the text as occurring in patients with the target disease
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

Express each phenotype using canonical medical terminology consistent with controlled biomedical vocabularies. Normalize non-canonical phrasing to its canonical form when the canonical form is unambiguous; otherwise reuse the text's wording.

Output exactly one phenotype per line, each line in the form:
PHENOTYPE: <term>

The output must consist solely of such PHENOTYPE lines. Do not produce any analytical commentary, reasoning, headers, bullet markers, numbering, or surrounding text.

# Source Type

This text comes from: {source_type}
Section: {section_name}

# Text

{text}

# Extracted Phenotypes for {disease}"""


def parse_phenotypes(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"PHENOTYPE\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            f = m.group(1).strip().rstrip(".,;:")
            f = re.sub(r"^[\*\-•\d\.\s]+", "", f).strip()
            if f and 2 < len(f) < 100:
                out.append(f)
    return out


def main():
    # Load sections
    sections = []
    with SEC_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                sections.append(json.loads(line))
            except Exception:
                continue
    print(f"Loaded {len(sections)} sections")

    # Truncate long texts (max 3000 chars to keep within model context)
    for s in sections:
        s["text_for_ie"] = s["text"][:3000]

    # Build prompts
    prompts = []
    for s in sections:
        prompts.append(IE_PROMPT.format(
            disease=s["disease"],
            source_type=s["source"],
            section_name=s.get("section_name", "unknown"),
            text=s["text_for_ie"],
        ))

    print(f"Built {len(prompts)} prompts")

    # vLLM batch
    from vllm import LLM, SamplingParams
    print("Loading gemma-4-E4B-it...")
    llm = LLM(
        model="google/gemma-4-E4B-it",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    sampling = SamplingParams(temperature=0, max_tokens=600)

    # Batch in groups of 1024 to avoid memory pressure
    print(f"Running IE batch...")
    convs = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(convs, sampling)

    n_edges = 0
    with OUT_PATH.open("w") as out:
        for s, o in zip(sections, outputs):
            try:
                resp_text = o.outputs[0].text
            except Exception:
                resp_text = ""
            findings = parse_phenotypes(resp_text)
            for f in findings:
                edge = {
                    "disease": s["disease"],
                    "umls_cui": s.get("umls_cui"),
                    "phenotype": f,
                    "source": s["source"],
                    "source_id": s.get("source_id"),
                    "section_name": s.get("section_name"),
                    "section_id": s.get("section_id"),
                    "extracted_by": "gemma-4-E4B",
                }
                # Keep additional provenance fields when available
                for k in ("pmid", "revid", "chapter_title", "topic_title", "title"):
                    if s.get(k):
                        edge[k] = s[k]
                out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                n_edges += 1

    print(f"\nExtracted {n_edges} disease-phenotype edges → {OUT_PATH}")


if __name__ == "__main__":
    main()
