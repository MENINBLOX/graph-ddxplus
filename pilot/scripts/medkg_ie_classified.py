#!/usr/bin/env python3
"""Classified IE: extract phenotypes with boolean classification flags.

Each extracted phenotype is annotated with:
- is_patient_reportable: patient can answer (lay vocab, observable)
- is_anatomical_location: anatomical site (location info)
- is_clinical_sign: physician-observed sign
- is_lab_or_imaging: laboratory test or imaging finding

This lets downstream benchmarks filter phens to match their question style.

Output: /windows/data/medkg/processed/edges_classified_ie.jsonl
  {"disease", "umls_cui", "phenotype",
   "is_patient_reportable", "is_anatomical_location", "is_clinical_sign", "is_lab_or_imaging",
   "source", "source_id", "extracted_by"}

Universal: UMLS preferred names, no DDXPlus-specific prompts.
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_pubmed import log

IE_PROMPT = """# Task: Classified Phenotype Extraction

Extract diagnostic phenotypes for the disease "{disease}" from the PubMed abstract below.
For EACH phenotype, classify it on FOUR dimensions:

1. **patient_reportable** (true/false): Can a patient answer this in a questionnaire WITHOUT medical training? (e.g., "fever", "headache", "pain in shoulder" = TRUE; "rhonchi", "elevated WBC" = FALSE)
2. **anatomical_location** (true/false): Is this a body part / location of symptom? (e.g., "shoulder", "right lower abdomen", "throat" = TRUE; "fever" = FALSE)
3. **clinical_sign** (true/false): Is this observable only by a clinician with examination (e.g., "rhonchi on auscultation", "positive Murphy's sign" = TRUE; "headache" = FALSE)
4. **lab_or_imaging** (true/false): Is this a laboratory test, imaging finding, or biochemical measurement? (e.g., "elevated troponin", "ground-glass opacity on CT" = TRUE; "shortness of breath" = FALSE)

# Inclusion Criteria
A candidate qualifies only if ALL of the following hold:
- Explicitly stated in the abstract as occurring in patients with the target disease
- Specific clinical observation, not a category
- Attribute of the patient's body
- Single phenotypic concept

# Exclusion Criteria
- The disease name itself or its synonyms
- Genes, gene products, antigens, antibodies, molecular entities
- Pathophysiological/molecular mechanisms
- Treatments or interventions
- Study design or epidemiological measurements
- Other diseases for differential comparison only

# Output Format
One phenotype per line in this exact JSON format:
{{"name": "phenotype text", "patient_reportable": true/false, "anatomical_location": true/false, "clinical_sign": true/false, "lab_or_imaging": true/false}}

# Abstract Title
{title}

# Abstract
{text}

# Phenotypes (one JSON object per line)
"""


def parse_classified(text: str):
    """Parse model output: one JSON object per line."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if not line.startswith('{'):
            # Strip leading bullets, numbers
            line = re.sub(r'^[-•\d.)\s]+', '', line)
            if not line.startswith('{'): continue
        try:
            obj = json.loads(line)
        except Exception:
            # Try to extract balanced braces
            m = re.search(r'\{[^{}]*\}', line)
            if not m: continue
            try: obj = json.loads(m.group(0))
            except: continue
        name = obj.get('name', '').strip()
        if not name or len(name) > 100: continue
        out.append({
            "name": name,
            "patient_reportable": bool(obj.get('patient_reportable', False)),
            "anatomical_location": bool(obj.get('anatomical_location', False)),
            "clinical_sign": bool(obj.get('clinical_sign', False)),
            "lab_or_imaging": bool(obj.get('lab_or_imaging', False)),
        })
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(Path('pilot/data/pubmed_ddx_extra/ddxplus_anchored_v2.jsonl')))
    ap.add_argument("--output", default=str(MEDKG_ROOT / 'processed' / 'edges_classified_ie.jsonl'))
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            try: records.append(json.loads(line))
            except: pass
    log(f"Loaded {len(records):,} abstracts from {args.input}")
    if not records:
        log("No records."); return

    # Get UMLS preferred names for disease CUIs
    with open('data/ddxplus/disease_icd10_cui_mapping.json') as f: icd = json.load(f)
    cui2name = {info['cui']: dn for dn, info in icd.items() if 'cui' in info}

    log("Loading vLLM (gemma-4-E4B-it)...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=600)

    BATCH = 256
    n_edges = 0; n_done = 0
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out_path.open('w') as out:
        for chunk_start in range(0, len(records), BATCH):
            chunk = records[chunk_start:chunk_start + BATCH]
            convs = []
            for r in chunk:
                dname = cui2name.get(r['cui'], r['cui'])
                text = r['text'][:2500]
                convs.append([{"role": "user", "content": IE_PROMPT.format(
                    disease=dname, title='', text=text)}])
            outputs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outputs):
                try: text = o.outputs[0].text
                except: text = ""
                for p in parse_classified(text):
                    edge = {
                        "disease": cui2name.get(r['cui'], r['cui']),
                        "umls_cui": r['cui'],
                        "phenotype": p['name'],
                        "is_patient_reportable": p['patient_reportable'],
                        "is_anatomical_location": p['anatomical_location'],
                        "is_clinical_sign": p['clinical_sign'],
                        "is_lab_or_imaging": p['lab_or_imaging'],
                        "source": "pubmed_classified",
                        "source_id": r.get('pmid'),
                        "extracted_by": "gemma-4-E4B",
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            log(f"  {n_done}/{len(records)}  edges={n_edges:,}  rate={rate:.1f}/s")
    log(f"Done. {n_edges:,} classified edges → {out_path}")


if __name__ == "__main__":
    main()
