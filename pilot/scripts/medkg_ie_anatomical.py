#!/usr/bin/env python3
"""Anatomical IE: extract typical body locations of symptoms per disease.

GT KG analysis showed our IE misses anatomical location (Pneumonia → Shoulder,
Elbow, Index finger). DDXPlus questionnaire asks "where on body is pain/lesion".

This prompt explicitly extracts ANATOMICAL LOCATIONS where symptoms occur.
"""
from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT
from medkg_ie_pubmed import log

IE_PROMPT = """# Task: Anatomical Location Extraction

For the disease "{disease}", extract ANATOMICAL LOCATIONS where symptoms (pain, swelling, rash, lesion, discomfort) typically occur.

# Output Format

For each location mentioned in the abstract for this disease, output:
LOCATION | SYMPTOM_TYPE

Where SYMPTOM_TYPE is one of: pain, swelling, rash, lesion, discomfort, tenderness, stiffness, bleeding, redness

# Examples
chest | pain
right lower abdomen | pain
left flank | pain
knee | swelling
trunk | rash
hands and feet | lesion
upper back | discomfort

# Inclusion
- Only anatomical body parts where the symptom occurs
- Side (left/right) and region (upper/lower) if specified
- Single anatomical concept per line

# Exclusion
- Generic terms ("somewhere", "the body")
- Anatomical structures inside the body that the patient cannot indicate (e.g., "myocardium", "bronchi")
- Imaging or laboratory findings

# Abstract
{text}

# Locations (LOCATION | SYMPTOM_TYPE, one per line):
"""


def parse(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or '|' not in line: continue
        line = re.sub(r'^[-•\d.)]+\s*', '', line).strip()
        parts = [p.strip() for p in line.split('|')]
        if len(parts) != 2: continue
        loc, sym = parts
        if not loc or len(loc) > 50: continue
        if not sym or len(sym) > 30: continue
        if any(b in loc.lower() for b in ['somewhere', 'the body', 'unspecified']): continue
        out.append({"location": loc, "symptom": sym.lower()})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(Path('pilot/data/pubmed_ddx_extra/ddxplus_anchored_v2.jsonl')))
    ap.add_argument("--output", default=str(MEDKG_ROOT / 'processed' / 'edges_anatomical_ie.jsonl'))
    args = ap.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            try: records.append(json.loads(line))
            except: pass
    log(f"Loaded {len(records):,} abstracts")
    if not records: return

    with open('data/ddxplus/disease_icd10_cui_mapping.json') as f: icd = json.load(f)
    cui2name = {info['cui']: dn for dn, info in icd.items() if 'cui' in info}

    log("Loading vLLM...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.95,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=300)

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
                convs.append([{"role": "user", "content": IE_PROMPT.format(
                    disease=dname, text=r['text'][:2500])}])
            outputs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outputs):
                try: text = o.outputs[0].text
                except: text = ""
                for p in parse(text):
                    edge = {
                        "disease": cui2name.get(r['cui'], r['cui']),
                        "umls_cui": r['cui'],
                        "location": p['location'],
                        "symptom": p['symptom'],
                        "phenotype": f"{p['symptom']} {p['location']}",  # combined for CUI match
                        "source": "pubmed_anatomical",
                        "source_id": r.get('pmid'),
                        "extracted_by": "gemma-4-E4B",
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
                n_done += 1
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            log(f"  {n_done}/{len(records)}  edges={n_edges:,}  rate={rate:.1f}/s")
    log(f"Done. {n_edges:,} anatomical edges")


if __name__ == "__main__":
    main()
