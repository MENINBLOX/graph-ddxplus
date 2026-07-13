#!/usr/bin/env python3
"""Re-IE sparse diseases with patient-questionnaire focused prompt.

Uses gemma-4-E4B via vLLM to extract patient-reportable symptoms specifically.
Different from generic IE — focuses on what patients say in a clinical Q&A:
location, character, intensity, history, common terms.
"""
from __future__ import annotations
import os, sys, json, time, math, pickle
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

OUT_EDGES = MEDKG_ROOT / "processed" / "edges_patient_focused_ie.jsonl"

# Patient-questionnaire focused IE prompt
PATIENT_IE_PROMPT = """# Task: Extract Patient-Reportable Symptoms

For the disease "{disease}", from the following text, extract clinical features that A PATIENT would self-report when answering a clinical questionnaire (NOT physician examination findings).

Focus on:
- Patient-perceived symptoms (e.g., "chest pain", "shortness of breath", "fatigue", "fever", "cough")
- Pain location/character/intensity as a patient describes
- Symptom duration/onset patterns
- Risk factor history that patient knows (smoking, prior diagnosis, family history)
- Common lay-vocabulary terms

EXCLUDE:
- Lab values, EKG findings, imaging findings (physician-observable only)
- Mechanistic terms (cytokines, receptors)
- Treatment names (medications, procedures)
- Disease name itself

# Text
Title: {title}
Abstract: {text}

# Output Format
One phenotype per line, in patient vocabulary. Max 20 items.
Example for "Pneumonia":
fever
cough
sputum production
shortness of breath
chest pain
fatigue
chills

# Output:
"""

# Sparse diseases — focus IE on these (eval CUIs)
SPARSE_DISEASES = [
    ("C0010072", "Possible NSTEMI / STEMI", ["C1304447", "C0151744", "C0027051"]),
    ("C0001175", "HIV (initial infection)", ["C0019693"]),
    ("C0478237", "Spontaneous rib fracture", ["C0035525", "C0016659"]),
    ("C0013609", "Localized edema", ["C0013604"]),
    ("C0340044", "Acute COPD exacerbation", ["C0741421", "C0024117"]),
    ("C0236832", "Acute dystonic reactions", ["C0013362"]),
    ("C0346647", "Pancreatic neoplasm", ["C0153466", "C0030297"]),
    ("C0023066", "Larygospasm", ["C0023068"]),
    ("C0039240", "PSVT", []),
    ("C0041912", "URTI", []),
    ("C0001344", "Viral pharyngitis", []),
    ("C0348343", "Pulmonary neoplasm", []),
]


def parse_phenotypes(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        if line.startswith("#") or line.startswith("//"): continue
        # Strip enumeration prefixes
        if line[0].isdigit() and line[1] in ".)-":
            line = line[2:].strip()
        if line.startswith("- "): line = line[2:]
        if line.startswith("* "): line = line[2:]
        if line.startswith("• "): line = line[2:]
        if len(line) < 3 or len(line) > 100: continue
        if line.startswith(("Disease", "Output", "Note", "Format")): continue
        out.append(line.strip(".,;:"))
    return out


def main():
    # Collect texts for sparse diseases
    records = []
    PUBMED_DIR = MEDKG_ROOT / "pubmed_alt"

    for eval_cui, disease_name, aliases in SPARSE_DISEASES:
        candidate_cuis = [eval_cui] + aliases
        n_papers = 0
        for c in candidate_cuis:
            fp = PUBMED_DIR / f"{c}.jsonl"
            if not fp.exists(): continue
            with fp.open() as f:
                for line in f:
                    try: e = json.loads(line)
                    except: continue
                    ab = (e.get("abstract") or "").strip()
                    if len(ab) < 100: continue
                    records.append({
                        "eval_cui": eval_cui,
                        "disease": disease_name,
                        "title": e.get("title", ""),
                        "abstract": ab[:2500],
                        "pmid": e.get("pmid", ""),
                    })
                    n_papers += 1
                    if n_papers >= 30: break
            if n_papers >= 30: break
        print(f"  {disease_name}: {n_papers} papers")

    print(f"\nTotal records: {len(records)}")

    if not records:
        print("No records — abort.")
        return

    print("\nLoading vLLM with gemma-4-E4B...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.90,
              enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0.0, max_tokens=350)

    n_edges = 0
    BATCH = 256
    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_EDGES.open("w") as out:
        t0 = time.time()
        for cs in range(0, len(records), BATCH):
            chunk = records[cs:cs+BATCH]
            convs = [[{"role": "user", "content": PATIENT_IE_PROMPT.format(
                disease=r["disease"], title=r["title"], text=r["abstract"]
            )}] for r in chunk]
            outs = llm.chat(convs, sampling)
            for r, o in zip(chunk, outs):
                try: text = o.outputs[0].text
                except: text = ""
                for p in parse_phenotypes(text):
                    edge = {
                        "eval_cui": r["eval_cui"],
                        "disease": r["disease"],
                        "phenotype": p,
                        "pmid": r["pmid"],
                        "title": r["title"],
                        "source": "patient_focused_ie",
                        "extracted_by": "gemma-4-E4B",
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
            print(f"  {cs+len(chunk)}/{len(records)} edges={n_edges:,} ({time.time()-t0:.0f}s)")
    print(f"\nDone. Patient-focused IE edges: {n_edges:,}")


if __name__ == "__main__":
    main()
