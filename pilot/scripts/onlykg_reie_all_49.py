#!/usr/bin/env python3
"""Re-IE ALL 49 DDXPlus diseases with patient-focused prompt.

v15 covered only 10 sparse. v16 covers all 49 for thorough KG patient-vocab.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

OUT_EDGES = MEDKG_ROOT / "processed" / "edges_patient_focused_all49_ie.jsonl"

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

# Output:
"""


def parse_phenotypes(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line: continue
        if line.startswith("#") or line.startswith("//"): continue
        if line[0].isdigit() and len(line) > 1 and line[1] in ".)-":
            line = line[2:].strip()
        if line.startswith("- "): line = line[2:]
        if line.startswith("* "): line = line[2:]
        if line.startswith("• "): line = line[2:]
        if len(line) < 3 or len(line) > 100: continue
        if line.lower().startswith(("disease", "output", "note", "format", "patient")): continue
        out.append(line.strip(".,;:"))
    return out


def main():
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd = json.load(f)
    ddx_cuis = {info["cui"]: dn for dn, info in icd.items() if "cui" in info}
    print(f"DDXPlus 49 disease CUIs: {len(ddx_cuis)}")

    # Manual aliases for diseases that lack pubmed_alt files
    ALIASES = {
        "C0010072": ["C1304447", "C0151744", "C0027051"],
        "C0001175": ["C0019693"],
        "C0478237": ["C0035525", "C0016659"],
        "C0013609": ["C0013604"],
        "C0340044": ["C0741421", "C0024117"],
        "C0236832": ["C0013362"],
        "C0346647": ["C0153466", "C0030297"],
        "C0023066": ["C0023068"],
        "C0001344": ["C0031350"],   # Viral pharyngitis → Pharyngitis
        "C0023067": ["C0023067"],
        "C0002871": ["C0002871"],
        "C0002792": ["C0002792"],
    }

    PUBMED_DIR = MEDKG_ROOT / "pubmed_alt"

    records = []
    for eval_cui, disease in ddx_cuis.items():
        candidate_cuis = [eval_cui] + ALIASES.get(eval_cui, [])
        for c in candidate_cuis:
            fp = PUBMED_DIR / f"{c}.jsonl"
            if not fp.exists(): continue
            with fp.open() as f:
                n = 0
                for line in f:
                    try: e = json.loads(line)
                    except: continue
                    ab = (e.get("abstract") or "").strip()
                    if len(ab) < 100: continue
                    records.append({
                        "eval_cui": eval_cui,
                        "disease": disease,
                        "title": e.get("title", ""),
                        "abstract": ab[:2500],
                        "pmid": e.get("pmid", ""),
                    })
                    n += 1
                    if n >= 50: break  # 50 papers per CUI variant
    print(f"Total records: {len(records)}")

    print("\nLoading vLLM...")
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.90,
              enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0.0, max_tokens=350)

    n_edges = 0
    BATCH = 256
    OUT_EDGES.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with OUT_EDGES.open("w") as out:
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
                    }
                    out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                    n_edges += 1
            print(f"  {cs+len(chunk)}/{len(records)} edges={n_edges:,} ({time.time()-t0:.0f}s)")
    print(f"\nDone. All-49 patient IE edges: {n_edges:,}")


if __name__ == "__main__":
    main()
