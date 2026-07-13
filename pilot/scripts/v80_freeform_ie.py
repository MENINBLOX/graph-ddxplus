#!/usr/bin/env python3
"""v80 — Benchmark-blind free-form LLM IE.

Academic justification:
- LLM used ONLY as medical knowledge base (equivalent to textbook/UpToDate IE)
- Input: disease name ONLY (NO DDXPlus questions, NO SymCat symptom list)
- Output: free-form medical phenotype text + prevalence rating
- Few-shot examples: generic medical diseases (NOT in DDXPlus 49)
- This is benchmark-blind: a single KG is built and tested on ALL benchmarks
- LLM is NOT used at inference time (only IE step)

Source diseases:
- DDXPlus 49 (English names)
- SymCat 801 (English names)
- Total: ~850 (with duplicates removed)

Output format per disease:
- phenotype_name (lay or clinical), prevalence_pct
- Will be mapped to CUI via scispaCy in next step
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


FEWSHOT = """\
## Examples (generic diseases for prompt calibration — NOT from any specific benchmark)

### Disease: Iron deficiency anemia
- Fatigue — 70
- Pale skin — 60
- Pale conjunctiva — 60
- Shortness of breath on exertion — 55
- Heavy menstrual bleeding (women) — 45
- Brittle nails — 35
- Headache — 30
- Cold hands and feet — 25
- Pica (craving ice or non-food) — 15
- Restless legs — 15
- Koilonychia (spoon-shaped nails) — 10
- Glossitis — 5

### Disease: Acute appendicitis
- Right lower quadrant abdominal pain — 95
- Periumbilical pain that shifts to right lower quadrant — 75
- Anorexia (loss of appetite) — 70
- Nausea — 65
- Vomiting — 55
- Low-grade fever — 50
- Rebound tenderness on physical exam — 60
- Guarding on palpation — 50
- Constipation — 25
- Diarrhea — 20

### Disease: Hypothyroidism
- Fatigue — 75
- Cold intolerance — 60
- Weight gain — 50
- Constipation — 50
- Dry skin — 50
- Hair loss — 40
- Bradycardia — 30
- Hoarseness of voice — 25
- Depression — 30
- Muscle cramps — 25
- Menstrual irregularity (women) — 25
- Goiter — 15
"""


PROMPT_TEMPLATE = """\
# Task
You are a medical knowledge base. For the disease "{disease}", list typical
symptoms, signs, and clinical features that a patient may report or that
would be noted on examination. For each, estimate the percentage (1-99) of
typical patients exhibiting it.

Output rules:
- One line per finding: `- <finding name> — <percentage>`
- Use standard medical or lay terminology (whichever is more common)
- Vary percentages across the FULL range (1-99), avoid clustering
- Include 15-30 most relevant findings
- Do NOT include benchmark-specific phrasing or question-style wording
- Do NOT include treatments, medications, or diagnostic tests

{fewshot}

### Disease: {disease}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--disease_source", default="all",
                    choices=["all", "ddxplus", "symcat"])
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    # Collect diseases
    diseases = []  # list of (name, source_tag)
    seen_names = set()

    if args.disease_source in ("all", "ddxplus"):
        with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
            icd = json.load(f)
        for dn in icd:
            if dn not in seen_names:
                diseases.append((dn, "ddxplus"))
                seen_names.add(dn)

    if args.disease_source in ("all", "symcat"):
        parsed = json.load(open("data/symcat/symcat_parsed_full.json"))
        for dn in parsed["disease_symptom_pairs"]:
            if dn not in seen_names:
                diseases.append((dn, "symcat"))
                seen_names.add(dn)

    if args.limit:
        diseases = diseases[:args.limit]
    print(f"Disease pool: {len(diseases)} ({sum(1 for _, s in diseases if s=='ddxplus')} DDXPlus, "
          f"{sum(1 for _, s in diseases if s=='symcat')} SymCat-only)", flush=True)

    # Build prompts: each disease × n_samples (self-consistency)
    prompts = []; meta = []
    for dn, src in diseases:
        p = PROMPT_TEMPLATE.format(disease=dn, fewshot=FEWSHOT)
        for s in range(args.n_samples):
            prompts.append(p)
            meta.append({"disease": dn, "source": src, "sample": s})
    print(f"Total prompts: {len(prompts)}", flush=True)

    print("Loading vLLM (Gemma-4-E4B)...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=args.temperature,
                              max_tokens=args.max_tokens, top_p=0.95)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Aggregate over samples per disease
    from collections import defaultdict
    disease_data = defaultdict(lambda: defaultdict(list))  # disease -> phen_name -> [pcts]
    line_re = re.compile(r"-\s*([^—\-]+?)\s*[—\-]\s*(\d+)")

    t0 = time.time()
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i+args.batch]
        meta_chunk = meta[i:i+args.batch]
        convs = [[{"role": "user", "content": p}] for p in chunk]
        outs = llm.chat(convs, sampling)
        for m, o in zip(meta_chunk, outs):
            try: text = o.outputs[0].text
            except: text = ""
            for line in text.split("\n"):
                line = line.strip()
                mat = line_re.match(line)
                if not mat: continue
                phen_name = mat.group(1).strip().rstrip(".,;:()").strip()
                if not phen_name or len(phen_name) < 3 or len(phen_name) > 100: continue
                # Skip if it looks like a test/medication/etc
                low = phen_name.lower()
                if any(skip in low for skip in [
                    "test", "treatment", "medication", "antibiotic", "therapy",
                    "diagnosis", "imaging", "biopsy", "x-ray", "ultrasound",
                    "scan", "blood test", "panel", "level of"
                ]):
                    continue
                pct = int(mat.group(2))
                if not (1 <= pct <= 99): continue
                disease_data[m["disease"]][phen_name.lower()].append(pct)
        elapsed = time.time() - t0
        if (i // args.batch) % 5 == 0:
            print(f"  {i+len(chunk)}/{len(prompts)} ({elapsed:.0f}s)", flush=True)

    # Average over samples
    n_records = 0
    with open(args.out, "w") as f:
        for dn, src in diseases:
            phens = disease_data.get(dn, {})
            agg = {}
            for phen, pcts in phens.items():
                if len(pcts) >= 1:  # appeared in at least 1/n_samples
                    agg[phen] = {
                        "prob": sum(pcts) / len(pcts) / 100.0,
                        "n_seen": len(pcts),
                    }
            rec = {"disease": dn, "source": src, "n_samples": args.n_samples,
                   "phenotypes": agg}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records += len(agg)
    print(f"\nSaved {len(diseases)} disease records → {args.out}", flush=True)
    print(f"Total phenotype entries: {n_records}", flush=True)


if __name__ == "__main__":
    main()
