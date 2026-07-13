#!/usr/bin/env python3
"""v74 — Direct IE: LLM estimates P(E|D) for DDXPlus evidence list.

Goal: produce a KG profile that mimics train-label NB statistics WITHOUT
using train labels. The LLM uses only:
  1. Disease name (from official DDXPlus 49)
  2. DDXPlus evidence list (208 binary questions)
  3. Few-shot examples (benchmark-blind, generic diseases NOT in DDXPlus 49)

For each disease, the LLM produces a prevalence rating (always/common/
frequent/occasional/uncommon/rare/never) for each binary evidence.
These ratings map to numeric P(E|D) and replace the PubMed-derived
profile.

NO train labels — only LLM prior medical knowledge.
"""
from __future__ import annotations
import os, sys, json, csv, argparse, re, time
from pathlib import Path
sys.path.insert(0, "pilot/scripts")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

EV_META = "data/ddxplus/release_evidences.json"

PREV_MAP = {
    "always": 0.95, "common": 0.75, "frequent": 0.55,
    "occasional": 0.30, "uncommon": 0.15, "rare": 0.05, "never": 0.001,
}


FEWSHOT_EXAMPLES = """\
## Examples (NOT from DDXPlus 49 list — generic diseases for prompt calibration)

### Disease: Iron deficiency anemia
- Do you feel tired or fatigued? — common (0.75)
- Have you noticed pale skin? — common (0.75)
- Do you experience shortness of breath on exertion? — frequent (0.55)
- Do you have heavy menstrual bleeding? — common (0.75)
- Have you been diagnosed with chronic kidney disease? — uncommon (0.15)
- Do you cough up blood? — never (0.001)

### Disease: Acute appendicitis
- Do you have abdominal pain? — always (0.95)
- Is the pain localized in the right lower quadrant? — common (0.75)
- Do you have a fever? — common (0.75)
- Have you experienced nausea or vomiting? — common (0.75)
- Do you have neck stiffness? — never (0.001)
- Have you noticed a skin rash? — rare (0.05)

### Disease: Hypothyroidism
- Do you feel tired or have low energy? — common (0.75)
- Have you gained weight unexpectedly? — frequent (0.55)
- Do you feel cold often? — frequent (0.55)
- Do you have constipation? — common (0.75)
- Do you have severe acute pain? — rare (0.05)
- Have you experienced bloody diarrhea? — never (0.001)
"""


PROMPT_TEMPLATE = """\
# Task
Estimate the clinical prevalence (probability that a TYPICAL patient with
the disease answers YES) for each diagnostic question below.

Use these prevalence categories with their numeric mappings:
- always (0.95): pathognomonic, ≥90% of patients
- common (0.75): 60-89% of patients
- frequent (0.55): 40-59% of patients
- occasional (0.30): 20-39% of patients
- uncommon (0.15): 5-19% of patients
- rare (0.05): 1-4% of patients
- never (0.001): clinically inappropriate or extremely rare (<1%)

{fewshot}

# Disease
{disease}

# Diagnostic questions (answer prevalence for each)
{questions}

# Output format
Output ONE line per question in this exact format (no extra text):
Q<index>: <category>

Example:
Q1: common
Q2: rare
Q3: never
...

Output your prevalence estimates now:
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_diseases", type=int, default=49, help="for quick testing")
    args = ap.parse_args()

    ev_meta = json.load(open(EV_META))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)

    # 49 diseases (English names)
    diseases = []
    for dn in icd:
        if "cond-name-fr" in cond.get(dn, {}):
            diseases.append((dn, icd[dn]["cui"]))
    print(f"Diseases: {len(diseases)}", flush=True)

    # All binary evidence questions (208)
    binary_evs = [(ev_id, m["question_en"])
                  for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0]
    print(f"Binary evidences: {len(binary_evs)}", flush=True)

    # Build prompts
    questions_block = "\n".join(f"Q{i+1}: {q}" for i, (_, q) in enumerate(binary_evs))
    prompts = []
    meta = []
    for dn, dcui in diseases[:args.n_diseases]:
        prompt = PROMPT_TEMPLATE.format(
            disease=dn, fewshot=FEWSHOT_EXAMPLES, questions=questions_block
        )
        prompts.append(prompt)
        meta.append({"disease": dn, "dcui": dcui})

    print(f"Prepared {len(prompts)} disease prompts", flush=True)
    print("Loading vLLM (Gemma-4-E4B)...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_parsed = 0
    t0 = time.time()
    with open(args.out, "w") as f:
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i+args.batch]
            meta_chunk = meta[i:i+args.batch]
            convs = [[{"role": "user", "content": p}] for p in chunk]
            outs = llm.chat(convs, sampling)
            for m, o in zip(meta_chunk, outs):
                try: text = o.outputs[0].text
                except: text = ""
                # Parse Q<i>: <category>
                profile = {}
                for line in text.split("\n"):
                    mat = re.match(r"Q(\d+):\s*([a-z_]+)", line.strip(), re.IGNORECASE)
                    if not mat: continue
                    idx = int(mat.group(1)) - 1
                    cat = mat.group(2).lower().strip()
                    if 0 <= idx < len(binary_evs) and cat in PREV_MAP:
                        ev_id = binary_evs[idx][0]
                        profile[ev_id] = (cat, PREV_MAP[cat])
                rec = {"disease": m["disease"], "dcui": m["dcui"],
                       "profile": profile, "raw_len": len(text)}
                f.write(json.dumps(rec) + "\n")
                n_parsed += len(profile)
            elapsed = time.time() - t0
            print(f"  {i+len(chunk)}/{len(prompts)} ({elapsed:.0f}s) "
                  f"parsed={n_parsed}", flush=True)
    print(f"Saved {len(prompts)} disease profiles → {args.out}", flush=True)
    print(f"Total parsed P(E|D) entries: {n_parsed}", flush=True)


if __name__ == "__main__":
    main()
