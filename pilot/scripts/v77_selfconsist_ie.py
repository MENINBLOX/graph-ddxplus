#!/usr/bin/env python3
"""v77 — Self-consistency percentage IE.

v76 single-pass (temp=0): P(E|D) noisy due to single sample.
v77: N samples (temp=0.7), average percentages. Reduces sampling noise,
stabilizes estimates. v318 N=10 self-consistency CoT pattern.

For each disease, batch N copies of the same prompt → average the
percentage outputs across samples per evidence.
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

EV_META = "data/ddxplus/release_evidences.json"

FEWSHOT_EXAMPLES = """\
## Examples (NOT from DDXPlus 49 — generic diseases for prompt calibration)

### Disease: Iron deficiency anemia
- Do you feel tired or fatigued? — 70
- Have you noticed pale skin? — 60
- Do you experience shortness of breath on exertion? — 55
- Do you have heavy menstrual bleeding? — 45
- Do you eat enough red meat? — 30 (low intake)
- Have you been diagnosed with chronic kidney disease? — 8
- Do you cough up blood? — 1

### Disease: Acute appendicitis
- Do you have abdominal pain? — 95
- Is the pain localized in the right lower quadrant? — 75
- Have you experienced nausea or vomiting? — 65
- Do you have a fever? — 55
- Has the pain shifted from periumbilical to right lower? — 45
- Do you have neck stiffness? — 1
- Have you noticed a skin rash? — 2

### Disease: Hypothyroidism
- Do you feel tired or have low energy? — 75
- Have you gained weight unexpectedly? — 50
- Do you feel cold often? — 50
- Do you have constipation? — 45
- Do you have dry skin? — 50
- Is your menstrual cycle irregular (female)? — 30
- Do you have severe acute pain? — 5
- Have you experienced bloody diarrhea? — 1

Note: realistic clinical prevalence varies smoothly across 1-99%. AVOID
clustering all evidences at the same percentage. Use the full range.
"""

PROMPT = """\
# Task
For each diagnostic question below, estimate the percentage (1-99) of
typical patients with the specified disease who would answer YES.

Output ONE line per question in this exact format (no extra text):
Q<index>: <percentage>

Guidelines:
- Use the FULL range (1-99), not just 15/30/75/etc.
- Vary your estimates — different evidences have different prevalences.
- 80+ = pathognomonic or near-universal
- 50-79 = common
- 20-49 = present in some patients
- 5-19 = uncommon but seen
- 1-4 = rare/atypical

{fewshot}

# Disease
{disease}

# Diagnostic questions
{questions}

Now estimate the percentage for each question of {disease}:
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_diseases", type=int, default=49)
    args = ap.parse_args()

    ev_meta = json.load(open(EV_META))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    diseases = []
    for dn in icd:
        if "cond-name-fr" in cond.get(dn, {}):
            diseases.append((dn, icd[dn]["cui"]))
    binary_evs = [(ev_id, m["question_en"])
                  for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0]
    print(f"Diseases: {len(diseases)}, binary evs: {len(binary_evs)}", flush=True)

    questions_block = "\n".join(f"Q{i+1}: {q}" for i, (_, q) in enumerate(binary_evs))

    # Build prompts (n_samples copies per disease)
    prompts = []; meta = []
    for dn, dcui in diseases[:args.n_diseases]:
        p = PROMPT.format(disease=dn, fewshot=FEWSHOT_EXAMPLES, questions=questions_block)
        for s in range(args.n_samples):
            prompts.append(p)
            meta.append({"disease": dn, "dcui": dcui, "sample": s})
    print(f"Total prompts: {len(prompts)} ({args.n_diseases} disease × {args.n_samples} samples)",
          flush=True)

    print(f"Loading vLLM (Gemma-4-E4B)...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                              top_p=0.95)

    # Collect samples per disease
    disease_samples = defaultdict(lambda: defaultdict(list))  # dcui -> ev -> [pcts]
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
                mat = re.match(r"Q(\d+):\s*(\d+)", line.strip())
                if not mat: continue
                idx = int(mat.group(1)) - 1
                pct = int(mat.group(2))
                if 0 <= idx < len(binary_evs) and 0 <= pct <= 100:
                    ev_id = binary_evs[idx][0]
                    disease_samples[m["dcui"]][ev_id].append(pct)
        elapsed = time.time() - t0
        print(f"  {i+len(chunk)}/{len(prompts)} ({elapsed:.0f}s)", flush=True)

    # Average samples
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_parsed = 0
    with open(args.out, "w") as f:
        for dn, dcui in diseases[:args.n_diseases]:
            samples = disease_samples.get(dcui, {})
            profile = {}
            for ev, pcts in samples.items():
                if pcts:
                    avg_pct = sum(pcts) / len(pcts)
                    profile[ev] = avg_pct / 100.0
            rec = {"disease": dn, "dcui": dcui, "profile": profile,
                   "n_samples": args.n_samples}
            f.write(json.dumps(rec) + "\n")
            n_parsed += len(profile)
    print(f"Saved {args.n_diseases} → {args.out}, total {n_parsed} P(E|D) entries",
          flush=True)


if __name__ == "__main__":
    main()
