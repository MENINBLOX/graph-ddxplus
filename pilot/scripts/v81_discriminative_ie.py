#!/usr/bin/env python3
"""v81 — Discriminative IE prompt: avoid generic, emphasize disease-specific.

v80 forensic: Anaphylaxis became broad attractor because LLM added generic
findings (fatigue, pain, swelling) to many diseases → over-recall.

v81 fix: prompt explicitly asks for DISCRIMINATIVE findings only,
de-emphasizing generic symptoms that appear in many diseases.

Same academic constraints:
- disease name only (no benchmark question/symptom)
- benchmark-blind few-shot
- LLM as medical knowledge base (textbook equivalent)
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


FEWSHOT = """\
## Examples (generic diseases for prompt calibration — NOT from any specific benchmark)

### Disease: Iron deficiency anemia
- Pallor of conjunctiva — 60
- Microcytic anemia on CBC — 80 (but exclude lab findings)
- Heavy menstrual bleeding (women of reproductive age) — 45
- Pica (craving ice or non-food) — 15
- Koilonychia (spoon-shaped nails) — 10
- Glossitis (smooth shiny tongue) — 15
- Angular cheilitis — 8
- Restless legs — 15
- Exertional dyspnea — 50
- Hair loss — 25

### Disease: Acute appendicitis
- Periumbilical pain shifting to right lower quadrant — 75
- Rebound tenderness in right lower quadrant — 60
- McBurney point tenderness — 70
- Rovsing sign positive — 30
- Psoas sign positive — 25
- Obturator sign positive — 15
- Anorexia followed by nausea — 70
- Low-grade fever — 50
- Right lower quadrant guarding — 50

### Disease: Hypothyroidism
- Cold intolerance — 60
- Bradycardia (slow heart rate) — 30
- Delayed deep tendon reflex relaxation — 35
- Periorbital edema — 25
- Hoarse voice — 25
- Dry coarse skin — 50
- Brittle hair — 30
- Goiter (enlarged thyroid) — 15
- Macroglossia (enlarged tongue) — 10
- Carpal tunnel syndrome — 15
"""


PROMPT_TEMPLATE = """\
# Task
You are a medical knowledge base. For the disease "{disease}", list
**discriminative** symptoms, signs, and clinical features — those that
help **distinguish this disease from other diseases**.

For each, estimate the percentage (1-99) of typical patients exhibiting it.

CRITICAL RULES (de-emphasize generic findings):
- AVOID overly generic findings unless they are unusually characteristic:
  - "fatigue", "headache", "pain", "feeling ill", "weakness" — only list if HIGHLY specific to this disease
- INSTEAD focus on:
  - Disease-specific signs (e.g. "Rovsing sign" for appendicitis, "Koilonychia" for iron deficiency)
  - Specific anatomical locations (e.g. "right lower quadrant pain", not just "abdominal pain")
  - Pathognomonic findings (e.g. "Cherry-red lips" for CO poisoning)
  - Specific qualifiers (e.g. "exertional dyspnea", not just "dyspnea")
  - Demographic/historical specifics (e.g. "smoking history" for COPD)
- DO NOT list:
  - Diagnostic tests (CBC, X-ray, biopsy, etc.)
  - Treatments or medications
  - Generic symptoms with no qualifier
- INCLUDE 12-25 discriminative findings

Output rules:
- One line per finding: `- <finding> — <percentage>`
- Use standard medical or lay terminology

{fewshot}

### Disease: {disease}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    diseases = []
    seen = set()
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    for dn in icd:
        if dn not in seen:
            diseases.append((dn, "ddxplus")); seen.add(dn)
    parsed = json.load(open("data/symcat/symcat_parsed_full.json"))
    for dn in parsed["disease_symptom_pairs"]:
        if dn not in seen:
            diseases.append((dn, "symcat")); seen.add(dn)
    print(f"Disease pool: {len(diseases)}", flush=True)

    prompts = []; meta = []
    for dn, src in diseases:
        p = PROMPT_TEMPLATE.format(disease=dn, fewshot=FEWSHOT)
        for s in range(args.n_samples):
            prompts.append(p)
            meta.append({"disease": dn, "source": src, "sample": s})
    print(f"Total prompts: {len(prompts)}", flush=True)

    print("Loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=args.temperature,
                              max_tokens=args.max_tokens, top_p=0.95)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    disease_data = defaultdict(lambda: defaultdict(list))
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
                phen = mat.group(1).strip().rstrip(".,;:()").strip()
                if not phen or len(phen) < 3 or len(phen) > 120: continue
                low = phen.lower()
                if any(skip in low for skip in [
                    "test", "treatment", "medication", "antibiotic", "therapy",
                    "diagnosis", "imaging", "biopsy", "x-ray", "ultrasound",
                    "scan", "blood test", "panel", "level of"
                ]): continue
                pct = int(mat.group(2))
                if not (1 <= pct <= 99): continue
                disease_data[m["disease"]][low].append(pct)
        if (i // args.batch) % 5 == 0:
            print(f"  {i+len(chunk)}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)

    n_records = 0
    with open(args.out, "w") as f:
        for dn, src in diseases:
            phens = disease_data.get(dn, {})
            agg = {phen: {"prob": sum(pcts)/len(pcts)/100.0, "n_seen": len(pcts)}
                   for phen, pcts in phens.items()}
            rec = {"disease": dn, "source": src, "n_samples": args.n_samples,
                   "phenotypes": agg}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records += len(agg)
    print(f"\nSaved {len(diseases)} disease records → {args.out}", flush=True)
    print(f"Total phenotype entries: {n_records}", flush=True)


if __name__ == "__main__":
    main()
