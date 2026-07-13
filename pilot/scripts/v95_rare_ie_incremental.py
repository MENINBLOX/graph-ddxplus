#!/usr/bin/env python3
"""v95 — Exhaustive LLM IE for rare diseases with INCREMENTAL save.

Same as v95_rare_ie.py but writes outputs every N batches (resumable on kill).
Each batch's results are appended to --out. If --out already exists, the
disease names found are skipped (resume mode).

CLAUDE.md compliance: identical to v95_rare_ie.py (benchmark-blind, exhaustive).
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


FEWSHOT = """\
## Examples (generic medical diseases for prompt calibration — NOT from any evaluation benchmark)

### Disease: Iron deficiency anemia
- Fatigue — 70
- Tiredness — 65
- Weakness — 60
- Pallor of skin — 60
- Pallor of conjunctiva — 60
- Pale lips — 50
- Pale palms — 40
- Heavy menstrual bleeding (women of reproductive age) — 45
- Heavy periods — 45
- Menorrhagia — 35
- Pica (craving ice or non-food) — 15
- Craving ice — 12
- Eating ice — 10
- Koilonychia (spoon-shaped nails) — 10
- Brittle nails — 25
- Spoon-shaped nails — 8
- Glossitis (smooth shiny tongue) — 15
- Sore tongue — 18
- Beefy red tongue — 8
- Angular cheilitis — 8
- Cracked corners of mouth — 12
- Restless legs syndrome — 15
- Leg restlessness at night — 15
- Exertional dyspnea — 50
- Shortness of breath on exertion — 50
- Shortness of breath with walking — 35
- Hair loss — 25
- Brittle hair — 20
- Dry skin — 30
- Itchy skin — 18
- Headache — 30
- Dizziness — 25
- Lightheadedness — 20
- Cold hands — 28
- Cold feet — 26
- Tinnitus (ringing in ears) — 10
- Decreased appetite — 18
- Difficulty concentrating — 30
- Brain fog — 20
- Chest pain (with severe anemia) — 12
- Palpitations — 22
- Rapid heartbeat — 20

### Disease: Acute appendicitis
- Periumbilical pain shifting to right lower quadrant — 75
- Right lower quadrant pain — 80
- Lower abdominal pain — 70
- Sharp abdominal pain — 55
- McBurney point tenderness — 70
- Rebound tenderness — 60
- Guarding on palpation — 50
- Rovsing sign positive — 30
- Psoas sign positive — 25
- Obturator sign positive — 15
- Anorexia (loss of appetite) — 70
- Loss of appetite — 70
- Decreased appetite — 65
- Nausea — 65
- Vomiting — 55
- Low-grade fever — 50
- Fever — 45
- Constipation — 25
- Diarrhea — 20
- Bloating — 18
- Abdominal distension — 15
- Pain worsened with movement — 50
- Pain worsened with coughing — 45
- Limping or hunched posture — 30
- Inability to lie flat — 25
- Tachycardia — 35
- Rapid heartbeat — 30
- Sweating — 20

### Disease: Hypothyroidism
- Fatigue — 75
- Tiredness — 70
- Lethargy — 50
- Cold intolerance — 60
- Feeling cold often — 55
- Cold hands — 35
- Cold feet — 35
- Weight gain — 50
- Unexplained weight gain — 45
- Constipation — 50
- Dry skin — 55
- Coarse skin — 30
- Hair loss — 40
- Hair thinning — 35
- Brittle hair — 30
- Brittle nails — 25
- Hoarse voice — 25
- Voice deepening — 18
- Bradycardia (slow heart rate) — 30
- Slow pulse — 28
- Periorbital edema — 25
- Puffy face — 30
- Macroglossia (enlarged tongue) — 10
- Goiter (enlarged thyroid) — 15
- Neck swelling — 18
- Depression — 30
- Sadness — 28
- Brain fog — 35
- Memory problems — 28
- Difficulty concentrating — 35
- Muscle cramps — 25
- Muscle stiffness — 22
- Joint stiffness — 25
- Joint pain — 22
- Menstrual irregularity (women) — 25
- Heavy periods — 18
- Carpal tunnel syndrome — 15
- Numbness in hands — 18
- Reduced libido — 25
- Delayed deep tendon reflex relaxation — 35
"""


PROMPT_TEMPLATE = """\
# Task
You are a medical knowledge base. For the disease "{disease}", list ALL
symptoms, signs, and clinical features that a patient may report or that
would be noted on examination. Be EXHAUSTIVE — include every relevant
finding regardless of frequency.

For each, estimate the percentage (1-99) of typical patients exhibiting it.

CRITICAL RULES:
- EXHAUSTIVE: Do NOT limit the number. List as many findings as you can
  recall, including rare and unusual presentations.
- MULTIPLE FORMS for the same concept (LAY + CLINICAL + SYNONYMS):
  - "Shortness of breath" + "Dyspnea" + "Difficulty breathing"
  - "Pale skin" + "Pallor"
  - "Belly pain" + "Abdominal pain"
- Include ANATOMICAL specificity:
  - "Right lower quadrant pain" (not just "abdominal pain")
  - "Periorbital edema" (not just "edema")
- Include ONSET/SEVERITY/DURATION qualifiers:
  - "Sudden onset chest pain"
  - "Severe headache"
  - "Chronic cough"
- Include DEMOGRAPHIC associations:
  - "More common in women over 50"
  - "History of smoking"
- Include EXAM findings (signs visible to clinician):
  - "Murphy sign positive", "Rebound tenderness", "Murmur"
- DO NOT include:
  - Diagnostic tests (CBC, X-ray, MRI, biopsy)
  - Treatments or medications
  - Disease names that are NOT phenotypes of this disease

Output rules:
- One line per finding: `- <finding> — <percentage>`
- Vary percentages across the FULL range (1-99)

{fewshot}

### Disease: {disease}
List exhaustively below.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pilot/data/cache/v95_rare_pool.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)
    ap.add_argument("--save_every", type=int, default=20,
                    help="Append-save every N disease groups")
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    diseases_all = [(p["name"], "rarebench") for p in pool]
    end = args.end_idx if args.end_idx >= 0 else len(diseases_all)
    diseases = diseases_all[args.start_idx:end]
    print(f"This shard: [{args.start_idx}:{end}] = {len(diseases)} diseases", flush=True)

    # Resume: skip diseases already in --out
    done_names = set()
    outp = Path(args.out)
    if outp.exists():
        with open(args.out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_names.add(r["disease"])
                except Exception: pass
        print(f"Resume: already done {len(done_names)} in {args.out}", flush=True)

    todo = [(dn, src) for dn, src in diseases if dn not in done_names]
    print(f"To process: {len(todo)} diseases", flush=True)
    if not todo:
        print("Nothing to do.", flush=True); return

    # Build all prompts (each disease repeated n_samples times)
    prompts = []; meta = []
    for dn, src in todo:
        p = PROMPT_TEMPLATE.format(disease=dn, fewshot=FEWSHOT)
        for s in range(args.n_samples):
            prompts.append(p)
            meta.append({"disease": dn, "source": src, "sample": s})
    print(f"Total prompts: {len(prompts)}", flush=True)

    print("Loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=6144, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=args.temperature,
                              max_tokens=args.max_tokens, top_p=0.95)

    outp.parent.mkdir(parents=True, exist_ok=True)
    line_re = re.compile(r"-\s*([^—\-]+?)\s*[—\-]\s*(\d+)")
    t0 = time.time()

    # Buffer: disease -> phen_lower -> [pcts]
    disease_data = defaultdict(lambda: defaultdict(list))
    # Track which sample-counts we've seen per disease — flush only when full
    disease_sample_seen = defaultdict(set)

    saved_count = 0
    fout = open(args.out, "a")

    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i+args.batch]
        meta_chunk = meta[i:i+args.batch]
        convs = [[{"role": "user", "content": p}] for p in chunk]
        outs = llm.chat(convs, sampling)
        for m, o in zip(meta_chunk, outs):
            try: text = o.outputs[0].text
            except Exception: text = ""
            for line in text.split("\n"):
                line = line.strip()
                mat = line_re.match(line)
                if not mat: continue
                phen = mat.group(1).strip().rstrip(".,;:()").strip()
                if not phen or len(phen) < 3 or len(phen) > 150: continue
                low = phen.lower()
                if any(skip in low for skip in [
                    " test", "x-ray", "ultrasound", "scan ", "biopsy",
                    "diagnostic ", "treatment", "medication", "antibiotic",
                    "therapy", "vaccine", " mri", " ct ", "panel", "level of",
                    "blood test"
                ]): continue
                pct = int(mat.group(2))
                if not (1 <= pct <= 99): continue
                disease_data[m["disease"]][low].append(pct)
            disease_sample_seen[m["disease"]].add(m["sample"])

        # Flush any disease where we've seen all n_samples
        ready = [dn for dn, seen in list(disease_sample_seen.items())
                 if len(seen) >= args.n_samples]
        if ready and len(ready) >= args.save_every:
            for dn in ready:
                phens = disease_data.get(dn, {})
                src = next((m["source"] for m in meta_chunk if m["disease"] == dn),
                           "rarebench")
                agg = {phen: {"prob": sum(pcts)/len(pcts)/100.0, "n_seen": len(pcts)}
                       for phen, pcts in phens.items()}
                rec = {"disease": dn, "source": src, "n_samples": args.n_samples,
                       "phenotypes": agg}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # cleanup (safe pop — disease_data may be empty if no phens parsed)
                disease_data.pop(dn, None)
                disease_sample_seen.pop(dn, None)
                saved_count += 1
            fout.flush()
            print(f"  flushed {len(ready)} diseases (total saved: {saved_count}) "
                  f"at prompt {i+len(chunk)}/{len(prompts)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Final flush
    for dn, seen in disease_sample_seen.items():
        phens = disease_data.get(dn, {})
        # find source
        src = "rarebench"
        for m in meta:
            if m["disease"] == dn:
                src = m["source"]; break
        agg = {phen: {"prob": sum(pcts)/len(pcts)/100.0, "n_seen": len(pcts)}
               for phen, pcts in phens.items()}
        rec = {"disease": dn, "source": src, "n_samples": args.n_samples,
               "phenotypes": agg}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        saved_count += 1
    fout.flush(); fout.close()
    print(f"\nSaved {saved_count} new disease records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
