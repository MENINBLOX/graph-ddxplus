#!/usr/bin/env python3
"""v101 prototype — JSON-structured IE with attribute schema.

seed=42 anaphylaxis case 검증용 prototype.

Pipeline:
1. Seed query: "Edema" (C0013604) — chief complaint of seed=42 patient
2. LLM IE: which diseases present with Edema? + full phenotype profile WITH attributes
3. 꼬리물기: collect all phenotype CUIs mentioned → recursive IE if needed
4. Build mini-KG with attribute-rich edges
5. Evaluate seed=42 patient (24 CUI vector + attributes from DDXPlus tokens)
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


# Controlled attribute vocabulary
ATTRIBUTE_SCHEMA = """\
Each phenotype must include an "attributes" object using ONLY these controlled keys:

- "location": list of body parts. Use these exact terms only:
  ["head","face","eye","eyelid","ear","nose","mouth","lip","tongue","throat",
   "larynx","neck","chest","abdomen","epigastric","back","pelvis","groin",
   "arm","shoulder","elbow","wrist","hand","finger","leg","thigh","hip",
   "knee","ankle","foot","toe","skin","joint","bone","muscle","lung","heart",
   "kidney","liver","brain","generalized","bilateral","unilateral","left","right"]
- "severity": one of ["mild","moderate","severe","critical","absent","variable"]
- "onset": one of ["sudden","rapid","gradual","chronic","intermittent","recurrent","absent"]
- "character": list of qualifiers (specific to phenotype). For pain use:
  ["sharp","dull","burning","cramping","throbbing","stabbing","radiating","tugging","crushing"]
  For cough: ["dry","productive","barking","whooping"]
  For swelling/edema: ["pitting","non_pitting","tender","painless"]
- "duration": one of ["minutes","hours","days","weeks","months","years","variable"]
- "frequency": float 0.0-1.0 (prevalence in patients with this disease)

OMIT a key if information is unknown or not applicable. Do NOT invent new keys or values.
"""


FEWSHOT = """\
### Example: Iron deficiency anemia (generic example for calibration)
{
  "disease": "Iron deficiency anemia",
  "phenotypes": [
    {
      "cui": "C0015672",
      "name": "Fatigue",
      "attributes": {
        "severity": "moderate",
        "onset": "gradual",
        "frequency": 0.70
      }
    },
    {
      "cui": "C0030193",
      "name": "Pain",
      "attributes": {
        "location": ["chest"],
        "character": ["dull"],
        "severity": "mild",
        "frequency": 0.12
      }
    }
  ]
}
"""


PROMPT_DISEASES_FOR_PHEN = """\
# Task
List medical diseases that commonly present with the phenotype "{phen_name}" (UMLS CUI {phen_cui}).
Return up to 8 most clinically relevant diseases.

For EACH disease, output a JSON object with:
- "disease": disease name (English)
- "phenotypes": list of typical phenotypes (5-15 per disease) WITH attributes

{schema}

Output VALID JSON only. No prose. Wrap output in <json>...</json> tags.

{fewshot}

Now, for diseases presenting with "{phen_name}", output:
<json>
[
  {{
    "disease": "<disease1>",
    "phenotypes": [...]
  }},
  ...
]
</json>
"""


def extract_json(text):
    m = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if m: return m.group(1).strip()
    # Fallback: find first [ ... ]
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m: return m.group(0)
    return None


def validate_json_record(rec):
    """Basic schema validation. Returns list of issues."""
    issues = []
    if "disease" not in rec: issues.append("missing disease")
    if "phenotypes" not in rec: issues.append("missing phenotypes")
    if isinstance(rec.get("phenotypes"), list):
        for i, p in enumerate(rec["phenotypes"]):
            if "cui" not in p: issues.append(f"phen[{i}] missing cui")
            if "name" not in p: issues.append(f"phen[{i}] missing name")
            if "attributes" not in p: issues.append(f"phen[{i}] missing attributes")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_cui", default="C0013604")
    ap.add_argument("--seed_name", default="Edema")
    ap.add_argument("--out", default="pilot/data/cache/v101_proto_ie.json")
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    print(f"=== v101 prototype IE: seed query = {args.seed_name} ({args.seed_cui}) ===", flush=True)

    prompt = PROMPT_DISEASES_FOR_PHEN.format(
        phen_name=args.seed_name, phen_cui=args.seed_cui,
        schema=ATTRIBUTE_SCHEMA, fewshot=FEWSHOT
    )

    print("Loading vLLM (Gemma-4-E4B-it)...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0.5, max_tokens=4096, top_p=0.95)

    print(f"Prompt length: {len(prompt)} chars", flush=True)
    t0 = time.time()
    out = llm.chat([[{"role":"user", "content": prompt}]], sampling)[0]
    text = out.outputs[0].text
    print(f"Generated in {time.time()-t0:.1f}s, {len(text)} chars", flush=True)

    # Save raw output
    raw_path = args.out.replace(".json", "_raw.txt")
    with open(raw_path, "w") as f: f.write(text)
    print(f"Raw output saved → {raw_path}", flush=True)

    # Extract JSON
    json_str = extract_json(text)
    if not json_str:
        print("ERROR: no JSON block extracted", flush=True)
        return

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        print(f"JSON parse error: {e}", flush=True)
        print(f"  json_str[:500]: {json_str[:500]}", flush=True)
        return

    # Validate + report
    print(f"\nExtracted {len(parsed)} disease records", flush=True)
    all_issues = []
    for rec in parsed:
        issues = validate_json_record(rec)
        if issues:
            all_issues.append((rec.get("disease","?"), issues))
    if all_issues:
        print(f"\n⚠️  Validation issues:", flush=True)
        for d, iss in all_issues[:5]:
            print(f"  {d}: {iss}", flush=True)
    else:
        print("✓ All records pass basic schema validation", flush=True)

    # Stats
    n_phens = sum(len(r.get("phenotypes",[])) for r in parsed)
    unique_phen_cuis = set()
    attr_keys_used = defaultdict(int)
    location_values_used = defaultdict(int)
    for r in parsed:
        for p in r.get("phenotypes",[]):
            unique_phen_cuis.add(p.get("cui"))
            for k, v in p.get("attributes",{}).items():
                attr_keys_used[k] += 1
                if k == "location" and isinstance(v, list):
                    for loc in v: location_values_used[loc] += 1

    print(f"\nTotal phenotypes: {n_phens}, unique CUIs: {len(unique_phen_cuis)}", flush=True)
    print(f"Attribute keys used: {dict(attr_keys_used)}", flush=True)
    print(f"Top location values: {dict(sorted(location_values_used.items(), key=lambda x:-x[1])[:10])}", flush=True)

    # Save
    with open(args.out, "w") as f:
        json.dump({"seed_cui": args.seed_cui, "seed_name": args.seed_name,
                   "diseases": parsed}, f, indent=2)
    print(f"Saved → {args.out}", flush=True)

    # Print diseases summary
    print(f"\n=== Diseases extracted ===", flush=True)
    for r in parsed:
        n = len(r.get("phenotypes",[]))
        print(f"  {r.get('disease')}: {n} phenotypes", flush=True)


if __name__ == "__main__":
    main()
