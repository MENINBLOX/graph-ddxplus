#!/usr/bin/env python3
"""v101 단계 2 — Recursive IE + compound query.

seed=42 anaphylaxis case 검증:
1. Multiple seed queries:
   - "Facial edema" (anatomy-specific)
   - "Pruritus" (key phenotype)
   - "Stridor" (key phenotype)
   - "Wheezing" (key phenotype)
   - "Hypersensitivity" (history)
2. JSON IE per query → diseases + attribute-rich phenotypes
3. Union all diseases — should include Anaphylaxis
4. 꼬리물기: collect all unique phenotype names → next round IE on missing
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")


ATTRIBUTE_SCHEMA = """\
Each phenotype must include "attributes" object using ONLY these keys:
- "location": list of body parts. Use exact terms only:
  ["head","face","eye","eyelid","ear","nose","mouth","lip","tongue","throat",
   "larynx","neck","chest","abdomen","epigastric","back","arm","hand","leg",
   "knee","ankle","foot","skin","joint","bone","muscle","lung","heart",
   "kidney","liver","brain","generalized","bilateral","unilateral","left","right"]
- "severity": one of ["mild","moderate","severe","critical","variable"]
- "onset": one of ["sudden","rapid","gradual","chronic","intermittent","recurrent"]
- "character": list of qualifiers (e.g., pain: ["sharp","dull","burning","cramping"], cough: ["dry","productive"])
- "frequency": float 0.0-1.0 (prevalence in this disease)
OMIT keys if unknown. Do NOT invent new keys/values.
"""


PROMPT_TPL = """\
List up to 6 medical diseases commonly presenting with the symptom/finding "{phen}".

For each disease, output a JSON object with:
- "disease": disease name (English)
- "phenotypes": list of 5-10 typical phenotypes WITH attributes

{schema}

Output VALID JSON in <json>...</json> tags. No prose.

<json>
[
  {{"disease": "<name>", "phenotypes": [{{"name": "<phen>", "attributes": {{...}}}}, ...]}},
  ...
]
</json>

Now for "{phen}":
"""


def extract_json(text):
    m = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m: return m.group(0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot/data/cache/v101_recursive_ie.json")
    args = ap.parse_args()

    # Seed queries from seed=42 patient evidence
    seed_queries = [
        ("Angioedema", "C0002994"),       # facial edema specific
        ("Pruritus", "C0033774"),
        ("Stridor", "C0035203"),
        ("Wheezing", "C0043144"),
        ("Generalized urticaria", "C0042109"),  # hives
        ("Hypersensitivity", "C0020517"),
    ]

    print(f"=== v101 recursive IE: {len(seed_queries)} seed queries ===", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=0.85,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0.4, max_tokens=4096, top_p=0.95)

    prompts = [PROMPT_TPL.format(phen=q, schema=ATTRIBUTE_SCHEMA) for q, _ in seed_queries]
    convs = [[{"role":"user", "content": p}] for p in prompts]

    t0 = time.time()
    outs = llm.chat(convs, sampling)
    print(f"  {len(outs)} queries generated in {time.time()-t0:.1f}s", flush=True)

    results = {}
    all_diseases = defaultdict(lambda: {"sources": [], "phenotypes_by_seed": {}})

    for (qname, qcui), out in zip(seed_queries, outs):
        text = out.outputs[0].text
        json_str = extract_json(text)
        if not json_str:
            print(f"\n[{qname}] ❌ No JSON extracted", flush=True)
            results[qname] = {"error": "no_json", "raw": text[:500]}
            continue
        try:
            parsed = json.loads(json_str)
        except Exception as e:
            print(f"\n[{qname}] ❌ Parse error: {e}", flush=True)
            results[qname] = {"error": str(e), "raw": text[:500]}
            continue

        n_dis = len(parsed)
        n_phen = sum(len(r.get("phenotypes",[])) for r in parsed)
        print(f"\n[{qname}] ✓ {n_dis} diseases, {n_phen} phenotypes", flush=True)
        for r in parsed:
            d = r.get("disease","?")
            print(f"   - {d}", flush=True)
            all_diseases[d]["sources"].append(qname)
            all_diseases[d]["phenotypes_by_seed"][qname] = r.get("phenotypes",[])

        results[qname] = {"diseases": parsed}

    # Save raw + structured
    with open(args.out, "w") as f:
        json.dump({"seed_queries": [q for q,_ in seed_queries],
                   "by_query": results,
                   "merged_diseases": {d: dict(info) for d, info in all_diseases.items()}}, f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)

    print(f"\n=== Disease union ({len(all_diseases)} diseases) ===", flush=True)
    # Sort by # sources (diseases mentioned in multiple queries are more relevant)
    sorted_d = sorted(all_diseases.items(), key=lambda x: -len(x[1]["sources"]))
    for d, info in sorted_d[:20]:
        n_phen_unique = set()
        for ps in info["phenotypes_by_seed"].values():
            for p in ps:
                n_phen_unique.add(p.get("name",""))
        print(f"  [{len(info['sources'])} queries] {d}: {len(n_phen_unique)} unique phen, sources={info['sources']}", flush=True)


if __name__ == "__main__":
    main()
