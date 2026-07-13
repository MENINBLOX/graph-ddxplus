#!/usr/bin/env python3
"""v103 batch IE — multiple PubMed abstracts per disease, aggregated.

Pipeline:
1. Load PubMed abstracts for each disease (CUI)
2. Run v103 grounded IE per abstract (batched via vLLM)
3. Post-validate (evidence_span keyword check)
4. Aggregate phenotypes per disease (distribution over attributes)
"""
from __future__ import annotations
import os, sys, json, argparse, re, time
from pathlib import Path
from typing import List, Optional, Literal
from collections import defaultdict, Counter
from pydantic import BaseModel, Field

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

sys.path.insert(0, str(Path(__file__).parent))
from v103_grounded_ie import (
    LocationEnum, SeverityEnum, OnsetPaceEnum,
    LocationAttribute, SeverityAttribute, OnsetPaceAttribute, CharacterAttribute,
    PhenotypeGrounded, IEOutputGrounded,
)


PROMPT_TPL = """\
You are extracting medical phenotypes from a PubMed abstract.

SOURCE TEXT:
\"\"\"
{source_text}
\"\"\"

DISEASE: {disease}

EXTRACTION RULES (strict):
1. Extract ONLY phenotypes (symptoms/signs/findings) EXPLICITLY mentioned in the source text above.
2. Extract attributes ONLY IF stated in the text. Use ONLY controlled enum values.
3. Do NOT use general medical knowledge. OMIT if not in text.
4. Each attribute requires evidence_span (verbatim text quote, min 5 chars).
5. ONSET_PACE: only if text states timing (e.g., "sudden", "rapid-onset", "gradual"). Do NOT use mechanism descriptions.
6. SEVERITY: only if text uses severity words ("mild","moderate","severe","profound","critical").
7. If text doesn't specify an attribute for a phenotype, output ONLY {{"name": "<phen>"}} with no attribute fields.
8. Output strict JSON.

CONTROLLED VOCABULARIES (EXACT values):
- location: ["head","face","eye","throat","larynx","neck","chest","abdomen","epigastric","back","leg","knee","foot","skin","lung","heart","kidney","liver","brain","generalized","systemic",...]
- severity: ["mild","moderate","severe","profound","critical","variable"]
- onset_pace: ["sudden","rapid","gradual","insidious","variable"]
"""


# Post-validation keywords
ONSET_KEYWORDS = {"sudden","rapid","rapidly","gradual","gradually","insidious",
                  "abrupt","immediate","minutes","hours","instant","acute","chronic",
                  "over time","progressive","onset","develops","develop"}
SEVERITY_KEYWORDS = {"mild","moderate","severe","profound","critical","intense",
                     "extreme","slight","marked","significant","minor"}


def has_kw(text, kws):
    t = text.lower()
    return any(kw in t for kw in kws)


def post_validate(phen: PhenotypeGrounded):
    """Drop attributes whose evidence_span lacks supporting keyword."""
    dropped = 0
    if phen.onset_pace and not has_kw(phen.onset_pace.evidence_span, ONSET_KEYWORDS):
        phen.onset_pace = None; dropped += 1
    if phen.severity and not has_kw(phen.severity.evidence_span, SEVERITY_KEYWORDS):
        phen.severity = None; dropped += 1
    return dropped


def aggregate_disease(per_abstract_results):
    """Aggregate phenotype outputs from N abstracts into disease-level distribution.

    Returns: dict[phen_name] = {
        "n_mentions": int,
        "location_dist": {loc: prob},
        "severity_dist": {sev: prob},
        "onset_dist": {onset: prob},
        "character_dist": {char: prob},
        "frequency_in_abstracts": n_mentions / total_abstracts
    }
    """
    n_abstracts = len(per_abstract_results)
    phen_data = defaultdict(lambda: {
        "n_mentions": 0,
        "location_counter": Counter(),
        "severity_counter": Counter(),
        "onset_counter": Counter(),
        "character_counter": Counter(),
    })

    for abstract_phens in per_abstract_results:
        seen_in_this_abstract = set()
        for phen in abstract_phens:
            name_normalized = phen.name.lower().strip()
            if name_normalized in seen_in_this_abstract: continue
            seen_in_this_abstract.add(name_normalized)
            d = phen_data[name_normalized]
            d["n_mentions"] += 1
            if phen.location:
                for loc in phen.location.values: d["location_counter"][loc] += 1
            if phen.severity:
                d["severity_counter"][phen.severity.value] += 1
            if phen.onset_pace:
                d["onset_counter"][phen.onset_pace.value] += 1
            if phen.character:
                for c in phen.character.values: d["character_counter"][c] += 1

    # Convert to distributions
    aggregated = {}
    for name, d in phen_data.items():
        n = d["n_mentions"]
        total_obs_loc = sum(d["location_counter"].values()) or 1
        total_obs_sev = sum(d["severity_counter"].values()) or 1
        total_obs_on = sum(d["onset_counter"].values()) or 1
        total_obs_ch = sum(d["character_counter"].values()) or 1
        aggregated[name] = {
            "n_mentions": n,
            "frequency_in_abstracts": n / n_abstracts,  # empirical, not LLM-hallucinated
            "location_dist": {k: v/total_obs_loc for k,v in d["location_counter"].items()},
            "severity_dist": {k: v/total_obs_sev for k,v in d["severity_counter"].items()},
            "onset_dist": {k: v/total_obs_on for k,v in d["onset_counter"].items()},
            "character_dist": {k: v/total_obs_ch for k,v in d["character_counter"].items()},
        }
    return aggregated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cui", required=True, help="Disease CUI (e.g., C0685898 for Anaphylaxis)")
    ap.add_argument("--disease_name", required=True)
    ap.add_argument("--pubmed_dir", default="/mnt/medkg/pubmed")
    ap.add_argument("--max_abstracts", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    pm_path = f"{args.pubmed_dir}/{args.cui}.jsonl"
    if not os.path.exists(pm_path):
        print(f"❌ PubMed file not found: {pm_path}")
        return

    abstracts = []
    with open(pm_path) as f:
        for line in f:
            try:
                a = json.loads(line)
                if a.get("abstract"):
                    abstracts.append(a)
                    if len(abstracts) >= args.max_abstracts: break
            except: continue
    print(f"Loaded {len(abstracts)} abstracts for {args.disease_name} ({args.cui})", flush=True)

    schema = IEOutputGrounded.model_json_schema()
    prompts = [PROMPT_TPL.format(source_text=a["abstract"][:2500],
                                   disease=args.disease_name) for a in abstracts]

    print("Loading vLLM...", flush=True)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=8192, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(
        temperature=0.2, max_tokens=3072, top_p=0.9,
        structured_outputs=StructuredOutputsParams(json=schema)
    )

    convs = [[{"role":"user", "content": p}] for p in prompts]
    t0 = time.time()
    outs = llm.chat(convs, sampling)
    print(f"Batch IE done in {time.time()-t0:.1f}s", flush=True)

    per_abstract_results = []
    n_validation_pass = 0
    n_attrs_dropped = 0
    for i, (a, out) in enumerate(zip(abstracts, outs)):
        text = out.outputs[0].text
        try:
            parsed = json.loads(text)
            validated = IEOutputGrounded(**parsed)
        except Exception as e:
            print(f"  [{i}/pmid={a.get('pmid')}] parse fail: {e}", flush=True)
            continue
        n_validation_pass += 1
        # Post-validate each phenotype
        for p in validated.phenotypes:
            n_attrs_dropped += post_validate(p)
        per_abstract_results.append(validated.phenotypes)

    print(f"\nValidation: {n_validation_pass}/{len(abstracts)} parsed", flush=True)
    print(f"Post-validation: dropped {n_attrs_dropped} unsupported attributes", flush=True)

    aggregated = aggregate_disease(per_abstract_results)
    print(f"\n=== Aggregated phenotypes ({len(aggregated)}) ===", flush=True)
    sorted_phens = sorted(aggregated.items(), key=lambda x: -x[1]["n_mentions"])
    for name, d in sorted_phens[:20]:
        print(f"  [{name}] mentions={d['n_mentions']}, freq={d['frequency_in_abstracts']:.2f}", flush=True)
        if d['location_dist']:
            top_loc = sorted(d['location_dist'].items(), key=lambda x:-x[1])[:3]
            print(f"      location: {top_loc}", flush=True)
        if d['severity_dist']:
            print(f"      severity: {dict(d['severity_dist'])}", flush=True)
        if d['onset_dist']:
            print(f"      onset: {dict(d['onset_dist'])}", flush=True)
        if d['character_dist']:
            top_ch = sorted(d['character_dist'].items(), key=lambda x:-x[1])[:3]
            print(f"      character: {top_ch}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "disease": args.disease_name, "cui": args.cui,
            "n_abstracts_total": len(abstracts), "n_validated": n_validation_pass,
            "aggregated": aggregated
        }, f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
