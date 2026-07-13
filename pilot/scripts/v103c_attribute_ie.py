"""Attribute-enriched IE: promote anatomical LOCATION (and key qualifiers) to
first-class phenotypes so they enter the KG vocabulary and become matchable.

Motivation (hernia case): DDXPlus patient's discriminative evidence is pain
LOCATION (groin/testicle/iliac fossa), encoded as separate location CUIs. The
strict IE buried location in an ignored edge attribute and extracted bare
"pain", so the location CUI was absent from the KG → no discrimination.

This IE extracts, with evidence_span, BOTH:
  (a) the finding (symptom/sign), and
  (b) its anatomical body location as a STANDALONE phenotype (e.g., "groin",
      "right iliac fossa", "scrotum"), plus location-qualified compound when
      natural (e.g., "groin swelling").

Same aggregated output schema as v103_run_shard → build with v103_build_kg_cui.
"""
import os, sys, json, argparse, re
from pathlib import Path
from collections import defaultdict

PROMPT = '''You are extracting clinical findings AND their anatomical locations from text.

SOURCE TEXT:
"""
{source_text}
"""

DISEASE: {disease}

TASK: List the symptoms/signs a patient with {disease} presents with. For EACH, if the text gives an anatomical body LOCATION, also output that location as a separate item. This makes location usable for diagnosis.

RULES (strict):
1. Extract ONLY findings/locations EXPLICITLY in the source text. Each needs a verbatim quote (evidence_span, min 5 chars).
2. For a finding with a stated location, output THREE items where applicable: the bare finding (e.g., "pain"), the anatomical location alone (e.g., "groin", "right iliac fossa", "scrotum"), and the location-qualified finding (e.g., "groin pain").
3. EXCLUDE the disease name "{disease}" and generic meta-terms (disease, treatment, surgery, prognosis, mortality, study, patients).
4. Use concrete anatomical terms for locations (groin, scrotum, flank, epigastrium, right lower quadrant, neck, etc.).
5. Output STRICT JSON: {{"items": [{{"name": "<finding or location>", "kind": "finding|location|qualified", "evidence_span": "<quote>"}}]}}'''


def parse_json(txt):
    m = re.search(r'\{.*\}', txt, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pubmed_dir", required=True)
    ap.add_argument("--max_abstracts", type=int, default=120)
    args = ap.parse_args()

    diseases = []
    with open(args.shard_file) as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) >= 2: diseases.append((p[0], p[1]))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Shard: {len(diseases)} diseases", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0.0, max_tokens=2048)

    for i, (cui, dn) in enumerate(diseases):
        out_path = f"{args.out_dir}/{cui}.json"
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(diseases)}] {dn}: skip", flush=True); continue
        pm = f"{args.pubmed_dir}/{cui}.jsonl"
        if not os.path.exists(pm):
            print(f"  [{i+1}/{len(diseases)}] {dn}: no text", flush=True); continue
        abstracts = []
        for line in open(pm):
            try:
                a = json.loads(line)
                if a.get("abstract"): abstracts.append(a["abstract"])
            except: continue
            if len(abstracts) >= args.max_abstracts: break
        if not abstracts:
            print(f"  [{i+1}/{len(diseases)}] {dn}: empty", flush=True); continue

        prompts = [PROMPT.format(source_text=a[:2500], disease=dn) for a in abstracts]
        convs = [[{"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(convs, sampling, use_tqdm=False)

        agg = defaultdict(lambda: {"n": 0})
        nparsed = 0
        for o in outs:
            obj = parse_json(o.outputs[0].text)
            if not obj or "items" not in obj: continue
            nparsed += 1
            for it in obj["items"]:
                name = (it.get("name", "") or "").strip().lower()
                span = (it.get("evidence_span", "") or "").strip()
                if not name or len(span) < 5: continue
                if name == dn.lower() or dn.lower() in name: continue
                agg[name]["n"] += 1
        n_abs = len(abstracts)
        aggregated = {name: {"n_mentions": d["n"],
                             "frequency_in_abstracts": d["n"]/max(n_abs,1),
                             "location_dist": {}, "severity_dist": {},
                             "onset_dist": {}, "character_dist": {}}
                      for name, d in agg.items()}
        json.dump({"disease": dn, "cui": cui, "n_abstracts": n_abs,
                   "n_parsed": nparsed, "aggregated": aggregated}, open(out_path, "w"))
        print(f"  [{i+1}/{len(diseases)}] {dn}: {n_abs}abs → {nparsed}parsed, {len(aggregated)}items", flush=True)


if __name__ == "__main__":
    main()
