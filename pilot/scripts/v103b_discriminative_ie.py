"""B-pilot: discriminative-findings IE (hybrid, verifiable).

Root cause from forensic: strict IE aggregates by raw mention count, so the
disease NAME self-mention (n~90) and generic terms swamp the actual
discriminative symptoms (n~2-3). This pass re-extracts, from the SAME abstracts,
the *characteristic presenting findings that distinguish the disease* — each
still requires an evidence_span (verifiable, not self-knowledge), and the
disease name / generic meta-terms are explicitly excluded.

Output per disease in the SAME aggregated schema as v103_run_shard, so
v103_build_kg_cui.py can build a KG from it directly.

Usage: CUDA_VISIBLE_DEVICES=0 python v103b_discriminative_ie.py \
         --shard_file /tmp/ddxplus49_shard.tsv \
         --out_dir pilot/data/cache/v103b_ddx49_per_disease \
         --pubmed_dir pilot/data/cache/pubmed_deep --max_abstracts 120
"""
import os, sys, json, argparse, re
from pathlib import Path
from collections import defaultdict

PROMPT = '''You are a clinician extracting DISCRIMINATIVE clinical findings from a PubMed abstract.

SOURCE TEXT:
"""
{source_text}
"""

DISEASE: {disease}

TASK: Extract the characteristic SYMPTOMS and SIGNS that a patient with {disease} presents with — the findings most useful to DISTINGUISH {disease} from other diseases.

RULES (strict):
1. Extract ONLY findings (symptoms/signs/exam/lab) EXPLICITLY present in the source text.
2. Each finding MUST be supported by a verbatim quote from the text (evidence_span, min 5 chars).
3. EXCLUDE the disease name itself and its synonyms (do NOT output "{disease}").
4. EXCLUDE generic/meta terms: disease, infection, inflammation, syndrome, mortality, prognosis, treatment, management, diagnosis, prevalence, incidence, risk factor, complication.
5. Prefer concrete patient-presenting findings (e.g., "barking cough", "stridor", "facial swelling") over abstract concepts.
6. Output STRICT JSON: {{"findings": [{{"name": "<finding>", "evidence_span": "<verbatim quote>"}}]}}. If none, {{"findings": []}}.'''


def parse_json(txt):
    # extract first JSON object
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
            print(f"  [{i+1}/{len(diseases)}] {dn}: no abstracts", flush=True); continue
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
            if not obj or "findings" not in obj: continue
            nparsed += 1
            for fd in obj["findings"]:
                name = (fd.get("name", "") or "").strip().lower()
                span = (fd.get("evidence_span", "") or "").strip()
                if not name or len(span) < 5: continue
                if name == dn.lower() or dn.lower() in name: continue  # self-name guard
                agg[name]["n"] += 1
        n_abs = len(abstracts)
        aggregated = {name: {"n_mentions": d["n"],
                             "frequency_in_abstracts": d["n"]/max(n_abs,1),
                             "location_dist": {}, "severity_dist": {},
                             "onset_dist": {}, "character_dist": {}}
                      for name, d in agg.items()}
        json.dump({"disease": dn, "cui": cui, "n_abstracts": n_abs,
                   "n_parsed": nparsed, "aggregated": aggregated},
                  open(out_path, "w"))
        print(f"  [{i+1}/{len(diseases)}] {dn}: {n_abs}abs → {nparsed}parsed, {len(aggregated)}findings", flush=True)


if __name__ == "__main__":
    main()
