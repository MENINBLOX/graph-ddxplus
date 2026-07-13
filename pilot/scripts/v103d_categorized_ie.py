"""Categorized IE: tag each finding at IE time with a CLINICAL category, so the
KG can be matched per evidence type (the email's 증상/징후/dual-service design).

Why (this session's finding): post-hoc UMLS-TUI filtering is too blunt — it
cannot separate a patient-reportable history-disease (anchor, keep) from a
clinician-only exam sign or a surgical complication (drop). An LLM tag at IE
time can. DDXPlus evidences are patient self-report → match against
{symptom, history}; the test-inclusive service uses {sign, test} too.

Categories:
  symptom : patient-reportable subjective complaint (pain, cough, nausea, swelling, lump)
  sign    : objective finding needing a clinician exam (stridor on auscultation, murmur, hepatomegaly on palpation)
  test    : laboratory / imaging / procedure result (elevated WBC, CT consolidation)
  history : prior condition / exposure / risk factor the patient can report (smoking, prior MI, travel)
  other   : treatment, surgery, pathophysiology, epidemiology, anatomy-only

Output schema (per disease) adds `category` to each aggregated entry, so a split
step can produce per-category dirs for v103_build_kg_cui.
"""
import os, sys, json, argparse, re
from pathlib import Path
from collections import defaultdict

PROMPT = '''You are a clinician annotating clinical findings from text with a BINARY label.

SOURCE TEXT:
"""
{source_text}
"""

DISEASE: {disease}

TASK: List the clinical findings of {disease} present in the text. For EACH, assign ONE binary label:
- "symptom": a SUBJECTIVE complaint the patient experiences and can self-report WITHOUT a clinician (e.g., pain, cough, nausea, itching, a lump or swelling the patient feels, fatigue, dizziness).
- "sign": anything that is NOT a patient self-reported symptom — i.e., an OBJECTIVE finding from physical examination (stridor on auscultation, murmur, hepatomegaly), OR a laboratory/imaging/procedure result (elevated white cell count, consolidation on X-ray), OR a prior condition / exposure / risk factor (history of smoking, previous heart attack, recent travel, known allergy).

RULES (strict):
1. Extract ONLY findings EXPLICITLY in the text; each needs a verbatim quote (evidence_span, min 5 chars).
2. EXCLUDE the disease name "{disease}", and exclude treatment/surgery/pathophysiology/epidemiology (do NOT output those).
3. Output STRICT JSON: {{"findings": [{{"name": "<finding>", "category": "symptom|sign", "evidence_span": "<quote>"}}]}}'''

CATS = {"symptom", "sign"}


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

        agg = defaultdict(lambda: {"n": 0, "cat": defaultdict(int)})
        nparsed = 0
        for o in outs:
            obj = parse_json(o.outputs[0].text)
            if not obj or "findings" not in obj: continue
            nparsed += 1
            for fd in obj["findings"]:
                name = (fd.get("name", "") or "").strip().lower()
                cat = (fd.get("category", "") or "").strip().lower()
                span = (fd.get("evidence_span", "") or "").strip()
                if not name or len(span) < 5 or cat not in CATS: continue
                if name == dn.lower() or dn.lower() in name: continue
                agg[name]["n"] += 1
                agg[name]["cat"][cat] += 1
        n_abs = len(abstracts)
        aggregated = {}
        for name, d in agg.items():
            cat = max(d["cat"].items(), key=lambda x: x[1])[0] if d["cat"] else "symptom"
            aggregated[name] = {"n_mentions": d["n"],
                                "frequency_in_abstracts": d["n"]/max(n_abs, 1),
                                "category": cat,
                                "location_dist": {}, "severity_dist": {},
                                "onset_dist": {}, "character_dist": {}}
        json.dump({"disease": dn, "cui": cui, "n_abstracts": n_abs,
                   "n_parsed": nparsed, "aggregated": aggregated}, open(out_path, "w"))
        cc = defaultdict(int)
        for v in aggregated.values(): cc[v["category"]] += 1
        print(f"  [{i+1}/{len(diseases)}] {dn}: {len(aggregated)} ({dict(cc)})", flush=True)


if __name__ == "__main__":
    main()
