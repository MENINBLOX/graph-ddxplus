#!/usr/bin/env python3
"""Stage 2 LLM disambiguation — vLLM batch on top-K candidates.

Pipeline:
1. Load v71 top-K dump (per patient: top-10 disease CUIs)
2. For each patient, build prompt with:
   - Patient evidence (yes + no)
   - 10 candidate diseases with KG features (top phenotypes per disease)
3. Gemma-4-E4B (vLLM batch) ranks candidates
4. Output: re-ranked top-K + new GTPA@k metrics

Strict zero-shot: no train labels, no few-shot examples in prompt.
"""
from __future__ import annotations
import os, sys, json, csv, ast, math, pickle, argparse, re
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
EV_META = "data/ddxplus/release_evidences.json"


def load_kg_features(graph_path, top_k_per_disease=15):
    """Per disease, top-K strongest phenotypes (English names)."""
    G = pickle.load(open(graph_path, "rb"))
    pr = set(json.load(open(PR_UNIVERSE)))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2en = {icd[dn]["cui"]: dn for dn in icd}

    # Get phenotype CUI → English name (from KG node attrs or MRCONSO fallback)
    cui_names = {}
    for node, data in G.nodes(data=True):
        if data.get("type") == "phenotype":
            n = data.get("name") or data.get("primary_name") or ""
            if n: cui_names[node] = n

    # Fallback: scan MRCONSO for missing names
    needed = set()
    features = {}
    for cui_d, _ in cui2en.items():
        if cui_d not in G: features[cui_d] = []; continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(cui_d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE": continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr: continue
            elif cat not in {"patient_reportable", "history", "demographic"}: continue
            ed_w[p] += ed.get("weight", 0.0)
        top = sorted(ed_w.items(), key=lambda x: -x[1])[:top_k_per_disease]
        feats = []
        for c, w in top:
            name = cui_names.get(c, "")
            if not name:
                needed.add(c)
            feats.append((c, name, w))
        features[cui_d] = feats

    # Load MRCONSO for missing
    if needed:
        import time
        t0 = time.time()
        mrconso = "/windows/data/umls_subset/MRCONSO.RRF"
        with open(mrconso) as f:
            for line in f:
                if not needed: break
                parts = line.split('|')
                if len(parts) < 15: continue
                c, lang = parts[0], parts[1]
                if lang != 'ENG': continue
                if c in needed:
                    cui_names[c] = parts[14]
                    needed.discard(c)
        print(f"  MRCONSO scan: {time.time()-t0:.0f}s, remaining missing: {len(needed)}",
              flush=True)
        # Refill features with names
        for cui_d, feats in features.items():
            features[cui_d] = [(c, cui_names.get(c, c), w) for c, _, w in feats]
    return features, cui2en


def parse_evidences(row, value_cuis, ev_meta, binary_evs):
    """Return list of (question_en, answer_text)."""
    evs = ast.literal_eval(row["EVIDENCES"])
    qa_list = []  # (q_text, a_text)
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            q = ev_meta.get(base, {}).get("question_en", base)
            qa_list.append((q, val))
        else:
            q = ev_meta.get(ev, {}).get("question_en", ev)
            qa_list.append((q, "yes"))
    return qa_list


def build_prompt(qa_list, age, sex, top_candidates, features, cui2en):
    """Single-pass differential diagnosis prompt. No few-shot."""
    lines = []
    lines.append("# Task: Differential diagnosis ranking")
    lines.append("Given the patient's evidence below, rank the 10 candidate diagnoses")
    lines.append("from most likely (1) to least likely (10). Output ONLY the ranking as")
    lines.append("a JSON list of disease names in order.")
    lines.append("")
    lines.append(f"# Patient demographics")
    lines.append(f"Age: {age}, Sex: {sex}")
    lines.append("")
    lines.append(f"# Patient evidence (yes answers)")
    for q, a in qa_list[:30]:
        if a == "yes":
            lines.append(f"- {q}")
        else:
            lines.append(f"- {q} → {a}")
    lines.append("")
    lines.append(f"# Candidate diagnoses (with characteristic phenotypes)")
    cand_names = []
    for i, (cui_d, score) in enumerate(top_candidates, 1):
        dname = cui2en.get(cui_d, cui_d)
        cand_names.append(dname)
        feats = features.get(cui_d, [])
        feat_str = ", ".join(name for _, name, _ in feats[:8])
        lines.append(f"{i}. {dname}: {feat_str}")
    lines.append("")
    lines.append("# Output")
    lines.append("Output a JSON list of 10 disease names ordered from most likely to least.")
    lines.append("Example format: [\"Disease A\", \"Disease B\", ...]")
    lines.append("Ranking:")
    return "\n".join(lines), cand_names


def parse_ranking(text, cand_names):
    """Parse LLM output back to ranked list of candidate names.
    Robust to formatting variations."""
    # Try JSON parse first
    m = re.search(r"\[(.*?)\]", text, re.DOTALL)
    if m:
        inner = m.group(1)
        items = re.findall(r'"([^"]+)"', inner)
        if items:
            # Match against cand_names
            ranked = []
            used = set()
            for item in items:
                # Find best matching candidate (case-insensitive substring)
                best = None
                for cn in cand_names:
                    if cn in used: continue
                    if item.lower() in cn.lower() or cn.lower() in item.lower():
                        best = cn; break
                if best:
                    ranked.append(best); used.add(best)
            # Append unused
            for cn in cand_names:
                if cn not in used: ranked.append(cn)
            return ranked
    # Fallback: line-by-line
    lines = text.split("\n")
    ranked = []; used = set()
    for ln in lines:
        ln = ln.strip().lstrip("0123456789.-) ")
        for cn in cand_names:
            if cn in used: continue
            if ln.lower().startswith(cn.lower()) or cn.lower() in ln.lower():
                ranked.append(cn); used.add(cn); break
    for cn in cand_names:
        if cn not in used: ranked.append(cn)
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_path", required=True, help="Stage 1 top-K dump")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=200, help="patients to process")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    # Load stage 1 dumps
    dumps = []
    with open(args.dump_path) as f:
        for line in f:
            dumps.append(json.loads(line))
    dumps = dumps[:args.n]
    print(f"Loaded {len(dumps)} stage1 results", flush=True)

    # Load KG features
    print("Loading KG features...", flush=True)
    features, cui2en = load_kg_features(args.graph)

    # Load patient evidence
    value_cuis = json.load(open(VALUE_CUIS))
    ev_meta = json.load(open(EV_META))
    binary_evs = {ev_id for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}

    pid_to_row = {}
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            pid_to_row[i] = row

    # Build prompts
    prompts = []; meta = []
    print("Building prompts...", flush=True)
    for d in dumps:
        pid = d["pid"]
        if pid not in pid_to_row: continue
        row = pid_to_row[pid]
        qa = parse_evidences(row, value_cuis, ev_meta, binary_evs)
        prompt, cand_names = build_prompt(qa, row["AGE"], row["SEX"],
                                           d["top"], features, cui2en)
        prompts.append(prompt)
        meta.append({"pid": pid, "true_cui": d["true_cui"], "top": d["top"],
                     "cand_names": cand_names,
                     "true_name": cui2en.get(d["true_cui"], d["true_cui"]),
                     "gt_diff": d["gt_diff"]})

    print(f"Prepared {len(prompts)} prompts", flush=True)
    # Launch vLLM
    print("Loading vLLM (Gemma-4-E4B)...", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    # Batch inference
    import time
    t0 = time.time()
    results = []
    for i in range(0, len(prompts), args.batch):
        chunk = prompts[i:i+args.batch]
        meta_chunk = meta[i:i+args.batch]
        convs = [[{"role": "user", "content": p}] for p in chunk]
        outs = llm.chat(convs, sampling)
        for m, o in zip(meta_chunk, outs):
            try: text = o.outputs[0].text
            except: text = ""
            ranked_names = parse_ranking(text, m["cand_names"])
            results.append({"pid": m["pid"], "true_cui": m["true_cui"],
                            "true_name": m["true_name"],
                            "ranked_names": ranked_names,
                            "raw_output": text[:500],
                            "gt_diff": m["gt_diff"]})
        elapsed = time.time() - t0
        print(f"  {i+len(chunk)}/{len(prompts)} ({elapsed:.0f}s)", flush=True)

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} → {args.out}", flush=True)

    # Compute metrics
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for r in results:
        if r["true_name"] not in r["ranked_names"]:
            rank = 10
        else:
            rank = r["ranked_names"].index(r["true_name"]) + 1
        n += 1
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0 / rank
    print(f"\n=== Stage 2 LLM rerank N={n} ===")
    print(f"  @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% "
          f"@10={100*c10/n:.2f}% MRR={rr/n:.4f}")


if __name__ == "__main__":
    main()
