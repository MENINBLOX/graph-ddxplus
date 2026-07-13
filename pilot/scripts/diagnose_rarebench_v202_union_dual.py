#!/usr/bin/env python3
"""RareBench v87: Bayesian top-10 + KG features in prompt + CoT tie-break.

Apply DDXPlus v87 framework to RareBench rare diseases.
Use Bayesian filter to reduce 440 candidates to top-10, then v87 scoring.
"""
from __future__ import annotations
import json, math, os, re, time, sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_rarebench_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
GENERIC_TERMS = {'symptom', 'sign', 'pain', 'patient', 'disease', 'syndrome', 'condition'}


def main():
    print("="*80, flush=True)
    print("RareBench v87 (Bayesian top-10 + KG features + CoT tie-break)", flush=True)
    print("="*80, flush=True)

    with open(KG_CACHE) as f: kg_data = json.load(f)
    pc = Counter()
    for k, v in kg_data["pair_counts"]: pc[tuple(k)] = v
    diseases = kg_data["diseases"]

    with open("data/rarebench/disease_umls_mapping.json") as f:
        dm = json.load(f)["mapping"]
    with open("data/rarebench/hpo_umls_mapping.json") as f:
        hm = json.load(f)["mapping"]

    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    dcs_set = set()
    for dn, info in diseases.items():
        if isinstance(info, dict) and info.get("cui"):
            dcs_set.add(info["cui"])
    dcs_list = sorted(dcs_set)
    dcs = set(dcs_list)
    cui_to_idx = {dc: i for i, dc in enumerate(dcs_list)}
    print(f"질환: {len(dcs_list)}", flush=True)

    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt
    ds = dict(ds)
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    # === v110 MEDKG hybrid for RareBench ===
    medkg_cui_path = Path("/mnt/medkg/kg/disease_features_dual_by_cui.json")
    medkg_path = Path("/mnt/medkg/kg/disease_features_dual.json")
    TOP_K = 8
    medkg_cui = json.load(medkg_cui_path.open()) if medkg_cui_path.exists() else {}
    medkg = json.load(medkg_path.open()) if medkg_path.exists() else {}
    cui_to_dname = {}
    for dn, info in diseases.items():
        if isinstance(info, dict) and info.get("cui"):
            cui_to_dname[info["cui"]] = dn

    # v87 PubMed-only baseline
    v87_features = {}
    for dc in dcs_list:
        feats = ds.get(dc, {})
        top_cuis = sorted(feats.items(), key=lambda x: -x[1])[:TOP_K * 3]
        names = []; seen = set()
        for cui, cnt in top_cuis:
            n_ = cp.get(cui, cui)
            nl = n_.lower().strip()
            if not nl or nl in seen or nl in GENERIC_TERMS: continue
            if len(nl) < 3 or len(nl) > 50: continue
            seen.add(nl); names.append(n_)
            if len(names) >= TOP_K: break
        v87_features[dc] = ", ".join(names) if names else "—"

    # v111 UNION
    disease_features = {}
    n_medkg = 0
    for dc in dcs_list:
        dn = cui_to_dname.get(dc, "")
        v87_str = v87_features.get(dc, "")
        v87_terms = [t.strip() for t in v87_str.split(",") if t.strip() and t.strip() != "—"]
        feats = medkg_cui.get(dc) or medkg.get(dn, [])
        medkg_terms = [f["phenotype"] for f in feats[:TOP_K]] if feats else []
        if medkg_terms: n_medkg += 1
        combined = []
        seen = set()
        for t in medkg_terms + v87_terms:
            tl = t.lower().strip()
            if not tl or tl in seen: continue
            seen.add(tl)
            combined.append(t)
            if len(combined) >= TOP_K * 2: break
        disease_features[dc] = ", ".join(combined) if combined else "—"
    print(f"[v202-union-dual-rarebench] {n_medkg}/{len(disease_features)} have medkg, all have v87. Combined top-{TOP_K*2}.", flush=True)

    patients = []
    for fname in ["HMS", "LIRICAL", "MME", "RAMEDIS"]:
        path = f"data/rarebench/data/{fname}.jsonl"
        if not os.path.exists(path): continue
        with open(path) as fp:
            for line in fp:
                d = json.loads(line)
                phen_cuis = []
                phen_names = []
                for hpo in d.get("Phenotype", []):
                    if hpo in hm and hm[hpo].get("umls_cui"):
                        phen_cuis.append(hm[hpo]["umls_cui"])
                        phen_names.append(hm[hpo].get("hpo_name", hpo))
                disease_cuis = set()
                for dx in d.get("RareDisease", []):
                    if dx in dm and dm[dx].get("umls_cui"):
                        disease_cuis.add(dm[dx]["umls_cui"])
                if phen_cuis and disease_cuis:
                    patients.append({"phen_cuis": phen_cuis, "phen_names": phen_names, "disease_cuis": disease_cuis})

    valid_patients = [p for p in patients if any(d in dcs for d in p["disease_cuis"])]
    SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else len(valid_patients)
    valid_patients = valid_patients[:SUBSET]
    n = len(valid_patients)
    print(f"평가: {n:,}", flush=True)

    # Bayesian top-10 filter
    print("\nBayesian top-10 filter...", flush=True)
    candidates = []; baseline = 0
    for p in valid_patients:
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in p["phen_cuis"])
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        topk = [dc for dc, _ in ranked[:10]]
        candidates.append({"patient": p, "topk": topk})
        if topk and topk[0] in p["disease_cuis"]: baseline += 1
    in_top10 = sum(1 for c in candidates if any(d in c["topk"] for d in c["patient"]["disease_cuis"]))
    print(f"  Bayesian @1={100*baseline/n:.1f}%, top10={100*in_top10/n:.1f}%", flush=True)

    print("\n[vLLM init]...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=8)

    # Stage 1: per-candidate scoring with KG features
    print("\nStage 1: per-candidate scoring with KG features...", flush=True)
    convs = []; pair_meta = []
    for c_idx, c in enumerate(candidates):
        phen_text = ", ".join(c["patient"]["phen_names"][:30])
        for d_idx, dc in enumerate(c["topk"]):
            name = cp.get(dc, dc)
            kg_feats = disease_features.get(dc, "—")
            prompt = f"""Patient phenotypes: {phen_text}

Diagnosis hypothesis: {name}
Typical features (medical literature): {kg_feats}

How well does the patient's presentation match this rare disease? Reply with ONLY a percentage 0-100 (0=no match, 100=textbook match)."""
            convs.append([{"role": "user", "content": prompt}])
            pair_meta.append((c_idx, d_idx))

    print(f"  Total LLM calls: {len(convs):,}", flush=True)
    CHUNK = 5000
    all_scores = {}
    t0 = time.time()
    for chunk_start in range(0, len(convs), CHUNK):
        chunk = convs[chunk_start:chunk_start+CHUNK]
        outs = llm.chat(chunk, sampling)
        for i, out in enumerate(outs):
            c_idx, d_idx = pair_meta[chunk_start + i]
            text = out.outputs[0].text.strip()
            m = re.search(r"(\d+)", text)
            score = int(m.group(1)) if m else 0
            score = min(score, 100)
            all_scores[(c_idx, d_idx)] = score
        print(f"  Stage 1: {chunk_start+len(chunk)}/{len(convs)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  Stage 1 완료: {time.time()-t0:.0f}초", flush=True)

    # Build score matrix per patient (over their top-10)
    score_per_patient = []
    for c_idx, c in enumerate(candidates):
        scores = [all_scores.get((c_idx, d_idx), 0) for d_idx in range(len(c["topk"]))]
        score_per_patient.append(scores)

    # Stage 1 baseline @1
    t1_s1 = 0
    for c_idx, c in enumerate(candidates):
        scores = score_per_patient[c_idx]
        ranked_idxs = sorted(range(len(scores)), key=lambda i: -scores[i])
        if c["topk"][ranked_idxs[0]] in c["patient"]["disease_cuis"]: t1_s1 += 1
    print(f"  Stage 1 (KG features) @1 = {100*t1_s1/n:.2f}%", flush=True)

    # ===== Stage 2: CoT tie-break =====
    tied_patients = []
    for c_idx, c in enumerate(candidates):
        scores = score_per_patient[c_idx]
        max_s = max(scores)
        ti = [i for i, s in enumerate(scores) if s == max_s]
        if len(ti) >= 2:
            tied_dcs = [c["topk"][i] for i in ti]
            tied_patients.append((c_idx, tied_dcs))
    print(f"\n  Tied: {len(tied_patients)} ({100*len(tied_patients)/n:.1f}%)", flush=True)

    cot_sampling = SamplingParams(temperature=0, max_tokens=400)
    chosen_dc = {}
    print(f"\n[Stage 2] CoT tie-break...", flush=True)
    tb_convs = []; tb_meta = []
    for c_idx, tied_dcs in tied_patients:
        c = candidates[c_idx]
        phen_text = ", ".join(c["patient"]["phen_names"][:30])
        lines = []
        for i, dc in enumerate(tied_dcs):
            name = cp.get(dc, dc)
            kg_feats = disease_features.get(dc, "—")
            lines.append(f"({i+1}) {name} — typical features: {kg_feats}")
        disease_list = "\n".join(lines)
        prompt = f"""Patient phenotypes: {phen_text}

The following candidate rare diseases are equally likely so far. For each candidate, briefly evaluate (1-2 sentences) which patient features support or contradict it. Then pick the SINGLE most likely.

Candidates:
{disease_list}

Format your reply EXACTLY as:
EVAL:
(1) brief evaluation
(2) brief evaluation
...
PICK: <number>"""
        tb_convs.append([{"role": "user", "content": prompt}])
        tb_meta.append((c_idx, tied_dcs))

    if tb_convs:
        print(f"  CoT calls: {len(tb_convs)}", flush=True)
        t2 = time.time()
        for chunk_start in range(0, len(tb_convs), CHUNK):
            outs = llm.chat(tb_convs[chunk_start:chunk_start+CHUNK], cot_sampling)
            for i, out in enumerate(outs):
                c_idx, tied_dcs = tb_meta[chunk_start + i]
                text = out.outputs[0].text.strip()
                m = re.search(r"PICK\s*:\s*\(?(\d+)\)?", text, re.IGNORECASE)
                if not m: m = re.search(r"\(?(\d+)\)?\s*$", text.strip())
                if m:
                    pick = int(m.group(1)) - 1
                    if 0 <= pick < len(tied_dcs):
                        chosen_dc[c_idx] = tied_dcs[pick]
                if c_idx not in chosen_dc:
                    chosen_dc[c_idx] = tied_dcs[0]
            print(f"  CoT: {chunk_start+len(outs)}/{len(tb_convs)} ({time.time()-t2:.0f}초)", flush=True)
        print(f"  CoT 완료: {time.time()-t2:.0f}초", flush=True)

    # Final
    t1 = t3 = t5 = t10 = 0
    for c_idx, c in enumerate(candidates):
        scores = score_per_patient[c_idx]
        # If has chosen, boost it
        if c_idx in chosen_dc:
            chosen = chosen_dc[c_idx]
            for i, dc in enumerate(c["topk"]):
                if dc == chosen:
                    scores[i] = max(scores) + 1.0
                    break
        ranked_idxs = sorted(range(len(scores)), key=lambda i: -scores[i])
        ranked_dcs = [c["topk"][i] for i in ranked_idxs]
        tdcs = c["patient"]["disease_cuis"]
        if ranked_dcs[0] in tdcs: t1 += 1
        if any(r in tdcs for r in ranked_dcs[:3]): t3 += 1
        if any(r in tdcs for r in ranked_dcs[:5]): t5 += 1
        if any(r in tdcs for r in ranked_dcs[:10]): t10 += 1

    print(f"\n  RareBench v87 @1={100*t1/n:.2f}% @3={100*t3/n:.1f}% @5={100*t5/n:.1f}% @10={100*t10/n:.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"RareBench v87 GTPA@1 = {100*t1/n:.2f}% (n={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
