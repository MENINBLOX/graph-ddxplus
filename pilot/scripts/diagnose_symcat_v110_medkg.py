#!/usr/bin/env python3
"""SymCat v87: KG features in prompt + CoT tie-break.

Apply DDXPlus v87 framework to SymCat 50 diseases.
Patient simulation from SymCat freq distribution (same as v54).
"""
from __future__ import annotations
import json, math, os, random, re, time, sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_symcat_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
GENERIC_TERMS = {'symptom', 'sign', 'pain', 'patient', 'disease', 'syndrome', 'condition'}

random.seed(42); np.random.seed(42)


def main():
    print("="*80, flush=True)
    print("SymCat v87 (KG features + CoT tie-break)", flush=True)
    print("="*80, flush=True)

    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    with open(KG_CACHE) as f: kg_data = json.load(f)
    pc = Counter()
    for k, v in kg_data["pair_counts"]: pc[tuple(k)] = v
    diseases = kg_data["diseases"]
    print(f"KG: {len(pc):,} 쌍", flush=True)

    with open("data/symcat/symcat_parsed.json") as f: sc = json.load(f)
    pairs = sc["disease_symptom_pairs"]

    dcs_list = []; name_to_cui = {}; cui_to_name = {}
    for dn in sorted(diseases):
        if diseases[dn].get("cui"):
            dcs_list.append(diseases[dn]["cui"])
            name_to_cui[dn] = diseases[dn]["cui"]
            cui_to_name[diseases[dn]["cui"]] = dn
    dcs = set(dcs_list)
    cui_to_idx = {dc: i for i, dc in enumerate(dcs_list)}
    print(f"Diseases: {len(dcs)}", flush=True)

    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt

    # === v110 MEDKG: load multi-source KG features (textbook + Orphanet) ===
    # Per-disease hybrid: use medkg if available, else fallback to v87 PubMed-only
    medkg_cui_path = Path("/home/max/Graph-DDXPlus/data/medkg/kg/disease_features_by_cui.json")
    medkg_path = Path("/home/max/Graph-DDXPlus/data/medkg/kg/disease_features.json")
    TOP_K = 8
    medkg_cui = json.load(medkg_cui_path.open()) if medkg_cui_path.exists() else {}
    medkg = json.load(medkg_path.open()) if medkg_path.exists() else {}

    # First compute v87 PubMed-only features as fallback baseline
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

    # Now overlay medkg features where available
    disease_features = dict(v87_features)  # start from v87 baseline
    n_medkg = 0
    for dn in diseases:
        cui = name_to_cui.get(dn)
        if not cui: continue
        feats = medkg_cui.get(cui) or medkg.get(dn, [])
        if feats:
            names = [f["phenotype"] for f in feats[:TOP_K]]
            if names:
                disease_features[cui] = ", ".join(names)
                n_medkg += 1
    print(f"[v110-medkg-symcat] Hybrid: {n_medkg}/{len(disease_features)} diseases use medkg, rest use v87 PubMed-only", flush=True)

    # Patient simulation (same as v54)
    print("\n환자 시뮬레이션 (SymCat 빈도 분포)...", flush=True)
    test_patients = []
    N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    for dn in sorted(diseases):
        if dn not in pairs: continue
        if dn not in name_to_cui: continue
        sym_list = pairs[dn]
        if not sym_list: continue
        true_dc = name_to_cui[dn]
        for _ in range(N_PER):
            patient_syms = []
            for sym, freq in sym_list:
                if random.random() * 100 < freq:
                    patient_syms.append(sym)
            if patient_syms:
                test_patients.append({"true_dc": true_dc, "true_name": dn, "symptoms": patient_syms})
    n = len(test_patients)
    print(f"  {n:,}명", flush=True)

    print("\n[vLLM init]...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    sampling = SamplingParams(temperature=0, max_tokens=8)

    # ===== Stage 1: 50-cand scoring with KG features =====
    print(f"\n[Stage 1] 50-cand × {n} patients...", flush=True)
    convs = []; pair_meta = []
    for c_idx, p in enumerate(test_patients):
        sym_text = ", ".join(p["symptoms"][:30])
        for d_idx, dc in enumerate(dcs_list):
            name = cui_to_name.get(dc, dc)
            kg_feats = disease_features[dc]
            prompt = f"""Patient symptoms: {sym_text}

Diagnosis hypothesis: {name}
Typical features (medical literature): {kg_feats}

How well does the patient's presentation match this diagnosis? Reply with ONLY a percentage 0-100 (0=no match, 100=textbook match)."""
            convs.append([{"role": "user", "content": prompt}])
            pair_meta.append((c_idx, d_idx))

    print(f"  Total LLM calls: {len(convs):,}", flush=True)
    CHUNK = 5000
    score_matrix = np.zeros((n, len(dcs_list)), dtype=np.float32)
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
            score_matrix[c_idx, d_idx] = score
        print(f"  Stage 1: {chunk_start+len(chunk)}/{len(convs)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  Stage 1 완료: {time.time()-t0:.0f}초", flush=True)
    np.save(f"pilot/results/symcat_v87_stage1.npy", score_matrix)

    t1_s1 = sum(1 for c_idx, p in enumerate(test_patients)
                if dcs_list[int(np.argmax(score_matrix[c_idx]))] == p["true_dc"])
    print(f"  Stage 1 @1 = {100*t1_s1/n:.2f}%", flush=True)

    # ===== Stage 2: CoT tie-break =====
    tied_patients = []
    for c_idx in range(n):
        scores = score_matrix[c_idx]
        max_s = scores.max()
        ti = np.where(scores == max_s)[0]
        if len(ti) >= 2:
            tied_patients.append((c_idx, [dcs_list[i] for i in ti]))
    small_ties = [(ci, td) for ci, td in tied_patients if len(td) <= 10]
    print(f"\n  Tied: {len(tied_patients)} ({100*len(tied_patients)/n:.1f}%), small≤10: {len(small_ties)}", flush=True)

    cot_sampling = SamplingParams(temperature=0, max_tokens=400)
    chosen_dc = {}
    print(f"\n[Stage 2] CoT tie-break...", flush=True)
    tb_convs = []; tb_meta = []
    for c_idx, tied_dcs in small_ties:
        p = test_patients[c_idx]
        sym_text = ", ".join(p["symptoms"][:30])
        lines = []
        for i, dc in enumerate(tied_dcs):
            name = cui_to_name.get(dc, dc)
            kg_feats = disease_features[dc]
            lines.append(f"({i+1}) {name} — typical features: {kg_feats}")
        disease_list = "\n".join(lines)
        prompt = f"""Patient symptoms: {sym_text}

The following candidate diagnoses are equally likely so far. For each candidate, briefly evaluate (1-2 sentences) which patient features support or contradict it. Then pick the SINGLE most likely.

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

    final_score = score_matrix.copy()
    for c_idx, picked_dc in chosen_dc.items():
        chosen_idx = cui_to_idx[picked_dc]
        max_s = score_matrix[c_idx].max()
        final_score[c_idx, chosen_idx] = max_s + 1.0

    t1c = t3c = t5c = t10c = 0
    for c_idx, p in enumerate(test_patients):
        ranked = np.argsort(-final_score[c_idx])
        if dcs_list[ranked[0]] == p["true_dc"]: t1c += 1
        if p["true_dc"] in [dcs_list[i] for i in ranked[:3]]: t3c += 1
        if p["true_dc"] in [dcs_list[i] for i in ranked[:5]]: t5c += 1
        if p["true_dc"] in [dcs_list[i] for i in ranked[:10]]: t10c += 1

    print(f"\n  SymCat v87 @1={100*t1c/n:.2f}% @3={100*t3c/n:.1f}% @5={100*t5c/n:.1f}% @10={100*t10c/n:.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"SymCat v87 GTPA@1 = {100*t1c/n:.2f}% (n={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
