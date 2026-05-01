#!/usr/bin/env python3
"""SymCat 진단 v54 방식 (per-candidate 0-100 LLM scoring).

Best DDXPlus method (60.4%) applied to SymCat 50 diseases.
Patient simulation from SymCat freq distribution.
"""
from __future__ import annotations
import json, math, os, random, re, time, sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_symcat_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'a','an','the','and','or','of','for','with','that','this','your','you'}

random.seed(42); np.random.seed(42)


def main():
    print("="*80, flush=True)
    print("SymCat v54 (per-candidate 0-100 scoring)", flush=True)
    print("="*80, flush=True)

    # KG load
    with open(KG_CACHE) as f: kg_data = json.load(f)
    pc = Counter()
    for k, v in kg_data["pair_counts"]: pc[tuple(k)] = v
    diseases = kg_data["diseases"]
    print(f"KG: {len(pc):,} 쌍", flush=True)

    with open("data/symcat/symcat_parsed.json") as f: sc = json.load(f)
    pairs = sc["disease_symptom_pairs"]

    dcs_list = []
    name_to_cui = {}
    cui_to_name = {}
    for dn in sorted(diseases):
        if diseases[dn].get("cui"):
            dcs_list.append(diseases[dn]["cui"])
            name_to_cui[dn] = diseases[dn]["cui"]
            cui_to_name[diseases[dn]["cui"]] = dn
    dcs = set(dcs_list)
    print(f"Diseases: {len(dcs)}", flush=True)

    can = defaultdict(set); cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cp: cp[p[0]] = p[14].strip()

    ds = defaultdict(dict); scuis = set()
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt; scuis.add(b)
        if b in dcs: ds[b][a] = cnt; scuis.add(a)
    ds = dict(ds)

    aho = ahocorasick.Automaton()
    for cui in scuis:
        for name in can.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) < 4 or lo in STOPWORDS: continue
            try: aho.add_word(lo, (lo, cui))
            except: pass
    aho.make_automaton()
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    # Patient simulation
    print("\n환자 시뮬레이션 (SymCat 빈도 분포에서)...", flush=True)
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
    print(f"  {len(test_patients):,}명", flush=True)

    # Bayesian baseline (no top-K filter for v54 with all candidates)
    print("\nBayesian baseline...", flush=True)
    n = baseline = 0
    cand_data = []
    for p in test_patients:
        n += 1
        text = " . ".join(s.lower() for s in p["symptoms"])
        cuis = set()
        for ei, (nm, cui) in aho.iter(text):
            si = ei - len(nm) + 1
            if si > 0 and text[si-1].isalpha(): continue
            if ei+1 < len(text) and text[ei+1].isalpha(): continue
            cuis.add(cui)
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in cuis)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        topk = [dc for dc, _ in ranked[:10]]
        cand_data.append({"patient": p, "topk": topk})
        if topk and topk[0] == p["true_dc"]: baseline += 1
    in_top10 = sum(1 for c in cand_data if c["patient"]["true_dc"] in c["topk"])
    print(f"  Baseline @1={100*baseline/n:.1f}%, top10={100*in_top10/n:.1f}%", flush=True)

    # vLLM init
    print("\nvLLM init...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    # v54 per-candidate scoring (top-10)
    print("\nv54 per-candidate 0-100 score...", flush=True)
    sampling = SamplingParams(temperature=0, max_tokens=8)
    convs = []
    pair_meta = []
    for c_idx, c in enumerate(cand_data):
        sym_text = ", ".join(c["patient"]["symptoms"][:30])
        for d_idx, dc in enumerate(c["topk"]):
            name = cui_to_name.get(dc, dc)
            prompt = f"""Patient symptoms: {sym_text}

Diagnosis hypothesis: {name}

How likely is this hypothesis given the patient's symptoms? Reply with ONLY a percentage 0-100 (0=impossible, 100=certain)."""
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
        print(f"  진행: {chunk_start+len(chunk)}/{len(convs)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # Rank by LLM score
    t1 = t3 = t5 = 0
    for c_idx, c in enumerate(cand_data):
        scored = [(d_idx, all_scores.get((c_idx, d_idx), 0)) for d_idx in range(len(c["topk"]))]
        scored.sort(key=lambda x: -x[1])
        reranked = [c["topk"][d_idx] for d_idx, _ in scored]
        true_dc = c["patient"]["true_dc"]
        if reranked[0] == true_dc: t1 += 1
        if true_dc in reranked[:3]: t3 += 1
        if true_dc in reranked[:5]: t5 += 1

    nt = len(cand_data)
    print(f"\n  Baseline @1={100*baseline/nt:.1f}%", flush=True)
    print(f"  v54 SymCat: @1={100*t1/nt:.1f}% @3={100*t3/nt:.1f}% @5={100*t5/nt:.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"SymCat v54 GTPA@1 = {100*t1/nt:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
