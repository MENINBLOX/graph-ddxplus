#!/usr/bin/env python3
"""SymCat LLM re-ranking 평가 (학습 없이).

DDXPlus v17 방식과 동일: Bayesian top-10 → LLM 단순 선택.
"""
from __future__ import annotations
import json, math, os, random, re, time
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_symcat_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'a','an','the','and','or','of','for','with','that','this','your','you'}

PROMPT = """A patient presents with the following symptoms:
{symptoms}

Which of these diseases is MOST LIKELY?
{candidates}

Answer with ONLY the number (1-{n})."""

random.seed(42); np.random.seed(42)


def main():
    print("="*80, flush=True)
    print("SymCat LLM re-ranking 평가", flush=True)
    print("="*80, flush=True)

    # KG load
    with open(KG_CACHE) as f: kg_data = json.load(f)
    pc = Counter()
    for k, v in kg_data["pair_counts"]: pc[tuple(k)] = v
    diseases = kg_data["diseases"]
    print(f"KG: {len(pc):,} 쌍", flush=True)

    # SymCat data
    with open("data/symcat/symcat_parsed.json") as f: sc = json.load(f)
    pairs = sc["disease_symptom_pairs"]

    # disease CUIs
    dcs_list = []
    name_to_cui = {}
    cui_to_name = {}
    for dn in sorted(diseases):
        if diseases[dn].get("cui"):
            dcs_list.append(diseases[dn]["cui"])
            name_to_cui[dn] = diseases[dn]["cui"]
            cui_to_name[diseases[dn]["cui"]] = dn
    dcs = set(dcs_list)

    # UMLS load
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

    # 환자 시뮬레이션 (SymCat 빈도 분포에서 샘플링)
    print("\n환자 시뮬레이션...", flush=True)
    test_patients = []
    for dn in sorted(diseases):
        if dn not in pairs: continue
        if dn not in name_to_cui: continue
        sym_list = pairs[dn]
        if not sym_list: continue
        true_dc = name_to_cui[dn]
        for _ in range(100):  # 50 질환 × 100 = 5000명
            patient_syms = []
            for sym, freq in sym_list:
                if random.random() * 100 < freq:
                    patient_syms.append(sym)
            if patient_syms:
                test_patients.append({"true_dc": true_dc, "true_name": dn, "symptoms": patient_syms})
    print(f"  {len(test_patients):,}명", flush=True)

    # Bayesian top-10
    print("\nBayesian top-10...", flush=True)
    candidates = []; baseline = 0; n = 0
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
        candidates.append({"patient": p, "topk": topk})
        if topk and topk[0] == p["true_dc"]: baseline += 1
    in_topk = sum(1 for c in candidates if c["patient"]["true_dc"] in c["topk"])
    print(f"  Baseline @1={100*baseline/n:.1f}%, top10={100*in_topk/n:.1f}%", flush=True)

    # LLM re-ranking
    print("\nLLM re-ranking 프롬프트...", flush=True)
    prompts = []
    for c in candidates:
        sym_text = "\n".join(f"- {s}" for s in c["patient"]["symptoms"][:20])
        cands = "\n".join(f"{i+1}. {cui_to_name.get(dc, dc)}" for i, dc in enumerate(c["topk"]))
        prompts.append(PROMPT.format(symptoms=sym_text, candidates=cands, n=len(c["topk"])))

    print(f"  프롬프트: {len(prompts):,}", flush=True)
    print("\nvLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=16)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # 평가
    t1 = t3 = t5 = pf = 0
    for c, out in zip(candidates, outputs):
        text = out.outputs[0].text.strip()
        m = re.search(r"(\d+)", text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(c["topk"]):
                reranked = list(c["topk"]); chosen = reranked.pop(idx); reranked.insert(0, chosen)
            else: reranked = c["topk"]; pf += 1
        else: reranked = c["topk"]; pf += 1

        tdc = c["patient"]["true_dc"]
        if reranked[0] == tdc: t1 += 1
        if tdc in reranked[:3]: t3 += 1
        if tdc in reranked[:5]: t5 += 1

    nt = len(candidates)
    print(f"\nRe-ranked: @1={100*t1/nt:.1f}% @3={100*t3/nt:.1f}% @5={100*t5/nt:.1f}%", flush=True)
    print(f"parse_fail={pf:,}", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"SymCat (LLM re-rank) GTPA@1 = {100*t1/nt:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
