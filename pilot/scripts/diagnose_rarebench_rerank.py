#!/usr/bin/env python3
"""RareBench LLM re-ranking 평가."""
from __future__ import annotations
import json, math, os, re, time
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_rarebench_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}

PROMPT = """A patient presents with the following clinical phenotypes:
{phenotypes}

Which of these rare diseases is MOST LIKELY?
{candidates}

Answer with ONLY the number (1-{n})."""


def main():
    print("="*80, flush=True)
    print("RareBench LLM re-ranking 평가", flush=True)
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

    # KG disease CUIs
    dcs_set = set()
    for dn, info in diseases.items():
        if isinstance(info, dict) and info.get("cui"):
            dcs_set.add(info["cui"])
    dcs_list = sorted(dcs_set)
    dcs = set(dcs_list)
    print(f"질환: {len(dcs_list)}", flush=True)

    ds = defaultdict(dict)
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt
        if b in dcs: ds[b][a] = cnt
    ds = dict(ds)
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    # 환자 로드
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
    print(f"평가 가능: {len(valid_patients)}/{len(patients)}", flush=True)

    # Bayesian top-10
    print("\nBayesian top-10...", flush=True)
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
    n = len(candidates)
    print(f"  Baseline @1={100*baseline/n:.1f}%", flush=True)

    # LLM re-ranking
    print("\nLLM 프롬프트...", flush=True)
    prompts = []
    for c in candidates:
        phen_text = "\n".join(f"- {n}" for n in c["patient"]["phen_names"][:30])
        cands = "\n".join(f"{i+1}. {cp.get(dc, dc)}" for i, dc in enumerate(c["topk"]))
        prompts.append(PROMPT.format(phenotypes=phen_text, candidates=cands, n=len(c["topk"])))

    print(f"  프롬프트: {len(prompts)}", flush=True)
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

        tdcs = c["patient"]["disease_cuis"]
        if reranked[0] in tdcs: t1 += 1
        if any(r in tdcs for r in reranked[:3]): t3 += 1
        if any(r in tdcs for r in reranked[:5]): t5 += 1

    print(f"\nRe-ranked: @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @5={100*t5/n:.1f}%", flush=True)
    print(f"parse_fail={pf}", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"RareBench (LLM re-rank) GTPA@1 = {100*t1/n:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
