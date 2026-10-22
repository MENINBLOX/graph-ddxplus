"""Pure-Python replay of archive/v1_ddxplus/scripts/experiment_final_75.py on kg_edges.json.

Neo4j tie order is unknown; ties are broken by CUI string (asc) as one deterministic choice.
Usage: python -I replay_v1.py <split> <deny_threshold> [n_limit]
"""
import ast
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool

D = "/home/max/thesis/data/ddxplus/"
SM = json.load(open(D + "umls_mapping.json"))["mapping"]
DMF = {v["name_fr"]: v["umls_cui"] for v in json.load(open(D + "disease_umls_mapping.json"))["mapping"].values()}
import os
KG = json.load(open(os.environ.get("KGFILE", D + "kg_edges.json")))
D2S = defaultdict(set)
S2D = defaultdict(set)
for s, d in KG["edges"]:
    D2S[d].add(s)
    S2D[s].add(d)
MAX_IL = 223
DENY = int(sys.argv[2])
TIE = sys.argv[4] if len(sys.argv) > 4 else "cui"
import hashlib
def tk(s):
    if TIE == "cui":
        return s
    if TIE == "rev":
        return tuple(-ord(ch) for ch in s)
    if TIE == "rand":
        return random.random()
    return hashlib.md5((TIE + s).encode()).hexdigest()
import random



def v15(c, d):
    return c / (c + d + 1.0) * c


def ranking(conf, den):
    rows = []
    for d in {d for s in conf for d in S2D.get(s, ())}:
        syms = D2S[d]
        c = len(syms & conf)
        if c == 0:
            continue
        rows.append((v15(c, len(syms & den)), d))
    tot = sum(r[0] for r in rows)
    rows = [(sc / tot if tot > 0 else 0.0, d) for sc, d in rows]
    rows.sort(key=lambda x: (-x[0], tk(x[1])))
    return rows


def next_question(initial, conf, den, asked):
    if not (conf - {initial}):
        cov = defaultdict(int)
        for d in S2D.get(initial, ()):
            for s in D2S[d]:
                if s != initial and s not in asked:
                    cov[s] += 1
        if not cov:
            return None
        return min(cov, key=lambda s: (-cov[s], tk(s)))
    diseases = {d for s in conf for d in S2D.get(s, ())}
    valid = [d for d in diseases if len(D2S[d] & den) < DENY] if DENY > 0 else list(diseases)
    if not valid:
        return None
    cov = defaultdict(int)
    co = defaultdict(set)
    for d in valid:
        cd = D2S[d] & conf
        for s in D2S[d]:
            if s in conf or s in den or s in asked:
                continue
            cov[s] += 1
            co[s] |= cd
    if not cov:
        return None
    return min(cov, key=lambda s: (-(len(co[s]) * cov[s]), tk(s)))


def run(row):
    random.seed(hash(row["EVIDENCES"]) & 0xffffffff)
    gt = DMF[row["PATHOLOGY"]]
    pos = {SM[e.split("_@_")[0]]["cui"] for e in ast.literal_eval(row["EVIDENCES"]) if SM.get(e.split("_@_")[0], {}).get("cui")}
    init = SM[row["INITIAL_EVIDENCE"]]["cui"]
    conf, den, asked = {init}, set(), {init}
    hist = []
    il = 0
    for _ in range(MAX_IL):
        q = next_question(init, conf, den, asked)
        if q is None:
            break
        asked.add(q)
        (conf if q in pos else den).add(q)
        il += 1
        hist.append(tuple(d for _, d in ranking(conf, den)[:3]))
        if len(hist) >= 5 and all(h == hist[-1] for h in hist[-5:]):
            break
    final = ranking(conf, den)
    top = [d for _, d in final[:10]]
    return (int(bool(top) and top[0] == gt), int(gt in top), il, row["PATHOLOGY"])


if __name__ == "__main__":
    split = sys.argv[1]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    rows = []
    with open(D + f"release_{split}_patients.csv", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if lim and i >= lim:
                break
            rows.append({k: r[k] for k in ("PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE")})
    with Pool(20) as p:
        res = p.map(run, rows, chunksize=500)
    n = len(res)
    print(f"tie={TIE} split={split} deny={DENY} n={n} GTPA@1={sum(r[0] for r in res)/n:.4f} "
          f"GTPA@10={sum(r[1] for r in res)/n:.4f} avgIL={sum(r[2] for r in res)/n:.2f}")
