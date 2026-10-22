"""v1 heuristic (experiment_final_75 logic) at DDXPlus code level, default values = negative.
Deterministic ties by code/disease name. Validation sample identical to adv_bayes.py (random.Random(0).sample).
Usage: python -I adv_v1code.py <n_val> <deny> <max_questions>
"""
import ast, csv, json, random, sys
from collections import defaultdict
from multiprocessing import Pool

D = "/home/max/thesis/data/ddxplus/"
import os
BUG = os.environ.get("BUG") == "1"
N_VAL, DENY, MAXQ = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
fr = json.load(open(D + "release_evidences.json"))
en = json.load(open(D + "release_evidences_en.json"))
cond = json.load(open(D + "release_conditions.json"))
q2fr = {v["question_en"].strip(): k for k, v in fr.items()}
e2f = {k: q2fr[v["question_en"].strip()] for k, v in en.items()}
DEF = {k: str(v.get("default_value")) for k, v in fr.items() if v["data_type"] in "MC"}
D2S = {c["cond-name-fr"]: {e2f[x] for x in list(c["symptoms"]) + list(c["antecedents"])} for c in cond.values()}
if os.environ.get("DROPSUB") == "1":
    SUB = {k for k, v in fr.items() if v["data_type"] in "MC" and k != "trav1"}
    D2S = {d: ss - SUB for d, ss in D2S.items()}
S2D = defaultdict(set)
for d, ss in D2S.items():
    for s in ss:
        S2D[s].add(d)

def pos_codes(ev):
    out = set()
    for e in ev:
        if "_@_" in e:
            c, v = e.split("_@_", 1)
            if DEF.get(c) == v and not BUG:
                continue
            out.add(c)
        else:
            out.add(e)
    return out

def ranking(conf, den):
    rows = []
    for d in {d for s in conf for d in S2D.get(s, ())}:
        c = len(D2S[d] & conf)
        if c:
            rows.append((c / (c + len(D2S[d] & den) + 1.0) * c, d))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return rows

def next_q(init, conf, den, asked):
    if not (conf - {init}):
        cov = defaultdict(int)
        for d in S2D.get(init, ()):
            for s in D2S[d]:
                if s not in asked:
                    cov[s] += 1
        return min(cov, key=lambda s: (-cov[s], s)) if cov else None
    valid = [d for d in {d for s in conf for d in S2D[s]} if len(D2S[d] & den) < DENY]
    if not valid:
        return None
    cov, co = defaultdict(int), defaultdict(set)
    for d in valid:
        cd = D2S[d] & conf
        for s in D2S[d]:
            if s in asked:
                continue
            cov[s] += 1
            co[s] |= cd
    return min(cov, key=lambda s: (-(len(co[s]) * cov[s]), s)) if cov else None

def run(row):
    pos = pos_codes(ast.literal_eval(row["EVIDENCES"]))
    init = row["INITIAL_EVIDENCE"]
    conf, den, asked, hist, nq = {init}, set(), {init}, [], 0
    while nq < MAXQ:
        q = next_q(init, conf, den, asked)
        if q is None:
            break
        asked.add(q)
        (conf if q in pos else den).add(q)
        nq += 1
        hist.append(tuple(d for _, d in ranking(conf, den)[:3]))
        if len(hist) >= 5 and all(h == hist[-1] for h in hist[-5:]):
            break
    r = ranking(conf, den)
    top = r[0][0]
    tied = [d for s, d in r if s == top]
    return (1 / len(tied) if row["PATHOLOGY"] in tied else 0.0, int(r[0][1] == row["PATHOLOGY"]), nq + 1)

if __name__ == "__main__":
    with open(D + "release_validate_patients.csv") as f:
        rows = list(csv.DictReader(f))
    rows = random.Random(0).sample(rows, N_VAL) if N_VAL < len(rows) else rows
    rows = [{k: r[k] for k in ("PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE")} for r in rows]
    with Pool(20) as p:
        res = p.map(run, rows, chunksize=200)
    n = len(res)
    import collections
    print("IL hist", sorted(collections.Counter(r[2] for r in res).items())[:12])
    print(f"v1-code deny={DENY} maxq={MAXQ} n={n} GTPA@1(tie-exp)={sum(r[0] for r in res)/n:.4f} "
          f"GTPA@1(name-tiebreak)={sum(r[1] for r in res)/n:.4f} IL(incl init)={sum(r[2] for r in res)/n:.2f}")
