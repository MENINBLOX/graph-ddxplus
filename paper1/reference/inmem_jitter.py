"""In-memory re-implementation of archive experiment_final_75.py (greedy, noante, co-occurrence).

usage: python -I inmem.py SPLIT N_ROWS DENY STOP TIEBREAK
TIEBREAK: lex | rev | rand<seed>
"""
import ast, csv, json, random, sys, time
from collections import defaultdict, deque

D = "/home/max/thesis/data/ddxplus/"
split, n_rows, deny, stop, tb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
MAX_IL = 223
JIT = random.Random(7)

umap = json.load(open(D + "umls_mapping.json"))["mapping"]
dmap = json.load(open(D + "disease_umls_mapping.json"))["mapping"]
cond = json.load(open(D + "release_conditions.json"))
fr2en = {v["cond-name-fr"]: k for k, v in cond.items()}
edges = json.load(open(D + "kg_edges.json"))["edges"]
d2s = defaultdict(set); s2d = defaultdict(set)
for s, d in edges:
    d2s[d].add(s); s2d[s].add(d)
allc = sorted(set(d2s) | set(s2d))
if tb == "lex":
    key = {c: i for i, c in enumerate(allc)}
elif tb == "rev":
    key = {c: -i for i, c in enumerate(allc)}
else:
    r = allc[:]; random.Random(int(tb[4:])).shuffle(r); key = {c: i for i, c in enumerate(r)}

def cui(code):
    v = umap.get(code)
    return v["cui"] if v else None

def candidates(init, conf, den, asked):
    if not (conf - {init}):
        cov = defaultdict(int)
        for d in s2d.get(init, ()):
            for s in d2s[d]:
                if s != init and s not in asked:
                    cov[s] += 1
        return sorted(cov, key=lambda s: (-cov[s], key[s]))[:10]
    dis = set()
    for c in conf:
        dis |= s2d.get(c, set())
    valid = [d for d in dis if deny <= 0 or len(d2s[d] & den) < deny]
    if not valid:
        return []
    cov = defaultdict(int); co = defaultdict(set)
    for d in valid:
        cs = d2s[d] & conf
        for s in d2s[d]:
            if s in conf or s in den or s in asked:
                continue
            cov[s] += 1; co[s] |= cs
    return sorted(cov, key=lambda s: (-(len(co[s]) * cov[s]), key[s]))[:10]

def diag(conf, den, k=10, jitter=None):
    dis = set()
    for c in conf:
        dis |= s2d.get(c, set())
    out = []
    for d in dis:
        c = len(d2s[d] & conf); n = len(d2s[d] & den)
        if c > 0:
            sc = c / (c + n + 1.0) * c
            if sc > 0:
                out.append((d, sc))
    out.sort(key=(lambda x: (-x[1], key[x[0]])) if jitter is None else (lambda x: (-x[1], jitter.random())))
    return out[:k]

def run(row):
    ev = ast.literal_eval(row["EVIDENCES"])
    pos = {cui(e.split("_@_")[0]) for e in ev} - {None}
    init = cui(row["INITIAL_EVIDENCE"])
    gt = dmap[fr2en[row["PATHOLOGY"]]]["umls_cui"]
    if not init:
        return None
    conf, den, asked = {init}, set(), {init}
    il = 0; hist = deque(maxlen=10)
    for _ in range(MAX_IL):
        c = candidates(init, conf, den, asked)
        if not c:
            break
        s = c[0]
        if s in pos:
            conf.add(s)
        else:
            den.add(s)
        asked.add(s); il += 1
        dd = diag(conf, den, jitter=JIT)
        hist.append(tuple(x[0] for x in dd[:3]))
        if stop == "top3_stable_5" and len(hist) >= 5 and len(set(list(hist)[-5:])) == 1:
            break
    fin = diag(conf, den)
    return (int(bool(fin) and fin[0][0] == gt), int(any(x[0] == gt for x in fin)), il, len(ev))

t0 = time.time()
res = []; skipped = 0
with open(D + f"release_{split}_patients.csv") as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= n_rows:
            break
        r = run(row)
        if r is None:
            skipped += 1
        else:
            res.append(r)
dt = time.time() - t0
n = len(res)
print(json.dumps({"split": split, "n": n, "skipped": skipped, "deny": deny, "stop": stop, "tiebreak": tb,
                  "gtpa1": round(sum(r[0] for r in res) / n, 4), "gtpa10": round(sum(r[1] for r in res) / n, 4),
                  "avg_il": round(sum(r[2] for r in res) / n, 2), "avg_n_evidence_tokens": round(sum(r[3] for r in res) / n, 2),
                  "sec": round(dt, 1), "ms_per_patient": round(1000 * dt / max(n, 1), 2)}))
