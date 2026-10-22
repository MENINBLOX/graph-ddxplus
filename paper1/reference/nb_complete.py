import ast
import collections
import csv
import json
import math
import sys
from pathlib import Path

D = Path(sys.argv[1])
N_TRAIN = int(sys.argv[2])

fr = json.load(open(D / "release_evidences.json"))
defaults = {k: str(v.get("default_value")) for k, v in fr.items() if v["data_type"] in "MC"}


def tokens(ev_list):
    out = set()
    for e in ev_list:
        if "_@_" in e:
            code, val = e.split("_@_", 1)
            if defaults.get(code) == val:
                continue
            out.add(e)
            out.add(code)
        else:
            out.add(e)
    return out


cls_n = collections.Counter()
tok_n = collections.defaultdict(collections.Counter)
vocab = set()
with open(D / "release_train_patients.csv") as f:
    for i, row in enumerate(csv.DictReader(f)):
        if i >= N_TRAIN:
            break
        y = row["PATHOLOGY"]
        t = tokens(ast.literal_eval(row["EVIDENCES"]))
        cls_n[y] += 1
        tok_n[y].update(t)
        vocab |= t
vocab = sorted(vocab)
ntot = sum(cls_n.values())
classes = sorted(cls_n)
logp = {}
base = {}
for y in classes:
    n = cls_n[y]
    lp = {}
    b = math.log(n / ntot)
    for t in vocab:
        p = (tok_n[y][t] + 0.5) / (n + 1.0)
        lp[t] = math.log(p) - math.log(1 - p)
        b += math.log(1 - p)
    logp[y] = lp
    base[y] = b

n = c1 = c3 = 0
with open(D / "release_validate_patients.csv") as f:
    for row in csv.DictReader(f):
        t = tokens(ast.literal_eval(row["EVIDENCES"])) & set(vocab)
        s = sorted(classes, key=lambda y: -(base[y] + sum(logp[y][x] for x in t)))
        n += 1
        c1 += s[0] == row["PATHOLOGY"]
        c3 += row["PATHOLOGY"] in s[:3]
print(f"train rows={ntot} vocab={len(vocab)} val n={n} complete-profile Bernoulli NB GTPA@1={c1 / n:.4f} @3={c3 / n:.4f}")
