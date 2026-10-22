"""Full-profile KB-Bayes (membership, uniform prior): which answers carry the accuracy? validation sample = random.Random(0).sample(.., N)."""
import ast, csv, json, random, sys
import numpy as np
D = "/home/max/thesis/data/ddxplus/"
N = int(sys.argv[1])
fr = json.load(open(D + "release_evidences.json")); en = json.load(open(D + "release_evidences_en.json"))
cond = json.load(open(D + "release_conditions.json"))
q2fr = {v["question_en"].strip(): k for k, v in fr.items()}; e2f = {k: q2fr[v["question_en"].strip()] for k, v in en.items()}
CODES = sorted(fr); CI = {c: i for i, c in enumerate(CODES)}
DEF = {k: str(v.get("default_value")) for k, v in fr.items() if v["data_type"] in "MC"}
DIS = sorted(c["cond-name-fr"] for c in cond.values()); DI = {d: i for i, d in enumerate(DIS)}
M = np.zeros((len(DIS), len(CODES)), dtype=bool)
for c in cond.values():
    for x in list(c["symptoms"]) + list(c["antecedents"]):
        M[DI[c["cond-name-fr"]], CI[e2f[x]]] = True
def pos_codes(ev):
    out = set()
    for e in ev:
        if "_@_" in e:
            c, v = e.split("_@_", 1)
            if DEF.get(c) == v: continue
            out.add(c)
        else: out.add(e)
    return out
a, eps = 0.5, 1e-3
LY, LN = np.log(np.where(M, a, eps)), np.log(np.where(M, 1 - a, 1 - eps))
rows = list(csv.DictReader(open(D + "release_validate_patients.csv")))
rows = random.Random(0).sample(rows, N)
acc = {"both": 0.0, "pos_only": 0.0, "neg_only": 0.0}
for r in rows:
    t = DI[r["PATHOLOGY"]]
    x = np.zeros(len(CODES), bool); x[[CI[c] for c in pos_codes(ast.literal_eval(r["EVIDENCES"]))]] = True
    sy, sn = LY[:, x].sum(1), LN[:, ~x].sum(1)
    for k, s in (("both", sy + sn), ("pos_only", sy), ("neg_only", sn)):
        tied = np.isclose(s, s.max(), rtol=0, atol=1e-9)
        acc[k] += tied[t] / tied.sum() if tied[t] else 0.0
print({k: round(v / N, 4) for k, v in acc.items()}, "n", N)
