import ast
import collections
import csv
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
SPLIT = sys.argv[2]
N = int(sys.argv[3])

fr = json.load(open(D / "release_evidences.json"))
en = json.load(open(D / "release_evidences_en.json"))
cond = json.load(open(D / "release_conditions.json"))
mp = json.load(open(D / "umls_mapping.json"))["mapping"]
dm = json.load(open(D / "disease_umls_mapping.json"))["mapping"]
kg = json.load(open(D / "kg_edges.json"))

q2fr = {v["question_en"].strip(): k for k, v in fr.items()}
e2f = {k: q2fr[v["question_en"].strip()] for k, v in en.items()}

dis_codes = {}
fr2eng = {}
for name, c in cond.items():
    codes = {e2f[x] for x in c["symptoms"]} | {e2f[x] for x in c["antecedents"]}
    dis_codes[name] = codes
    fr2eng[c["cond-name-fr"]] = name

gen_edges = set()
for name, codes in dis_codes.items():
    dc = dm[name]["umls_cui"]
    for code in codes:
        cui = mp.get(code, {}).get("cui")
        if cui:
            gen_edges.add((cui, dc))
kg_edges = {tuple(e) for e in kg["edges"]}
print("generator-projected CUI edges", len(gen_edges), "kg_edges.json", len(kg_edges),
      "intersection", len(gen_edges & kg_edges), "kg-only", len(kg_edges - gen_edges),
      "gen-only", len(gen_edges - kg_edges))

dis_cuisets = {
    name: {mp[c]["cui"] for c in codes if mp.get(c, {}).get("cui")}
    for name, codes in dis_codes.items()
}

n = 0
subset_viol = 0
outside_codes = collections.Counter()
trav_present = 0
trav_N = 0
consistent_counts = collections.Counter()
unique_correct = 0
cui_consistent_counts = collections.Counter()
init_in_list = 0
with open(D / f"release_{SPLIT}_patients.csv") as f:
    for row in csv.DictReader(f):
        n += 1
        if n > N:
            break
        ev = ast.literal_eval(row["EVIDENCES"])
        base = {e.split("_@_")[0] for e in ev}
        path = fr2eng[row["PATHOLOGY"]]
        out = base - dis_codes[path]
        if out:
            subset_viol += 1
            outside_codes.update(out)
        if "trav1" in base:
            trav_present += 1
            if "trav1_@_N" in ev:
                trav_N += 1
        if row["INITIAL_EVIDENCE"] in dis_codes[path]:
            init_in_list += 1
        cons = [d for d, codes in dis_codes.items() if base <= codes]
        consistent_counts[len(cons)] += 1
        if cons == [path]:
            unique_correct += 1
        pos_cuis = {mp[c]["cui"] for c in base if mp.get(c, {}).get("cui")}
        ccons = [d for d, cs in dis_cuisets.items() if pos_cuis <= cs]
        cui_consistent_counts[min(len(ccons), 6)] += 1
n = min(n, N)
print(f"split={SPLIT} n={n}")
print("patients with any positive evidence outside true pathology's generator list:", subset_viol, outside_codes.most_common(5))
print("initial evidence in true pathology list:", init_in_list)
print("trav1 present:", trav_present, " of which trav1_@_N (no travel):", trav_N)
print("# diseases whose generator list contains ALL positive evidence codes:", sorted(consistent_counts.items())[:10])
print("hard set-inclusion filter alone leaves only the true disease:", unique_correct, f"{unique_correct / n:.4f}")
print("# diseases consistent at mapped-CUI level (capped 6):", sorted(cui_consistent_counts.items()))
