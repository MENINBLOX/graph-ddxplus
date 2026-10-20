"""Stage tables of D and the v3 graphs, and triple-weighted transitions from D's verdict per pair."""
import json
import sys
from collections import Counter

STAGES = ["no_docs", "not_in_text", "not_extracted", "not_linked", "filtered", "offtopic_only", "near", "edge"]
D = sys.argv[1]                     # D pairs run carrying "n"
runs = {"D": D} | {g: f"pilot/cache/v3/recall/decomp_{g}" for g in sys.argv[2:]}
res = {g: json.load(open(p + ".json")) for g, p in runs.items()}
raw = {g: json.load(open(p + "_pairs.json")) for g, p in runs.items()}
w = {(x["d"], x["hpo"]): x["n"] for x in raw["D"]}
pairs = {g: {(x["d"], x["hpo"]): x["v"] for x in v} for g, v in raw.items()}
assert all(p.keys() == w.keys() for p in pairs.values())
assert sum(w.values()) == 82751

for unit in ("by_triple", "by_pair"):
    print(unit)
    print(f"{'':9}" + "".join(f"{s:>17}" for s in STAGES))
    for g, r in res.items():
        print(f"{g:9}" + "".join(f"{r[unit][s][0]:>10} {r[unit][s][1]:>5.1f}" for s in STAGES))
for g, r in res.items():
    t = r["by_triple"]
    print(f"{g:7} collection {t['no_docs'][0] + t['not_in_text'][0]:>6} ({t['no_docs'][1] + t['not_in_text'][1]:.1f}%)  "
          f"extract+link {t['not_extracted'][0] + t['not_linked'][0]:>6} ({t['not_extracted'][1] + t['not_linked'][1]:.1f}%)  "
          f"filtered+offtopic {t['filtered'][0] + t['offtopic_only'][0]:>6} ({t['filtered'][1] + t['offtopic_only'][1]:.1f}%)  "
          f"near+edge {t['near'][0] + t['edge'][0]:>6} ({t['near'][1] + t['edge'][1]:.1f}%)")
for g in list(runs)[1:]:
    print(f"\n== D -> {g}, triples (rows: D verdict, columns: {g} verdict)")
    tr = Counter()
    for k, v in pairs["D"].items():
        tr[(v, pairs[g][k])] += w[k]
    print(f"{'':14}" + "".join(f"{s[:9]:>10}" for s in STAGES))
    for s in STAGES:
        print(f"{s:14}" + "".join(f"{tr[(s, t)]:>10}" for t in STAGES))
