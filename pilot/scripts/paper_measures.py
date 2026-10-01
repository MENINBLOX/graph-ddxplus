"""Analysis measurements reported in the paper that are not diagnostic accuracy.

These used to live as throw-away scripts in a session scratch directory, and a reboot deleted them
while the paper still cited their numbers. Every figure the paper reports must be regenerable from
the repository, so they are collected here, each reading only the final KG plus the benchmark files.

    python paper_measures.py testability --kg KG.json   # attribute opportunity counts (Table 7)
    python paper_measures.py curated     --kg KG.json   # recovery of DDXPlus's curated graph (§4.5)
    python paper_measures.py oracle                     # DDXPlus separability upper bound (§4.2)
    python paper_measures.py topology    --kg KG.json   # disease coverage and degree (§4.1)

`curated` reads DDXPlus's released disease–symptom graph. That graph is a disease–symptom mapping,
which constraint C2 forbids as a KG source, so it is used here for analysis only and nothing it
contains flows back into the KG.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys

sys.path.insert(0, "pilot/scripts")

ATTRS = ("location", "character", "severity", "onset", "timing", "aggravating", "relieving")


def _edge_index(kg: dict) -> dict[str, dict[str, dict]]:
    idx: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for e in kg["edges"]:
        idx[e["d"]][e["p"]] = e.get("attrs", {}) or {}
    return idx


def _states(obj: dict, attr: str) -> bool:
    """Does this finding or edge state a value for the attribute?"""
    if attr == "location":
        v = obj.get("location_cui") or obj.get("location")
    else:
        v = obj.get(attr)
    if isinstance(v, (set, list, tuple)):
        return len(v) > 0
    return bool(v) and bool(str(v).strip())


def testability(a: argparse.Namespace) -> None:
    """Opportunity count: gold-disease findings where BOTH the patient record and the KG edge state
    the attribute. A benchmark with zero opportunities for an attribute cannot test it at all."""
    import attr_loader_ddxplus as AD
    kg = json.load(open(a.kg, encoding="utf-8"))
    edge = _edge_index(kg)

    def count(patients):
        pat, both, n = collections.Counter(), collections.Counter(), 0
        for gold, findings in patients:
            em = edge.get(gold, {})
            for f in findings:
                n += 1
                cell = em.get(f["cui"])
                for at in ATTRS:
                    if not _states(f, at):
                        continue
                    pat[at] += 1
                    if cell is not None and _states(cell, at):
                        both[at] += 1
        return n, pat, both

    import attr_loader_rarebench as AR
    import attr_loader_symcat as AS

    def gold_rows(cand, pats):
        return [(cand[p[0][0]][1], p[1]) for p in pats]

    q = json.load(open("pilot/cache/prod/nlice_queries_v2.json", encoding="utf-8"))
    benches = {"NLICE": [(p["true_cui"], p["findings"]) for p in q["patients"]]}
    benches["DDXPlus"] = gold_rows(*AD.load(3000))
    for ds in ("RAMEDIS", "LIRICAL", "HMS", "MME"):
        benches[ds] = gold_rows(*AR.load([ds]))
    benches["SymCat"] = gold_rows(*AS.load(a.ppd))
    res = {b: count(rows) for b, rows in benches.items()}
    print("patients / gold findings: " + ", ".join(f"{b} {len(benches[b])}/{res[b][0]}" for b in benches))
    print("opportunity = gold-disease findings where both the patient and the KG edge state the attribute")
    print(f"{'attribute':12}" + "".join(f"{b:>10}" for b in benches))
    for at in ATTRS:
        print(f"{at:12}" + "".join(f"{res[b][2][at]:10}" for b in benches))
    print("patient-side only (the benchmark states the attribute at all)")
    for at in ATTRS:
        print(f"{at:12}" + "".join(f"{res[b][1][at]:10}" for b in benches))


def curated(a: argparse.Namespace) -> None:
    import onlykg_eval_unified as U
    ref = json.load(open("data/ddxplus/kg_edges.json", encoding="utf-8"))
    dz_ref = set(ref["diseases"])
    cur: dict[str, set] = collections.defaultdict(set)
    for x, y in ref["edges"]:                      # released as [symptom, disease]
        if y in dz_ref and x not in dz_ref:
            cur[y].add(x)
        elif x in dz_ref and y not in dz_ref:
            cur[x].add(y)
        elif x in dz_ref and y in dz_ref:
            cur[y].add(x)                          # disease as evidence (antecedent)
    sym_name = {c: v["name"] for c, v in ref["symptoms"].items()}

    kg, prof, _idf, dz = U.load_kg(a.kg)
    children = json.load(open("pilot/cache/umls_maps/disease_children.json", encoding="utf-8"))
    R = U.Resolver(prof, dz, children)
    cand, pats = U.load_ddxplus(3000)
    reported: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for tk, ev, _ in pats:
        for k in tk:
            reported[k].update(ev)
    cu2key = {cand[k][1]: k for k in cand}

    tot = hit = extra = extra_used = 0
    sizes = []
    miss = collections.Counter()
    for dcui, syms in cur.items():
        k = cu2key.get(dcui)
        if not k:
            continue
        nm, cu = cand[k]
        p = set(U.merged_profile(R.cluster(nm, cu), prof))
        sizes.append((len(syms), len(p)))
        tot += len(syms)
        hit += len(syms & p)
        ex = p - syms
        extra += len(ex)
        extra_used += sum(1 for c in ex if reported[k].get(c))
        for s in syms - p:
            miss[sym_name.get(s, s)] += 1
    print(f"curated edges {tot}; recovered by the KG {hit} ({100 * hit / max(1, tot):.1f}%)")
    print(f"per-disease concepts: curated mean {sum(s[0] for s in sizes) / len(sizes):.1f}, "
          f"KG profile mean {sum(s[1] for s in sizes) / len(sizes):.1f}")
    print(f"KG concepts absent from the curation {extra}; reported by these patients {extra_used}")
    print("most-missed curated items (diseases):", miss.most_common(12))


def oracle(a: argparse.Namespace) -> None:
    """Profile each disease as the union of its own patients' evidence and rank by overlap. This
    uses the patient-derived mapping C2 forbids, so it is an upper bound on separability only."""
    import onlykg_eval_unified as U
    cand, pats = U.load_ddxplus(3000)
    prof: dict[str, set] = collections.defaultdict(set)
    for tk, ev, _ in pats:
        for k in tk:
            prof[k] |= set(ev)
    keys = sorted(cand)
    h1 = h3 = 0
    for tk, ev, _ in pats:
        sc = {k: len(ev & prof[k]) / max(1.0, math.sqrt(len(prof[k]))) for k in keys}
        order = sorted(keys, key=lambda k: (-sc[k], k))
        r = min(order.index(k) + 1 for k in tk)
        h1 += r == 1
        h3 += r <= 3
    n = len(pats)
    print(f"DDXPlus oracle over {len(keys)} diseases, {n} patients: "
          f"Top-1 {100 * h1 / n:.2f}  Top-3 {100 * h3 / n:.2f}")


def topology(a: argparse.Namespace) -> None:
    kg = json.load(open(a.kg, encoding="utf-8"))
    deg = collections.Counter(e["d"] for e in kg["edges"])
    all_dz = set(kg["diseases"])
    with_edges = [deg[d] for d in all_dz if deg[d] > 0]
    print(f"diseases {len(all_dz)}; with at least one edge {len(with_edges)}; "
          f"with none {len(all_dz) - len(with_edges)}")
    print(f"degree over diseases with edges: median {statistics.median(with_edges):.0f}; "
          f"<=5 evidence {sum(1 for x in with_edges if x <= 5)} "
          f"({100 * sum(1 for x in with_edges if x <= 5) / len(with_edges):.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("testability", "curated", "topology"):
        sp = sub.add_parser(name)
        sp.add_argument("--kg", required=True)
        if name == "testability":
            sp.add_argument("--ppd", type=int, default=30, help="SymCat patients per disease (evaluator default)")
    sub.add_parser("oracle")
    a = ap.parse_args()
    {"testability": testability, "curated": curated, "oracle": oracle, "topology": topology}[a.cmd](a)


if __name__ == "__main__":
    main()
