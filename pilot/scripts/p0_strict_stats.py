"""P0-3 (research design §9.32): strict versus lineage judging of the E-T2 comparisons, confirmation set E, POST HOC.

The re-ranked runs (run_p0_strict.sh) must reproduce the confirmation run first: the same arguments, per-patient
ranks_lineage equal to pilot/cache/v2/confirmE/<S>.json, the same stored strict@k and lineage@k, and the lineage
macro rows equal to the confirmation paired files. Then for S-mix - S-llm, - S-dict, - S-union under each rule
and k in (10, 1): paired_stats.compare per set (patient CI, exact McNemar with Holm over the two sets, gold-disease
cluster CI) and paired_stats.macro_compare over both sets. §9.32 rule: a comparison whose strict difference has the
other sign and whose strict CI excludes 0 depends on the judging rule.

Lineage bonus per graph and set: Top-k hits that are lineage-only (a non-gold member of the gold's lineage ranked
within k, no gold key), the number of lineage members in the Top-10, and the gold diseases that collect the most
lineage-only hits with their lineage size (Disease IS_A*0..3 below the gold, the same in every graph). Per gold
disease, the Top-10 strict and lineage-only hits of every system and the other gold classes of the same set that lie
inside its lineage.

    CONFIRM_E_UNSEAL=1 python pilot/scripts/p0_strict_stats.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import opendx_eval
import v2_ids
from graph_dx import LINEAGE
from neo4j import GraphDatabase
from paired_stats import compare, macro_compare

CONF = Path("pilot/cache/v2/confirmE")
RUNS = Path("pilot/cache/v3/p0/strict")
OUT = RUNS / "stats.json"
URI = "bolt://localhost:7688"
SYSTEMS = ("S-mix", "S-llm", "S-dict", "S-union")
PAIRS = [("S-llm", "S-mix"), ("S-dict", "S-mix"), ("S-union", "S-mix")]     # (A, B), B - A as in run_confirm_E.sh
RULES = ("lineage", "strict")
SAME_ARGS = ("bench", "ddx_n", "symcat_ppd", "alpha", "mu", "decay", "unified_w", "support_beta", "offtopic_w",
             "full_unified", "uri", "closed")
BUCKETS = ("0", "1", "2", "3-4", "5+")


def hit(p: int | None, k: int) -> bool:
    return p is not None and p <= k


def check_reproduction(system: str, new: dict, old: dict) -> dict:
    bad = [a for a in SAME_ARGS if new["args"][a] != old["args"][a]]
    if bad:
        raise SystemExit(f"{system}: arguments differ from the confirmation run: {bad}")
    old_r = {r["bench"]: r for r in old["results"]}
    if sorted(old_r) != sorted(r["bench"] for r in new["results"]):
        raise SystemExit(f"{system}: sets differ from the confirmation run")
    out = {}
    for r in new["results"]:
        o = old_r[r["bench"]]
        if r["ranks_lineage"] != o["ranks_lineage"]:
            diff = sum(x != y for x, y in zip(r["ranks_lineage"], o["ranks_lineage"], strict=True))
            raise SystemExit(f"{system} {r['bench']}: ranks_lineage differ for {diff} patients")
        keys = ["evaluated"] + [f"{rule}@{k}" for rule in RULES for k in opendx_eval.KS]
        if any(r[key] != o[key] for key in keys):
            raise SystemExit(f"{system} {r['bench']}: stored rates differ from the confirmation run")
        for g, d, s, lin in zip(r["top_gold"], r["top_descendants"], r["ranks_strict"], r["ranks_lineage"],
                                strict=True):
            if (g[0] if g else None) != s or (min(g + d) if g + d else None) != lin:
                raise SystemExit(f"{system} {r['bench']}: top positions disagree with the ranks")
        out[r["bench"]] = {"patients": r["evaluated"], "ranks_lineage_equal": True, "stored_rates_equal": True}
    return out


def as_rule(run: dict, rule: str) -> dict:
    """paired_stats reads ranks_lineage; this run carries the chosen rule's ranks under that key."""
    return {"results": [{**r, "ranks_lineage": r[f"ranks_{rule}"]} for r in run["results"]]}


def side(ci: list[float]) -> int:
    return 1 if ci[0] > 0 else -1 if ci[1] < 0 else 0


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


def bucket(m: int) -> str:
    return str(m) if m <= 2 else "3-4" if m <= 4 else "5+"


def lineage_bonus(r: dict, clusters: list, lineage_size: dict, names: dict) -> dict:
    n = r["evaluated"]
    res: dict = {"n": n}
    for k in opendx_eval.KS:
        strict = sum(hit(p, k) for p in r["ranks_strict"])
        lin = sum(hit(p, k) for p in r["ranks_lineage"])
        res[f"@{k}"] = {"strict_pct": round(100 * strict / n, 2), "lineage_pct": round(100 * lin / n, 2),
                        "lineage_only": lin - strict, "lineage_only_pct": round(100 * (lin - strict) / n, 2),
                        "lineage_only_share_of_lineage_hits_pct": round(100 * (lin - strict) / max(1, lin), 1)}
    pairs = list(zip(r["top_gold"], r["top_descendants"], strict=True))
    members = [len(g) + len(d) for g, d in pairs]
    only = [not g and bool(d) for g, d in pairs]
    lin_hits = sum(m > 0 for m in members)
    several = sum(m >= 2 for m in members)
    res["top10"] = {
        "lineage_members_distribution": {b: Counter(bucket(m) for m in members).get(b, 0) for b in BUCKETS},
        "several_members": several, "several_members_pct": round(100 * several / n, 2),
        "several_members_share_of_lineage_hits_pct": round(100 * several / max(1, lin_hits), 1),
        "mean_members_when_any": round(sum(members) / max(1, lin_hits), 2),
        "gold_and_descendant": sum(bool(g) and bool(d) for g, d in pairs),
        "lineage_only_with_several_descendants": sum(o and len(d) >= 2 for o, (_, d) in zip(only, pairs, strict=True)),
        "descendant_slots_pct": round(100 * sum(len(d) for d in r["top_descendants"]) / (10 * n), 2),
    }
    by_gold: dict = defaultdict(lambda: [0, 0, 0])            # patients, strict hits, lineage-only hits at Top-10
    for key, p, o in zip(clusters, r["ranks_strict"], only, strict=True):
        by_gold[key][0] += 1
        by_gold[key][1] += hit(p, 10)
        by_gold[key][2] += o
    total_only = sum(v[2] for v in by_gold.values())
    top = sorted(by_gold.items(), key=lambda kv: (-kv[1][2], kv[0]))[:5]
    res["golds_with_lineage_only_hits"] = sum(v[2] > 0 for v in by_gold.values())
    res["golds"] = len(by_gold)
    res["top_golds"] = [{"gold": list(key), "name": " / ".join(names.get(g, g) for g in key),
                         "lineage_size_non_gold": lineage_size[key], "patients": v[0], "strict_hits": v[1],
                         "lineage_only_hits": v[2],
                         "share_of_lineage_only_pct": round(100 * v[2] / max(1, total_only), 1)}
                        for key, v in top]
    with_desc = [key for key in by_gold if lineage_size[key] > 0]
    res["golds_with_descendants"] = len(with_desc)
    res["patients_with_descendant_golds_pct"] = round(100 * sum(by_gold[key][0] for key in with_desc) / n, 1)
    return res


def lineages(clusters: dict[str, list]) -> dict[tuple, set[str]]:
    keys = sorted({key for v in clusters.values() for key in v})
    drv = GraphDatabase.driver(URI, auth=("neo4j", "thesis-dev-only"))
    with drv.session() as s:
        return {key: set(s.run(LINEAGE, gold=list(key)).single()["ids"]) - set(key) for key in keys}


def per_gold(runs: dict[str, dict], bench: str, keys: list, lin: dict, names: dict) -> list[dict]:
    """Top-10 strict and lineage-only hits per gold disease and system. other_classes_in_lineage: gold diseases of
    the same set inside this gold's lineage, so a lineage-only hit may be the benchmark's own other class."""
    classes = {k for key in keys for k in key}
    out = {key: {"gold": list(key), "name": " / ".join(names.get(g, g) for g in key), "patients": n,
                 "lineage_size_non_gold": len(lin[key]),
                 "other_classes_in_lineage": sorted(names.get(c, c) for c in lin[key] & classes),
                 "strict_hits": dict.fromkeys(runs, 0), "lineage_only_hits": dict.fromkeys(runs, 0)}
           for key, n in Counter(keys).items()}
    for s, run in runs.items():
        r = next(x for x in run["results"] if x["bench"] == bench)
        for key, g, d in zip(keys, r["top_gold"], r["top_descendants"], strict=True):
            out[key]["strict_hits"][s] += bool(g)
            out[key]["lineage_only_hits"][s] += not g and bool(d)
    return sorted(out.values(), key=lambda v: (-v["patients"], v["name"]))


def main() -> None:
    if not (CONF / "DONE").exists():
        raise SystemExit("confirmation E has not finished")
    runs = {s: json.loads((RUNS / f"{s}.json").read_text()) for s in SYSTEMS}
    repro = {s: check_reproduction(s, runs[s], json.loads((CONF / f"{s}.json").read_text())) for s in SYSTEMS}

    opendx_eval.RAW_GOLD = True
    conf = opendx_eval.load_confirm_e()
    clusters = {b: [tuple(sorted(v2_ids.gold_keys(g))) for g, ev in pats if v2_ids.gold_keys(g) and ev]
                for b, pats in conf.items()}
    sets = sorted(conf)
    for s in SYSTEMS:
        for r in runs[s]["results"]:
            if len(clusters[r["bench"]]) != r["evaluated"]:
                raise SystemExit(f"{s} {r['bench']}: {len(clusters[r['bench']])} gold keys "
                                 f"for {r['evaluated']} patients")

    rows, macros = [], []
    for a, b in PAIRS:
        for rule in RULES:
            ra, rb = as_rule(runs[a], rule), as_rule(runs[b], rule)
            for k in (10, 1):
                for r in compare(ra, rb, k, clusters=clusters, require=sets):
                    rows.append({"comparison": f"{b} - {a}", "rule": rule, "k": k, "bench": r["bench"], "n": r["n"],
                                 "A_pct": r[f"A@{k}"], "B_pct": r[f"B@{k}"], "diff": r["diff"], "ci": r["ci"],
                                 "ci_cluster": r["ci_cluster"], "diseases": r["clusters"], "only_A": r["only_A"],
                                 "only_B": r["only_B"], "p": r["p"], "p_holm": r["p_holm"]})
                m = macro_compare(ra, rb, k, sets)
                macros.append({"comparison": f"{b} - {a}", "rule": rule, "k": k, "A": m["A"], "B": m["B"],
                               "diff": m["diff"], "ci": m["ci"], "p": m["p"]})
                if rule == "lineage":
                    reg = json.loads((CONF / f"paired_{a}_vs_{b}_k{k}.json").read_text())["macro"]
                    if (reg["diff"], reg["ci"]) != (m["diff"], m["ci"]):
                        raise SystemExit(f"{b} - {a} @{k}: lineage macro {m['diff']} {m['ci']} != confirmation "
                                         f"{reg['diff']} {reg['ci']}")

    def find(table: list[dict], comp: str, rule: str, k: int, bench: str | None = None) -> dict:
        return next(x for x in table if x["comparison"] == comp and x["rule"] == rule and x["k"] == k
                    and (bench is None or x["bench"] == bench))

    rule_check = []
    for a, b in PAIRS:
        comp = f"{b} - {a}"
        for k in (10, 1):
            for unit in [None, *sets]:
                lin = find(macros if unit is None else rows, comp, "lineage", k, unit)
                st = find(macros if unit is None else rows, comp, "strict", k, unit)
                flip = sign(st["diff"]) != sign(lin["diff"])
                rule_check.append({
                    "comparison": comp, "k": k, "unit": unit or "macro", "e_t2_judged_unit": unit is None and k == 10,
                    "lineage_diff": lin["diff"], "lineage_ci": lin["ci"],
                    "strict_diff": st["diff"], "strict_ci": st["ci"],
                    "sign_flip": flip, "strict_ci_excludes_0": side(st["ci"]) != 0,
                    "depends_on_rule": flip and side(st["ci"]) != 0,
                    "non_inferior_lineage": lin["ci"][0] > -1, "non_inferior_strict": st["ci"][0] > -1})

    names = {key: v["name"] for key, v in v2_ids.universe().items()}
    lin = lineages(clusters)
    size = {key: len(v) for key, v in lin.items()}
    golds = {bench: per_gold(runs, bench, clusters[bench], lin, names) for bench in sets}
    bonus = {s: {r["bench"]: lineage_bonus(r, clusters[r["bench"]], size, names) for r in runs[s]["results"]}
             for s in SYSTEMS}
    decomposition = []
    for a, b in PAIRS:
        for bench in sets:
            x, y = bonus[a][bench]["@10"], bonus[b][bench]["@10"]
            decomposition.append({"comparison": f"{b} - {a}", "bench": bench,
                                  "lineage_diff": round(y["lineage_pct"] - x["lineage_pct"], 2),
                                  "strict_diff": round(y["strict_pct"] - x["strict_pct"], 2),
                                  "lineage_only_diff": round(y["lineage_only_pct"] - x["lineage_only_pct"], 2)})

    res = {"note": "POST HOC on confirmation set E (research design §9.32 P0-3). diff = B - A in percentage points of "
                   "Top-k under the named rule; ci = patient-paired bootstrap 95% (2,000, seed 0); ci_cluster = gold-"
                   "disease cluster bootstrap 95%; p = exact McNemar, p_holm = Holm across the two sets of one "
                   "comparison, rule and k; macro = paired_stats.macro_compare (equal weight, set-stratified). "
                   "lineage-only = a non-gold member of the gold's lineage (Disease IS_A*0..3 below a gold key) "
                   "ranked within k while no gold key is",
           "runs": {s: str(RUNS / f"{s}.json") for s in SYSTEMS}, "reproduction": repro,
           "rows": rows, "macro": macros, "rule_check": rule_check, "lineage_bonus": bonus,
           "difference_decomposition_top10": decomposition, "per_gold_top10": golds}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))

    print("reproduction: ranks_lineage, stored rates and lineage macro rows equal the confirmation run for "
          + ", ".join(SYSTEMS))
    print(f"\n{'comparison':17s} {'rule':7s} {'k':>3s} {'set':17s} {'A':>6s} {'B':>6s} {'diff':>6s} "
          f"{'patient CI':>16s} {'disease CI':>16s} {'Holm p':>9s}")
    for r in rows:
        print(f"{r['comparison']:17s} {r['rule']:7s} @{r['k']:<2d} {r['bench']:17s} "
              f"{r['A_pct']:6.2f} {r['B_pct']:6.2f} {r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] "
              f"[{r['ci_cluster'][0]:+6.2f},{r['ci_cluster'][1]:+6.2f}] {r['p_holm']:9.2g}")
    print("\nmacro (both sets, equal weight, set-stratified patient bootstrap):")
    for m in macros:
        print(f"  {m['comparison']:17s} {m['rule']:7s} @{m['k']:<2d} A {m['A']:6.2f} B {m['B']:6.2f} "
              f"{m['diff']:+6.2f} [{m['ci'][0]:+6.2f},{m['ci'][1]:+6.2f}]")
    print("\n§9.32 rule (strict sign differs from lineage and strict CI excludes 0):")
    for c in rule_check:
        print(f"  {c['comparison']:17s} @{c['k']:<2d} {c['unit']:25s} lineage {c['lineage_diff']:+6.2f} "
              f"strict {c['strict_diff']:+6.2f} [{c['strict_ci'][0]:+6.2f},{c['strict_ci'][1]:+6.2f}] "
              f"flip {c['sign_flip']!s:5s} depends_on_rule {c['depends_on_rule']}")
    print("\nlineage bonus at Top-10 (per graph and set):")
    for s in SYSTEMS:
        for bench, v in bonus[s].items():
            t, x = v["@10"], v["top10"]
            print(f"  {s:8s} {bench:17s} strict {t['strict_pct']:5.2f} lineage {t['lineage_pct']:5.2f} "
                  f"lineage-only {t['lineage_only']:4d} ({t['lineage_only_pct']:5.2f}%, "
                  f"{t['lineage_only_share_of_lineage_hits_pct']:4.1f}% of hits) "
                  f"| members {x['lineage_members_distribution']} several {x['several_members_pct']:5.2f}% "
                  f"({x['several_members_share_of_lineage_hits_pct']:4.1f}% of hits)")
            for g in v["top_golds"][:3]:
                print(f"      {g['name'][:45]:45s} lineage {g['lineage_size_non_gold']:3d} n {g['patients']:4d} "
                      f"strict {g['strict_hits']:4d} lineage-only {g['lineage_only_hits']:4d} "
                      f"({g['share_of_lineage_only_pct']}%)")
    print("\nTop-10 difference = strict difference + lineage-only difference:")
    for d in decomposition:
        print(f"  {d['comparison']:17s} {d['bench']:17s} {d['lineage_diff']:+6.2f} = {d['strict_diff']:+6.2f} "
              f"+ {d['lineage_only_diff']:+6.2f}")
    print("\nlineage-only Top-10 hits of golds whose lineage holds another gold class of the same set:")
    for bench in sets:
        nested = [g for g in golds[bench] if g["other_classes_in_lineage"]]
        print(f"  {bench}: {len(nested)} of {len(golds[bench])} golds; lineage-only hits "
              + ", ".join(f"{s} {sum(g['lineage_only_hits'][s] for g in nested)}/"
                          f"{sum(g['lineage_only_hits'][s] for g in golds[bench])}" for s in SYSTEMS))
    print("\nDDXPlus-E golds driving each Top-10 difference (B - A patients: strict, lineage-only):")
    ddx = next(b for b in sets if "DDXPlus" in b)
    for a, b in PAIRS:
        print(f"  {b} - {a}")
        drive = sorted(golds[ddx], key=lambda g: -(abs(g["strict_hits"][b] - g["strict_hits"][a])
                                                     + abs(g["lineage_only_hits"][b] - g["lineage_only_hits"][a])))
        for g in drive[:6]:
            print(f"    {g['name'][:32]:32s} n {g['patients']:4d} lineage {g['lineage_size_non_gold']:3d} "
                  f"strict {g['strict_hits'][b] - g['strict_hits'][a]:+4d} "
                  f"lineage-only {g['lineage_only_hits'][b] - g['lineage_only_hits'][a]:+4d} "
                  f"other classes {g['other_classes_in_lineage']}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
