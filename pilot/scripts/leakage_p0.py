"""Development results without the §9.24 leakage-flagged patients (research design §9.32 P0-1).

Flags come from leakage_mirror.py (one document supports >= 50% and >= 5 of the patient's concepts for a
gold disease). A comparison drops the patients flagged in either of its two graphs, so both runs keep the
same patients.

(a) Collection depth: the seven same-extraction pairs of variants_analysis.py (v2 graph A -> its v2+v3
    counterpart B, each at the offtopic_w the D3 rule selected), equal-weight macro lineage Top-10 over the
    nine development sets. Rule: the depth effect stays a development-level claim when every pair keeps at
    least half of its original difference; otherwise it is reported as leakage-sensitive.
(b) S-mix - S-dict over the seven rare sets (frozen design E offtopic_w .5 both). Rule: kept when the
    set-stratified patient bootstrap CI (paired_stats.macro_compare) has a lower bound > 0, dropped otherwise.

Sensitivity rows (not part of the rule): article unit (an article's table document counts as the article)
and one patient set for every comparison (patients flagged in any of the fifteen graphs).

    python pilot/scripts/leakage_p0.py
"""
from __future__ import annotations

import json
from pathlib import Path

from paired_stats import compare, macro_compare
from variants_analysis import gold_clusters

VARIANTS = Path("pilot/cache/v2/devD/variants")
OUT = Path("pilot/cache/v3/p0/leakage")
DEPTH_PAIRS = [("v2dict", "v3dict"), ("v2umls", "v3umls"), ("v2dict_ns", "v3dict_ns"), ("v2umls_ns", "v3umls_ns"),
               ("base_split_tg_ns", "v3llm_ns"), ("v2union_ns", "v3union_ns"), ("v2unionu_ns", "v3unionu_ns")]
SMIX, SDICT = "v3llmd_ns", "v3umls_ns"
FROZEN_OT = {SMIX: "0.5", SDICT: "0.5"}
NINE = ["DDXPlus-test", "PUMCH-ADM", "PUMCH_L", "Phenopacket-store", "RareBench-HMS", "RareBench-LIRICAL",
        "RareBench-MME", "RareBench-RAMEDIS", "SymCat"]
RARE = [s for s in NINE if s not in ("DDXPlus-test", "SymCat")]
KEEP_RATIO = 0.5


def subset(run: dict, keep: dict[str, list[bool]]) -> dict:
    res = []
    for r in run["results"]:
        k = keep[r["bench"]]
        if len(k) != len(r["ranks_lineage"]):
            raise SystemExit(f"{r['bench']}: {len(r['ranks_lineage'])} ranked patients, {len(k)} flags")
        res.append({"bench": r["bench"], "ranks_lineage": [p for p, x in zip(r["ranks_lineage"], k) if x]})
    return {"results": res}


def keep_mask(flags: dict, graphs: list[str], key: str) -> dict[str, list[bool]]:
    sets = flags["graphs"][graphs[0]]["sets"]
    return {b: [not any(flags["graphs"][g]["sets"][b][key][i] for g in graphs) for i in range(len(sets[b][key]))]
            for b in sets}


def hits(run: dict, bench: str, k: int, sel: list[bool]) -> tuple[float | None, int]:
    rk = next(r["ranks_lineage"] for r in run["results"] if r["bench"] == bench)
    xs = [p is not None and p <= k for p, s in zip(rk, sel) if s]
    return (round(100 * sum(xs) / len(xs), 2) if xs else None), len(xs)


def macro_row(a: dict, b: dict, sets: list[str], keep: dict | None, k: int) -> dict:
    if keep is not None:
        a, b = subset(a, keep), subset(b, keep)
    m = macro_compare(a, b, k, sets)
    return {"A": m["A"], "B": m["B"], "diff": m["diff"], "ci": m["ci"], "p": m["p"],
            "n": {s: v["n"] for s, v in m["per_set"].items()},
            "per_set_diff": {s: v["diff"] for s, v in m["per_set"].items()}}


def main() -> None:
    flags = json.loads((OUT / "flags.json").read_text())
    bad = {g: e["mismatch"] for g, e in flags["graphs"].items() if not e["summary_match"]}
    mc = flags["graphs"][SMIX]["mention_check"]
    if bad or mc["pairs_differing"]:
        raise SystemExit(f"mirror does not reproduce the builds: {bad}, S-mix Mention check {mc}")
    sel = json.loads((VARIANTS / "analysis.json").read_text())["dev"]
    ot = {g: sel[g]["selected_offtopic_w"] for pair in DEPTH_PAIRS for g in pair}
    if any(ot.get(g, w) != w for g, w in FROZEN_OT.items()):
        raise SystemExit(f"selected offtopic_w differs from design E: {ot}")
    ot |= FROZEN_OT
    runs = {g: json.loads((VARIANTS / f"dev_{g}_ot{w}.json").read_text()) for g, w in ot.items()}
    every = list(flags["graphs"])
    res = {"note": "diff = B - A, macro lineage Top-k over the named sets (equal weight), percentage points; "
                   "ci = set-stratified patient bootstrap 95% (paired_stats.macro_compare, 2,000, seed 0); "
                   "excluded = patients flagged in A or B by the §9.24 rule (document unit)",
           "offtopic_w": ot, "flagged": {g: e["flagged"] for g, e in flags["graphs"].items()},
           "flagged_article": {g: e["flagged_article"] for g, e in flags["graphs"].items()},
           "check": {g: e["mirror"] for g, e in flags["graphs"].items()}, "mention_check": mc}

    depth = []
    for a, b in DEPTH_PAIRS:
        row = {"A": a, "B": b}
        for k in (10, 1):
            orig = macro_row(runs[a], runs[b], NINE, None, k)
            excl = macro_row(runs[a], runs[b], NINE, keep_mask(flags, [a, b], "flag"), k)
            art = macro_row(runs[a], runs[b], NINE, keep_mask(flags, [a, b], "article_flag"), k)
            glob = macro_row(runs[a], runs[b], NINE, keep_mask(flags, every, "flag"), k)
            row[f"@{k}"] = {"all": orig, "excluded": excl, "excluded_article": art, "excluded_any_graph": glob,
                            "kept_ratio": round(excl["diff"] / orig["diff"], 3) if orig["diff"] else None,
                            "kept_ratio_article": round(art["diff"] / orig["diff"], 3) if orig["diff"] else None,
                            "kept_ratio_any_graph": round(glob["diff"] / orig["diff"], 3) if orig["diff"] else None}
        depth.append(row)
    keep_depth = all(r["@10"]["kept_ratio"] is not None and r["@10"]["kept_ratio"] >= KEEP_RATIO for r in depth)
    res["depth_pairs"] = depth
    res["verdict_depth"] = ("keep: every pair retains >= half of its macro Top-10 depth effect"
                            if keep_depth else "leakage-sensitive: a pair retains < half of its macro Top-10 depth effect")

    clusters = gold_clusters()
    km = keep_mask(flags, [SMIX, SDICT], "flag")
    rare = {}
    for k in (10, 1):
        rare[f"@{k}"] = {"all": macro_row(runs[SDICT], runs[SMIX], RARE, None, k),
                         "excluded": macro_row(runs[SDICT], runs[SMIX], RARE, km, k),
                         "excluded_article": macro_row(runs[SDICT], runs[SMIX], RARE,
                                                       keep_mask(flags, [SMIX, SDICT], "article_flag"), k),
                         "excluded_any_graph": macro_row(runs[SDICT], runs[SMIX], RARE,
                                                         keep_mask(flags, every, "flag"), k)}
        cl = {s: [c for c, x in zip(clusters[s], km[s]) if x] for s in RARE}
        rows = compare(subset(runs[SDICT], km), subset(runs[SMIX], km), k, clusters=cl, require=RARE)
        rare[f"@{k}"]["per_set_excluded"] = [r for r in rows if r["bench"] in RARE]
    flagged_acc = {}
    for g in (SMIX, SDICT):
        for s in NINE:
            f = flags["graphs"][g]["sets"][s]["flag"]
            flagged_acc[f"{g} {s}"] = {
                "flagged": sum(f), **{f"{w}@10": hits(runs[g], s, 10, [bool(x) == (w == "flagged") for x in f])[0]
                                      for w in ("flagged", "clean")}}
    rare["flagged_patients_top10"] = flagged_acc
    res["smix_minus_sdict_rare"] = rare
    lo = rare["@10"]["excluded"]["ci"][0]
    res["verdict_rare"] = (f"keep: S-mix - S-dict rare macro Top-10 CI lower bound {lo:+.2f} > 0 without flagged patients"
                           if lo > 0 else f"drop: CI lower bound {lo:+.2f} <= 0 without flagged patients")
    (OUT / "result.json").write_text(json.dumps(res, indent=1))

    print("flagged patients per set (document unit):")
    for g, f in res["flagged"].items():
        print(f"  {g:17s} " + " ".join(f"{s.split('-')[-1][:8]} {f[s]}" for s in NINE) + f"  total {sum(f.values())}")
    print("\n(a) collection depth, macro lineage Top-10 over nine sets (B - A):")
    for r in depth:
        x = r["@10"]
        print(f"  {r['B']:12s} - {r['A']:17s} all {x['all']['diff']:+6.2f} [{x['all']['ci'][0]:+.2f},{x['all']['ci'][1]:+.2f}]"
              f"  excluded {x['excluded']['diff']:+6.2f} [{x['excluded']['ci'][0]:+.2f},{x['excluded']['ci'][1]:+.2f}]"
              f"  ratio {x['kept_ratio']}  (article {x['kept_ratio_article']}, any graph {x['kept_ratio_any_graph']})")
    print(f"  verdict: {res['verdict_depth']}")
    print("\n(b) S-mix - S-dict, rare macro (7 sets):")
    for k in (10, 1):
        x = rare[f"@{k}"]
        print(f"  @{k:<2d} all {x['all']['diff']:+6.2f} [{x['all']['ci'][0]:+.2f},{x['all']['ci'][1]:+.2f}]"
              f"  excluded {x['excluded']['diff']:+6.2f} [{x['excluded']['ci'][0]:+.2f},{x['excluded']['ci'][1]:+.2f}]"
              f"  article {x['excluded_article']['diff']:+6.2f}  any graph {x['excluded_any_graph']['diff']:+6.2f}")
    print(f"  verdict: {res['verdict_rare']}")
    print(f"\nsaved {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
