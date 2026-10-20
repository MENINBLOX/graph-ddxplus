"""SemMedDB with the same ranker against S-mix and S-dict on the development sets (research design §9.32 P0-5).

run_semmeddb.sh loads the SemMedDB disease-finding graph into the comparator instance and ranks the nine
development sets with the frozen gemma 1-hop configuration and with the configuration tune_dev.py selects
on that graph (design D3 rule; the amendment of §9.32 makes the selected one the main result, as for HPOA
and PrimeKG in §9.22). Per set: paired_stats.compare with the gold-disease cluster CI; macro:
paired_stats.macro_compare. The §9.32 rule reads the development macro Top-10 (graph_dx.py rounding, as
tune_dev.py): SemMedDB at or above S-dict -> the text may say a public rule-based resource is as high or
higher. The confirmation sets are not used.

    python pilot/scripts/semmeddb_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

from paired_stats import compare, macro_compare
from variants_analysis import gold_clusters, macro

DEV = Path("pilot/cache/v2/devD/variants")
REF = Path("pilot/cache/v2/devD")
OUT = Path("pilot/cache/v3/p0/semmeddb")
MAIN = "semantic_drop"
VARIANTS = ("semantic_drop", "semantic_keep", "subject_drop")
COMMON = ["DDXPlus-test", "SymCat"]
RARE = ["PUMCH-ADM", "PUMCH_L", "Phenopacket-store", "RareBench-HMS", "RareBench-LIRICAL",
        "RareBench-MME", "RareBench-RAMEDIS"]
GROUPS = {"all nine": COMMON + RARE, "common (DDXPlus, SymCat)": COMMON, "rare (7 sets)": RARE}
# frozen E3 systems at the offtopic_w design E fixed; earlier comparators at their D3 selection (§9.22)
BASE = {"S-mix": DEV / "dev_v3llmd_ns_ot0.5.json", "S-dict": DEV / "dev_v3umls_ns_ot0.5.json"}
CONTEXT = {"HPOA (D3)": REF / "ref_hpoa/dev.json", "PrimeKG (D3)": REF / "ref_primekg/dev.json",
           "MeSH co-occurrence (D3)": REF / "ref_cooc/dev.json"}


def paired(runs: dict[str, dict], pairs: list[tuple[str, str]], clusters: dict[str, list],
           groups: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    rows, macros = [], []
    for a, b in pairs:
        for k in (10, 1):
            for r in compare(runs[a], runs[b], k, clusters=clusters, require=COMMON + RARE):
                rows.append({"comparison": f"{b} - {a}", "k": k, "bench": r["bench"], "n": r["n"],
                             "A_pct": r[f"A@{k}"], "B_pct": r[f"B@{k}"], "diff": r["diff"], "ci": r["ci"],
                             "ci_cluster": r.get("ci_cluster"), "diseases": r.get("clusters"),
                             "only_A": r["only_A"], "only_B": r["only_B"], "p_holm": r["p_holm"]})
            for g, gs in groups.items():
                m = macro_compare(runs[a], runs[b], k, gs)
                macros.append({"comparison": f"{b} - {a}", "k": k, "group": g, "A": m["A"], "B": m["B"],
                               "diff": m["diff"], "ci_patient_stratified": m["ci"], "p_bootstrap": m["p"]})
    return rows, macros


def per_set(run: dict) -> dict[str, list[float]]:
    return {r["bench"]: [r["lineage@10"], r["lineage@1"]] for r in run["results"]}


def main() -> None:
    runs = {name: json.loads(p.read_text()) for name, p in BASE.items()}
    semmed: dict[str, dict] = {}
    for v in VARIANTS:
        for cfg in ("tuned", "frozen"):
            if (p := OUT / f"dev_{v}_{cfg}.json").exists():
                semmed[f"SemMedDB {v} {cfg}"] = json.loads(p.read_text())
    main_t, main_f = f"SemMedDB {MAIN} tuned", f"SemMedDB {MAIN} frozen"
    if main_t not in semmed or main_f not in semmed:
        raise SystemExit(f"missing {OUT}/dev_{MAIN}_tuned.json or _frozen.json: run run_semmeddb.sh")
    runs.update(semmed)
    tune = json.loads((OUT / f"tune_{MAIN}.json").read_text())
    graphs = {v: json.loads(p.read_text()) for v in VARIANTS if (p := OUT / f"graph_{v}.json").exists()}

    dev_macro = {k: dict(zip(("top10", "top1"), macro(r["results"]))) for k, r in runs.items()}
    for k, p in CONTEXT.items():
        dev_macro[k] = dict(zip(("top10", "top1"), macro(json.loads(p.read_text())["results"])))
    gap_t = round(dev_macro[main_t]["top10"] - dev_macro["S-dict"]["top10"], 2)
    gap_f = round(dev_macro[main_f]["top10"] - dev_macro["S-dict"]["top10"], 2)
    verdict = ("write: a public rule-based resource is as high or higher" if gap_t >= 0
               else "do not write: SemMedDB is below S-dict")

    pairs = [(main_t, "S-mix"), (main_t, "S-dict"), (main_f, "S-mix"), (main_f, "S-dict")]
    rows, macros = paired(runs, pairs, gold_clusters(), GROUPS)
    for name in semmed:
        if name not in (main_t, main_f):
            _, m = paired(runs, [(name, "S-dict")], {}, {"all nine": COMMON + RARE})
            macros += m

    res = {"note": "diff = B - A in percentage points of lineage Top-k on the nine development sets; ci = patient-"
                   "paired bootstrap 95% (2,000, seed 0); ci_cluster = gold-disease cluster bootstrap; p_holm = exact "
                   "McNemar, Holm across the nine sets of one comparison at one k; macro = equal-weight mean of the "
                   "set values (graph_dx.py rounding) and paired_stats.macro_compare.",
           "rule": {"semmeddb_tuned_dev_macro_top10": dev_macro[main_t]["top10"],
                    "semmeddb_frozen_dev_macro_top10": dev_macro[main_f]["top10"],
                    "s_dict_dev_macro_top10": dev_macro["S-dict"]["top10"],
                    "gap_tuned": gap_t, "gap_frozen": gap_f, "verdict": verdict},
           "tuned_config": tune["selected"], "tuned_trail_len": len(tune["trail"]),
           "graphs": graphs, "dev_macro": dev_macro,
           "per_set": {k: per_set(r) for k, r in runs.items()}, "rows": rows, "macro": macros}
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))

    for v, g in graphs.items():
        print(f"graph {v}: edges {g['edges']:,}, diseases with findings {g['diseases_with_findings']:,}, "
              f"concepts {g['concepts']:,}, PMID-pair support {g['n_docs_total']:,}")
        print(f"  predications {g['predications']}")
    print(f"tuned configuration: {tune['selected']} ({len(tune['trail'])} configurations evaluated)")
    print("\ndevelopment macro Top-10 / Top-1 (nine sets):")
    for k, v in dev_macro.items():
        print(f"  {k:34s} {v['top10']:6.2f} / {v['top1']:6.2f}")
    sets = COMMON + RARE
    print(f"\nper set Top-10 / Top-1:\n  {'':34s} " + " ".join(f"{s[-12:]:>13s}" for s in sets))
    for k, r in res["per_set"].items():
        print(f"  {k:34s} " + " ".join(f"{r[s][0]:6.2f}/{r[s][1]:6.2f}" for s in sets))
    print(f"\nrule: SemMedDB (tuned) - S-dict = {gap_t:+.2f} pp (frozen {gap_f:+.2f}) -> {verdict}")
    print(f"\n{'comparison':48s} {'k':>3s} {'set':22s} {'n':>6s} {'A':>6s} {'B':>6s} {'diff':>7s} "
          f"{'patient CI':>16s} {'disease CI':>16s} {'Holm p':>8s}")
    for r in rows:
        cc = r["ci_cluster"] or [float("nan")] * 2
        print(f"{r['comparison']:48s} @{r['k']:<2d} {r['bench']:22s} {r['n']:6d} {r['A_pct']:6.2f} {r['B_pct']:6.2f} "
              f"{r['diff']:+7.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] [{cc[0]:+6.2f},{cc[1]:+6.2f}] "
              f"{r['p_holm']:8.2g}")
    print("\nmacro (equal weight, set-stratified patient bootstrap):")
    for m in macros:
        print(f"  {m['comparison']:48s} @{m['k']:<2d} {m['group']:26s} {m['A']:6.2f} {m['B']:6.2f} {m['diff']:+7.2f} "
              f"[{m['ci_patient_stratified'][0]:+6.2f},{m['ci_patient_stratified'][1]:+6.2f}]")
    print(f"\nsaved {OUT / 'stats.json'}")


if __name__ == "__main__":
    main()
