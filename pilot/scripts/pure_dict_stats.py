"""The pure dictionary graph against S-dict (research design §9.32 P0-2).

v3umlsnt_ns (run_pure_dict.sh) is S-dict (v3umls_ns) without the LLM phrase table. Development: the nine selection
sets at offtopic_w .5 (S-dict's frozen value) and, when the whole tune_v2 grid ran, the offtopic_w the design D3
rule picks (macro lineage Top-10 over the nine sets, ties Top-1, an exact tie to the lower value). Confirmation
set E: POST HOC at offtopic_w .5. Per set: paired_stats.compare with the gold-disease cluster CI; macro:
paired_stats.macro_compare. The §9.32 rule reads the development macro Top-10 (graph_dx.py rounding, as tune_dev.py):
within 1 percentage point of S-dict -> the cost comparison is written as the LLM-free dictionary path, otherwise as
S-dict as built. S-mix - pure is given as context (not part of the rule).

    CONFIRM_E_UNSEAL=1 python pilot/scripts/pure_dict_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import opendx_eval
import v2_ids
from paired_stats import compare, macro_compare
from variants_analysis import GRID, gold_clusters, macro

DEV = Path("pilot/cache/v2/devD/variants")
CONF = Path("pilot/cache/v2/confirmE")
OUT = Path("pilot/cache/v3/p0/pure_dict")
PURE = "v3umlsnt_ns"
OT = "0.5"
COMMON = ["DDXPlus-test", "SymCat"]
RARE = ["PUMCH-ADM", "PUMCH_L", "Phenopacket-store", "RareBench-HMS", "RareBench-LIRICAL",
        "RareBench-MME", "RareBench-RAMEDIS"]
PAIRS = [("S-dict", "pure"), ("pure", "S-mix")]     # (A, B): diff = B - A
RULE_PP = 1.0


def d3_pick(runs: dict[str, dict]) -> str | None:
    if any(w not in runs for w in GRID):
        return None
    return max(GRID, key=lambda w: (*macro(runs[w]["results"]), -float(w)))


def paired_part(part: str, runs: dict[str, dict], clusters: dict[str, list], sets: list[str],
                groups: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    rows, macros = [], []
    for a, b in PAIRS:
        for k in (10, 1):
            for r in compare(runs[a], runs[b], k, clusters=clusters, require=sets):
                rows.append({"part": part, "comparison": f"{b} - {a}", "k": k, "bench": r["bench"], "n": r["n"],
                             "A_pct": r[f"A@{k}"], "B_pct": r[f"B@{k}"], "diff": r["diff"], "ci": r["ci"],
                             "ci_cluster": r.get("ci_cluster"), "diseases": r.get("clusters"),
                             "only_A": r["only_A"], "only_B": r["only_B"], "p_holm": r["p_holm"]})
            for g, gs in groups.items():
                m = macro_compare(runs[a], runs[b], k, gs)
                macros.append({"part": part, "comparison": f"{b} - {a}", "k": k, "group": g, "A": m["A"],
                               "B": m["B"], "diff": m["diff"], "ci_patient_stratified": m["ci"], "p_bootstrap": m["p"]})
    return rows, macros


def main() -> None:
    pure_dev = {w: json.loads(p.read_text()) for w in GRID if (p := OUT / f"dev_{PURE}_ot{w}.json").exists()}
    if OT not in pure_dev:
        raise SystemExit(f"missing {OUT}/dev_{PURE}_ot{OT}.json")
    dev = {"S-dict": json.loads((DEV / "dev_v3umls_ns_ot0.5.json").read_text()),
           "S-mix": json.loads((DEV / "dev_v3llmd_ns_ot0.5.json").read_text()), "pure": pure_dev[OT]}
    grid = {w: dict(zip(("top10", "top1"), macro(r["results"]))) for w, r in pure_dev.items()}
    pick = d3_pick(pure_dev)
    s_dict_macro = dict(zip(("top10", "top1"), macro(dev["S-dict"]["results"])))
    s_mix_macro = dict(zip(("top10", "top1"), macro(dev["S-mix"]["results"])))
    gap = round(grid[OT]["top10"] - s_dict_macro["top10"], 2)
    verdict = "LLM-free dictionary path" if abs(gap) <= RULE_PP else "S-dict as built"

    dev_rows, dev_macro = paired_part("development", dev, gold_clusters(), COMMON + RARE,
                                      {"all nine": COMMON + RARE, "common (DDXPlus, SymCat)": COMMON,
                                       "rare (7 sets)": RARE})
    if pick and pick != OT:
        alt = {**dev, "pure": pure_dev[pick]}
        r, m = paired_part(f"development, pure at D3 pick {pick}", alt, gold_clusters(), COMMON + RARE,
                           {"all nine": COMMON + RARE})
        dev_rows += r
        dev_macro += m

    conf_rows: list[dict] = []
    conf_macro: list[dict] = []
    e_file = OUT / f"E_{PURE}_ot{OT}.json"
    if e_file.exists():
        opendx_eval.RAW_GOLD = True
        conf = opendx_eval.load_confirm_e()
        clusters = {b: [tuple(sorted(v2_ids.gold_keys(g))) for g, ev in pats if v2_ids.gold_keys(g) and ev]
                    for b, pats in conf.items()}
        runs = {"S-dict": json.loads((CONF / "S-dict.json").read_text()),
                "S-mix": json.loads((CONF / "S-mix.json").read_text()), "pure": json.loads(e_file.read_text())}
        conf_rows, conf_macro = paired_part("confirmation E (post hoc)", runs, clusters, sorted(conf),
                                            {"both sets": sorted(conf)})

    res = {"note": "diff = B - A in percentage points of lineage Top-k; ci = patient-paired bootstrap 95% (2,000, "
                   "seed 0); ci_cluster = gold-disease cluster bootstrap; p_holm = exact McNemar, Holm across the sets "
                   "of one comparison at one k. Development macro = mean of the nine graph_dx.py set values (D3).",
           "rule": {"pure_dev_macro_top10_ot0.5": grid[OT]["top10"], "s_dict_dev_macro_top10": s_dict_macro["top10"],
                    "gap": gap, "threshold_pp": RULE_PP, "verdict": verdict},
           "dev_macro": {"S-dict ot .5": s_dict_macro, "S-mix ot .5": s_mix_macro,
                         **{f"pure ot {w}": v for w, v in grid.items()}},
           "d3_pick": pick, "rows": dev_rows + conf_rows, "macro": dev_macro + conf_macro}
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))

    print("development macro Top-10 / Top-1 (nine sets):")
    for k, v in res["dev_macro"].items():
        print(f"  {k:14s} {v['top10']:6.2f} / {v['top1']:6.2f}")
    print(f"  D3 pick for pure: {pick}")
    print(f"rule: pure - S-dict = {gap:+.2f} pp -> {verdict}")
    print(f"\n{'part':28s} {'comparison':16s} {'k':>3s} {'set':22s} {'n':>6s} {'A':>6s} {'B':>6s} {'diff':>6s} "
          f"{'patient CI':>16s} {'disease CI':>16s} {'Holm p':>8s}")
    for r in res["rows"]:
        cc = r["ci_cluster"] or [float("nan")] * 2
        print(f"{r['part'][:28]:28s} {r['comparison']:16s} @{r['k']:<2d} {r['bench']:22s} {r['n']:6d} {r['A_pct']:6.2f} "
              f"{r['B_pct']:6.2f} {r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] "
              f"[{cc[0]:+6.2f},{cc[1]:+6.2f}] {r['p_holm']:8.2g}")
    print("\nmacro (equal weight, set-stratified patient bootstrap):")
    for m in res["macro"]:
        print(f"  {m['part'][:28]:28s} {m['comparison']:16s} @{m['k']:<2d} {m['group']:26s} {m['A']:6.2f} {m['B']:6.2f} "
              f"{m['diff']:+6.2f} [{m['ci_patient_stratified'][0]:+6.2f},{m['ci_patient_stratified'][1]:+6.2f}]")
    print(f"\nsaved {OUT / 'stats.json'}")


if __name__ == "__main__":
    main()
