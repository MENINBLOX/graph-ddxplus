"""Dictionary versus mixed extraction, per set, at the patient and the gold-disease level.

For each comparison (B - A), set and k in (10, 1): the patient-paired bootstrap CI, an exact McNemar test
with Holm correction across the sets of that comparison (one family per comparison and k), and the
gold-disease cluster bootstrap CI with the number of diseases (paired_stats.compare). Equal-weight macro
differences over the common and the rare sets use paired_stats.macro_compare (patient-level CI only).

Development: the nine selection sets, so every difference carries selection optimism. Confirmation set E:
post hoc. The design E pre-registration judged S-mix - S-dict only as a macro Top-10 non-inferiority test
(E-T2) and S-mix - S-D only at Top-10 per set (E-T1); every other row here was not pre-registered.

    CONFIRM_E_UNSEAL=1 python pilot/scripts/dict_mix_stats.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import opendx_eval
import v2_ids
from paired_stats import compare, macro_compare
from variants_analysis import gold_clusters

DEV = Path("pilot/cache/v2/devD/variants")
CONF = Path("pilot/cache/v2/confirmE")
OUT = Path("pilot/cache/v3/review/dict_mix_stats.json")
# the frozen E3 systems at the offtopic_w that design E fixed
DEV_RUNS = {"S-mix": "dev_v3llmd_ns_ot0.5.json", "S-dict": "dev_v3umls_ns_ot0.5.json",
            "S-union": "dev_v3unionu_ns_ot0.5.json", "S-llm": "dev_v3llm_ns_ot1.0.json"}
DEV_PAIRS = [("S-dict", "S-mix"), ("S-dict", "S-union"), ("S-llm", "S-mix"), ("S-mix", "S-union")]   # (A, B)
CONF_PAIRS = [("S-dict", "S-union"), ("S-dict", "S-mix"), ("S-D", "S-mix")]
COMMON = ["DDXPlus-test", "SymCat"]
RARE = ["PUMCH-ADM", "PUMCH_L", "Phenopacket-store", "RareBench-HMS", "RareBench-LIRICAL",
        "RareBench-MME", "RareBench-RAMEDIS"]


def registration(part: str, a: str, b: str, k: int) -> str:
    if part == "development":
        return "development (selection sets)"
    if (a, b, k) == ("S-D", "S-mix", 10):
        return "pre-registered E-T1 (patient CI, Holm); disease CI auxiliary"
    return "post hoc (not pre-registered)"


def status(r: dict, a: str, b: str) -> str:
    def side(ci: list[float]) -> str | None:
        return b if ci[0] > 0 else a if ci[1] < 0 else None

    pat, dis = side(r["ci"]), side(r["ci_cluster"])
    if dis and pat:
        return f"{dis} higher: disease CI excludes 0"
    if pat:
        return f"{pat} higher: patient CI only"
    if dis:
        return f"{dis} higher: disease CI only"
    return "both CIs include 0"


def cluster_sizes(keys: list) -> dict:
    sizes: dict = {}
    for key in keys:
        sizes[key] = sizes.get(key, 0) + 1
    v = sorted(sizes.values())
    return {"patients": len(keys), "diseases": len(v), "patients_per_disease_min": v[0],
            "patients_per_disease_median": statistics.median(v), "patients_per_disease_max": v[-1],
            "largest_disease_share_pct": round(100 * v[-1] / len(keys), 1)}


def run_part(part: str, runs: dict[str, dict], pairs: list[tuple[str, str]], clusters: dict[str, list],
             sets: list[str], groups: dict[str, list[str]]) -> tuple[list[dict], list[dict]]:
    rows, macros = [], []
    for a, b in pairs:
        for k in (10, 1):
            for r in compare(runs[a], runs[b], k, clusters=clusters, require=sets):
                if "ci_cluster" not in r:
                    raise SystemExit(f"{part} {b} - {a} {r['bench']}: {len(clusters.get(r['bench'], ()))} gold keys "
                                     f"for {r['n']} patients")
                rows.append({"part": part, "comparison": f"{b} - {a}", "A": a, "B": b, "k": k,
                             "registration": registration(part, a, b, k), "bench": r["bench"], "n": r["n"],
                             "A_pct": r[f"A@{k}"], "B_pct": r[f"B@{k}"], "diff": r["diff"], "ci": r["ci"],
                             "ci_cluster": r["ci_cluster"], "diseases": r["clusters"], "only_A": r["only_A"],
                             "only_B": r["only_B"], "p": r["p"], "p_holm": r["p_holm"],
                             "status": status(r, a, b)})
            for g, gs in groups.items():
                m = macro_compare(runs[a], runs[b], k, gs)
                macros.append({"part": part, "comparison": f"{b} - {a}", "k": k, "group": g, "sets": m["sets"],
                               "A": m["A"], "B": m["B"], "diff": m["diff"], "ci_patient_stratified": m["ci"],
                               "p_bootstrap": m["p"]})
    return rows, macros


def main() -> None:
    if not (CONF / "DONE").exists():
        raise SystemExit("confirmation E has not finished")
    dev_clusters = gold_clusters()        # sets RAW_GOLD = True
    dev_runs = {s: json.loads((DEV / f).read_text()) for s, f in DEV_RUNS.items()}
    dev_sets = COMMON + RARE
    dev_rows, dev_macro = run_part("development", dev_runs, DEV_PAIRS, dev_clusters, dev_sets,
                                   {"common (DDXPlus, SymCat)": COMMON, "rare (7 sets)": RARE})

    opendx_eval.RAW_GOLD = True
    conf = opendx_eval.load_confirm_e()
    conf_clusters = {b: [tuple(sorted(v2_ids.gold_keys(g))) for g, ev in pats if v2_ids.gold_keys(g) and ev]
                     for b, pats in conf.items()}
    conf_runs = {s: json.loads((CONF / f"{s}.json").read_text()) for s in ("S-mix", "S-dict", "S-union", "S-D")}
    conf_rows, conf_macro = run_part("confirmation E", conf_runs, CONF_PAIRS, conf_clusters, sorted(conf),
                                     {"both sets": sorted(conf)})

    res = {"note": "diff = B - A in percentage points of lineage Top-k; ci = patient-paired bootstrap 95% "
                   "(2,000, seed 0); ci_cluster = gold-disease cluster bootstrap 95% (whole diseases resampled); "
                   "p = exact McNemar, p_holm = Holm across the sets of one comparison at one k",
           "runs": {"development": {s: str(DEV / f) for s, f in DEV_RUNS.items()},
                    "confirmation E": {s: str(CONF / f"{s}.json") for s in conf_runs}},
           "gold_diseases": {**{b: cluster_sizes(v) for b, v in dev_clusters.items()},
                             **{b: cluster_sizes(v) for b, v in conf_clusters.items()}},
           "rows": dev_rows + conf_rows, "macro": dev_macro + conf_macro}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))

    print("gold diseases (cluster = distinct sorted gold-key set):")
    for b, c in res["gold_diseases"].items():
        print(f"  {b:19s} n {c['patients']:6d}  diseases {c['diseases']:5d}  patients/disease "
              f"{c['patients_per_disease_min']}-{c['patients_per_disease_median']}-{c['patients_per_disease_max']}"
              f"  largest {c['largest_disease_share_pct']}%")
    print(f"\n{'comparison':18s} {'k':>3s} {'set':19s} {'n':>6s} {'A':>6s} {'B':>6s} {'diff':>6s} "
          f"{'patient CI':>16s} {'disease CI':>16s} {'dis':>5s} {'Holm p':>9s}  status")
    for r in res["rows"]:
        tag = "" if r["part"] == "development" else " [E, " + r["registration"].split(" (")[0] + "]"
        print(f"{r['comparison']:18s} @{r['k']:<2d} {r['bench']:19s} {r['n']:6d} {r['A_pct']:6.2f} {r['B_pct']:6.2f} "
              f"{r['diff']:+6.2f} [{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] "
              f"[{r['ci_cluster'][0]:+6.2f},{r['ci_cluster'][1]:+6.2f}] "
              f"{r['diseases']:5d} {r['p_holm']:9.2g}  {r['status']}{tag}")
    print("\nmacro (equal-weight over sets, set-stratified patient bootstrap):")
    for m in res["macro"]:
        print(f"  {m['part']:14s} {m['comparison']:18s} @{m['k']:<2d} {m['group']:25s} {m['diff']:+6.2f} "
              f"[{m['ci_patient_stratified'][0]:+6.2f},{m['ci_patient_stratified'][1]:+6.2f}]")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
