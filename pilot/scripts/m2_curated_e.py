"""Frozen design-D curated comparators on confirmation set E (research design §9.33 M2, POST HOC).

run_m2_curated_e.sh rebuilds HPOA and PrimeKG on the comparator instance and ranks E once with the 1-hop
configuration frozen for design D (frozen_config_D.json comparators.<src>.onehop, the setting of the D
confirmation reference run). `check` runs before E is ranked: the rebuilt graph must give the stored development
ranks (devD/ref_<src>/dev.json) for every patient and hold exactly the HAS_FINDING edges of devD/ref_<src>/graph.npz,
the graph the development strata were computed on.

`stats` compares S-mix (confirmE/S-mix.json, the frozen E3 system) with each comparator (diff = S-mix - comparator,
lineage Top-k): per set paired_stats.compare (patient-paired CI, gold-disease cluster CI, exact McNemar with Holm
across the two E sets of one comparison at one k), paired_stats.macro_compare over both sets, and the
disease-equal-weight difference (each gold cluster counted once, clusters resampled). Strata, as computed on the
development sets before E was ranked (boundaries fixed in §9.33):
  lineage coverage   the gold or one of its IS_A descendants up to depth 3 (the evaluation's lineage rule) has at
                     least one HPOA edge
  profile thickness  HPOA edges of the gold disease, the maximum over its gold keys: 0, 1-9, 10-29, >=30
Both comparators are stratified by the HPOA strata; S-mix - PrimeKG by PrimeKG's own lineage coverage is auxiliary.
The development DDXPlus and SymCat (stored ranks) give the same strata for reference; S-dict, S-D and the pure
dictionary graph against both comparators are auxiliary rows. M2 rule (§9.33): for each E set, the S-mix - HPOA
Top-10 gold-disease cluster CI lower bound > 0 adds the E value to that set's H2 sentence; otherwise that set's H2
sentence is removed. The >=30 stratum is reported without a judgment.

    python pilot/scripts/m2_curated_e.py check --src hpoa --dev pilot/cache/v3/framing/m2/dev_hpoa.json
    CONFIRM_E_UNSEAL=1 python pilot/scripts/m2_curated_e.py stats
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import opendx_eval
import v2_ids
from neo4j import GraphDatabase
from paired_stats import compare, holm, macro_compare
from variants_analysis import gold_clusters

DEVD = Path("pilot/cache/v2/devD")
CONF = Path("pilot/cache/v2/confirmE")
OUT = Path("pilot/cache/v3/framing/m2")
SOURCES = {"hpoa": "HPOA", "primekg": "PrimeKG"}
E_SETS = ["ConfirmE-DDXPlus", "ConfirmE-SymCat"]
DEV_SETS = ["DDXPlus-test", "SymCat"]
THICK = ["0", "1-9", "10-29", ">=30"]
AUX_E = {"S-dict": CONF / "S-dict.json", "S-D": CONF / "S-D.json",
         "pure-dict": Path("pilot/cache/v3/p0/pure_dict/E_v3umlsnt_ns_ot0.5.json")}


def npz_edges(src: str) -> tuple[set[tuple[str, str]], np.lib.npyio.NpzFile]:
    z = np.load(DEVD / f"ref_{src}" / "graph.npz")
    d, c = z["diseases"], z["concepts"]
    nd = len(d)          # ppr_dx.py export numbers concepts after the diseases
    return {(str(d[i]), str(c[j - nd])) for i, j in zip(z["hf_d"].tolist(), z["hf_c"].tolist())}, z


def check(src: str, dev: Path, uri: str, auth: str) -> None:
    stored = {r["bench"]: r["ranks_lineage"] for r in json.loads((DEVD / f"ref_{src}" / "dev.json").read_text())["results"]}
    new = {r["bench"]: r["ranks_lineage"] for r in json.loads(dev.read_text())["results"]}
    bad = []
    for b, rk in stored.items():
        if new.get(b) != rk:
            n = sum(x != y for x, y in zip(rk, new.get(b, []))) if b in new and len(new[b]) == len(rk) else "all"
            bad.append(f"{b}: {n} patients differ")
        else:
            print(f"{b:20s} {len(rk):6d} patients: stored development ranks reproduced")
    want, _ = npz_edges(src)
    drv = GraphDatabase.driver(uri, auth=tuple(auth.split("/", 1)))
    with drv.session() as s:
        got = {(r["d"], r["c"]) for r in s.run("MATCH (d:Disease)-[:HAS_FINDING]->(c:Concept) RETURN d.id AS d, c.cui AS c")}
    drv.close()
    if got != want:
        bad.append(f"HAS_FINDING edges: {len(got - want)} not in the development graph, {len(want - got)} missing")
    else:
        print(f"HAS_FINDING edges equal to {DEVD / f'ref_{src}' / 'graph.npz'} ({len(got)})")
    if bad:
        sys.exit("check failed: " + "; ".join(bad))


def lineage_fn(pairs: list[tuple[str, str]]):
    kids = defaultdict(set)
    for c, p in pairs:
        kids[p].add(c)

    def lineage(gold: tuple[str, ...]) -> set[str]:
        out, fr = set(gold), set(gold)
        for _ in range(3):
            fr = {c for f in fr for c in kids.get(f, ())} - out
            out |= fr
        return out
    return lineage


def thickness(n: int) -> str:
    return "0" if n == 0 else "1-9" if n < 10 else "10-29" if n < 30 else ">=30"


def disease_equal(ha: list[bool], hb: list[bool], keys: list, boot: int = 2000, seed: int = 0) -> dict:
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    gl = list(groups.values())
    per = [sum(hb[i] - ha[i] for i in x) / len(x) for x in gl]
    rng = random.Random(seed)
    dist = sorted(100 * sum(per[rng.randrange(len(per))] for _ in per) / len(per) for _ in range(boot))
    return {"A": round(100 * sum(sum(ha[i] for i in x) / len(x) for x in gl) / len(gl), 2),
            "B": round(100 * sum(sum(hb[i] for i in x) / len(x) for x in gl) / len(gl), 2),
            "diff": round(100 * sum(per) / len(per), 2),
            "ci": [round(dist[int(0.025 * boot)], 2), round(dist[int(0.975 * boot) - 1], 2)], "diseases": len(gl)}


def ranks(run: dict) -> dict[str, list]:
    return {r["bench"]: r["ranks_lineage"] for r in run["results"]}


def sub(run_ranks: list, bench: str, idx: list[int]) -> dict:
    return {"results": [{"bench": bench, "ranks_lineage": [run_ranks[i] for i in idx]}]}


def side(ci: list[float]) -> str:
    return "S-mix higher" if ci[0] > 0 else "comparator higher" if ci[1] < 0 else "includes 0"


def row(part: str, a_name: str, b_name: str, bench: str, k: int, stratum: str, ra: list, rb: list,
        keys: list, idx: list[int] | None = None) -> dict | None:
    idx = list(range(len(keys))) if idx is None else idx
    if not idx:
        return None
    kk = [keys[i] for i in idx]
    r = compare(sub(ra, bench, idx), sub(rb, bench, idx), k, clusters={bench: kk}, require=[bench])[0]
    ha = [ra[i] is not None and ra[i] <= k for i in idx]
    hb = [rb[i] is not None and rb[i] <= k for i in idx]
    de = disease_equal(ha, hb, kk)
    return {"part": part, "comparison": f"{b_name} - {a_name}", "A": a_name, "B": b_name, "bench": bench, "k": k,
            "stratum": stratum, "n": r["n"], "diseases": r["clusters"], "A_pct": r[f"A@{k}"], "B_pct": r[f"B@{k}"],
            "diff": r["diff"], "ci": r["ci"], "ci_cluster": r["ci_cluster"], "p_mcnemar": r["p"],
            "only_A": r["only_A"], "only_B": r["only_B"], "disease_equal": de,
            "patient_ci": side(r["ci"]), "cluster_ci": side(r["ci_cluster"]), "disease_equal_ci": side(de["ci"])}


def strata(part: str, sets: list[str], runs: dict[str, dict], clusters: dict[str, list], prof: dict[str, Counter],
           cover: dict[str, set], lineage) -> tuple[list[dict], dict]:
    rows, table = [], {}
    for b in sets:
        keys = clusters[b]
        n = len(keys)
        rk = {s: ranks(r)[b] for s, r in runs.items()}
        if any(len(v) != n for v in rk.values()):
            sys.exit(f"{part} {b}: rank lengths {[len(v) for v in rk.values()]} vs {n} gold keys")
        cov = {s: [bool(lineage(g) & cover[s]) for g in keys] for s in cover}
        exact = [bool(set(g) & cover["HPOA"]) for g in keys]
        th = [thickness(max(prof["HPOA"].get(x, 0) for x in g)) for g in keys]
        dz = sorted(set(keys))
        first = {g: keys.index(g) for g in dz}
        table[b] = {"patients": n, "diseases": len(dz),
                    "HPOA_lineage_covered": {"patients": sum(cov["HPOA"]),
                                             "diseases": sum(cov["HPOA"][first[g]] for g in dz)},
                    "HPOA_exact_covered": {"patients": sum(exact), "diseases": sum(exact[first[g]] for g in dz)},
                    "PrimeKG_lineage_covered": {"patients": sum(cov["PrimeKG"]),
                                                "diseases": sum(cov["PrimeKG"][first[g]] for g in dz)},
                    "thickness": {t: {"patients": th.count(t), "diseases": sum(th[first[g]] == t for g in dz),
                                      "lineage_covered_patients": sum(c for c, x in zip(cov["HPOA"], th) if x == t)}
                                  for t in THICK}}
        groups = [("all", None)]
        groups += [(f"HPOA lineage {'covered' if v else 'not covered'}", [i for i in range(n) if cov["HPOA"][i] == v])
                   for v in (True, False)]
        groups += [(f"HPOA thickness {t}", [i for i in range(n) if th[i] == t]) for t in THICK]
        for ref in ("HPOA", "PrimeKG"):
            for k in (10, 1):
                for label, idx in groups:
                    r = row(part, ref, "S-mix", b, k, label, rk[ref], rk["S-mix"], keys, idx)
                    if r:
                        rows.append(r)
                if ref == "PrimeKG":
                    for v in (True, False):
                        idx = [i for i in range(n) if cov["PrimeKG"][i] == v]
                        r = row(part, ref, "S-mix", b, k, f"aux: PrimeKG lineage {'covered' if v else 'not covered'}",
                                rk[ref], rk["S-mix"], keys, idx)
                        if r:
                            rows.append(r)
    return rows, table


def stats() -> None:
    if os.environ.get("CONFIRM_E_UNSEAL") != "1":
        sys.exit("set CONFIRM_E_UNSEAL=1 (post hoc use of confirmation set E)")
    prof, cover = {}, {}
    for src, name in SOURCES.items():
        edges, _ = npz_edges(src)
        prof[name] = Counter(d for d, _ in edges)
        cover[name] = set(prof[name])
    z = np.load(DEVD / "ref_hpoa" / "graph.npz")
    di_pairs = [(str(z["diseases"][a]), str(z["diseases"][b])) for a, b in zip(z["di_a"].tolist(), z["di_b"].tolist())]
    lin_dev = lineage_fn(list(v2_ids.disease_isa()))       # the development strata's rule
    lin_graph = lineage_fn(di_pairs)                        # Disease IS_A as loaded in Neo4j (universe only)

    dev_clusters = gold_clusters()                          # sets RAW_GOLD = True
    dev_runs = {"S-mix": json.loads((DEVD / "variants" / "dev_v3llmd_ns_ot0.5.json").read_text()),
                "HPOA": json.loads((DEVD / "ref_hpoa" / "dev.json").read_text()),
                "PrimeKG": json.loads((DEVD / "ref_primekg" / "dev.json").read_text())}
    opendx_eval.RAW_GOLD = True
    conf = opendx_eval.load_confirm_e()
    e_clusters = {b: [tuple(sorted(v2_ids.gold_keys(g))) for g, ev in pats if v2_ids.gold_keys(g) and ev]
                  for b, pats in conf.items()}
    e_runs = {"S-mix": json.loads((CONF / "S-mix.json").read_text()),
              "HPOA": json.loads((OUT / "E_hpoa.json").read_text()),
              "PrimeKG": json.loads((OUT / "E_primekg.json").read_text())}
    for name in ("HPOA", "PrimeKG"):
        args = e_runs[name]["args"]
        want = json.loads((DEVD / "frozen_config_D.json").read_text())["comparators"][name.lower()]["onehop"]
        if any(args[k] != v for k, v in want.items()) or args["bench"] != "confirm_e":
            sys.exit(f"{name}: E run args {args} differ from the frozen D configuration {want}")

    rule_disagree = {b: sum(bool(lin_dev(g) & cover["HPOA"]) != bool(lin_graph(g) & cover["HPOA"]) for g in keys)
                     for b, keys in {**{s: dev_clusters[s] for s in DEV_SETS}, **e_clusters}.items()}

    e_rows, e_table = strata("confirmation E [post hoc]", E_SETS, e_runs, e_clusters, prof, cover, lin_dev)
    d_rows, d_table = strata("development (reference)", DEV_SETS, dev_runs, dev_clusters, prof, cover, lin_dev)
    for comp in ("S-mix - HPOA", "S-mix - PrimeKG"):
        for k in (10, 1):
            fam = [r for r in e_rows if r["comparison"] == comp and r["k"] == k and r["stratum"] == "all"]
            for r, p in zip(fam, holm([r["p_mcnemar"] for r in fam])):
                r["p_holm"] = p

    macros = []
    for ref in ("HPOA", "PrimeKG"):
        for k in (10, 1):
            m = macro_compare(e_runs[ref], e_runs["S-mix"], k, E_SETS)
            macros.append({"comparison": f"S-mix - {ref}", "k": k, "sets": m["sets"], "A": m["A"], "B": m["B"],
                           "diff": m["diff"], "ci_patient_stratified": m["ci"], "p_bootstrap": m["p"]})

    aux = []
    for sname, path in AUX_E.items():
        run = json.loads(path.read_text())
        for ref in ("HPOA", "PrimeKG"):
            for k in (10, 1):
                for b in E_SETS:
                    r = row("confirmation E [post hoc] auxiliary", ref, sname, b, k, "all", ranks(e_runs[ref])[b],
                            ranks(run)[b], e_clusters[b])
                    aux.append(r)

    decision = {}
    for b in E_SETS:
        r = next(x for x in e_rows if x["comparison"] == "S-mix - HPOA" and x["k"] == 10 and x["bench"] == b
                 and x["stratum"] == "all")
        keep = r["ci_cluster"][0] > 0
        decision[b] = {"S-mix - HPOA @10": r["diff"], "cluster_ci": r["ci_cluster"], "patient_ci": r["ci"],
                       "disease_equal": r["disease_equal"], "lower_bound_gt_0": keep,
                       "action": "add the E value to this set's H2 sentence" if keep
                       else "remove this set's H2 dominance sentence"}
    both = all(v["lower_bound_gt_0"] for v in decision.values())
    decision["overall"] = ("both E sets: lower bound > 0 -> H2 ('the curated annotations leave this space empty or "
                           "thin') gains the [post hoc] sealed-set values" if both else
                           "at least one E set fails -> that set's H2 dominance sentence is removed")

    res = {"note": "POST HOC use of confirmation set E (§9.33 M2). diff = S-mix (or the auxiliary graph) - comparator, "
                   "percentage points of lineage Top-k. ci = patient-paired bootstrap 95% (2,000, seed 0); ci_cluster = "
                   "gold-disease cluster bootstrap 95% of the patient-weighted difference; disease_equal = mean over "
                   "gold clusters of the per-cluster difference, clusters resampled (2,000, seed 0). Gold cluster = "
                   "sorted gold-key tuple (as dict_mix_stats.py). Strata: HPOA lineage coverage (gold or IS_A "
                   "descendants to depth 3 has >= 1 HPOA edge) and HPOA profile thickness (max over gold keys of the "
                   "gold's HPOA edges; 0, 1-9, 10-29, >=30). Comparator configuration: frozen_config_D.json "
                   "comparators.<src>.onehop. Holm (p_holm) only on the overall E rows, across the two E sets.",
           "runs": {"E": {"S-mix": str(CONF / "S-mix.json"), "HPOA": str(OUT / "E_hpoa.json"),
                          "PrimeKG": str(OUT / "E_primekg.json"), **{k: str(v) for k, v in AUX_E.items()}},
                    "development": {"S-mix": str(DEVD / "variants" / "dev_v3llmd_ns_ot0.5.json"),
                                    "HPOA": str(DEVD / "ref_hpoa" / "dev.json"),
                                    "PrimeKG": str(DEVD / "ref_primekg" / "dev.json")}},
           "lineage_rule_check": {"note": "patients whose HPOA lineage coverage differs between the development "
                                          "rule (all Disease IS_A pairs) and the IS_A loaded in Neo4j",
                                  "differ": rule_disagree},
           "coverage": {"confirmation E": e_table, "development": d_table},
           "decision_M2": decision, "rows": e_rows + d_rows, "macro_E": macros, "auxiliary": aux}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))

    print("lineage rule check (patients differing):", rule_disagree)
    for part, tab in res["coverage"].items():
        for b, t in tab.items():
            print(f"{part:15s} {b:17s} n {t['patients']:5d} dz {t['diseases']:4d} | HPOA lineage-covered "
                  f"{t['HPOA_lineage_covered']['patients']} pts / {t['HPOA_lineage_covered']['diseases']} dz, exact "
                  f"{t['HPOA_exact_covered']['patients']} / {t['HPOA_exact_covered']['diseases']}; PrimeKG lineage "
                  f"{t['PrimeKG_lineage_covered']['patients']} / {t['PrimeKG_lineage_covered']['diseases']} | thickness "
                  + ", ".join(f"{s}: {v['patients']}/{v['diseases']}dz" for s, v in t["thickness"].items()))
    hdr = (f"\n{'part':8s} {'comparison':16s} {'k':>3s} {'set':17s} {'stratum':32s} {'n':>5s} {'dz':>4s} {'A':>6s} "
           f"{'B':>6s} {'diff':>6s} {'patient CI':>16s} {'cluster CI':>16s} | {'dz-equal':>7s} {'CI':>16s}")
    print(hdr)
    for r in e_rows + d_rows + aux:
        de = r["disease_equal"]
        print(f"{'E' if r['part'].startswith('conf') else 'dev':8s} {r['comparison']:16s} @{r['k']:<2d} {r['bench']:17s} "
              f"{r['stratum']:32s} {r['n']:5d} {r['diseases']:4d} {r['A_pct']:6.2f} {r['B_pct']:6.2f} {r['diff']:+6.2f} "
              f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] [{r['ci_cluster'][0]:+6.2f},{r['ci_cluster'][1]:+6.2f}] | "
              f"{de['diff']:+7.2f} [{de['ci'][0]:+6.2f},{de['ci'][1]:+6.2f}]"
              + (f"  Holm p {r['p_holm']:.2g}" if "p_holm" in r else ""))
    print("\nmacro over both E sets (equal weight, set-stratified patient bootstrap):")
    for m in macros:
        print(f"  {m['comparison']:16s} @{m['k']:<2d} {m['A']:6.2f} {m['B']:6.2f} {m['diff']:+6.2f} "
              f"[{m['ci_patient_stratified'][0]:+6.2f},{m['ci_patient_stratified'][1]:+6.2f}]")
    print("\nM2 decision:")
    for b, v in decision.items():
        print(f"  {b}: {v}")
    print(f"\nsaved {OUT / 'stats.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "stats"])
    ap.add_argument("--src", choices=list(SOURCES))
    ap.add_argument("--dev", type=Path)
    ap.add_argument("--uri", default="bolt://localhost:7688")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    a = ap.parse_args()
    if a.cmd == "check":
        if not (a.src and a.dev):
            sys.exit("check needs --src and --dev")
        check(a.src, a.dev, a.uri, a.auth)
    else:
        stats()


if __name__ == "__main__":
    main()
