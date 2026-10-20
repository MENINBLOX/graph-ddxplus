"""P1-3 (docs/positive_framing_review_2026-10-20.md §5.1 row 8; research design §9.33 auxiliary), POST HOC:
the E-T2 comparisons on confirmation set E stratified by whether the patient's input already names the gold disease.

Flag, as in the development analysis behind H3 caveat 4: with RAW_GOLD, gold = v2_ids.gold_keys(gold ids); the input
names the gold when a gold key is in the input keys (direct) or when a direct parent of a gold key
(v2_ids.disease_isa, one step up) is. Strata: "input names gold or parent" and "neither". The development flag is
recomputed first and must give the stored counts (DDXPlus-test 3,000: 458 direct, 681 gold or parent, 2,319 neither).

Stored per-patient ranks only, no re-ranking: lineage from pilot/cache/v2/confirmE/<S>.json and the pure dictionary
run (P0-2, pilot/cache/v3/p0/pure_dict/E_v3umlsnt_ns_ot0.5.json); strict from the P0-3 re-ranked runs
(pilot/cache/v3/p0/strict/<S>.json), whose ranks_lineage must equal the confirmation run. The pure dictionary run
has no strict ranks, so its strict rows are absent.

Per stratum, rule, comparison (B - A, B = S-mix) and k in (10, 1): paired_stats.compare per set (patient-paired CI,
gold-disease cluster CI, exact McNemar with Holm over the stratum's sets) and paired_stats.macro_compare over both E
sets with a set-stratified gold-disease cluster CI. Per set: each stratum's share of the set's overall difference
(n_stratum / n * diff_stratum, which sums to the overall difference) and the stratum interaction
(diff in "input names gold or parent" - diff in "neither", patient and cluster CI). On E, the gold diseases of the
named stratum with their Top-10 hits there. The development DDXPlus-test and SymCat runs are stratified the same way
(lineage only) as a reference.

P1-3 rule (judged at lineage Top-10 on DDXPlus-E): if, in the "neither" stratum, the S-dict - S-mix CI includes 0 or
is negative, H3 caveat 4 is written at the sealed-set level; otherwise it stays development-level.

    CONFIRM_E_UNSEAL=1 python pilot/scripts/input_gold_strata_e.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import opendx_eval
import v2_ids
from paired_stats import compare, macro_compare

CONF = Path("pilot/cache/v2/confirmE")
STRICT = Path("pilot/cache/v3/p0/strict")
PURE = Path("pilot/cache/v3/p0/pure_dict/E_v3umlsnt_ns_ot0.5.json")
DEV = Path("pilot/cache/v2/devD/variants")
DEV_RUNS = {"S-mix": DEV / "dev_v3llmd_ns_ot0.5.json", "S-dict": DEV / "dev_v3umls_ns_ot0.5.json",
            "S-union": DEV / "dev_v3unionu_ns_ot0.5.json",
            "pure dict": Path("pilot/cache/v3/p0/pure_dict/dev_v3umlsnt_ns_ot0.5.json")}
OUT = Path("pilot/cache/v3/framing/input_gold/stats.json")
NAMED, NEITHER = "input names gold or parent", "neither"
STRATA = ("all", NAMED, NEITHER)
PAIRS = [("S-dict", "S-mix"), ("S-union", "S-mix"), ("pure dict", "S-mix")]     # (A, B): B - A
DEV_COUNTS = {"direct": 458, NAMED: 681, NEITHER: 2319}
BOOT = 2000


def parent_map() -> dict[str, set[str]]:
    par: dict[str, set[str]] = {}
    for child, parent in v2_ids.disease_isa():
        par.setdefault(child, set()).add(parent)
    return par


def flags(pats: list, par: dict[str, set[str]]) -> list[dict]:
    """One entry per patient graph_dx.evaluate ranks (gold keys and input both non-empty), in its order."""
    out = []
    for gold_raw, ev in pats:
        gold = v2_ids.gold_keys(gold_raw)
        if not gold or not ev:
            continue
        direct = bool(gold & ev)
        parent = any(par.get(g, set()) & ev for g in gold)
        out.append({"direct": direct, "parent_only": parent and not direct,
                    "stratum": NAMED if direct or parent else NEITHER, "cluster": tuple(sorted(gold))})
    return out


def ranks(path: Path, rule: str) -> dict[str, list]:
    return {r["bench"]: r[f"ranks_{rule}"] for r in json.loads(path.read_text())["results"]}


def hit(p: int | None, k: int) -> bool:
    return p is not None and p <= k


def sub_run(rk: dict[str, list], idx: dict[str, list[int]]) -> dict:
    return {"results": [{"bench": b, "ranks_lineage": [rk[b][i] for i in ix]} for b, ix in idx.items()]}


def macro_cluster(a: dict, b: dict, k: int, sets: list[str], clusters: dict[str, list]) -> list[float]:
    """Equal-weight macro of the patient-weighted differences with whole gold diseases drawn within each set
    (sets in sorted order, one random.Random(0)), as m1_combine.macro_cluster."""
    ra = {r["bench"]: r["ranks_lineage"] for r in a["results"]}
    rb = {r["bench"]: r["ranks_lineage"] for r in b["results"]}
    data = []
    for s in sorted(sets):
        groups: dict = {}
        for key, x, y in zip(clusters[s], ra[s], rb[s], strict=True):
            groups.setdefault(key, []).append(int(hit(y, k)) - int(hit(x, k)))
        data.append(list(groups.values()))
    rng = random.Random(0)
    dist = []
    for _ in range(BOOT):
        tot = 0.0
        for gl in data:
            draw = [gl[rng.randrange(len(gl))] for _ in gl]
            tot += 100 * sum(map(sum, draw)) / sum(map(len, draw))
        dist.append(tot / len(data))
    dist.sort()
    return [round(dist[int(0.025 * BOOT)], 2), round(dist[int(0.975 * BOOT) - 1], 2)]


def side(ci: list[float], a: str, b: str) -> str:
    return f"{b} higher" if ci[0] > 0 else f"{a} higher" if ci[1] < 0 else "includes 0"


def stratum_rows(part: str, rule: str, stratum: str, a: str, b: str, ra: dict, rb: dict,
                 idx: dict[str, list[int]], clusters: dict[str, list]) -> tuple[list[dict], list[dict]]:
    sets = sorted(s for s, ix in idx.items() if ix)
    idx = {s: idx[s] for s in sets}
    run_a, run_b = sub_run(ra, idx), sub_run(rb, idx)
    cl = {s: [clusters[s][i] for i in idx[s]] for s in sets}
    rows, macros = [], []
    for k in (10, 1):
        for r in compare(run_a, run_b, k, boot=BOOT, clusters=cl, require=sets):
            rows.append({"part": part, "rule": rule, "stratum": stratum, "comparison": f"{b} - {a}", "A": a, "B": b,
                         "k": k, "bench": r["bench"], "n": r["n"], "diseases": r["clusters"], "A_pct": r[f"A@{k}"],
                         "B_pct": r[f"B@{k}"], "diff": r["diff"], "ci": r["ci"], "ci_cluster": r["ci_cluster"],
                         "only_A": r["only_A"], "only_B": r["only_B"], "p": r["p"], "p_holm": r["p_holm"],
                         "patient_ci": side(r["ci"], a, b), "cluster_ci": side(r["ci_cluster"], a, b)})
        if len(sets) > 1:
            m = macro_compare(run_a, run_b, k, sets, boot=BOOT)
            ci_cl = macro_cluster(run_a, run_b, k, sets, cl)
            macros.append({"part": part, "rule": rule, "stratum": stratum, "comparison": f"{b} - {a}", "k": k,
                           "sets": m["sets"], "n": {s: len(idx[s]) for s in sets}, "A": m["A"], "B": m["B"],
                           "diff": m["diff"], "ci_patient_stratified": m["ci"], "ci_cluster": ci_cl,
                           "p_bootstrap": m["p"], "patient_ci": side(m["ci"], a, b),
                           "cluster_ci": side(ci_cl, a, b)})
    return rows, macros


def decompose(part: str, rule: str, a: str, b: str, bench: str, ra: list, rb: list, fl: list[dict], k: int) -> dict:
    """Each stratum's share of the set's B - A and the interaction (B - A in NAMED) - (B - A in NEITHER)."""
    items = [(f["stratum"], int(hit(y, k)) - int(hit(x, k))) for f, x, y in zip(fl, ra, rb, strict=True)]
    n = len(items)

    def stat(its: list[tuple[str, int]]) -> float | None:
        tot = {NAMED: [0, 0], NEITHER: [0, 0]}
        for st, d in its:
            tot[st][0] += d
            tot[st][1] += 1
        if not tot[NAMED][1] or not tot[NEITHER][1]:
            return None
        return 100 * tot[NAMED][0] / tot[NAMED][1] - 100 * tot[NEITHER][0] / tot[NEITHER][1]

    def pct(xs: list[float | None]) -> tuple[list[float], int]:
        ok = sorted(x for x in xs if x is not None)
        return [round(ok[int(0.025 * len(ok))], 2), round(ok[int(0.975 * len(ok)) - 1], 2)], BOOT - len(ok)

    rng = random.Random(0)
    pat, pat_skip = pct([stat([items[rng.randrange(n)] for _ in range(n)]) for _ in range(BOOT)])
    groups: dict = {}
    for f, it in zip(fl, items, strict=True):
        groups.setdefault(f["cluster"], []).append(it)
    gl = list(groups.values())
    clu, clu_skip = pct([stat([it for _ in gl for it in gl[rng.randrange(len(gl))]]) for _ in range(BOOT)])
    share = {}
    for st in (NAMED, NEITHER):
        ds = [d for s, d in items if s == st]
        share[st] = {"n": len(ds), "diff": round(100 * sum(ds) / max(1, len(ds)), 2),
                     "contribution_pp": round(100 * sum(ds) / n, 2)}
    overall = 100 * sum(d for _, d in items) / n
    for v in share.values():
        v["share_of_overall_pct"] = round(100 * v["contribution_pp"] / overall, 1) if overall else None
    point = stat(items)
    return {"part": part, "rule": rule, "comparison": f"{b} - {a}", "bench": bench, "k": k, "n": n,
            "overall_diff": round(overall, 2), "strata": share,
            "interaction": {"diff": None if point is None else round(point, 2), "ci": pat, "ci_cluster": clu,
                            "replicates_without_both_strata": {"patient": pat_skip, "cluster": clu_skip}}}


def flag_counts(fl: list[dict]) -> dict:
    return {"n": len(fl), "direct": sum(f["direct"] for f in fl), "parent_only": sum(f["parent_only"] for f in fl),
            NAMED: sum(f["stratum"] == NAMED for f in fl), NEITHER: sum(f["stratum"] == NEITHER for f in fl),
            "diseases": len({f["cluster"] for f in fl}),
            "diseases_with_named": len({f["cluster"] for f in fl if f["stratum"] == NAMED}),
            "diseases_with_neither": len({f["cluster"] for f in fl if f["stratum"] == NEITHER})}


def named_diseases(fl: list[dict], runs: dict[str, dict[str, list]], names: dict[str, str]) -> list[dict]:
    """Gold diseases with patients in the named stratum: counts and Top-10 hits there per rule and system."""
    out: dict = {}
    for i, f in enumerate(fl):
        d = out.setdefault(f["cluster"], {"gold": list(f["cluster"]),
                                          "name": " / ".join(names.get(g, g) for g in f["cluster"]),
                                          "patients": 0, "named": 0, "direct": 0,
                                          "named_top10_hits": {rule: dict.fromkeys(rk, 0) for rule, rk in runs.items()}})
        d["patients"] += 1
        if f["stratum"] == NAMED:
            d["named"] += 1
            d["direct"] += f["direct"]
            for rule, rk in runs.items():
                for s, r in rk.items():
                    d["named_top10_hits"][rule][s] += hit(r[i], 10)
    return sorted((d for d in out.values() if d["named"]), key=lambda d: (-d["named"], d["name"]))


def analyse(part: str, fls: dict[str, list[dict]], runs: dict[str, dict[str, dict]],
            pairs: dict[str, list[tuple[str, str]]]) -> tuple[list, list, list]:
    clusters = {b: [f["cluster"] for f in fl] for b, fl in fls.items()}
    rows, macros, dec = [], [], []
    for rule, rule_pairs in pairs.items():
        for a, b in rule_pairs:
            ra, rb = runs[rule][a], runs[rule][b]
            for st in STRATA:
                idx = {s: [i for i, f in enumerate(fl) if st == "all" or f["stratum"] == st] for s, fl in fls.items()}
                r, m = stratum_rows(part, rule, st, a, b, ra, rb, idx, clusters)
                rows += r
                macros += m
            for bench, fl in fls.items():
                for k in (10, 1):
                    dec.append(decompose(part, rule, a, b, bench, ra[bench], rb[bench], fl, k))
    return rows, macros, dec


def development(par: dict[str, set[str]]) -> dict:
    """The two common development sets as graph_dx.py loaded them (--ddx_n 3000, --symcat_ppd 10); lineage only,
    the development runs store no strict ranks."""
    pats = {**opendx_eval.load_ddxplus("test", 3000), **opendx_eval.load_symcat(10)}
    fls = {b: flags(pats[b], par) for b in ("DDXPlus-test", "SymCat")}
    counts = flag_counts(fls["DDXPlus-test"])
    got = {key: counts[key] for key in DEV_COUNTS}
    if got != DEV_COUNTS:
        raise SystemExit(f"development flag counts {got} differ from the stored analysis {DEV_COUNTS}")
    runs = {"lineage": {s: ranks(p, "lineage") for s, p in DEV_RUNS.items()}}
    for s, rk in runs["lineage"].items():
        for b, fl in fls.items():
            if len(rk[b]) != len(fl):
                raise SystemExit(f"development {s} {b}: {len(rk[b])} ranks for {len(fl)} patients")
    rows, macros, dec = analyse("development (reference)", fls, runs, {"lineage": PAIRS})
    return {"flags": {b: flag_counts(fl) for b, fl in fls.items()}, "rows": rows, "macro": macros,
            "decomposition": dec}


def confirmation(par: dict[str, set[str]]) -> dict:
    conf = opendx_eval.load_confirm_e()
    fls = {b: flags(pats, par) for b, pats in sorted(conf.items())}
    runs = {"lineage": {s: ranks(CONF / f"{s}.json", "lineage") for s in ("S-mix", "S-dict", "S-union")},
            "strict": {s: ranks(STRICT / f"{s}.json", "strict") for s in ("S-mix", "S-dict", "S-union")}}
    runs["lineage"]["pure dict"] = ranks(PURE, "lineage")
    for s in ("S-mix", "S-dict", "S-union"):
        if ranks(STRICT / f"{s}.json", "lineage") != runs["lineage"][s]:
            raise SystemExit(f"{s}: P0-3 strict run ranks_lineage differ from the confirmation run")
    for rule, rs in runs.items():
        for s, rk in rs.items():
            for b, fl in fls.items():
                if len(rk[b]) != len(fl):
                    raise SystemExit(f"E {rule} {s} {b}: {len(rk[b])} ranks for {len(fl)} patients")
    pairs = {"lineage": PAIRS, "strict": [p for p in PAIRS if p[0] != "pure dict"]}
    rows, macros, dec = analyse("confirmation E [post hoc]", fls, runs, pairs)
    names = {key: v["name"] for key, v in v2_ids.universe().items()}
    named = {b: named_diseases(fl, {rule: {s: rk[b] for s, rk in rs.items()} for rule, rs in runs.items()}, names)
             for b, fl in fls.items()}
    return {"flags": {b: flag_counts(fl) for b, fl in fls.items()}, "rows": rows, "macro": macros,
            "decomposition": dec, "named_diseases": named}


def p13_rule(rows: list[dict]) -> dict:
    r = next(x for x in rows if x["rule"] == "lineage" and x["stratum"] == NEITHER and x["bench"] == "ConfirmE-DDXPlus"
             and x["A"] == "S-dict" and x["k"] == 10)
    sdict = {"diff": -r["diff"], "ci": [-r["ci"][1], -r["ci"][0]], "ci_cluster": [-r["ci_cluster"][1],
                                                                                   -r["ci_cluster"][0]]}

    def met(ci: list[float]) -> bool:
        return ci[0] <= 0

    return {"rule": "DDXPlus-E, neither stratum, lineage Top-10: S-dict - S-mix CI includes 0 or is negative -> H3 "
                    "caveat 4 at the sealed-set level; otherwise development-level only",
            "S-dict - S-mix": sdict, "n": r["n"], "diseases": r["diseases"],
            "met_patient_ci": met(sdict["ci"]), "met_cluster_ci": met(sdict["ci_cluster"])}


def fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:+6.2f},{ci[1]:+6.2f}]"


def report(res: dict) -> None:
    print("input-gold flag counts:")
    for part in ("development", "confirmation"):
        for b, c in res[part]["flags"].items():
            print(f"  {b:17s} n {c['n']:5d}  direct {c['direct']:5d}  parent only {c['parent_only']:4d}  "
                  f"named {c[NAMED]:5d} ({100 * c[NAMED] / c['n']:4.1f}%)  neither {c[NEITHER]:5d}  diseases "
                  f"{c['diseases']} (named {c['diseases_with_named']}, neither {c['diseases_with_neither']})")
    print(f"\n{'part':4s} {'rule':7s} {'comparison':18s} {'k':>3s} {'set':17s} {'stratum':27s} {'n':>5s} {'dz':>4s} "
          f"{'A':>6s} {'B':>6s} {'diff':>6s} {'patient CI':>16s} {'cluster CI':>16s} {'Holm p':>8s}")
    for part in ("development", "confirmation"):
        for r in res[part]["rows"]:
            print(f"{part[:4]:4s} {r['rule']:7s} {r['comparison']:18s} @{r['k']:<2d} {r['bench']:17s} "
                  f"{r['stratum']:27s} {r['n']:5d} {r['diseases']:4d} {r['A_pct']:6.2f} {r['B_pct']:6.2f} "
                  f"{r['diff']:+6.2f} {fmt_ci(r['ci'])} {fmt_ci(r['ci_cluster'])} {r['p_holm']:8.2g}")
    print("\nmacro (equal weight over the two common sets; patient CI set-stratified, cluster CI diseases within sets):")
    for part in ("development", "confirmation"):
        for m in res[part]["macro"]:
            print(f"  {part[:4]:4s} {m['rule']:7s} {m['comparison']:18s} @{m['k']:<2d} {m['stratum']:27s} "
                  f"n {'/'.join(str(v) for v in m['n'].values()):10s} {m['A']:6.2f} {m['B']:6.2f} {m['diff']:+6.2f} "
                  f"{fmt_ci(m['ci_patient_stratified'])} {fmt_ci(m['ci_cluster'])}")
    print("\nstratum shares of each set's difference and the interaction (named - neither):")
    for part in ("development", "confirmation"):
        for d in res[part]["decomposition"]:
            s, it = d["strata"], d["interaction"]
            inter = "n/a" if it["diff"] is None else f"{it['diff']:+6.2f} {fmt_ci(it['ci'])} {fmt_ci(it['ci_cluster'])}"
            print(f"  {part[:4]:4s} {d['rule']:7s} {d['comparison']:18s} @{d['k']:<2d} {d['bench']:17s} overall "
                  f"{d['overall_diff']:+6.2f} = named {s[NAMED]['contribution_pp']:+6.2f} (diff {s[NAMED]['diff']:+6.2f},"
                  f" share {s[NAMED]['share_of_overall_pct']}%) + neither {s[NEITHER]['contribution_pp']:+6.2f} "
                  f"(diff {s[NEITHER]['diff']:+6.2f}) | interaction {inter}")
    print("\nE gold diseases with patients in the named stratum (named n / patients; Top-10 hits there, lineage | "
          "strict):")
    for b, ds in res["confirmation"]["named_diseases"].items():
        for d in ds[:15]:
            hits = " | ".join(" ".join(f"{s} {v}" for s, v in h.items()) for h in d["named_top10_hits"].values())
            print(f"  {b:17s} {d['name'][:40]:40s} {d['named']:4d}/{d['patients']:4d} (direct {d['direct']:4d})  {hits}")
    p = res["p13_rule"]
    print(f"\nP1-3 rule: S-dict - S-mix (neither, DDXPlus-E, lineage @10) {p['S-dict - S-mix']['diff']:+.2f} "
          f"{fmt_ci(p['S-dict - S-mix']['ci'])} cluster {fmt_ci(p['S-dict - S-mix']['ci_cluster'])}; "
          f"met (patient CI) {p['met_patient_ci']}, met (cluster CI) {p['met_cluster_ci']}")


def main() -> None:
    opendx_eval.RAW_GOLD = True
    par = parent_map()
    dev = development(par)
    conf = confirmation(par)
    res = {"note": "POST HOC on confirmation set E. diff = B - A in percentage points of Top-k (rule = lineage or "
                   "strict); ci = patient-paired bootstrap 95% (2,000, seed 0); ci_cluster = gold-disease cluster "
                   "bootstrap 95% (whole diseases resampled; gold cluster = sorted gold-key tuple); macro: equal-weight "
                   "over the two common sets, ci_patient_stratified = paired_stats.macro_compare, ci_cluster = diseases drawn "
                   "within each set. Stored ranks only. The pure dictionary run has no strict ranks.",
           "flag": "input names gold or parent: gold_keys(gold) & ev, or a direct parent of a gold key "
                   "(v2_ids.disease_isa) in ev; RAW_GOLD = True; patients without gold keys or input are skipped as "
                   "graph_dx.evaluate skips them",
           "runs": {"lineage": {**{s: str(CONF / f"{s}.json") for s in ("S-mix", "S-dict", "S-union")},
                                "pure dict": str(PURE)},
                    "strict": {s: str(STRICT / f"{s}.json") for s in ("S-mix", "S-dict", "S-union")},
                    "development": {s: str(p) for s, p in DEV_RUNS.items()}},
           "development": dev, "confirmation": conf, "p13_rule": p13_rule(conf["rows"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    report(res)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
