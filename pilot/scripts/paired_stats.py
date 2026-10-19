"""Paired comparison of two evaluation runs on the same patients (research design §9.23, E4).

For each benchmark: difference in lineage Top-k (B - A) with a patient-level paired bootstrap 95% CI,
an exact two-sided McNemar test on discordant patients, and Holm correction across benchmarks.
Across benchmarks: a two-sided sign test on the direction of the Top-k differences, and with --macro
the equal-weight macro difference over the named sets with a set-stratified paired bootstrap CI.

    python paired_stats.py A.json B.json --k 10 [--label "frozen vs no_topic"]
        [--require SET1,SET2] [--macro SET1,SET2] [--out summary.json]
"""
from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def sign_test(pos: int, neg: int) -> float:
    return mcnemar_exact(pos, neg)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def _check_sets(ra: dict[str, list], rb: dict[str, list], sets: list[str]) -> None:
    bad = []
    for s in sets:
        if s not in ra or s not in rb:
            bad.append(f"{s}: missing in {'A' if s not in ra else 'B'}")
        elif len(ra[s]) != len(rb[s]):
            bad.append(f"{s}: n differs (A {len(ra[s])}, B {len(rb[s])})")
        elif not ra[s]:
            bad.append(f"{s}: no patients")
    if bad:
        raise ValueError("unpaired sets: " + "; ".join(bad))


def cluster_ci(ha: list[bool], hb: list[bool], keys: list, boot: int, rng: random.Random) -> tuple[list[float], int]:
    groups: dict = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    gl = list(groups.values())
    dist = []
    for _ in range(boot):
        idx = [i for _ in gl for i in gl[rng.randrange(len(gl))]]
        dist.append(100 * sum(hb[i] - ha[i] for i in idx) / len(idx))
    dist.sort()
    return [round(dist[int(0.025 * boot)], 2), round(dist[int(0.975 * boot) - 1], 2)], len(gl)


def compare(a: dict, b: dict, k: int, boot: int = 2000,
            clusters: dict[str, list] | None = None, require: list[str] | None = None) -> list[dict]:
    """clusters: per benchmark, one key per patient (e.g. the gold disease). When given, a second CI
    resamples whole clusters, since patients of one disease are not independent draws.
    require: sets that must be present in both runs with the same patient count (ValueError otherwise);
    other unpaired benchmarks are still skipped."""
    ra = {r["bench"]: r["ranks_lineage"] for r in a["results"]}
    rb = {r["bench"]: r["ranks_lineage"] for r in b["results"]}
    if require:
        _check_sets(ra, rb, require)
    rows = []
    for bench in ra:
        if bench not in rb or len(ra[bench]) != len(rb[bench]):
            continue
        ha = [p is not None and p <= k for p in ra[bench]]
        hb = [p is not None and p <= k for p in rb[bench]]
        n = len(ha)
        d = 100 * (sum(hb) - sum(ha)) / n
        rng = random.Random(0)
        dist = []
        for _ in range(boot):
            idx = [rng.randrange(n) for _ in range(n)]
            dist.append(100 * sum(hb[i] - ha[i] for i in idx) / n)
        dist.sort()
        only_a = sum(x and not y for x, y in zip(ha, hb))
        only_b = sum(y and not x for x, y in zip(ha, hb))
        rows.append({"bench": bench, "n": n, f"A@{k}": round(100 * sum(ha) / n, 2),
                     f"B@{k}": round(100 * sum(hb) / n, 2), "diff": round(d, 2),
                     "ci": [round(dist[int(0.025 * boot)], 2), round(dist[int(0.975 * boot) - 1], 2)],
                     "only_A": only_a, "only_B": only_b, "p": mcnemar_exact(only_a, only_b)})
        if clusters and len(clusters.get(bench, ())) == n:
            rows[-1]["ci_cluster"], rows[-1]["clusters"] = cluster_ci(ha, hb, clusters[bench], boot, rng)
    family = [r for r in rows if require is None or r["bench"] in require]     # Holm family = judged sets
    for r, p in zip(family, holm([r["p"] for r in family])):
        r["p_holm"] = p
    return rows


def macro_compare(a: dict, b: dict, k: int, sets: list[str], boot: int = 2000, seed: int = 0) -> dict:
    """Equal-weight macro Top-k difference (B - A) over `sets`: the mean of the per-set differences, so a
    large set does not outweigh a small one. ValueError unless every set is in both runs with the same n.

    Bootstrap convention (fixed by design E4; it reproduces the design D T4 numbers, graph -> fusion
    @10 +2.67 [+1.81, +3.58]): one random.Random(seed) for the whole run; in each replicate the sets are
    visited in sorted name order and each draws n patient indices with replacement, shared by A and B.
    Changing the set order or giving each set its own RNG changes the CI. Percentile CI with the same
    index convention as compare(); p = 2 * min(P(d <= 0), P(d >= 0)) over replicates, capped at 1."""
    if len(set(sets)) != len(sets):
        raise ValueError(f"duplicate sets: {sets}")
    sets = sorted(sets)
    if not sets:
        raise ValueError("macro_compare needs at least one set")
    ra = {r["bench"]: r["ranks_lineage"] for r in a["results"]}
    rb = {r["bench"]: r["ranks_lineage"] for r in b["results"]}
    _check_sets(ra, rb, sets)
    per_set, deltas = {}, []
    for s in sets:
        ha = [p is not None and p <= k for p in ra[s]]
        hb = [p is not None and p <= k for p in rb[s]]
        n = len(ha)
        per_set[s] = {"n": n, "A": 100 * sum(ha) / n, "B": 100 * sum(hb) / n,
                      "diff": 100 * (sum(hb) - sum(ha)) / n}
        deltas.append([int(y) - int(x) for x, y in zip(ha, hb)])
    rng = random.Random(seed)
    dist = []
    for _ in range(boot):
        tot = 0.0
        for dl in deltas:
            n = len(dl)
            tot += 100 * sum(dl[rng.randrange(n)] for _ in range(n)) / n
        dist.append(tot / len(sets))
    dist.sort()
    p = min(1.0, 2 * min(sum(x <= 0 for x in dist), sum(x >= 0 for x in dist)) / boot)

    def macro(key: str) -> float:
        return sum(v[key] for v in per_set.values()) / len(sets)

    return {"sets": sets, "k": k, "A": round(macro("A"), 2), "B": round(macro("B"), 2),
            "diff": round(macro("diff"), 2),
            "ci": [round(dist[int(0.025 * boot)], 2), round(dist[int(0.975 * boot) - 1], 2)], "p": p,
            "per_set": {s: {key: v if key == "n" else round(v, 2) for key, v in d.items()}
                        for s, d in per_set.items()}}


def _set_list(arg: str) -> list[str]:
    return [s for s in arg.split(",") if s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--require", type=_set_list, default=None,
                    help="comma-separated sets that must be paired in A and B; stop otherwise")
    ap.add_argument("--macro", type=_set_list, default=None,
                    help="comma-separated sets for the equal-weight macro comparison (also required)")
    x = ap.parse_args()
    a, b = json.loads(Path(x.a).read_text()), json.loads(Path(x.b).read_text())
    require = sorted(set(x.require or []) | set(x.macro or [])) or None
    rows = compare(a, b, x.k, require=require)
    pos = sum(r["diff"] > 0 for r in rows)
    neg = sum(r["diff"] < 0 for r in rows)
    summary = {"label": x.label, "k": x.k, "A": x.a, "B": x.b, "rows": rows,
               "B_better": pos, "A_better": neg, "sign_test_p": sign_test(pos, neg)}
    print(f"== {x.label} (Top-{x.k}, B - A)")
    for r in rows:
        print(f"{r['bench']:20} n={r['n']:6}  A {r[f'A@{x.k}']:5.1f}  B {r[f'B@{x.k}']:5.1f}  "
              f"diff {r['diff']:+5.1f} [{r['ci'][0]:+5.1f}, {r['ci'][1]:+5.1f}]  "
              f"p={r['p']:.3g} holm={r['p_holm']:.3g}")
    print(f"B better on {pos}, A better on {neg}; sign test p={summary['sign_test_p']:.3g}")
    if x.macro:
        m = macro_compare(a, b, x.k, x.macro)
        summary["macro"] = m
        print(f"macro over {len(m['sets'])} sets ({', '.join(m['sets'])}): A {m['A']:5.2f}  B {m['B']:5.2f}  "
              f"diff {m['diff']:+.2f} [{m['ci'][0]:+.2f}, {m['ci'][1]:+.2f}]  p={m['p']:.3g}")
    if x.out:
        Path(x.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
