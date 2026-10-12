"""Paired comparison of two evaluation runs on the same patients (research design §9.23).

For each benchmark: difference in lineage Top-k (B - A) with a patient-level paired bootstrap 95% CI,
an exact two-sided McNemar test on discordant patients, and Holm correction across benchmarks.
Across benchmarks: a two-sided sign test on the direction of the Top-k differences.

    python paired_stats.py A.json B.json --k 10 [--label "frozen vs no_topic"]
"""
from __future__ import annotations

import argparse
import json
import random
from math import comb


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


def compare(a: dict, b: dict, k: int, boot: int = 2000) -> list[dict]:
    ra = {r["bench"]: r["ranks_lineage"] for r in a["results"]}
    rb = {r["bench"]: r["ranks_lineage"] for r in b["results"]}
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
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    x = ap.parse_args()
    rows = compare(json.load(open(x.a)), json.load(open(x.b)), x.k)
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
    if x.out:
        json.dump(summary, open(x.out, "w"), indent=1)


if __name__ == "__main__":
    main()
