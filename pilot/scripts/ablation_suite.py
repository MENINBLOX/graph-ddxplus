"""Component ablation with patient-level uncertainty, on one KG with one frozen configuration.

Two things the results previously lacked are produced here together:

  * a component ablation run on the FINAL KG -- every lever was evaluated during development, but on
    whichever KG was current at the time, so no single table said what each component contributes;
  * patient-bootstrap 95% confidence intervals on every reported top-k, and a paired bootstrap test
    of each ablated variant against the full system. Paired, because the variants rank the same
    patients: resampling patients jointly cancels the between-patient variance and isolates the
    effect of the component. Small benchmarks (MME has 40 patients, one patient = 2.5 points) are
    exactly where unpaired intervals would hide a real difference or invent one.

Each variant differs from the frozen configuration in ONE switch. Variants run as independent
processes (they share nothing but the read-only KG), then statistics are computed from the
per-patient rank dumps, so the statistics can be recomputed without re-running any scoring.

    python ablation_suite.py run   --kg KG.json --out-dir DIR [--jobs 3]
    python ablation_suite.py stats --out-dir DIR [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

P = "pilot/cache/prod"

# The configuration frozen on the DEVELOPMENT KG. Kept because config_reselect.py starts from it.
FROZEN_DEV = {
    "--parents": f"{P}/phenotype_parents_rich.json", "--hier": "1", "--sapbert": "0", "--tau": "0.78",
    "--hpo_bridge": f"{P}/hpo_bridge.json", "--hpo_anc": "0", "--hpo_w": "0.5",
    "--bio_bridge": f"{P}/biochem_bridge.json",
    "--syn_bridge": f"{P}/synonym_bridge.json", "--syn_w": "1.0",
    "--cov_mu": "0.5", "--normexp": "0.28", "--local_idf": "1.0", "--neg_w": "15",
    "--lsa": "1", "--lsa_w": "2.0", "--fuse": "minrank",
    "--attr": "1", "--attr_w": "10", "--attr_mu": "0.0",
    "--attr_use": "location,character,severity,onset,timing,aggravating,relieving",
    "--unified": "1",
}

# The frozen configuration shared by every benchmark (§C5), re-selected on the final KG by
# config_reselect.py with one pre-declared criterion (unweighted mean Top-1 over the six diagnostic
# benchmarks). Kept identical to reproduce.sh. Scorer environment switches live in FROZEN_ENV.
FROZEN = {**FROZEN_DEV, "--normexp": "0.35", "--verb_w": "0.5", "--fuse": "mult"}
FROZEN_ENV = {"KG_ATTR_COMBINE": "mean", "KG_ATTR_SEL": "0"}

# One switch each. A value of None removes the flag entirely (the component then falls back to the
# evaluator's own "off" default), which is how the bridges are disabled.
VARIANTS: dict[str, dict[str, str | None]] = {
    "full": {},
    "-unified_evidence": {"--unified": "0"},
    "-local_idf": {"--local_idf": "0"},
    "-absence_evidence": {"--neg_w": "0"},
    "-length_retune": {"--normexp": "0.5"},
    "-two_stage_length": {"--verb_w": "0"},
    "-coverage_smoothing": {"--cov_mu": "0"},
    "-attributes": {"--attr": "0"},
    "-latent_prior": {"--lsa": "0"},
    # Weighted-sum fusion reads --lsa_w as a convex weight, (1-w)*evidence + w*prior. The frozen
    # 2.0 is a MULTIPLIER for the multiplicative/min-rank forms; fed to the sum it puts weight -1 on
    # the evidence channel and inverts the ranking (an artefact, not a property of fusion). The
    # coherent weighted-sum system is the one with the weight it was actually used at.
    "alt:weighted_sum_fusion": {"--fuse": "sum", "--lsa_w": "0.4"},
    "-synonym_bridge": {"--syn_bridge": None},
    "-biochem_bridge": {"--bio_bridge": None},
    "-hpo_bridge": {"--hpo_bridge": None},
    "-hierarchy": {"--hier": "0"},
    # alternatives rather than removals, requested by the paper's method section
    "alt:minrank_fusion": {"--fuse": "minrank"},
    "alt:sum_attribute_combine": {"_env:KG_ATTR_COMBINE": "sum"},
    "alt:attr_selectivity": {"_env:KG_ATTR_SEL": "1"},   # value-IDF weighting of attribute agreements
}

BENCHES = ("ddxplus", "rarebench", "symcat")
KS = (1, 3, 10)


def argv_for(variant: str, kg: str, bench: str, dump: str) -> list[str]:
    cfg = dict(FROZEN)
    for k, v in VARIANTS[variant].items():
        if k.startswith("_env:"):
            continue                                   # environment switches are applied in run()
        if v is None:
            cfg.pop(k, None)
        else:
            cfg[k] = v
    args = [sys.executable, "pilot/scripts/onlykg_eval_unified2.py", "--kg", kg,
            "--benchmarks", bench, "--dump_ranks", dump]
    for k, v in cfg.items():
        args += [k, v]
    return args


def run(a: argparse.Namespace) -> None:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # PYTHONHASHSEED is pinned because some scorer paths sum over sets of concept strings, and the
    # iteration order (hence float summation order, hence near-ties) follows the hash seed: without
    # it DDXPlus Top-1 moved by one patient between runs of an identical configuration.
    env = dict(os.environ, **FROZEN_ENV, PYTHONHASHSEED="0")
    jobs = []
    for v in VARIANTS:
        for b in BENCHES:
            stem = out / f"{v}__{b}"
            if list(out.glob(f"{v}__{b}.*.json")) and (out / f"{v}__{b}.log").exists():
                continue                               # resumable: skip finished (variant, bench)
            jobs.append((v, b, stem))
    print(f"{len(jobs)} (variant, benchmark) runs to do, {a.jobs} at a time", flush=True)

    def one(job):
        v, b, stem = job
        e = dict(env)
        for k, val in VARIANTS[v].items():
            if k.startswith("_env:"):
                e[k[5:]] = val
        if b == "ddxplus":
            e["DDX_COMPOSE_PAIN"] = "1"
        with open(f"{stem}.log", "w") as log:
            rc = subprocess.run(argv_for(v, a.kg, b, str(stem)), stdout=log, stderr=subprocess.STDOUT,
                                env=e).returncode
        return v, b, rc

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for v, b, rc in ex.map(one, jobs):
            print(f"  {'ok ' if rc == 0 else 'FAIL'} {v:22} {b}", flush=True)


# ---------------------------------------------------------------------------------------------
def load_ranks(out: Path) -> dict[str, dict[str, list[int]]]:
    """{variant: {benchmark_name: [rank per patient]}} from the per-patient dumps."""
    res: dict[str, dict[str, list[int]]] = {}
    for f in sorted(out.glob("*__*.*.json")):
        v, rest = f.name.split("__", 1)
        bench = rest.split(".", 1)[1][:-5]             # "<bench-arg>.<BenchName>.json"
        res.setdefault(v, {})[bench] = [int(r[0]) if isinstance(r, list) else int(r)
                                        for r in json.load(open(f))]
    return res


def topk(ranks: list[int], idx: list[int], k: int) -> float:
    return 100.0 * sum(1 for i in idx if ranks[i] <= k) / len(idx)


def stats(a: argparse.Namespace) -> None:
    out = Path(a.out_dir)
    R = load_ranks(out)
    if "full" not in R:
        sys.exit("no 'full' variant dumps found — run first")
    rng = random.Random(42)
    benches = sorted(R["full"])
    boots = {b: [[rng.randrange(len(R["full"][b])) for _ in R["full"][b]] for _ in range(a.boot)]
             for b in benches}

    rows = []
    for b in benches:
        full = R["full"][b]
        n = len(full)
        for v in VARIANTS:
            if b not in R.get(v, {}):
                continue
            ranks = R[v][b]
            if len(ranks) != n:
                print(f"  SKIP {v} on {b}: {len(ranks)} patients vs {n} in full — not paired")
                continue
            everyone = list(range(n))
            row = {"bench": b, "variant": v, "n": n}
            for k in KS:
                pt = topk(ranks, everyone, k)
                dist = sorted(topk(ranks, idx, k) for idx in boots[b])
                lo, hi = dist[int(0.025 * a.boot)], dist[int(0.975 * a.boot) - 1]
                row[f"@{k}"], row[f"@{k}_ci"] = round(pt, 2), [round(lo, 2), round(hi, 2)]
                if v != "full":
                    d = sorted(topk(ranks, idx, k) - topk(full, idx, k) for idx in boots[b])
                    delta = pt - topk(full, everyone, k)
                    # two-sided paired bootstrap p: how often the resampled difference crosses zero
                    below = sum(1 for x in d if x <= 0) / a.boot
                    above = sum(1 for x in d if x >= 0) / a.boot
                    row[f"d@{k}"] = round(delta, 2)
                    row[f"d@{k}_ci"] = [round(d[int(0.025 * a.boot)], 2),
                                        round(d[int(0.975 * a.boot) - 1], 2)]
                    row[f"p@{k}"] = round(min(1.0, 2 * min(below, above)), 4)
            rows.append(row)

    json.dump(rows, open(out / "ablation_stats.json", "w"), indent=1)

    # main table: full system with CIs
    print(f"\n=== Full system, patient-bootstrap 95% CI (B={a.boot}) ===")
    print(f"{'benchmark':22}{'n':>6}   {'@1 [95% CI]':>22}{'@3 [95% CI]':>24}{'@10 [95% CI]':>24}")
    for r in rows:
        if r["variant"] != "full":
            continue
        cells = "".join(f"{r[f'@{k}']:>8.2f} [{r[f'@{k}_ci'][0]:5.1f},{r[f'@{k}_ci'][1]:5.1f}]"
                        .rjust(24) for k in KS)
        print(f"{r['bench']:22}{r['n']:>6}{cells}")

    # ablation table: Δ@1 per component per benchmark
    print("\n=== Component ablation: Δ@1 vs full (paired bootstrap; * p<0.05) ===")
    head = "".join(f"{b.replace('RareBench_', '')[:9]:>11}" for b in benches)
    print(f"{'removed component':22}{head}")
    for v in VARIANTS:
        if v == "full":
            continue
        cells = ""
        for b in benches:
            r = next((x for x in rows if x["bench"] == b and x["variant"] == v), None)
            cells += (f"{r['d@1']:>+9.2f}{'*' if r['p@1'] < 0.05 else ' '} " if r else f"{'-':>11}")
        print(f"{v:22}{cells}")
    print(f"\n-> {out / 'ablation_stats.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kg", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--jobs", type=int, default=3,
                   help="parallel evaluator processes; bounded by RAM (each loads the KG)")
    s = sub.add_parser("stats")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else stats(a)


if __name__ == "__main__":
    main()
