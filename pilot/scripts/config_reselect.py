"""Re-select the frozen configuration on the final KG with one pre-declared, benchmark-agnostic criterion.

The frozen configuration was chosen on the development KG. On the final KG two alternatives beat it
on single benchmarks (two-stage length normalization on RAMEDIS, no attribute selectivity on
DDXPlus), so it is no longer the best single configuration. Re-selecting is allowed under §7 only
if ONE configuration is chosen for every benchmark by ONE criterion fixed before looking:

    criterion = unweighted mean Top-1 over the six diagnostic benchmarks
                (DDXPlus, SymCat, RAMEDIS, LIRICAL, HMS, MME); ties -> mean Top-3.

Search is coordinate-wise from the current frozen configuration: each step varies one switch over a
fixed grid while the others hold their best value so far, and keeps a change only if the criterion
improves. Every evaluated configuration and its per-benchmark scores are written to the log, so the
selection is auditable and nothing is hidden.

    python config_reselect.py --kg KG.json --out-dir DIR [--jobs 5]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_suite import BENCHES, FROZEN_DEV as FROZEN  # noqa: E402  (search starts from it)

# one switch per step; "_env:" keys are environment variables of the scorer
GRID: list[tuple[str, list[str]]] = [
    ("--verb_w", ["0", "0.5", "1.0"]),
    ("--normexp", ["0.2", "0.28", "0.35", "0.5"]),
    ("_env:KG_ATTR_SEL", ["1", "0"]),
    ("--cov_mu", ["0", "0.5", "1.0"]),
    ("--fuse", ["minrank", "mult"]),
]
DEFAULT = {"--verb_w": "0", "_env:KG_ATTR_SEL": "1"}


def key(cfg: dict) -> str:
    return "__".join(f"{k.strip('-').replace('_env:', 'env-')}={v}" for k, v in sorted(cfg.items()))


def run_one(kg: str, cfg: dict, bench: str, out: Path) -> Path:
    stem = out / f"{key(cfg)}__{bench}"
    if list(out.glob(f"{stem.name}.*.json")) and Path(f"{stem}.log").exists():
        return stem
    args = [sys.executable, "pilot/scripts/onlykg_eval_unified2.py", "--kg", kg,
            "--benchmarks", bench, "--dump_ranks", str(stem)]
    env = dict(os.environ, KG_ATTR_COMBINE="mean", PYTHONHASHSEED="0")
    for k, v in {**FROZEN, **cfg}.items():
        if k.startswith("_env:"):
            env[k[5:]] = v
        else:
            args += [k, v]
    if bench == "ddxplus":
        env["DDX_COMPOSE_PAIN"] = "1"
    with open(f"{stem}.log", "w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    return stem


def score(out: Path, cfg: dict) -> dict:
    top1, top3 = {}, {}
    k = key(cfg)
    for f in out.glob(f"{k}__*.*.json"):
        # "<key>__<bench-arg>.<BenchName>.json"; the key itself may contain dots (verb_w=0.5)
        name = f.name[len(k) + 2:].split(".", 1)[1][:-5]
        ranks = [int(r[0]) if isinstance(r, list) else int(r) for r in json.load(open(f))]
        top1[name] = 100 * sum(r <= 1 for r in ranks) / len(ranks)
        top3[name] = 100 * sum(r <= 3 for r in ranks) / len(ranks)
    assert len(top1) == 6, f"expected 6 benchmarks, got {sorted(top1)}"
    return {"mean@1": sum(top1.values()) / 6, "mean@3": sum(top3.values()) / 6,
            "top1": {k: round(v, 2) for k, v in sorted(top1.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def evaluate(cfgs: list[dict]) -> list[dict]:
        jobs = [(c, b) for c in cfgs for b in BENCHES]
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(lambda j: run_one(a.kg, j[0], j[1], out), jobs))
        return [score(out, c) for c in cfgs]

    best = dict(DEFAULT)
    best_s = evaluate([best])[0]
    trail = [{"step": "start (current frozen configuration)", "cfg": dict(best), **best_s}]
    print(f"start {best_s['mean@1']:.3f} {best_s['top1']}", flush=True)
    for switch, values in GRID:
        cands = [{**best, switch: v} for v in values if v != best.get(switch, FROZEN.get(switch))]
        for c, s in zip(cands, evaluate(cands)):
            trail.append({"step": switch, "cfg": c, **s})
            print(f"  {switch}={c[switch]:8} mean@1 {s['mean@1']:.3f} mean@3 {s['mean@3']:.3f} {s['top1']}",
                  flush=True)
            if (s["mean@1"], s["mean@3"]) > (best_s["mean@1"], best_s["mean@3"]):
                best, best_s = c, s
        print(f"after {switch}: {best} -> {best_s['mean@1']:.3f}", flush=True)
    json.dump({"criterion": "unweighted mean Top-1 over 6 benchmarks; ties -> mean Top-3",
               "selected": best, "selected_scores": best_s, "trail": trail},
              open(out / "reselect.json", "w"), indent=1)
    print(f"SELECTED {best} mean@1 {best_s['mean@1']:.3f} {best_s['top1']}")


if __name__ == "__main__":
    main()
