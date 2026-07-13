#!/usr/bin/env python3
"""Compare v87 (PubMed-only baseline) vs v110 (multi-source medkg) across 3 benchmarks.

Looks for result files at:
  v87:  pilot/results/v87_*_results.json
  v110: pilot/results/v110_*_results.json

Outputs comparison table in markdown.
"""
import json, os
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("/home/max/Graph-DDXPlus/pilot/results")
BENCHMARKS = ["ddxplus", "symcat", "rarebench"]


def find_result(prefix, benchmark):
    """Find latest result file for given prefix + benchmark."""
    candidates = list(RESULTS_DIR.glob(f"*{prefix}*{benchmark}*.json"))
    candidates += list(RESULTS_DIR.glob(f"*{benchmark}*{prefix}*.json"))
    if not candidates: return None
    # Pick most recent
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_metric(path, metric_keys=("top1_accuracy", "GTPA@1", "@1", "accuracy")):
    if not path or not path.exists(): return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    for k in metric_keys:
        if k in d: return d[k]
        # nested
        for v in d.values():
            if isinstance(v, dict) and k in v:
                return v[k]
    # fallback: any number
    return None


def main():
    print("# v87 vs v110 (medkg) Comparison\n")
    print("| Benchmark | v87 (PubMed-only) | v110 (Multi-source) | Δ |")
    print("|-----------|------------------|--------------------|---|")
    for b in BENCHMARKS:
        r87 = find_result("v87", b)
        r110 = find_result("v110", b)
        m87 = extract_metric(r87)
        m110 = extract_metric(r110)
        delta = (m110 - m87) if (m87 is not None and m110 is not None) else None
        d_str = f"{delta:+.2%}" if delta is not None else "N/A"
        print(f"| {b:12s} | {m87 if m87 is not None else 'N/A'} | {m110 if m110 is not None else 'N/A'} | {d_str} |")
    print()
    print("v87 SOTA (current README): DDXPlus 66.48%, SymCat 43.27%, RareBench 26.49%")


if __name__ == "__main__":
    main()
