#!/usr/bin/env python3
"""v93 KG on PhenoBrain Zenodo subset (HPO-CUI cosine + IDF eval).

Reuses the SAME algorithm (cosine + IDF reweight, identical hyper-params)
as ``eval_symcat_full_v71.py``. Only the *patient loader* and
*candidate-set construction* differ.

PhenoBrain JSON layout
----------------------
Each file is a list of cases, each case = [hpo_list, disease_id_list]
  hpos      : list[str] of HP:0000xxx
  diseases  : list[str] of OMIM / ORPHA / CCRD prefixed IDs

We map:
  HPO  -> UMLS CUI  via data/rarebench/hpo_umls_mapping.json
  OMIM/ORPHA -> UMLS CUI  via data/rarebench/disease_umls_mapping.json

Candidate set = union of all PhenoBrain target CUIs that exist in v93 KG.
This is the same "fair full-pool" pattern used by v71.

Reports: per-dataset and overall GTPA@1/3/5/10 and MRR.
"""
from __future__ import annotations
import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/max/Graph-DDXPlus")
sys.path.insert(0, str(ROOT / "pilot" / "scripts"))

PR_UNIVERSE = ROOT / "pilot" / "data" / "pr_universe.json"
HPO_MAP = ROOT / "data" / "rarebench" / "hpo_umls_mapping.json"
DIS_MAP = ROOT / "data" / "rarebench" / "disease_umls_mapping.json"
PB_BASE = ROOT / "data" / "external_benchmarks" / "phenobrain" / "Test cases"

DEFAULT_GRAPH = ROOT / "pilot" / "data" / "onlykg_graph_v93_s3.pkl"


# ---- v71 algorithm primitives (duplicated to avoid importing v71 CLI) ---

def build_profile(G, dcs, kappa, pr_set):
    profile = {}
    all_evs = set()
    allowed = {"patient_reportable", "history", "demographic"}
    for d in dcs:
        if d not in G:
            profile[d] = {}
            continue
        ed_w = defaultdict(float)
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") != "HAS_PHENOTYPE":
                continue
            cat = ed.get("category")
            if cat is None:
                if p not in pr_set:
                    continue
            elif cat not in allowed:
                continue
            ed_w[p] += ed.get("weight", 0.0)
        prof = {p: w / (w + kappa) for p, w in ed_w.items() if w > 0}
        profile[d] = prof
        all_evs.update(prof.keys())
    return profile, all_evs


def compute_idf(profile, df_threshold):
    n = len(profile)
    df = defaultdict(int)
    for prof in profile.values():
        for e, p in prof.items():
            if p >= df_threshold:
                df[e] += 1
    return {e: math.log((n + 1) / (df_e + 1)) + 1.0 for e, df_e in df.items()}


def reweight(profile, idf, alpha, beta):
    return {d: {e: (p ** alpha) * (idf.get(e, 1.0) ** beta)
                for e, p in prof.items()}
            for d, prof in profile.items()}


def score_cosine(pcuis, profile, idf, beta):
    scores = {}
    pat_vec = {e: idf.get(e, 1.0) ** beta for e in pcuis}
    p_norm = math.sqrt(sum(v * v for v in pat_vec.values())) or 1e-9
    for d, prof in profile.items():
        if not prof:
            scores[d] = -1e9
            continue
        dot = sum(pat_vec[e] * prof[e] for e in pcuis if e in prof)
        d_norm = math.sqrt(sum(v * v for v in prof.values())) or 1e-9
        scores[d] = dot / (p_norm * d_norm)
    return scores


# ---- PhenoBrain loader --------------------------------------------------

def load_phenobrain(datasets: list[str], hpo_map: dict, dis_map: dict):
    """Return list of (target_cuis, evidence_cuis, ds_name)."""
    patients = []
    for ds in datasets:
        path = PB_BASE / f"{ds}.json"
        if not path.exists():
            print(f"  [skip] missing: {path}", file=sys.stderr)
            continue
        cases = json.load(open(path))
        for c in cases:
            if not (isinstance(c, list) and len(c) >= 2):
                continue
            hpos, dis = c[0], c[1]
            # disease ids -> UMLS CUI (filter prefix; CCRD ignored — no map)
            target_cuis = set()
            for did in dis:
                info = dis_map.get(did)
                if info and info.get("umls_cui"):
                    target_cuis.add(info["umls_cui"])
            # HPO -> CUI
            pcuis = set()
            for hp in hpos:
                info = hpo_map.get(hp)
                if isinstance(info, dict) and info.get("umls_cui"):
                    pcuis.add(info["umls_cui"])
                elif isinstance(info, str):
                    pcuis.add(info)
            if target_cuis and pcuis:
                patients.append((target_cuis, pcuis, ds))
    return patients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--datasets",
                    default="HMS,LIRICAL,MME,RAMEDIS,PUMCH-ADM,PUMCH_L")
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=20.0)
    ap.add_argument("--df_thr", type=float, default=0.12)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"Loading v93 KG: {args.graph}", flush=True)
    G = pickle.load(open(args.graph, "rb"))
    print(f"  nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}",
          flush=True)
    pr_set = set(json.load(open(PR_UNIVERSE)))
    hpo_map = json.load(open(HPO_MAP))["mapping"]
    dis_map = json.load(open(DIS_MAP))["mapping"]

    print(f"Loading PhenoBrain ({datasets}) ...", flush=True)
    patients = load_phenobrain(datasets, hpo_map, dis_map)
    print(f"  patients with target+pcuis: {len(patients)}", flush=True)
    by_ds = Counter(p[2] for p in patients)
    for ds, n in by_ds.items():
        print(f"    {ds}: {n}")

    # Candidate disease pool = union of all targets ∩ KG.
    all_targets = set()
    for ts, _, _ in patients:
        all_targets.update(ts)
    in_kg = sorted(c for c in all_targets if c in G)
    not_in_kg = sorted(all_targets - set(in_kg))
    print(f"  unique target CUIs: {len(all_targets)}; in v93 KG: {len(in_kg)}; "
          f"missing: {len(not_in_kg)}", flush=True)

    # Build profile + IDF (v71 hyper-params)
    print("Building disease profiles + IDF ...", flush=True)
    base, all_evs = build_profile(G, in_kg, args.kappa, pr_set)
    idf = compute_idf(base, args.df_thr)
    profile = reweight(base, idf, args.alpha, args.beta)
    n_nonempty = sum(1 for d in in_kg if profile[d])
    print(f"  non-empty profiles: {n_nonempty}/{len(in_kg)} ; "
          f"|all_evs|={len(all_evs)}", flush=True)

    # Evaluate
    overall = {"n": 0, "c1": 0, "c3": 0, "c5": 0, "c10": 0, "rr": 0.0,
               "empty": 0, "unrank": 0}
    per_ds = {ds: {"n": 0, "c1": 0, "c3": 0, "c5": 0, "c10": 0, "rr": 0.0}
              for ds in by_ds}
    for target_cuis, pcuis, ds in patients:
        overall["n"] += 1
        per_ds[ds]["n"] += 1
        target_in_pool = [t for t in target_cuis if t in in_kg]
        if not target_in_pool:
            overall["unrank"] += 1
            continue
        pmatch = pcuis & all_evs
        if not pmatch:
            overall["empty"] += 1
            # can't score → rank = end of pool
            rank = len(in_kg)
        else:
            scores = score_cosine(pmatch, profile, idf, args.beta)
            ranked = sorted(in_kg, key=lambda d: -scores.get(d, -1e9))
            ranks = []
            for t in target_in_pool:
                try:
                    ranks.append(ranked.index(t) + 1)
                except ValueError:
                    ranks.append(len(in_kg))
            rank = min(ranks)
        # tally
        if rank == 1:
            overall["c1"] += 1; per_ds[ds]["c1"] += 1
        if rank <= 3:
            overall["c3"] += 1; per_ds[ds]["c3"] += 1
        if rank <= 5:
            overall["c5"] += 1; per_ds[ds]["c5"] += 1
        if rank <= 10:
            overall["c10"] += 1; per_ds[ds]["c10"] += 1
        overall["rr"] += 1.0 / rank
        per_ds[ds]["rr"] += 1.0 / rank

    def _pct(c, n): return f"{(100 * c / n):.2f}%" if n else "n/a"

    print(f"\n=== v93 KG on PhenoBrain ({overall['n']} patients, "
          f"pool={len(in_kg)} diseases) ===")
    n = overall["n"]
    print(f"  Empty-evidence (KG vocab miss) : {overall['empty']} "
          f"({100*overall['empty']/n:.1f}%)" if n else "  n=0")
    print(f"  Unranked (target not in pool)  : {overall['unrank']}")
    print(f"  Random baseline @1             : "
          f"{100/len(in_kg):.3f}%" if in_kg else "n/a")
    print(f"  GTPA: @1={_pct(overall['c1'], n)}  @3={_pct(overall['c3'], n)}  "
          f"@5={_pct(overall['c5'], n)}  @10={_pct(overall['c10'], n)}")
    print(f"  MRR : {overall['rr']/n:.4f}" if n else "  MRR: n/a")
    print("\n  Per-dataset:")
    for ds in by_ds:
        r = per_ds[ds]; m = r["n"]
        if not m: continue
        print(f"    {ds:10s} n={m:4d}  @1={_pct(r['c1'],m)}  "
              f"@5={_pct(r['c5'],m)}  @10={_pct(r['c10'],m)}  "
              f"MRR={r['rr']/m:.4f}")


if __name__ == "__main__":
    main()
