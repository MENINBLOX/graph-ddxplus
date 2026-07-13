#!/usr/bin/env python3
"""v74 evaluation — NB-style scoring with LLM-derived P(E|D).

For each patient:
  log P(patient | D) = Σ over binary evs:
                        log P(E|D)     if patient answered yes
                        log(1 - P(E|D)) if patient answered no

P(E|D) comes from v74 direct IE (LLM medical knowledge, no train labels).

This is the NB framework that scored 99.43% with train labels. Here we
test whether LLM medical knowledge can match that.
"""
from __future__ import annotations
import sys, json, csv, ast, math, argparse
from pathlib import Path
sys.path.insert(0, "pilot/scripts")

EV_META = "data/ddxplus/release_evidences.json"


def load_v74_profiles(path):
    profiles = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prof = {}
            for ev, val in r["profile"].items():
                if isinstance(val, (list, tuple)):
                    prof[ev] = val[1]  # v74 format: [cat, prob]
                else:
                    prof[ev] = float(val)  # v76 format: prob directly
            profiles[r["dcui"]] = prof
    return profiles


def load_ddxplus(n_max):
    ev_meta = json.load(open(EV_META))
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    binary_evs = {ev_id for ev_id, m in ev_meta.items()
                  if m.get("data_type") == "B" and m.get("default_value") == 0}
    patients = []; n = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= n_max: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            yes_binary = set()
            for ev in evs:
                base = ev.split("_@_")[0] if "_@_" in ev else ev
                if base in binary_evs:
                    yes_binary.add(base)
            no_binary = binary_evs - yes_binary
            patients.append((true_cui, yes_binary, no_binary, n)); n += 1
    return dcs_list, patients, binary_evs


def nb_score(yes_set, no_set, profiles, smooth=1e-3):
    """log P(patient | D) for each D."""
    scores = {}
    for d, prof in profiles.items():
        s = 0.0
        for ev in yes_set:
            p = prof.get(ev, 0.01)
            p = max(smooth, min(1-smooth, p))
            s += math.log(p)
        for ev in no_set:
            p = prof.get(ev, 0.01)
            p = max(smooth, min(1-smooth, p))
            s += math.log(1 - p)
        scores[d] = s
    return scores


def cosine_score(yes_set, profiles):
    """Compare to v71-like cosine on binary evidences only."""
    scores = {}
    p_norm = math.sqrt(len(yes_set)) or 1e-9
    for d, prof in profiles.items():
        dot = sum(prof.get(e, 0) for e in yes_set)
        d_norm = math.sqrt(sum(v*v for v in prof.values())) or 1e-9
        scores[d] = dot / (p_norm * d_norm)
    return scores


def evaluate(profiles, patients, dcs_list, method='nb'):
    n = c1 = c3 = c5 = c10 = 0; rr = 0.0
    for true_cui, yes, no, pid in patients:
        if method == 'nb':
            scores = nb_score(yes, no, profiles)
        else:
            scores = cosine_score(yes, profiles)
        ranked = sorted(profiles.keys(), key=lambda d: -scores[d])
        n += 1
        try: rank = ranked.index(true_cui)+1
        except: rank = len(dcs_list)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0/rank
    return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n, "at5": 100*c5/n,
            "at10": 100*c10/n, "mrr": rr/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v74_path", required=True)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    profiles = load_v74_profiles(args.v74_path)
    print(f"Loaded {len(profiles)} disease profiles", flush=True)
    coverage = [len(p) for p in profiles.values()]
    print(f"Avg P(E|D) entries per disease: {sum(coverage)/len(coverage):.1f}")

    dcs_list, patients, binary_evs = load_ddxplus(args.n)
    print(f"DDXPlus patients: {len(patients)}", flush=True)

    print("=== v74 LLM-NB scoring ===")
    r = evaluate(profiles, patients, dcs_list, method='nb')
    print(f"  NB: @1={r['at1']:.2f}% @3={r['at3']:.2f}% @5={r['at5']:.2f}% "
          f"@10={r['at10']:.2f}% MRR={r['mrr']:.4f}")

    print("=== v74 LLM-Cosine scoring (yes only) ===")
    r = evaluate(profiles, patients, dcs_list, method='cosine')
    print(f"  Cos: @1={r['at1']:.2f}% @3={r['at3']:.2f}% @5={r['at5']:.2f}% "
          f"@10={r['at10']:.2f}% MRR={r['mrr']:.4f}")


if __name__ == "__main__":
    main()
