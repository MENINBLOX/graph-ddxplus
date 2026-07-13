#!/usr/bin/env python3
"""v42: Zero-shot Bayesian per-evidence likelihood diagnosis.

Architecture change: instead of "weighted sum of matched CUIs",
use proper Bayesian log-likelihood per evidence question.

For each binary evidence E (B-type, 208 of them):
- patient: yes (E_name in evidence list) or no (not in evidence list)
- per disease D: P(E=yes|D) estimated from KG = disease's weight for E's Q-CUI (normalized)

log P(D|all answers) ∝ log P(D) + Σ_E [if yes_E: log P(E=yes|D)/P(E=yes|~D); else: log P(E=no|D)/P(E=no|~D)]

This naturally handles BOTH positive AND negative evidence symmetrically,
unlike current model which treats negatives as soft penalty only.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
EVIDENCES_DEF = "data/ddxplus/release_evidences.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.7)
    ap.add_argument("--alpha_smooth", type=float, default=0.1,
                    help="Laplace smoothing for P(yes|D)")
    ap.add_argument("--positive_only", action="store_true", default=False,
                    help="Only use positive evidence (no negative terms)")
    ap.add_argument("--neg_weight", type=float, default=0.3,
                    help="Weight of negative evidence terms (0=disabled, 1=full Bayesian)")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))
    ev_def = json.load(open(EVIDENCES_DEF))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2name = {icd[dn]["cui"]: dn for dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    # Build d_q (disease -> {Q-CUI: weight}) with hop2 propagation
    d_q = {}
    for d in dcs_list:
        if d not in G: d_q[d] = {}; continue
        phen_w = {}
        for _, p, ed in G.out_edges(d, data=True):
            if ed.get("etype") == "HAS_PHENOTYPE":
                phen_w[p] = phen_w.get(p, 0) + ed.get("weight", 0)
        for p_direct in list(phen_w.keys()):
            dw = phen_w[p_direct]
            for _, p2, ed2 in G.out_edges(p_direct, data=True):
                if ed2.get("etype") == "HIERARCHY":
                    phen_w[p2] = phen_w.get(p2, 0) + args.hop2_decay * dw * ed2.get("weight", 0)
        d_q[d] = {p: w for p, w in phen_w.items() if p in Q}

    # Map each evidence name to its representative CUIs (from value_cuis)
    # B-type: use _question CUI
    # M-type (location/character): use _question CUI + value CUIs (compound)
    ev_to_cuis = {}  # ev_name → set of CUIs
    for ev_name, mapping in value_cuis.items():
        cuis = set()
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): cuis.update(v)
        if cuis:
            ev_to_cuis[ev_name] = cuis

    # All Binary evidences (B-type) — ev_def keys are already French names
    binary_evs = [ev for ev in ev_def if ev_def[ev].get('data_type') == 'B']
    bin_fr_names = set(ev for ev in binary_evs if ev in ev_to_cuis)
    print(f"Binary evidences with CUIs: {len(bin_fr_names)}/{len(binary_evs)}", flush=True)

    # For each binary evidence and disease, compute disease's max weight for any of E's CUIs
    # This serves as proxy for P(E=yes|D)
    disease_ev_weight = defaultdict(dict)  # (d, ev_name) → max weight
    for ev_fr in bin_fr_names:
        cuis = ev_to_cuis.get(ev_fr, set())
        cuis_in_q = cuis  # already in Q by construction
        if not cuis_in_q: continue
        for d in dcs_list:
            qp = d_q.get(d, {})
            w_max = max((qp.get(c, 0) for c in cuis_in_q), default=0)
            disease_ev_weight[d][ev_fr] = w_max

    # Normalize: P(E=yes|D) proxy = min(1, weight/max_weight) with smoothing
    # Better: use weight relative to total weight of disease
    # Or: convert weight to probability via softmax-like

    # Approach: P(E=yes|D) = (weight + smooth) / (sum_weight_D + smooth*N) capped at [smooth, 1-smooth]
    smooth = args.alpha_smooth
    # Compute per-disease sum of all evidence weights
    disease_total_w = {d: sum(disease_ev_weight[d].values()) for d in dcs_list}

    # Compute marginal P(E=yes) across diseases
    p_yes_marginal = {}
    for ev_fr in bin_fr_names:
        # Disease가 E를 가지면 1, 아니면 0으로 처리 (binary indicator)
        n_pos = sum(1 for d in dcs_list if disease_ev_weight[d].get(ev_fr, 0) > 0.1)
        p_yes_marginal[ev_fr] = (n_pos + smooth) / (len(dcs_list) + 2*smooth)

    # Compute P(E=yes|D) — continuous from weight, with smoothing
    # Use logarithmic normalization: p = sigmoid(log(1 + weight) - mean_log)
    import math as _m
    p_yes_given_d = defaultdict(dict)
    # Per-evidence: max weight across diseases for normalization
    ev_max_w = {ev: max((disease_ev_weight[d].get(ev, 0) for d in dcs_list), default=1.0) or 1.0
                for ev in bin_fr_names}
    for d in dcs_list:
        for ev_fr in bin_fr_names:
            w = disease_ev_weight[d].get(ev_fr, 0)
            # Normalized weight [0, 1]
            w_norm = w / ev_max_w[ev_fr] if ev_max_w[ev_fr] > 0 else 0
            # Map to probability with smoothing: [smooth, 1-smooth]
            p = smooth + w_norm * (1 - 2*smooth)
            p_yes_given_d[d][ev_fr] = p

    def get_patient_evidence_set(evs):
        """Return set of base evidence names that patient answered (Y for binary)."""
        positive = set()
        for ev in evs:
            if "_@_" in ev:
                base, _ = ev.split("_@_", 1)
                positive.add(base)
            else:
                positive.add(ev)
        return positive

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    log_prior = math.log(1.0 / len(dcs_list))
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            yes_evs = get_patient_evidence_set(evs)

            scores = {}
            for d in dcs_list:
                log_post = log_prior
                for ev_fr in bin_fr_names:
                    p_d = p_yes_given_d[d].get(ev_fr, smooth)
                    p_m = p_yes_marginal.get(ev_fr, smooth)
                    if ev_fr in yes_evs:
                        # log likelihood ratio for yes (full weight)
                        log_post += math.log(p_d) - math.log(p_m)
                    elif not args.positive_only and args.neg_weight > 0:
                        # log likelihood ratio for no (weighted by neg_weight)
                        log_post += args.neg_weight * (math.log(1 - p_d) - math.log(1 - p_m))
                scores[d] = log_post

            ranked = sorted(dcs_list, key=lambda d: -scores.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"\nv42 Bayesian (smooth={smooth}): @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
