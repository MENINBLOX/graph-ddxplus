#!/usr/bin/env python3
"""v43: Dimensional scoring — separate channels by CUI semantic type.

Architecture change: instead of treating all Q-CUIs as one bag, split by semantic dimension:
- D1 Symptom: T184 (Sign/Symptom), T033 (Finding patient-reportable)
- D2 Anatomy: T029 (Body Location), T023 (Body Part), T030 (Body Space), T024 (Tissue)
- D3 History: T047 (Disease), T046 (PathFunction), T048 (MentalDisorder)
- D4 Function: T039 (Physiologic Function)
- D5 Other: everything else

Each dimension scored independently → weighted sum.

Hypothesis: different dimensions carry different discriminative value.
Patient with (symptom: cough) + (anatomy: chest) + (history: smoking) gives
different signal weight per disease compared to bag-of-CUIs sum.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"

DIMENSIONS = {
    "symptom": {"T184", "T033"},  # Sign/Symptom + Finding
    "anatomy": {"T029", "T023", "T030", "T024", "T031", "T022", "T025"},  # Body
    "history": {"T047", "T046", "T048", "T037", "T191", "T190", "T020", "T019"},  # Disease + Mental + Path
    "function": {"T039", "T032", "T201", "T034"},  # Physio + Attribute
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--hop2_decay", type=float, default=0.7)
    ap.add_argument("--idf_pow", type=float, default=0.5)
    ap.add_argument("--core_k", type=int, default=35)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--identity_boost", type=float, default=1.5)
    ap.add_argument("--sig_k", type=int, default=10)
    ap.add_argument("--sig_w", type=float, default=9.0)
    # Dimension weights
    ap.add_argument("--w_symptom", type=float, default=1.0)
    ap.add_argument("--w_anatomy", type=float, default=0.5)
    ap.add_argument("--w_history", type=float, default=1.5,
                    help="Medical history is highly discriminative")
    ap.add_argument("--w_function", type=float, default=0.5)
    ap.add_argument("--w_other", type=float, default=0.3)
    ap.add_argument("--normalize", action="store_true")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    # Load MRSTY for TUI assignment to Q CUIs
    cui_tuis = defaultdict(set)
    with open('data/umls_extracted/MRSTY.RRF') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) < 4: continue
            if parts[0] in Q:
                cui_tuis[parts[0]].add(parts[1])

    # Assign each Q-CUI to a dimension (priority: symptom > history > anatomy > function > other)
    cui_dim = {}
    for c, tuis in cui_tuis.items():
        if tuis & DIMENSIONS["symptom"]:
            cui_dim[c] = "symptom"
        elif tuis & DIMENSIONS["history"]:
            cui_dim[c] = "history"
        elif tuis & DIMENSIONS["anatomy"]:
            cui_dim[c] = "anatomy"
        elif tuis & DIMENSIONS["function"]:
            cui_dim[c] = "function"
        else:
            cui_dim[c] = "other"
    for c in Q:
        if c not in cui_dim: cui_dim[c] = "other"

    dim_count = Counter(cui_dim.values())
    print(f"Q CUIs by dimension: {dict(dim_count)}", flush=True)

    # Build d_q (Q-restricted disease phens with hop2)
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

    # Per-dimension IDF
    dim_phen_freq = defaultdict(Counter)
    for d, qp in d_q.items():
        for p in qp:
            dim_phen_freq[cui_dim.get(p, "other")][p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(dim_phen_freq[cui_dim.get(p, "other")][p], 1)) ** args.idf_pow for p in Q}

    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}
    # Per-dimension top-K core for negative
    d_core_dim = defaultdict(dict)
    for d, qp in d_q_idf.items():
        for dim in ["symptom","anatomy","history","function","other"]:
            dim_qp = {p: w for p, w in qp.items() if cui_dim.get(p) == dim}
            d_core_dim[dim][d] = set(sorted(dim_qp.keys(), key=lambda p: -dim_qp[p])[:args.core_k])
    # Per-dimension signature top-sig_k
    d_sig_dim = defaultdict(dict)
    for d, qp in d_q_idf.items():
        for dim in ["symptom","anatomy","history","function","other"]:
            dim_qp = {p: w for p, w in qp.items() if cui_dim.get(p) == dim}
            d_sig_dim[dim][d] = set(sorted(dim_qp.keys(), key=lambda p: -dim_qp[p])[:args.sig_k])

    def get_pcuis(evs):
        cuis = set()
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                cuis.update(m.get("_question", []))
                cuis.update(m.get(val, []))
            else:
                m = value_cuis.get(ev, {})
                cuis.update(m.get("_question", []))
        return cuis

    DIM_WEIGHTS = {
        "symptom": args.w_symptom,
        "anatomy": args.w_anatomy,
        "history": args.w_history,
        "function": args.w_function,
        "other": args.w_other,
    }

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis = get_pcuis(evs)
            identity_diseases = pcuis & dcs_set

            # Per-dimension pcuis
            pcuis_by_dim = defaultdict(set)
            for c in pcuis:
                pcuis_by_dim[cui_dim.get(c, "other")].add(c)

            final = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                total_score = 0
                for dim, pset in pcuis_by_dim.items():
                    # Pos: matched phens in this dimension
                    pos = sum(qp.get(p, 0) for p in pset if cui_dim.get(p) == dim)
                    # Neg: core_k missing in this dimension
                    core = d_core_dim[dim].get(d, set())
                    neg = sum(qp.get(c, 0) for c in core if c not in pset)
                    # Per-dim normalization
                    dim_total = sum(w for p, w in qp.items() if cui_dim.get(p) == dim)
                    s = pos - args.alpha * neg
                    s = s / (math.sqrt(dim_total) if dim_total > 0 else 1)
                    # Per-dim signature match
                    sig = d_sig_dim[dim].get(d, set())
                    if sig:
                        s += args.sig_w * (sum(1 for p in sig if p in pset) / len(sig)) / 4  # divided 4 because 4-5 dims
                    total_score += DIM_WEIGHTS.get(dim, 1.0) * s

                # Identity boost (universal)
                if d in identity_diseases:
                    total_score += args.identity_boost

                final[d] = total_score

            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    weights_str = f"sym={args.w_symptom},anat={args.w_anatomy},hist={args.w_history},fn={args.w_function},oth={args.w_other}"
    print(f"v43 dim [{weights_str}]: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
