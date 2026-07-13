#!/usr/bin/env python3
"""v50: Structured feature representation + zero-shot disease profile.

Architectural overhaul: instead of bag-of-CUIs, represent both patient and
disease as structured features:

Patient = {
  "pain": {"present": bool, "intensity": int 0-10, "sudden": int 0-10, "char": set, ...},
  "general": {"fever": bool, "chills": bool, ...},
  ...
}

Disease profile (zero-shot from KG):
  "intensity_high": count of severe-pain CUIs in disease (C0238995 Sharp etc.)
  "sudden_high": count of acute-onset markers (acute name + sudden-related CUIs)
  "chronic": count of chronic-onset markers

Distance = weighted dimension-wise difference.

Combined with v41 baseline (standard CUI matching) as additional channels.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse, re
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
COMPOUND_PATH = "pilot/data/compound_pain_lookup_lt5.json"

# Disease profile dimensions — CUIs that mark each dimension
DIM_MARKERS = {
    "severe_pain": {'C0238995', 'C0008033', 'C0151826', 'C0235710'},  # Sharp/Pleuritic/Retrosternal/Pleuritic
    "radiating_pain": {'C0234254', 'C2318664'},  # Radiating, Radiating to left arm
    "chronic_features": set(),  # Will populate based on disease NAME
    "acute_features": set(),  # Same
    "infectious_features": {'C0015967', 'C0011991', 'C0027497', 'C0009676'},  # Fever, diarrhea, nausea, chills
}


def normalize_scores(d):
    vals = list(d.values())
    if not vals: return d
    lo, hi = min(vals), max(vals)
    if hi == lo: return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def parse_patient_features(evs, value_cuis):
    """Parse EVIDENCES into structured features."""
    features = {
        "pain_present": False,
        "pain_intensity": 0,
        "pain_sudden": 0,
        "pain_precision": 0,
        "pain_radiation": False,
        "pain_characters": set(),  # FR strings
        "pain_locations": set(),  # FR strings
        "skin_present": False,
        "fever": False,
        "cough": False,
        "dyspnea": False,
        "edema_present": False,
        "pcuis": set(),
        "compound_targets": set(),
    }
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            m = value_cuis.get(base, {})
            q_cuis = m.get("_question", [])
            v_cuis = m.get(val, [])
            features["pcuis"].update(q_cuis)
            features["pcuis"].update(v_cuis)
            # Structured extraction
            if base == 'douleurxx_intens':
                try: features["pain_intensity"] = int(val); features["pain_present"] = True
                except: pass
            elif base == 'douleurxx_soudain':
                try: features["pain_sudden"] = int(val); features["pain_present"] = True
                except: pass
            elif base == 'douleurxx_precis':
                try: features["pain_precision"] = int(val)
                except: pass
            elif base == 'douleurxx_carac':
                features["pain_characters"].add(val)
                features["pain_present"] = True
            elif base == 'douleurxx_endroitducorps':
                features["pain_locations"].add(val)
                features["pain_present"] = True
            elif base == 'douleurxx_irrad' and val != 'nulle_part':
                features["pain_radiation"] = True
            elif base == 'lesions_peau_endroitducorps':
                features["skin_present"] = True
            elif base == 'oedeme_endroitducorps' and val != 'nulle_part':
                features["edema_present"] = True
        else:
            m = value_cuis.get(ev, {})
            features["pcuis"].update(m.get("_question", []))
            if ev == 'douleurxx': features["pain_present"] = True
            elif ev == 'lesions_peau': features["skin_present"] = True
            elif ev == 'fievre': features["fever"] = True
            elif ev == 'toux': features["cough"] = True
            elif ev == 'dyspn': features["dyspnea"] = True
            elif ev == 'oedeme': features["edema_present"] = True
    return features


def build_disease_profile(d_cui, d_full_phens, d_name):
    """Zero-shot disease profile from KG signature + disease name."""
    profile = {
        "severe_pain": sum(1 for c in DIM_MARKERS["severe_pain"] if c in d_full_phens),
        "radiating_pain": sum(1 for c in DIM_MARKERS["radiating_pain"] if c in d_full_phens),
        "is_acute": int("acute" in d_name.lower() or "spontaneous" in d_name.lower()),
        "is_chronic": int("chronic" in d_name.lower()),
        "is_emergency": int(any(kw in d_name.lower() for kw in ["embolism", "pneumothorax", "anaphylaxis", "stemi", "anaphyl"])),
        "infectious": sum(1 for c in DIM_MARKERS["infectious_features"] if c in d_full_phens),
    }
    return profile


def feature_match_score(patient, disease_profile):
    """Compute dimension-wise match score between patient features and disease profile."""
    score = 0
    # Intensity match: high-intensity patient → high-severity disease
    if patient["pain_intensity"] > 0:
        # Severe pain match
        intensity_score = (patient["pain_intensity"] / 10.0) * disease_profile["severe_pain"]
        score += intensity_score * 0.5
    # Sudden onset match: high-sudden patient → acute/emergency disease
    if patient["pain_sudden"] > 0:
        sudden_score = (patient["pain_sudden"] / 10.0) * (disease_profile["is_acute"] + disease_profile["is_emergency"])
        score += sudden_score * 0.5
        # Chronic mismatch: high-sudden patient → NOT chronic disease
        score -= (patient["pain_sudden"] / 10.0) * disease_profile["is_chronic"] * 0.5
    # Radiation match
    if patient["pain_radiation"]:
        score += disease_profile["radiating_pain"] * 0.5
    elif patient["pain_present"]:  # no radiation but pain
        score -= disease_profile["radiating_pain"] * 0.3
    # Infectious dimension
    if patient["fever"] and patient["cough"]:
        score += disease_profile["infectious"] * 0.3
    return score


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
    ap.add_argument("--w_s1", type=float, default=0.7)
    ap.add_argument("--w_cov", type=float, default=0.1)
    ap.add_argument("--w_prcov", type=float, default=0.1)
    ap.add_argument("--w_compound", type=float, default=0.1)
    ap.add_argument("--w_struct", type=float, default=0.2)
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    value_cuis = json.load(open(VALUE_CUIS))

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd[dn]["cui"] for dn,info in cond.items() if dn in icd}
    cui2name = {info['cui']: dn for dn, info in icd.items() if 'cui' in info}
    dcs_list = sorted(set(fr2cui.values()))
    dcs_set = set(dcs_list)

    Q = set()
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)
    PR = set(json.load(open(PR_UNIVERSE))) if Path(PR_UNIVERSE).exists() else set()

    compound = defaultdict(set)
    raw = json.load(open(COMPOUND_PATH))
    for k, v_list in raw.items():
        q, v = k.split('|')
        compound[(q, v)].update(v_list)

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

    phen_freq = Counter()
    for d, qp in d_q.items():
        for p in qp: phen_freq[p] += 1
    N_D = len(dcs_list)
    idf = {p: math.log(N_D / max(c, 1)) ** args.idf_pow for p, c in phen_freq.items()}
    d_q_idf = {d: {p: w * idf.get(p, 1.0) for p, w in qp.items()} for d, qp in d_q.items()}
    d_core = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.core_k]) for d, qp in d_q_idf.items()}
    d_sig = {d: set(sorted(qp.keys(), key=lambda p: -qp[p])[:args.sig_k]) for d, qp in d_q_idf.items()}

    disease_full_phens = {d: {p for _, p, ed in G.out_edges(d, data=True) if ed.get("etype")=="HAS_PHENOTYPE"} if d in G else set() for d in dcs_list}
    compound_cuis_all = set()
    for cuis in compound.values(): compound_cuis_all.update(cuis)
    compound_doc_freq = {c: sum(1 for p in disease_full_phens.values() if c in p) for c in compound_cuis_all}
    compound_idf = {c: math.log(49 / max(compound_doc_freq.get(c, 1), 1)) for c in compound_cuis_all}

    # Build disease profiles
    disease_profiles = {d: build_disease_profile(d, disease_full_phens[d], cui2name.get(d, '?')) for d in dcs_list}

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            features = parse_patient_features(evs, value_cuis)
            pcuis = features["pcuis"]
            identity_diseases = pcuis & dcs_set

            s1_scores = {}; cov_scores = {}; prcov_scores = {}; comp_scores = {}; struct_scores = {}
            for d in dcs_list:
                qp = d_q_idf.get(d, {})
                pos = sum(w for q, w in qp.items() if q in pcuis)
                core = d_core.get(d, set())
                neg = sum(qp.get(c, 0) for c in core if c not in pcuis)
                s1 = pos - args.alpha * neg
                total = sum(qp.values()) if qp else 1
                s1 = s1 / (math.sqrt(total) or 1)
                sig = d_sig.get(d, set())
                if sig:
                    s1 += args.sig_w * (sum(1 for p in sig if p in pcuis) / len(sig))
                if d in identity_diseases:
                    s1 += args.identity_boost
                s1_scores[d] = s1

                cov_scores[d] = sum(1 for p in pcuis if p in qp) / max(len(pcuis), 1) if pcuis and qp else 0
                if PR and pcuis and qp:
                    pr_pcuis = pcuis & PR
                    pr_qp = {p: w for p, w in qp.items() if p in PR}
                    prcov_scores[d] = sum(1 for p in pr_pcuis if p in pr_qp) / max(len(pr_pcuis), 1) if (pr_pcuis and pr_qp) else 0
                else:
                    prcov_scores[d] = 0

                comp = 0
                if features["compound_targets"] and disease_full_phens[d]:
                    comp = sum(compound_idf.get(c, 0) for c in (features["compound_targets"] & disease_full_phens[d]))
                comp_scores[d] = comp

                # STRUCTURED: dimension-wise match
                struct_scores[d] = feature_match_score(features, disease_profiles[d])

            s1_n = normalize_scores(s1_scores)
            cov_n = normalize_scores(cov_scores)
            prcov_n = normalize_scores(prcov_scores)
            comp_n = normalize_scores(comp_scores)
            struct_n = normalize_scores(struct_scores)

            final = {d: args.w_s1*s1_n[d] + args.w_cov*cov_n[d] + args.w_prcov*prcov_n[d] + args.w_compound*comp_n[d] + args.w_struct*struct_n[d] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"v50 [w_struct={args.w_struct}]: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
