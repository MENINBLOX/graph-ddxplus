#!/usr/bin/env python3
"""v53: Few-shot Naive Bayes — per-evidence likelihood ratio.

Architectural upgrade over v52 (mean-distance):
- For each evidence E, estimate P(E_value | D) from few-shot patients
- For each patient response, compute log P(D | answers) = log P(D) + Σ log P(Ei|D)
- Laplace smoothing for unseen (evidence, value) combinations

v52 only used aggregate stats (mean intensity, mean sudden). v53 uses FULL distribution
of each evidence answer across few-shot patients.
"""
from __future__ import annotations
import sys, json, csv, ast, math, pickle, argparse, random
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
PR_UNIVERSE = "pilot/data/pr_universe.json"
COMPOUND_PATH = "pilot/data/compound_pain_lookup_lt5.json"


def normalize_scores(d):
    vals = list(d.values())
    if not vals: return d
    lo, hi = min(vals), max(vals)
    if hi == lo: return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def build_nb_profile(N_per_disease, fr2cui, dcs_list, seed=42):
    """Sample N patients per disease. Build per-evidence value distribution P(E_val | D).

    Returns: dict[disease] = {evidence_base: Counter({value: count, ...}, total_count}}
    """
    random.seed(seed)
    disease_evs_str = defaultdict(list)
    with open('data/ddxplus/release_train_patients.csv') as f:
        for row in csv.DictReader(f):
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in fr2cui.values(): continue
            disease_evs_str[true_cui].append(row["EVIDENCES"])

    profile = {}
    for d in dcs_list:
        evs_list = disease_evs_str.get(d, [])
        if not evs_list:
            profile[d] = {'evs': defaultdict(Counter), 'patient_count': 0}
            continue
        sample = random.sample(evs_list, min(N_per_disease, len(evs_list)))
        ev_counters = defaultdict(Counter)  # base → Counter({value: count})
        n_patients = len(sample)
        for evs_str in sample:
            evs = ast.literal_eval(evs_str)
            answered_bases = set()
            for ev in evs:
                if '_@_' in ev:
                    base, val = ev.split('_@_', 1)
                    ev_counters[base][val] += 1
                    answered_bases.add(base)
                else:
                    ev_counters[ev]['__yes__'] += 1
                    answered_bases.add(ev)
        profile[d] = {'evs': dict(ev_counters), 'patient_count': n_patients}
    return profile


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
    ap.add_argument("--w_s1", type=float, default=0.3)
    ap.add_argument("--w_cov", type=float, default=0.1)
    ap.add_argument("--w_prcov", type=float, default=0.1)
    ap.add_argument("--w_compound", type=float, default=0.1)
    ap.add_argument("--w_nb", type=float, default=1.0,
                    help="Naive Bayes log-likelihood weight")
    ap.add_argument("--n_shot", type=int, default=50)
    ap.add_argument("--laplace", type=float, default=1.0, help="Laplace smoothing alpha")
    ap.add_argument("--seed", type=int, default=42)
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
    PR = set(json.load(open(PR_UNIVERSE))) if Path(PR_UNIVERSE).exists() else set()

    compound = defaultdict(set)
    raw = json.load(open(COMPOUND_PATH))
    for k, v_list in raw.items():
        q, v = k.split('|')
        compound[(q, v)].update(v_list)

    # Build few-shot NB profile
    nb_profile = build_nb_profile(args.n_shot, fr2cui, dcs_list, seed=args.seed)

    # Precompute log P(E_val | D) for each (disease, base, value)
    # Use Laplace smoothing: P(val|D) = (count + α) / (N_D + α * V)
    # For binary evidence (no value): P(yes|D) = (yes_count + α) / (N_D + 2α)
    # For multi-value: P(val|D) = (count + α) / (N_D + α * |vocab|)

    # Build vocabulary per evidence
    ev_vocab = defaultdict(set)
    for d, prof in nb_profile.items():
        for base, ctr in prof['evs'].items():
            ev_vocab[base].update(ctr.keys())

    # Pre-compute log P(val|D) and log P(notseen|D) for each base
    log_p_val_given_d = defaultdict(dict)  # base → dict[(d, val)] → log_prob
    log_p_notseen_given_d = defaultdict(dict)  # base → dict[d] → log_prob (for absent answers)

    EPS = 1e-9
    for base, vocab in ev_vocab.items():
        V = len(vocab)
        if V == 1 and '__yes__' in vocab:
            # Binary: yes vs no
            for d in dcs_list:
                prof = nb_profile.get(d, {})
                N_d = prof.get('patient_count', 0)
                yes_c = prof.get('evs', {}).get(base, Counter()).get('__yes__', 0)
                p_yes = max(EPS, min(1-EPS, (yes_c + args.laplace) / max(N_d + 2 * args.laplace, EPS)))
                p_no = 1 - p_yes
                log_p_val_given_d[base][(d, '__yes__')] = math.log(p_yes)
                log_p_notseen_given_d[base][d] = math.log(max(p_no, EPS))
        else:
            # Multi-value: each value vs not-answered
            for d in dcs_list:
                prof = nb_profile.get(d, {})
                N_d = prof.get('patient_count', 0)
                ev_ctr = prof.get('evs', {}).get(base, Counter())
                total_answered = sum(ev_ctr.values())
                for val in vocab:
                    c = ev_ctr.get(val, 0)
                    p = (c + args.laplace) / (total_answered + args.laplace * V) if total_answered > 0 else 1.0/V
                    p_answered = (total_answered + args.laplace) / (N_d + 2 * args.laplace) if N_d > 0 else 0.5
                    log_p_val_given_d[base][(d, val)] = math.log(max(p_answered * p, EPS))
                p_notanswered = (N_d - total_answered + args.laplace) / (N_d + 2 * args.laplace) if N_d > 0 else 0.5
                log_p_notseen_given_d[base][d] = math.log(max(p_notanswered, EPS))

    # Standard d_q for v41 baseline channels
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

    def parse_features(evs):
        pcuis = set(); compound_targets = set()
        patient_answers = {}  # base → val (or '__yes__' for binary)
        for ev in evs:
            if "_@_" in ev:
                base, val = ev.split("_@_", 1)
                m = value_cuis.get(base, {})
                q_cuis = m.get("_question", [])
                v_cuis = m.get(val, [])
                for q in q_cuis:
                    for v in v_cuis:
                        if (q, v) in compound: compound_targets.update(compound[(q, v)])
                pcuis.update(q_cuis); pcuis.update(v_cuis)
                # For multi-value: track ALL values answered (not just first)
                if base not in patient_answers:
                    patient_answers[base] = []
                patient_answers[base].append(val)
            else:
                m = value_cuis.get(ev, {})
                pcuis.update(m.get("_question", []))
                patient_answers[ev] = ['__yes__']
        return pcuis, compound_targets, patient_answers

    n = 0; c1=c3=c5=c10=0; rr_sum=0
    log_prior = math.log(1.0 / len(dcs_list))

    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs_list: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pcuis, compound_targets, patient_answers = parse_features(evs)
            identity_diseases = pcuis & dcs_set

            s1_scores = {}; cov_scores = {}; prcov_scores = {}; comp_scores = {}; nb_scores = {}
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
                if compound_targets and disease_full_phens[d]:
                    comp = sum(compound_idf.get(c, 0) for c in (compound_targets & disease_full_phens[d]))
                comp_scores[d] = comp

                # NAIVE BAYES log-likelihood
                log_post = log_prior
                for base in ev_vocab:
                    if base in patient_answers:
                        # Patient answered this evidence
                        for val in patient_answers[base]:
                            ll = log_p_val_given_d[base].get((d, val))
                            if ll is None:
                                # Unseen value in few-shot — use Laplace smoothing
                                V = len(ev_vocab[base])
                                ll = math.log(args.laplace / max(args.n_shot + args.laplace * V, 1))
                            log_post += ll
                    else:
                        # Patient did NOT answer this evidence base
                        log_post += log_p_notseen_given_d[base].get(d, math.log(0.5))
                nb_scores[d] = log_post

            s1_n = normalize_scores(s1_scores)
            cov_n = normalize_scores(cov_scores)
            prcov_n = normalize_scores(prcov_scores)
            comp_n = normalize_scores(comp_scores)
            nb_n = normalize_scores(nb_scores)

            final = {d: args.w_s1*s1_n[d] + args.w_cov*cov_n[d] + args.w_prcov*prcov_n[d] + args.w_compound*comp_n[d] + args.w_nb*nb_n[d] for d in dcs_list}
            ranked = sorted(dcs_list, key=lambda d: -final.get(d, -1e9))
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank

    print(f"v53 NB n_shot={args.n_shot:>5d} w_nb={args.w_nb}: @1={100*c1/n:.2f}% @3={100*c3/n:.2f}% @5={100*c5/n:.2f}% @10={100*c10/n:.2f}% MRR={rr_sum/n:.4f}")


if __name__ == "__main__":
    main()
