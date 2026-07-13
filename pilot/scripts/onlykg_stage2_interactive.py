#!/usr/bin/env python3
"""only-KG Phase 3: Interactive Q&A simulation with information gain.

Pipeline (per patient):
  1. Initial chief complaint → seed CUI(s)
  2. Stage 1 candidates: graph-score over disease nodes
  3. Loop:
     a. Compute info gain for each candidate phenotype (across DDXPlus questions)
     b. Pick max-gain question
     c. Simulate patient answer (Y/N from DDXPlus evidence dict)
     d. Update candidate posterior (Bayesian)
     e. Stop if: top-1 confidence ≥ THRESH OR question_count ≥ MAX_Q
  4. Final pick: argmax posterior

Cypher-equivalent for Stage 1 + posterior update:
  MATCH (d:Disease)-[r:HAS_PHENOTYPE]->(p:Phenotype)
  WHERE p.cui IN $patient_cuis_so_far
  WITH d, SUM(r.weight) AS log_lik
  RETURN d ORDER BY log_lik + log(prior(d)) DESC LIMIT $top_k

Patient simulator: for DDXPlus, patient HAS evidence_X iff evidence_X in patient.evidences.
Question candidate = any DDXPlus evidence (223 total) not yet asked.
Match question CUI(s) with KG phenotype CUIs to compute candidate-phenotype relevance.
"""
from __future__ import annotations
import sys, json, csv, ast, math, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT

import networkx as nx
import numpy as np

GRAPH = MEDKG_ROOT / "kg" / "onlykg_graph.pkl"
EVIDENCE_CUI = MEDKG_ROOT / "kg" / "ddxplus_evidence_cuis.json"
DISEASE_PRIOR = MEDKG_ROOT / "kg" / "disease_prior_by_cui.json"


def load_data():
    print("Loading graph...")
    with GRAPH.open("rb") as f:
        G = pickle.load(f)
    ev_cuis = json.load(open(EVIDENCE_CUI))
    prior = json.load(open(DISEASE_PRIOR)) if DISEASE_PRIOR.exists() else {}
    return G, ev_cuis, prior


def graph_score(G, patient_cuis: set, candidate: str, variant: str = "B") -> float:
    """Sum weighted edges from candidate to phenotypes in patient_cuis."""
    if candidate not in G: return 0.0
    s = 0.0
    for _, p, edata in G.out_edges(candidate, data=True):
        if p in patient_cuis:
            s += edata.get("weight", 0.0)
    if variant == "B":
        deg = G.out_degree(candidate) or 1
        s = s / math.sqrt(deg)
    return s


def info_gain(G, candidates: list[str], cand_posteriors: dict, question_cuis: set) -> float:
    """Estimate how much the question would split current candidate distribution.

    For each candidate, P(answer=Yes|D) = (any of question_cuis appears in D's phenotypes).
    More specifically: weighted overlap mass / total mass.
    Information gain = H(D) - H(D | answer).
    """
    if not question_cuis or not candidates:
        return 0.0
    # Compute P(Yes | D) for each candidate
    p_yes = {}
    for d in candidates:
        if d not in G:
            p_yes[d] = 0.5; continue
        total_w = 0.0; match_w = 0.0
        for _, p, edata in G.out_edges(d, data=True):
            w = edata.get("weight", 0.0)
            total_w += w
            if p in question_cuis:
                match_w += w
        p_yes[d] = match_w / total_w if total_w > 0 else 0.5
    # Current entropy
    posts = np.array([cand_posteriors.get(d, 0.0) for d in candidates])
    posts = posts / max(posts.sum(), 1e-9)
    H = -(posts * np.log2(posts + 1e-9)).sum()
    # Expected posterior entropy after answer
    p_yes_overall = sum(cand_posteriors.get(d, 0) * p_yes[d] for d in candidates)
    p_yes_overall = max(min(p_yes_overall, 1-1e-9), 1e-9)
    # Posterior given Yes
    post_yes = np.array([cand_posteriors.get(d, 0) * p_yes[d] for d in candidates])
    post_yes_sum = post_yes.sum()
    if post_yes_sum > 0:
        post_yes = post_yes / post_yes_sum
        H_yes = -(post_yes * np.log2(post_yes + 1e-9)).sum()
    else:
        H_yes = H
    # Posterior given No
    post_no = np.array([cand_posteriors.get(d, 0) * (1 - p_yes[d]) for d in candidates])
    post_no_sum = post_no.sum()
    if post_no_sum > 0:
        post_no = post_no / post_no_sum
        H_no = -(post_no * np.log2(post_no + 1e-9)).sum()
    else:
        H_no = H
    H_after = p_yes_overall * H_yes + (1 - p_yes_overall) * H_no
    return H - H_after


def update_posterior(G, candidates: list[str], cand_posteriors: dict,
                     question_cuis: set, answer_yes: bool) -> dict:
    """Bayesian update of candidate posteriors given Y/N answer."""
    new_post = {}
    for d in candidates:
        if d not in G:
            p_yes_d = 0.5
        else:
            total_w = 0.0; match_w = 0.0
            for _, p, edata in G.out_edges(d, data=True):
                w = edata.get("weight", 0.0)
                total_w += w
                if p in question_cuis:
                    match_w += w
            p_yes_d = match_w / total_w if total_w > 0 else 0.5
            # smooth to avoid 0/1
            p_yes_d = 0.05 + 0.9 * p_yes_d
        prob = p_yes_d if answer_yes else (1 - p_yes_d)
        new_post[d] = cand_posteriors.get(d, 0.0) * prob
    # Normalize
    total = sum(new_post.values())
    if total > 0:
        for d in new_post: new_post[d] /= total
    return new_post


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--max_q", type=int, default=15, help="max interactive questions")
    ap.add_argument("--conf_thresh", type=float, default=0.7, help="stop if top-1 posterior >=")
    ap.add_argument("--variant", default="B")
    args = ap.parse_args()

    G, ev_cuis, prior = load_data()
    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # DDXPlus mapping
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_info = json.load(f)
    fr2cui = {info.get("cond-name-fr",""): icd_map[dn]["cui"]
              for dn,info in cond.items() if dn in icd_map}
    cui2name = {icd_map[dn]["cui"]: dn for dn in icd_map}
    dcs = sorted(set(fr2cui.values()))
    # Disease prior by CUI; if missing assume uniform
    cui_prior = {}
    for fr_name, cui in fr2cui.items():
        cui_prior[cui] = prior.get(fr_name, 1.0/len(dcs))
    # Normalize prior to candidates
    total_pr = sum(cui_prior.values())
    if total_pr > 0:
        for c in cui_prior: cui_prior[c] /= total_pr

    # All DDXPlus questions (= 223 evidences, but exclude pain detail sub-questions clobber)
    all_questions = list(ev_cuis.keys())  # 223
    print(f"Candidate DDXPlus questions: {len(all_questions)}")

    # Eval
    print(f"\nEvaluating interactive Q&A on {args.n} patients (max_q={args.max_q}, conf_thresh={args.conf_thresh})...")
    t0 = time.time()
    n = 0; c1=c3=c5=c10=0; rr_sum=0
    n_questions_used = []
    n_correct = 0
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if n >= args.n: break
            true_cui = fr2cui.get(row["PATHOLOGY"])
            if true_cui not in dcs: continue
            patient_evs = set(ev.split("_@_")[0] for ev in ast.literal_eval(row["EVIDENCES"]))
            # Initial: chief complaint (initial_evidence)
            initial_ev = row.get("INITIAL_EVIDENCE", "")
            asked = set()
            patient_cuis_so_far = set(ev_cuis.get(initial_ev, []))
            asked.add(initial_ev)
            # Initialize posteriors with prior + Stage 1 graph score
            posteriors = {}
            for d in dcs:
                gs = graph_score(G, patient_cuis_so_far, d, variant=args.variant)
                # Log-space: prior * exp(gs)
                posteriors[d] = cui_prior.get(d, 1/len(dcs)) * math.exp(min(gs, 50))
            total_p = sum(posteriors.values())
            if total_p > 0:
                for d in posteriors: posteriors[d] /= total_p
            # Interactive loop
            q_count = 1  # initial counted
            for _ in range(args.max_q - 1):
                # Pick top-1 candidate confidence
                top_d = max(posteriors, key=posteriors.get)
                top_conf = posteriors[top_d]
                if top_conf >= args.conf_thresh: break
                # Compute info gain for each unasked question
                best_q = None; best_ig = -1
                # Only consider top-30 candidates for speed
                top_cands = sorted(posteriors, key=posteriors.get, reverse=True)[:30]
                for q in all_questions:
                    if q in asked: continue
                    q_cuis = set(ev_cuis.get(q, []))
                    if not q_cuis: continue
                    ig = info_gain(G, top_cands, posteriors, q_cuis)
                    if ig > best_ig:
                        best_ig = ig; best_q = q
                if best_q is None or best_ig <= 0: break
                # Simulate answer
                ans_yes = (best_q in patient_evs)
                q_cuis = set(ev_cuis.get(best_q, []))
                posteriors = update_posterior(G, list(posteriors.keys()), posteriors, q_cuis, ans_yes)
                asked.add(best_q)
                if ans_yes:
                    patient_cuis_so_far.update(q_cuis)
                q_count += 1
            # Final ranking
            ranked = sorted(posteriors, key=posteriors.get, reverse=True)
            try: rank = ranked.index(true_cui) + 1
            except: rank = 50
            n += 1
            if rank == 1: c1 += 1; n_correct += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank
            n_questions_used.append(q_count)
            if n % 500 == 0:
                avg_q = sum(n_questions_used)/len(n_questions_used)
                print(f"  {n}/{args.n} @1={100*c1/n:.2f}% @5={100*c5/n:.2f}% MRR={rr_sum/n:.3f}  avg_q={avg_q:.1f} ({time.time()-t0:.0f}s)")

    print(f"\n=== only-KG Interactive Q&A Results ===")
    print(f"  GTPA@1  = {100*c1/n:.2f}%")
    print(f"  GTPA@3  = {100*c3/n:.2f}%")
    print(f"  GTPA@5  = {100*c5/n:.2f}%")
    print(f"  GTPA@10 = {100*c10/n:.2f}%")
    print(f"  MRR     = {rr_sum/n:.4f}")
    print(f"  Avg questions used: {sum(n_questions_used)/len(n_questions_used):.1f}")
    print(f"  n       = {n}")


if __name__ == "__main__":
    main()
