#!/usr/bin/env python3
"""SymCat NB evaluation — transfer test of v53 architecture.

SymCat structure:
- 50 diseases × N_d symptoms each, with probability P(symptom|disease) %

NB architecture:
- Profile: P(symptom_i | D) from SymCat directly (already supervised-equivalent)
- Patient: simulated from disease's symptom probabilities (binomial sampling)
- Scoring: log P(D|patient) = log P(D) + Σ_i log P(sym_i|D)
  for present symptoms, plus log(1 - P(sym_i|D)) for absent.

Test variants:
1. Self-evaluation: simulate from SAME distribution used for profile (sanity)
2. Held-out: split each disease's symptoms 70/30, profile from 70%, test on 30%
3. Noise-perturbed: add Gaussian noise to probabilities then test
"""
from __future__ import annotations
import json, random, math
from collections import defaultdict


def load_symcat():
    d = json.load(open('data/symcat/symcat_parsed.json'))
    pairs = d['disease_symptom_pairs']
    # Convert to: disease -> {symptom: prob (0-1)}
    disease_profile = {}
    for dname, sym_list in pairs.items():
        disease_profile[dname] = {s[0]: s[1]/100.0 for s in sym_list}
    return disease_profile


def simulate_patient(disease, profile, seed=None):
    """Sample patient symptom set from disease's symptom probabilities."""
    if seed is not None: random.seed(seed)
    sym_prob = profile[disease]
    return {s for s, p in sym_prob.items() if random.random() < p}


def nb_score(patient, profile, all_symptoms, log_prior, smooth=0.01):
    """Compute log P(D|patient) for each disease."""
    scores = {}
    for d, sym_prob in profile.items():
        log_p = log_prior
        for s in all_symptoms:
            p = sym_prob.get(s, 0)
            p = max(smooth, min(1-smooth, p))
            if s in patient:
                log_p += math.log(p)
            else:
                log_p += math.log(1 - p)
        scores[d] = log_p
    return scores


def eval_patients(profile_train, profile_test, all_symptoms, n_patients_per_d=100, seed=42):
    """Generate patients from profile_test, score against profile_train."""
    random.seed(seed)
    diseases = sorted(profile_test.keys())
    log_prior = math.log(1.0 / len(diseases))
    c1 = c3 = c5 = c10 = 0; n = 0; rr_sum = 0.0
    for d_true in diseases:
        for _ in range(n_patients_per_d):
            patient = {s for s, p in profile_test[d_true].items() if random.random() < p}
            if not patient: continue
            scores = nb_score(patient, profile_train, all_symptoms, log_prior)
            ranked = sorted(scores.keys(), key=lambda x: -scores[x])
            n += 1
            try: rank = ranked.index(d_true) + 1
            except: rank = len(diseases)
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr_sum += 1.0/rank
    return {
        '@1': 100*c1/n, '@3': 100*c3/n, '@5': 100*c5/n, '@10': 100*c10/n,
        'MRR': rr_sum/n, 'n': n
    }


def main():
    profile = load_symcat()
    all_syms = sorted(set(s for d in profile for s in profile[d]))
    print(f"=== SymCat: {len(profile)} diseases, {len(all_syms)} symptoms ===\n")

    # Disease size stats
    sizes = [len(profile[d]) for d in profile]
    print(f"Symptoms per disease: avg={sum(sizes)/len(sizes):.1f}, min={min(sizes)}, max={max(sizes)}")

    # ============ Test 1: Self-evaluation (sanity check) ============
    print("\n=== Test 1: SELF-EVAL (profile = test, sanity check) ===")
    res = eval_patients(profile, profile, all_syms, n_patients_per_d=100, seed=42)
    print(f"  @1={res['@1']:.2f}% @3={res['@3']:.2f}% @5={res['@5']:.2f}% @10={res['@10']:.2f}% MRR={res['MRR']:.4f}")

    # ============ Test 2: Held-out split (70/30) per disease ============
    # Split each disease's symptom list 70/30
    random.seed(42)
    profile_train_70 = {}
    profile_test_30 = {}
    for d, sym_prob in profile.items():
        syms = list(sym_prob.items())
        random.shuffle(syms)
        n_train = int(len(syms) * 0.7)
        train = dict(syms[:n_train])
        test = dict(syms[n_train:])
        if not train: train = sym_prob  # fallback
        if not test: test = sym_prob
        profile_train_70[d] = train
        profile_test_30[d] = test

    print("\n=== Test 2: HELD-OUT (train 70% symptoms, test from 30%) ===")
    res = eval_patients(profile_train_70, profile_test_30, all_syms, n_patients_per_d=100, seed=42)
    print(f"  @1={res['@1']:.2f}% @3={res['@3']:.2f}% @5={res['@5']:.2f}% @10={res['@10']:.2f}% MRR={res['MRR']:.4f}")

    # ============ Test 3: Noise perturbed (add 30% prob noise) ============
    random.seed(42)
    profile_noisy = {}
    for d, sym_prob in profile.items():
        noisy = {}
        for s, p in sym_prob.items():
            noise = random.gauss(0, 0.15)
            np_ = max(0.05, min(0.95, p + noise))
            noisy[s] = np_
        profile_noisy[d] = noisy

    print("\n=== Test 3: NOISY profile (Gaussian σ=0.15 on each P) ===")
    res = eval_patients(profile_noisy, profile, all_syms, n_patients_per_d=100, seed=42)
    print(f"  @1={res['@1']:.2f}% @3={res['@3']:.2f}% @5={res['@5']:.2f}% @10={res['@10']:.2f}% MRR={res['MRR']:.4f}")

    # ============ Test 4: Few-shot equivalent ============
    # Simulate N patients per disease from FULL profile, then estimate profile from those samples
    print("\n=== Test 4: FEW-SHOT (estimate profile from N simulated patients) ===")
    for n_shot in [5, 10, 50, 100, 500, 1000]:
        random.seed(42)
        # Generate n_shot patients per disease using true profile
        fs_profile = {}
        for d in profile:
            counts = defaultdict(int)
            for _ in range(n_shot):
                p = {s for s, prob in profile[d].items() if random.random() < prob}
                for s in p:
                    counts[s] += 1
            # Estimate P(symptom|disease) with Laplace smoothing
            fs_profile[d] = {s: (counts[s] + 1) / (n_shot + 2) for s in all_syms if counts[s] > 0}

        res = eval_patients(fs_profile, profile, all_syms, n_patients_per_d=100, seed=43)
        print(f"  n_shot={n_shot:>4d}: @1={res['@1']:.2f}% @3={res['@3']:.2f}% @5={res['@5']:.2f}% @10={res['@10']:.2f}% MRR={res['MRR']:.4f}")


if __name__ == "__main__":
    main()
