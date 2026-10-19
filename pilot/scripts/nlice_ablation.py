"""Attribute leave-one-out ablation with a permutation-null significance test.

Question this answers: do the normalized Bates attributes on our KG edges carry REAL
diagnostic signal, or would any apparent gain be indistinguishable from chance?

Design (randomization inference / permutation feature importance, Breiman 2001):
  * Candidate set = closed NLICE disease set (SymCat-grounded, in our KG).
  * Base scorer   = IDF-weighted phenotype-CUI overlap (the attribute-blind onlykg model).
  * +attr scorer  = base + LAMBDA * w  whenever a query finding's attribute value equals the
                    KG edge's attribute value for the same (disease, phenotype). LAMBDA fixed=1.0
                    (single global default; no per-benchmark tuning -- CLAUDE.md §7).
  * Observed delta = GTPA@1(+attr) - GTPA@1(base), same for MRR.
  * Permutation null: shuffle the attribute's values across the edges that carry it (marginal
    preserved, disease<->attribute association destroyed) -> recompute delta. K reps -> null
    distribution. Empirical p = P(null delta >= observed). z = (obs - mean_null)/sd_null.
  * Bootstrap 95% CI on the observed delta by resampling patients.

A channel is credited only if the observed delta clears its own permutation null (p<0.05)
AND the CI excludes 0. This converts the 4 gold-less attributes from "unverified" into
channels that are either causally supported or honestly reported as null.
"""
from __future__ import annotations
import os, json, math, argparse, random, sys
from collections import defaultdict, Counter
sys.path.insert(0, "pilot/scripts")
import attr_score as A

LAMBDA = 1.0          # attribute-match bonus, fixed global default
SEED = 42
NORM_EXP = float(os.environ.get("KG_NORM_EXP", "0.5"))


def load_kg(path: str):
    kg = json.load(open(path))
    N = len(kg["diseases"])
    df = Counter()
    for e in kg["edges"]:
        df[e["p"]] += 1
    idf = {p: math.log(N / c) for p, c in df.items()}
    # selectivity weights for attribute agreements, so the ablation credits a channel by the same
    # rule the production scorer uses (a value most candidates share separates nothing)
    A.build_attr_df(kg["edges"])
    # prof[d][p] = {"w":idf, "location_cui":.., "character_cui":.., "character":.., "onset":..}
    prof: dict[str, dict[str, dict]] = defaultdict(dict)
    for e in kg["edges"]:
        d, p, at = e["d"], e["p"], e.get("attrs", {})
        cell = {"w": idf.get(p, 0.0)}
        # Load ALL six candidate attributes. Only location/character/onset used to be read here, so
        # severity, timing, aggravating and relieving could not be ablated at all -- three of the
        # six were structurally untestable, which is exactly the primary metric of the thesis.
        if at.get("location_cui"):
            cell["location_cui"] = at["location_cui"]
        if at.get("location"):
            cell["location_txt"] = str(at["location"]).lower()
        for k in ("character", "onset", "severity", "timing", "aggravating", "relieving"):
            if at.get(k):
                cell[k] = str(at[k]).lower()
        prof[d][p] = cell
    return kg, prof, idf


def score_patient(findings, cand, prof, models):
    """Return {model: ranked_list}. models: dict name -> set(active attrs)."""
    out = {}
    for mname, attrs in models.items():
        sc = {}
        for d in cand:
            pf = prof.get(d)
            if not pf:
                sc[d] = 0.0
                continue
            s = 0.0
            for f in findings:
                cell = pf.get(f["cui"])
                if cell is None:
                    continue
                w = cell["w"]
                s += w
                # Per-finding attribute agreements are AVERAGED, not summed, matching the
                # production scorer (attr_score.ATTR_COMBINE="mean"). Summing lets one finding
                # whose location, character, severity and onset all agree outweigh four separately
                # matched findings, so the multi-attribute model scores WORSE than its best single
                # channel -- measured here (+all 21.13 vs +location 21.94) and in production
                # (DDXPlus all-four 46.33 vs onset alone 49.17). Averaging states "this finding's
                # qualifiers agree" once, however many channels carry it.
                parts = []
                if "location" in attrs and f.get("location_cui") and \
                        cell.get("location_cui") == f["location_cui"]:
                    parts.append(LAMBDA * w)
                if "onset" in attrs and f.get("onset") and cell.get("onset") == f["onset"]:
                    parts.append(LAMBDA * w)
                if "severity" in attrs and f.get("severity") and cell.get("severity") \
                        and f["severity"] == cell["severity"]:
                    parts.append(LAMBDA * w)
                # free-text channels: collapse both sides to the controlled classes used by the
                # production scorer, so the ablation tests the signal the diagnosis actually sees
                for ch, fn in (("character", A.char_classes), ("timing", A.timing_classes),
                               ("aggravating", A.trigger_classes),
                               ("relieving", A.trigger_classes)):
                    if ch in attrs and cell.get(ch) and f.get(ch):
                        inter = fn(f[ch]) & fn(cell[ch])
                        if inter:
                            parts.append(LAMBDA * w * A.attr_sel(ch, inter))
                if parts:
                    s += sum(parts) / len(parts)
            # Length normalization exponent, not the hardcoded 0.5 (=sqrt) it used to be. A
            # profile is long partly because the disease manifests widely and partly because we
            # read more about it (Robertson's scope vs verbosity); a square-root penalty charges
            # both alike. Measured on DDXPlus, the gold at ranks 4-10 covers MORE of the patient's
            # evidence than the candidates beating it (62.0% vs 57.1%) and loses only for being
            # larger, and lowering the exponent to 0.28 moved @1 46.5->50.5 and @3 77.4->80.1.
            norm = (len(pf) ** NORM_EXP) if pf else 1.0
            sc[d] = s / max(norm, 1.0)
        out[mname] = sorted(cand, key=lambda d: -sc.get(d, 0.0))
    return out


def metrics(ranks_per_patient, truths):
    n = len(truths)
    c1 = mrr = 0.0
    for ranked, t in zip(ranks_per_patient, truths):
        r = ranked.index(t) + 1 if t in ranked else len(ranked)
        if r == 1:
            c1 += 1
        mrr += 1.0 / r
    return 100 * c1 / n, mrr / n


def eval_model(patients, cand, prof, attrs):
    """Per-patient top1-correct (0/1) and reciprocal rank, for a given attr set."""
    top1, rr = [], []
    for p in patients:
        ranked = score_patient(p["findings"], cand, prof,
                               {"m": attrs})["m"]
        t = p["true_cui"]
        r = ranked.index(t) + 1 if t in ranked else len(ranked)
        top1.append(1 if r == 1 else 0)
        rr.append(1.0 / r)
    return top1, rr


def permute_attribute(prof, attr_key, rng):
    """Return a shallow-copied prof with `attr_key` values shuffled across edges that have it."""
    vals = []
    slots = []
    for d, pf in prof.items():
        for p, cell in pf.items():
            if attr_key in cell:
                vals.append(cell[attr_key])
                slots.append((d, p))
    rng.shuffle(vals)
    new = {d: {p: dict(cell) for p, cell in pf.items()} for d, pf in prof.items()}
    for (d, p), v in zip(slots, vals):
        new[d][p][attr_key] = v
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--queries", default="pilot/cache/prod/nlice_queries.json")
    ap.add_argument("--perm", type=int, default=50, help="permutation-null reps")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap reps")
    a = ap.parse_args()
    rng = random.Random(SEED)

    kg, prof, idf = load_kg(a.kg)
    Q = json.load(open(a.queries))
    patients = Q["patients"]
    cand = sorted(set(Q["meta"]["diseases"].values()) & set(prof.keys()))
    truths = [p["true_cui"] for p in patients]
    print(f"candidates={len(cand)}  patients={len(patients)}  "
          f"findings/pt={sum(len(p['findings']) for p in patients)/len(patients):.1f}")

    # ---- headline models ---------------------------------------------------
    CHANNELS = ["location", "character", "severity", "onset", "timing",
                "aggravating", "relieving"]
    ALL = set(CHANNELS)
    model_sets = {"base": set()}
    model_sets.update({"+" + ch: {ch} for ch in CHANNELS})
    model_sets["+all"] = set(ALL)
    per = {}
    print("\n=== absolute performance (NLICE closed-set DDx) ===")
    print(f"  {'model':12} {'GTPA@1':>8} {'MRR':>7}")
    for m, attrs in model_sets.items():
        top1, rr = eval_model(patients, cand, prof, attrs)
        per[m] = (top1, rr)
        print(f"  {m:12} {100*sum(top1)/len(top1):8.2f} {sum(rr)/len(rr):7.4f}")

    base_top1, base_rr = per["base"]
    base_acc = sum(base_top1) / len(base_top1)

    # ---- per-attribute: observed delta + permutation null + bootstrap ------
    print(f"\n=== leave-one-in delta vs permutation null (K={a.perm}) + bootstrap ({a.boot}) ===")
    print(f"  {'attr':10} {'dObs@1':>8} {'null_mean':>10} {'null_sd':>8} "
          f"{'z':>7} {'p':>7} {'boot95CI':>18}")
    attr_key = {ch: ("location_cui" if ch == "location" else ch) for ch in CHANNELS}
    results = {}
    for ch in CHANNELS:
        top1, rr = per["+" + ch]
        obs = sum(top1) / len(top1) - base_acc          # delta in accuracy (fraction)
        # bootstrap CI on delta (resample patients)
        diffs = [t - b for t, b in zip(top1, base_top1)]
        n = len(diffs)
        boots = []
        for _ in range(a.boot):
            s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
            boots.append(s)
        boots.sort()
        lo, hi = boots[int(0.025 * a.boot)], boots[int(0.975 * a.boot)]
        # permutation null: shuffle this attr's KG values, recompute +attr delta
        nulls = []
        for _ in range(a.perm):
            pprof = permute_attribute(prof, attr_key[ch], rng)
            pt1, _ = eval_model(patients, cand, pprof, {ch})
            nulls.append(sum(pt1) / len(pt1) - base_acc)
        nm = sum(nulls) / len(nulls)
        nsd = (sum((x - nm) ** 2 for x in nulls) / max(1, len(nulls) - 1)) ** 0.5
        z = (obs - nm) / nsd if nsd > 0 else float("nan")
        pval = (1 + sum(1 for x in nulls if x >= obs)) / (len(nulls) + 1)
        results[ch] = {"obs": obs, "null_mean": nm, "null_sd": nsd, "z": z,
                       "p": pval, "ci": [lo, hi]}
        print(f"  {ch:10} {100*obs:8.3f} {100*nm:10.3f} {100*nsd:8.3f} "
              f"{z:7.2f} {pval:7.3f} [{100*lo:+.3f},{100*hi:+.3f}]")

    print("\n  (delta in percentage points of GTPA@1; p = permutation P(null>=obs); "
          "CI excludes 0 AND p<0.05 => channel credited)")
    json.dump({"candidates": cand, "abs": {m: (100*sum(t)/len(t), sum(r)/len(r))
                                           for m, (t, r) in per.items()},
               "ablation": results, "lambda": LAMBDA, "perm": a.perm, "boot": a.boot},
              open("pilot/cache/prod/nlice_ablation_result.json", "w"))
    print("\n-> pilot/cache/prod/nlice_ablation_result.json")


if __name__ == "__main__":
    main()
