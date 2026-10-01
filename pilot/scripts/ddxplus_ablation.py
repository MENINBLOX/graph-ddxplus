"""DDXPlus attribute leave-one-in ablation with a permutation-null significance test.

Mirrors nlice_ablation.py but on the REAL DDXPlus benchmark (not synthetic NLICE). DDXPlus
queries each symptom across Bates axes: pain -> location (E_55), radiation (E_57), character
(E_54), intensity/severity (E_56), onset (E_59); rash -> colour (E_130), location (E_133)...
attr_loader_ddxplus decodes these per finding, so 79-82% of patients carry an attribute.

Question: on a real benchmark, does a normalized attribute (esp. location) carry diagnostic
signal beyond chance? Design = randomization inference (Altmann 2010): observed leave-one-in
delta GTPA@1 vs a null from shuffling that attribute's values across the KG edges that carry
it (marginal preserved, disease<->attribute link destroyed). Bootstrap 95% CI over patients.
A channel is credited iff observed delta clears its permutation null (p<0.05) AND CI excludes 0.

Usage: python pilot/scripts/ddxplus_ablation.py --kg pilot/cache/prod/kg_v2.json --perm 200 --boot 2000
"""
from __future__ import annotations
import json, math, argparse, random, sys, ast
from collections import defaultdict, Counter

sys.path.insert(0, "pilot/scripts")
import attr_loader_ddxplus as DL

LAMBDA = 1.0          # attribute-match bonus, fixed global default (CLAUDE.md §7)
SEED = 42


def load_kg(path):
    kg = json.load(open(path))
    N = len(kg["diseases"])
    df = Counter()
    for e in kg["edges"]:
        df[e["p"]] += 1
    idf = {p: math.log(N / c) for p, c in df.items()}
    prof = defaultdict(dict)
    for e in kg["edges"]:
        d, p, at = e["d"], e["p"], e.get("attrs", {})
        if isinstance(at, str):
            try:
                at = ast.literal_eval(at)
            except Exception:
                at = {}
        cell = {"w": idf.get(p, 0.0)}
        if at.get("location_cui"):
            cell["location_cui"] = at["location_cui"]
        if at.get("character"):
            cell["character"] = str(at["character"]).lower()
        if at.get("onset"):
            cell["onset"] = str(at["onset"]).lower()
        if at.get("severity"):
            cell["severity"] = str(at["severity"]).lower()
        prof[d][p] = cell
    return kg, prof, idf


def score_patient(findings, cand, prof, attrs):
    ranked = {}
    for d in cand:
        pf = prof.get(d)
        if not pf:
            ranked[d] = 0.0
            continue
        s = 0.0
        for f in findings:
            cell = pf.get(f["cui"])
            if cell is None:
                continue
            w = cell["w"]
            s += w
            if "location" in attrs and cell.get("location_cui") and \
                    cell["location_cui"] in f.get("location", ()):      # set membership
                s += LAMBDA * w
            if "character" in attrs and cell.get("character") and f.get("character"):
                cc = cell["character"]
                if any(cc == q or cc in q or q in cc for q in f["character"]):
                    s += LAMBDA * w
            if "severity" in attrs and cell.get("severity") and f.get("severity"):
                if cell["severity"] == f["severity"]:
                    s += LAMBDA * w
            if "onset" in attrs and cell.get("onset") and f.get("onset"):
                if cell["onset"] == f["onset"]:
                    s += LAMBDA * w
        ranked[d] = s / max(math.sqrt(len(pf)), 1.0)
    return sorted(cand, key=lambda d: -ranked.get(d, 0.0))


def eval_model(patients, cand, prof, attrs):
    top1, rr = [], []
    for tset, findings in patients:
        ranked = score_patient(findings, cand, prof, attrs)
        r = min((ranked.index(t) + 1 for t in tset if t in ranked), default=len(ranked))
        top1.append(1 if r == 1 else 0)
        rr.append(1.0 / r)
    return top1, rr


def permute_attr(prof, key, rng):
    vals, slots = [], []
    for d, pf in prof.items():
        for p, cell in pf.items():
            if key in cell:
                vals.append(cell[key]); slots.append((d, p))
    rng.shuffle(vals)
    new = {d: {p: dict(c) for p, c in pf.items()} for d, pf in prof.items()}
    for (d, p), v in zip(slots, vals):
        new[d][p][key] = v
    return new


ATTR_KEY = {"location": "location_cui", "character": "character",
            "severity": "severity", "onset": "onset"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg_v2.json")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    rng = random.Random(SEED)

    kg, prof, idf = load_kg(a.kg)
    cand_map, raw = DL.load(a.n)
    cand = sorted({cu for _, cu in cand_map.values()} & set(prof.keys()))
    # patients: (target_cui_set, findings) — keep only findings whose base CUI is in KG vocab
    patients = []
    for tk, finds, _ in raw:
        tset = {cand_map[k][1] for k in tk if k in cand_map}
        if tset:
            patients.append((tset, finds))
    print(f"kg={a.kg}  candidates={len(cand)}  patients={len(patients)}", flush=True)

    models = {"base": set(), "+location": {"location"}, "+character": {"character"},
              "+severity": {"severity"}, "+onset": {"onset"},
              "+all": {"location", "character", "severity", "onset"}}
    per = {}
    print(f"\n{'model':12} {'GTPA@1':>8} {'MRR':>8}")
    for m, at in models.items():
        t1, rr = eval_model(patients, cand, prof, at)
        per[m] = (t1, rr)
        print(f"{m:12} {100*sum(t1)/len(t1):8.2f} {sum(rr)/len(rr):8.4f}", flush=True)
    base_t1 = per["base"][0]
    base_acc = 100 * sum(base_t1) / len(base_t1)

    print(f"\n=== leave-one-in delta vs permutation null (K={a.perm}) + bootstrap ({a.boot}) ===")
    print(f"{'attr':10} {'dObs(pp)':>9} {'null(mean±sd)':>16} {'z':>7} {'p_perm':>8} {'boot95CI':>18}")
    for attr in ["location", "character", "severity", "onset"]:
        obs_t1 = per["+" + attr][0]
        d_obs = 100 * (sum(obs_t1) - sum(base_t1)) / len(base_t1)
        # permutation null: shuffle this attr's KG values, recompute delta
        null = []
        for _ in range(a.perm):
            pp = permute_attr(prof, ATTR_KEY[attr], rng)
            t1, _ = eval_model(patients, cand, pp, {attr})
            null.append(100 * (sum(t1) - sum(base_t1)) / len(base_t1))
        mu = sum(null) / len(null)
        sd = (sum((x - mu) ** 2 for x in null) / len(null)) ** 0.5
        z = (d_obs - mu) / sd if sd > 0 else float("nan")
        p = (sum(1 for x in null if x >= d_obs) + 1) / (len(null) + 1)
        # bootstrap CI on observed delta (paired per-patient)
        diffs = [o - b for o, b in zip(obs_t1, base_t1)]
        boot = []
        for _ in range(a.boot):
            s = sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs)))
            boot.append(100 * s / len(diffs))
        boot.sort()
        lo, hi = boot[int(.025 * len(boot))], boot[int(.975 * len(boot))]
        print(f"{attr:10} {d_obs:9.2f} {mu:7.2f}±{sd:5.2f}   {z:7.2f} {p:8.3f}   [{lo:+.2f}, {hi:+.2f}]",
              flush=True)


if __name__ == "__main__":
    main()
