"""Extrinsic (downstream-task) evaluation of IE prompt variants — KGrEaT paradigm
(Heist & Paulheim, arXiv:2308.10537): evaluate a KG by its downstream task
performance. The downstream task here is DDXPlus top-k diagnosis.

Controlled experiment — the ONLY variable is the IE prompt:
  - same sources (pilot/data/cache/v105_sources/, Wikipedia clinical sections)
  - same 49 DDXPlus diseases
  - same scispaCy UMLS linker (shared name->CUI map, held constant)
  - same scoring algorithm + same hyperparameters (onlykg_eval_v71_selfaware)
Whatever @1/@10 a variant yields is attributable to its extracted finding set.

Complements the intrinsic NLI faithfulness (precision/hallucination) already run:
faithfulness tied at 83%; this measures recall+utility via the downstream task.
"""
import sys, json, glob, pickle, math, random, argparse
from collections import defaultdict

sys.path.insert(0, "pilot/scripts")
from onlykg_eval_v71_selfaware import (compute_idf, reweight,
                                       precompute_signal_v71, score,
                                       load_ddxplus_full)

VARIANTS = ["v105_grounded_ie", "v106_grounded_ie", "v107_grounded_ie"]
LINK_CACHE = "pilot/data/cache/v110_name2cui.json"


def link_names(all_names, threshold):
    """Shared scispaCy UMLS linker — same system for every variant."""
    try:
        cached = json.load(open(LINK_CACHE))
        if all(n in cached for n in all_names):
            print(f"  name->CUI: loaded {len(cached)} cached", flush=True)
            return cached
    except FileNotFoundError:
        cached = {}
    import spacy
    from scispacy.linking import EntityLinker  # noqa: F401
    print("  loading scispaCy en_core_sci_lg + UMLS linker...", flush=True)
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
        "k": 3, "threshold": threshold, "max_entities_per_mention": 1})
    todo = sorted(set(all_names) - set(cached))
    print(f"  linking {len(todo)} new names...", flush=True)
    for doc, name in zip(nlp.pipe(todo, batch_size=128), todo):
        best, bsc = None, 0.0
        for ent in doc.ents:
            for cui, sc in ent._.kb_ents:
                if sc > bsc: bsc, best = sc, cui
        cached[name] = best  # may be None
    json.dump(cached, open(LINK_CACHE, "w"))
    print(f"  name->CUI cached -> {LINK_CACHE}", flush=True)
    return cached


def build_profile(variant, name2cui):
    """Disease->CUI presence profile from one variant's findings (p=1.0 presence,
    IDF-weighted cosine does the discrimination — counts are ~1 in grounded IE)."""
    P = defaultdict(dict)
    nf = 0
    for f in sorted(glob.glob(f"pilot/data/cache/{variant}/*.json")):
        o = json.load(open(f))
        d = o["cui"]
        for fd in o.get("findings", []):
            nm = (fd.get("name") or fd.get("finding") or "").strip().lower()
            if not nm: continue
            pc = name2cui.get(nm)
            if not pc or pc == d: continue
            P[d][pc] = 1.0
            nf += 1
    return dict(P), nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.85)
    # fixed algorithm hyperparameters — identical across all variants
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--sharp", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.2)
    a = ap.parse_args()

    # proper random sample (memory: first-N is biased toward easy diseases)
    print(f"Loading DDXPlus (random {a.n}, seed={a.seed})...", flush=True)
    dcs_list, all_patients, binary_evs = load_ddxplus_full(200000)
    random.seed(a.seed)
    patients = random.sample(all_patients, min(a.n, len(all_patients)))
    value_cuis = json.load(open("/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"))

    # shared linker over the union of all variants' finding names
    names = set()
    for v in VARIANTS:
        for f in glob.glob(f"pilot/data/cache/{v}/*.json"):
            for fd in json.load(open(f)).get("findings", []):
                nm = (fd.get("name") or fd.get("finding") or "").strip().lower()
                if nm: names.add(nm)
    print(f"Union finding names: {len(names)}", flush=True)
    name2cui = link_names(names, a.threshold)
    mapped = sum(1 for n in names if name2cui.get(n))
    print(f"  mapped {mapped}/{len(names)} ({100*mapped/len(names):.1f}%)\n", flush=True)

    print(f"=== EXTRINSIC downstream eval (DDXPlus, N={len(patients)} random) ===")
    print(f"    fixed algo: beta={a.beta} tau={a.tau} sharp={a.sharp} lam={a.lam}")
    print(f"{'variant':<20} {'findings':>8} {'CUIs':>6} {'@1':>7} {'@5':>7} {'@10':>7} {'MRR':>7}")
    results = {}
    for v in VARIANTS:
        base, nf = build_profile(v, name2cui)
        ncui = len(set().union(*[set(p) for p in base.values()])) if base else 0
        idf = compute_idf(base, 0.12)
        prof = reweight(base, idf, 1.0, a.beta)
        sig = precompute_signal_v71(prof, value_cuis, binary_evs, idf, a.tau, a.sharp)
        all_evs = set().union(*[set(p) for p in prof.values()]) if prof else set()
        n = c1 = c5 = c10 = 0; rr = 0.0
        for true_cui, pos_raw, neg_binary in patients:
            pos = pos_raw & all_evs
            if not pos or true_cui not in prof: continue
            s = score(pos, neg_binary, prof, idf, a.beta, sig, a.lam)
            ranked = sorted(prof.keys(), key=lambda d: -s[d])
            n += 1
            try: rank = ranked.index(true_cui) + 1
            except ValueError: rank = len(prof)
            c1 += rank == 1; c5 += rank <= 5; c10 += rank <= 10; rr += 1.0/rank
        r = {"n": n, "at1": 100*c1/n, "at5": 100*c5/n, "at10": 100*c10/n, "mrr": rr/n}
        results[v] = r
        print(f"{v:<20} {nf:>8} {ncui:>6} {r['at1']:>6.2f}% {r['at5']:>6.2f}% "
              f"{r['at10']:>6.2f}% {r['mrr']:>7.4f}", flush=True)

    best1 = max(results, key=lambda v: results[v]["at1"])
    best10 = max(results, key=lambda v: results[v]["at10"])
    print(f"\nbest @1:  {best1} ({results[best1]['at1']:.2f}%)")
    print(f"best @10: {best10} ({results[best10]['at10']:.2f}%)")
    json.dump(results, open("pilot/data/cache/v110_extrinsic_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
