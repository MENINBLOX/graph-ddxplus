"""Unified only-KG Stage-1 evaluator across ALL benchmarks, with corrected disease-CUI
resolution (the eval-pipeline flaw found by nlice_forensic.py).

Root cause of the low base: our KG is built from a rare-disease-inclusive UMLS DISO seed,
so it indexes diseases as SPECIFIC subtype CUIs. Benchmark diseases (common umbrellas)
link to a generic CUI with profile=0, or mislink ('Appendicitis' -> 'symptoms of
appendicitis'). Fix: resolve each benchmark disease to a CLUSTER of KG subtype CUIs and
merge their phenotype profiles.

Resolution (single rule, all benchmarks -- CLAUDE.md §7):
  cluster(name, mapped_cui) =
      {mapped_cui}                                if in KG-with-profile
    ∪ {kg_cui : content-tokens(name) ⊆ tokens(kg_name)}   # lexical specialization (isa)
    ∪ {MRREL descendants(mapped_cui)} ∩ KG        # taxonomic subtypes
  All signals are UMLS/lexical structure, NOT benchmark data -> strict zero-shot safe.

Scoring: IDF-weighted phenotype-CUI overlap over the merged cluster profile, size-normalized.
Metric: GTPA@1/3/5/10 + MRR, per benchmark and aggregate.
"""
from __future__ import annotations
import os, sys, json, math, re, random, argparse
from collections import defaultdict, Counter

random.seed(42)
STOP = {"of", "the", "a", "and", "with", "disease", "disorder", "syndrome", "acute",
        "chronic", "infection", "most", "likely", "type", "other", "specified",
        "unspecified", "nos", "symptoms", "symptom", "due", "to", "in", "and related",
        "conditions", "condition", "primary", "secondary", "congenital"}


def toks(name: str) -> set:
    name = re.sub(r"\(.*?\)", " ", name.lower())
    return {w for w in re.findall(r"[a-z0-9]+", name) if w not in STOP and len(w) > 2}


# How much an edge's salience is allowed to modulate its weight. The IE gives a cardinal symptom
# and a passing mention the same weight, which is what makes profiles broad and costs Top-1; the
# source's own ordering separates them (a finding named in the opening tenth is reported by
# patients 14.8% of the time vs 3.8% for the closing tenth). 0 = ignore salience.
SAL_W = float(os.environ.get("KG_SALIENCE_W", "0"))


def load_kg(path):
    kg = json.load(open(path))
    PH = kg["phenotypes"]
    prof = defaultdict(dict)
    for e in kg["edges"]:
        w = PH.get(e["p"], {}).get("idf", 0.0)
        if SAL_W:
            sal = e.get("sal", 1.0)
            w *= (1.0 - SAL_W) + SAL_W * sal
        prof[e["d"]][e["p"]] = w
    idf = {c: PH[c].get("idf", 0.0) for c in PH}
    dz_names = {c: d["name"] for c, d in kg["diseases"].items()}
    return kg, prof, idf, dz_names


def build_name_index(prof, dz_names):
    """token -> set(kg disease CUIs with profile) for lexical specialization lookup."""
    inv = defaultdict(set)
    tset = {}
    for c in prof:
        t = toks(dz_names.get(c, ""))
        tset[c] = t
        for w in t:
            inv[w].add(c)
    return inv, tset


class Resolver:
    def __init__(self, prof, dz_names, children):
        self.prof = prof
        self.dz_names = dz_names
        self.children = children
        self.inv, self.tset = build_name_index(prof, dz_names)
        self.cache = {}

    def descendants(self, cui, cap=3):
        seen, frontier, d = set(), {cui}, 0
        while frontier and d < cap:
            nxt = set()
            for c in frontier:
                for k in self.children.get(c, []):
                    if k not in seen:
                        seen.add(k); nxt.add(k)
            frontier, d = nxt, d + 1
        return seen

    def cluster(self, name, mapped_cui):
        key = (name, mapped_cui)
        if key in self.cache:
            return self.cache[key]
        cl = set()
        if mapped_cui and mapped_cui in self.prof:
            cl.add(mapped_cui)
        # lexical specialization: KG names whose tokens ⊇ name tokens
        nt = toks(name)
        if nt:
            cand = None
            for w in nt:
                s = self.inv.get(w, set())
                cand = s if cand is None else (cand & s)
                if not cand:
                    break
            for c in (cand or ()):
                if nt <= self.tset[c]:
                    cl.add(c)
        # taxonomic descendants
        if mapped_cui:
            for c in self.descendants(mapped_cui):
                if c in self.prof:
                    cl.add(c)
        self.cache[key] = cl
        return cl


# Cluster-consensus weighting for merged profiles (0 = off, i.e. plain union-max).
# A cluster's union profile inherits every member's noise: Bronchitis resolves to 19 subtype
# nodes whose union holds 841 phenotypes, 82% carried by a SINGLE member (comorbidity mentions
# like "insulin resistance" from one chronic-bronchitis article). Union-max gives that lone
# mention the same weight as cough, which every member carries. gamma>0 scales each phenotype
# by (members carrying it / cluster size)^gamma -- consensus keeps full weight, a lone mention
# is discounted in proportion to how unrepresentative it is. Threshold-free.
CONSENSUS_G = float(os.environ.get("KG_CONSENSUS_G", "0"))


def merged_profile(cluster, prof):
    m = {}
    if CONSENSUS_G and len(cluster) > 1:
        carry = {}
        for c in cluster:
            for p, w in prof.get(c, {}).items():
                if w > m.get(p, 0):
                    m[p] = w
                carry[p] = carry.get(p, 0) + 1
        n = float(len(cluster))
        return {p: w * ((carry[p] / n) ** CONSENSUS_G) for p, w in m.items()}
    for c in cluster:
        for p, w in prof.get(c, {}).items():
            if w > m.get(p, 0):
                m[p] = w
    return m


# --- phenotype hierarchy expansion (shared-ancestor granularity bridging) -------------
PARENTS = {}
_anc_cache = {}


def ancestors(cui, cap=2, decay=0.5):
    """{cui:1.0} ∪ {ancestor: decay^hop}, capped depth."""
    if cui in _anc_cache:
        return _anc_cache[cui]
    out = {cui: 1.0}
    frontier = {cui}
    w = 1.0
    for _ in range(cap):
        w *= decay
        nxt = set()
        for c in frontier:
            for pa in PARENTS.get(c, []):
                if pa not in out:
                    out[pa] = w
                    nxt.add(pa)
        frontier = nxt
        if not frontier:
            break
    _anc_cache[cui] = out
    return out


def expand_profile(prof_map, cap, decay):
    """profile {p:idf} -> {c: weight} adding ancestors of each p (idf * decay^hop)."""
    if cap == 0:
        return prof_map
    exp = {}
    for p, w in prof_map.items():
        for a, aw in ancestors(p, cap, decay).items():
            v = w * aw
            if v > exp.get(a, 0):
                exp[a] = v
    return exp


def expand_query(ev, cap, decay):
    if cap == 0:
        return {c: 1.0 for c in ev}
    q = {}
    for e in ev:
        for a, aw in ancestors(e, cap, decay).items():
            if aw > q.get(a, 0):
                q[a] = aw
    return q


# ---------------- benchmark loaders: yield (disease_key, name, mapped_cui) candidates
#                  and patients (target_key, evidence_cui_set) -------------------------

def load_rarebench(datasets):
    dis_map = json.load(open("data/rarebench/disease_umls_mapping.json"))["mapping"]
    hpo_map = json.load(open("data/rarebench/hpo_umls_mapping.json"))["mapping"]
    cand = {}     # key -> (name, cui)
    patients = []
    for ds in datasets:
        p = f"data/rarebench/data/{ds}.jsonl"
        try:
            fh = open(p)
        except FileNotFoundError:
            continue
        for line in fh:
            r = json.loads(line)
            tkeys = []
            for t in r["RareDisease"]:
                info = dis_map.get(t)
                if info and info.get("umls_cui"):
                    cand[t] = (info.get("disease_name", t), info["umls_cui"])
                    tkeys.append(t)
            if not tkeys:
                continue
            ev = set()
            for hp in r["Phenotype"]:
                info = hpo_map.get(hp)
                c = info.get("umls_cui") if isinstance(info, dict) else info if isinstance(info, str) else None
                if c:
                    ev.add(c)
            if ev:
                patients.append((tkeys, ev, ds))
    return cand, patients


def load_symcat(ppd=50):
    sc = json.load(open("data/symcat/symcat_parsed.json"))
    sym_umls = json.load(open("data/symcat/symptom_umls_mapping.json"))["mapping"]
    dis_umls = json.load(open("data/symcat/disease_umls_mapping.json"))["mapping"]
    sym2cui = {s: i["umls_cui"] for s, i in sym_umls.items() if i.get("umls_cui")}
    cand = {}
    for dn, info in dis_umls.items():
        if info.get("umls_cui"):
            cand[dn] = (dn, info["umls_cui"])
    patients = []
    for dn, sym_list in sc["disease_symptom_pairs"].items():
        if dn not in cand or not sym_list:
            continue
        sc_cuis = [(sym2cui.get(s), f) for s, f in sym_list if sym2cui.get(s)]
        if not sc_cuis:
            continue
        for _ in range(ppd):
            ev = {c for c, fr in sc_cuis if random.random() * 100 < fr}
            if ev:
                patients.append(([dn], ev, "SymCat"))
    return cand, patients


def cui_names(kg):
    """CUI -> preferred name, aggregated from the KG and every benchmark's own mapping files.
    Used to embed CUIs for SapBERT-style vocabulary alignment."""
    nm = {c: d.get("name", "") for c, d in kg["phenotypes"].items()}
    for c, d in kg["diseases"].items():
        nm.setdefault(c, d.get("name", ""))
    try:
        um = json.load(open("data/ddxplus/umls_mapping.json"))["mapping"]
        for v in um.values():
            nm.setdefault(v["cui"], v["name"])
    except Exception:
        pass
    for path, key in [("data/rarebench/hpo_umls_mapping.json", "mapping"),
                      ("data/rarebench/disease_umls_mapping.json", "mapping"),
                      ("data/symcat/symptom_umls_mapping.json", "mapping"),
                      ("data/symcat/disease_umls_mapping.json", "mapping")]:
        try:
            m = json.load(open(path))[key]
            for k, v in m.items():
                if isinstance(v, dict) and v.get("umls_cui"):
                    label = v.get("hpo_name") or v.get("disease_name") or v.get("name") or k
                    nm.setdefault(v["umls_cui"], label)
        except Exception:
            pass
    return nm


def load_ddxplus(n=3000):
    """DDXPlus: 49 diseases, patients = list of evidence codes (symptoms + antecedents)
    + PATHOLOGY (French). Unified evidence: antecedent history maps to disease CUIs (§11)."""
    import csv, ast
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    dmap = json.load(open("data/ddxplus/disease_umls_mapping.json"))["mapping"]
    umap = json.load(open("data/ddxplus/umls_mapping.json"))["mapping"]
    fr2en = {info.get("cond-name-fr") or info.get("condition_name"): en
             for en, info in cond.items()}
    cand = {en: (info.get("name_en", en), info["umls_cui"])
            for en, info in dmap.items() if info.get("umls_cui")}

    def ev_cui(e):
        for c in (e, e.split("_@_")[0], e.split("_")[0]):
            if c in umap:
                return umap[c]["cui"]
        return None

    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        rd = csv.DictReader(f)
        for i, row in enumerate(rd):
            if len(patients) >= n:
                break
            en = fr2en.get(row["PATHOLOGY"])
            if en not in cand:
                continue
            evs = ast.literal_eval(row["EVIDENCES"])
            ev = {c for c in (ev_cui(e) for e in evs) if c}
            if ev:
                patients.append(([en], ev, "DDXPlus"))
    return cand, patients


def evaluate(name, cand, patients, resolver, prof, hier_cap=0, hier_decay=0.5):
    keys = sorted(cand)
    prof_of = {}
    for k in keys:
        nm, cu = cand[k]
        pf = merged_profile(resolver.cluster(nm, cu), prof)
        prof_of[k] = expand_profile(pf, hier_cap, hier_decay) if pf else pf
    rankable = [k for k in keys if prof_of[k]]
    n = len(patients)
    c1 = c3 = c5 = c10 = rr = 0
    covered = 0
    for tkeys, ev, ds in patients:
        tset = set(tkeys)
        if not any(prof_of.get(t) for t in tset):
            n_rank = len(keys)
            rr += 1.0 / n_rank
            continue
        covered += 1
        q = expand_query(ev, hier_cap, hier_decay)
        scores = {}
        for k in rankable:
            pf = prof_of[k]
            s = sum(qw * pf[c] for c, qw in q.items() if c in pf)
            scores[k] = s / max(math.sqrt(len(pf)), 1.0)
        ranked = sorted(rankable, key=lambda k: -scores.get(k, 0))
        ranks = [ranked.index(t) + 1 if t in ranked else len(keys) for t in tset if t in rankable]
        rank = min(ranks) if ranks else len(keys)
        if rank == 1: c1 += 1
        if rank <= 3: c3 += 1
        if rank <= 5: c5 += 1
        if rank <= 10: c10 += 1
        rr += 1.0 / rank
    print(f"\n=== {name}  ({n} patients, {len(rankable)}/{len(keys)} rankable candidates) ===")
    print(f"  disease-coverage (target resolvable to KG cluster): {100*covered/n:.1f}%")
    print(f"  GTPA@1={100*c1/n:5.2f}  @3={100*c3/n:5.2f}  @5={100*c5/n:5.2f}  "
          f"@10={100*c10/n:5.2f}  MRR={rr/n:.4f}")
    return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n, "at5": 100*c5/n,
            "at10": 100*c10/n, "mrr": rr/n, "cov": 100*covered/n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--parents", default="pilot/cache/umls_maps/phenotype_parents.json")
    ap.add_argument("--benchmarks", default="rarebench,symcat")
    ap.add_argument("--ppd", type=int, default=50)
    ap.add_argument("--hier", type=int, default=0, help="phenotype ancestor-expansion depth")
    ap.add_argument("--decay", type=float, default=0.5)
    a = ap.parse_args()

    kg, prof, idf, dz_names = load_kg(a.kg)
    children = json.load(open(a.children))
    resolver = Resolver(prof, dz_names, children)
    if a.hier:
        PARENTS.update(json.load(open(a.parents)))
    print(f"KG diseases-with-profile={len(prof)}  children-map parents={len(children)}  "
          f"hier={a.hier} pheno-parents={len(PARENTS)}")

    def ev(name, cand, pats):
        return evaluate(name, cand, pats, resolver, prof, a.hier, a.decay)

    summary = {}
    for b in a.benchmarks.split(","):
        if b == "rarebench":
            for ds in ["RAMEDIS", "LIRICAL", "MME", "HMS"]:
                cand, pats = load_rarebench([ds])
                if pats:
                    summary[f"RareBench:{ds}"] = ev(f"RareBench:{ds}", cand, pats)
        elif b == "symcat":
            cand, pats = load_symcat(a.ppd)
            summary["SymCat"] = ev("SymCat", cand, pats)

    print("\n" + "=" * 70)
    print(f"  {'benchmark':18} {'cov%':>6} {'@1':>6} {'@3':>6} {'@5':>6} {'@10':>6} {'MRR':>7}")
    for k, v in summary.items():
        print(f"  {k:18} {v['cov']:6.1f} {v['at1']:6.2f} {v['at3']:6.2f} "
              f"{v['at5']:6.2f} {v['at10']:6.2f} {v['mrr']:7.4f}")
    json.dump(summary, open("pilot/cache/prod/unified_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
