"""TRACEABLE DIFFERENTIAL — the provenance showcase (thesis traceability axis).

Given one RareBench patient, this:
  1. ranks candidate diseases against the patient's evidence with the KG's structural scorer
     (unified-evidence: disease-hierarchy prior + phenotype similarity, CLAUDE.md §11), and
  2. for the CORRECT disease, prints every matched disease->phenotype edge together with its
     materialized `prov` — the source document (PubMed PMID / GeneReviews or StatPearls NBK) and the
     original finding surface text the edge was extracted from.

This demonstrates that a diagnosis is not a black box: each supporting edge traces back to a citable
source passage. Uses the provenanced KG produced by v159_provenance.py.

Scoring here is the base structural channel of onlykg_eval_unified2 (hier=2). The LSA/SapBERT
late-fusion re-rank used in the headline eval is additive and does not change which edges a disease
is built from, so the traced edges are identical; we omit it to keep the showcase self-contained.

Usage: python trace_differential.py [--kg ...kg_merged2_prov.json] [--dataset RAMEDIS]
       [--patient IDX]   # default: auto-pick a clean metabolic case ranked #1"""
from __future__ import annotations
import sys, json, math, argparse
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U

HIER, DECAY, DZCAP, NORMEXP, DZDECAY = 2, 0.5, 4, 0.5, 1.0
METABOLIC = ("phenylketon", "hyperphenyl", "maple syrup", "homocystin", "tyrosin",
             "aciduria", "acidemia", "galactos", "urea cycle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg_merged2_prov.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--parents", default="pilot/cache/umls_maps/phenotype_parents.json")
    ap.add_argument("--dataset", default="LIRICAL")
    ap.add_argument("--patient", type=int, default=267,
                    help="patient index; -1 = auto-pick a metabolic case ranked #1 "
                         "(default 267 = Tuberous Sclerosis, provenance across all 3 corpora)")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    kg, prof, idf, dz_names = U.load_kg(a.kg)
    children = json.load(open(a.children))
    U.PARENTS.update(json.load(open(a.parents)))
    resolver = U.Resolver(prof, dz_names, children)
    PH = kg["phenotypes"]
    # edge prov lookup: (disease_cui, phenotype_cui) -> prov list
    edge_prov = {(e["d"], e["p"]): e.get("prov", []) for e in kg["edges"]}

    child_adj = children
    parent_adj = defaultdict(set)
    for p, chs in children.items():
        for c in chs:
            parent_adj[c].add(p)
    KGDZ = set(kg["diseases"]); N_DZ = len(KGDZ)

    def closure(cui, adj, cap):
        seen, fr, d = set(), {cui}, 0
        while fr and d < cap:
            nx = set()
            for c in fr:
                for k in adj.get(c, ()):
                    if k not in seen:
                        seen.add(k); nx.add(k)
            fr, d = nx, d + 1
        return seen

    def closure_dist(cui, adj, cap):
        dist = {cui: 0}; fr, d = {cui}, 0
        while fr and d < cap:
            nx = set()
            for c in fr:
                for k in adj.get(c, ()):
                    if k not in dist:
                        dist[k] = d + 1; nx.add(k)
            fr, d = nx, d + 1
        return dist

    ic_cache = {}
    def ic_dz(cui):
        if cui not in ic_cache:
            nd = len(closure(cui, child_adj, 6) & KGDZ)
            ic_cache[cui] = math.log(N_DZ / (1 + nd))
        return ic_cache[cui]

    sigcache = {}
    def sig(name, cui):
        cl = resolver.cluster(name, cui)
        raw = U.merged_profile(cl, prof)
        pf = U.expand_profile(raw, HIER, DECAY) if raw else {}
        dsig = {}; base = ({cui} if cui else set()) | cl
        for b in base:
            dd = closure_dist(b, parent_adj, DZCAP)
            for r, dist in closure_dist(b, child_adj, DZCAP).items():
                if dist < dd.get(r, 99):
                    dd[r] = dist
            for r, dist in dd.items():
                w = ic_dz(r) * (DZDECAY ** dist)
                if w > dsig.get(r, 0):
                    dsig[r] = w
        return pf, dsig, set(raw), cl

    cand, pats = U.load_rarebench([a.dataset])
    keys = sorted(cand)
    for k in keys:
        sigcache[k] = sig(*cand[k])

    def rank_patient(ev):
        q = U.expand_query(ev, HIER, DECAY)
        scores = {}
        for k in keys:
            pf, dsig, raw, cl = sigcache[k]
            if not pf and not dsig:
                continue
            dm = pm = 0.0
            for c, qw in q.items():
                d = dsig.get(c, 0.0); p = pf.get(c, 0.0)
                dm += qw * d
                if p > d:
                    pm += qw * p
            scores[k] = dm / max(len(dsig) ** NORMEXP, 1.0) + pm / max(len(pf) ** NORMEXP, 1.0)
        ranked = sorted(scores, key=lambda k: -scores[k])
        return ranked, scores

    # choose patient
    idx = a.patient
    if idx < 0:
        for i, (tkeys, ev, ds) in enumerate(pats):
            nm = " ".join(dz_names.get(cand[t][1], cand[t][0]).lower() for t in tkeys)
            if not any(w in nm for w in METABOLIC) or len(ev) < 6:
                continue
            ranked, _ = rank_patient(ev)
            tr = min([ranked.index(t) + 1 for t in tkeys if t in ranked] or [999])
            if tr == 1:
                idx = i; break
        if idx < 0:
            idx = 0
    tkeys, ev, ds = pats[idx]
    ranked, scores = rank_patient(ev)

    def cui_label(c):
        if c in PH and PH[c].get("name"):
            return PH[c]["name"]
        return dz_names.get(c, c)

    print("=" * 78)
    print(f"TRACEABLE DIFFERENTIAL  —  {ds} patient #{idx}")
    print("=" * 78)
    print(f"\nPatient evidence ({len(ev)} findings):")
    for c in sorted(ev):
        print(f"    {c}  {cui_label(c)}")

    print(f"\nTop-{a.topk} ranked diseases:")
    tset = set(tkeys)
    for r, k in enumerate(ranked[:a.topk], 1):
        nm, cu = cand[k]
        mark = "  <-- CORRECT" if k in tset else ""
        print(f"  {r}. [{scores[k]:.3f}] {dz_names.get(cu, nm)}  ({cu}){mark}")

    # correct disease + its rank (best-ranked co-referent target code)
    tgt = None
    for k in tkeys:
        if k in ranked:
            tgt = k if tgt is None or ranked.index(k) < ranked.index(tgt) else tgt
    if tgt is None:
        print("\n(correct disease not rankable)")
        return
    tgt_rank = ranked.index(tgt) + 1
    tnm, tcu = cand[tgt]
    print(f"\nCorrect disease: {dz_names.get(tcu, tnm)} ({tcu})  ranked #{tgt_rank}")

    # matched phenotype edges with provenance. The benchmark declares the patient's target codes
    # co-referent (OMIM+ORPHA of one disease); the evaluator merges them, so we trace edges over the
    # UNION of every target code's disease cluster (all subtypes of the correct disease).
    q = U.expand_query(ev, HIER, DECAY)
    ev_set = set(ev)
    dz_members = set()
    for k in tkeys:
        nm, cu = cand[k]
        dz_members |= resolver.cluster(nm, cu) | ({cu} if cu else set())
    # a matched edge = (cluster-member disease, phenotype p) where p intersects the patient's
    # evidence (direct) or its hierarchy-expanded query (indirect).
    dm_set = set(dz_members)
    matched = []
    for (ed, ep), pv in edge_prov.items():
        if ed not in dm_set:
            continue
        if ep in ev_set:
            matched.append((ed, ep, "direct", pv))
        elif ep in q:
            matched.append((ed, ep, "hierarchy", pv))
    # de-dup by phenotype, prefer direct
    best = {}
    for dm, ep, kind, pv in matched:
        if ep not in best or (kind == "direct" and best[ep][1] != "direct"):
            best[ep] = (dm, kind, pv)

    print(f"\nMatched disease->phenotype edges (traced to source): {len(best)} phenotype(s)")
    for ep, (dm, kind, pv) in sorted(best.items(), key=lambda x: -idf.get(x[0], 0)):
        print(f"\n  [{kind}] phenotype {ep}  {cui_label(ep)}   (idf={idf.get(ep,0):.2f})")
        if dm != tcu:
            print(f"      via subtype {dm} {dz_names.get(dm, '')}")
        for pr in pv:
            src = pr["source"]; doc = pr.get("doc_id"); fnd = pr.get("finding")
            docs = pr.get("doc_ids")
            if src == "pubmed":
                extra = f" (+{len(docs)-1} more docs)" if docs else ""
                cite = f"PubMed {doc}{extra}" if doc else "PubMed (disease source set)"
            elif src == "genereviews":
                cite = f"GeneReviews {doc}"
            else:
                cite = f"StatPearls {doc}"
            print(f"        - {cite:32}  finding: \"{fnd}\"")


if __name__ == "__main__":
    main()
