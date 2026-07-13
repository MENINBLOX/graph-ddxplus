#!/usr/bin/env python3
"""v25: add pubmed_alt-corpus anatomical + patient-focused IE edges to v23 SOTA.

pubmed_alt = 49 DDXPlus disease alt-search PubMed corpus (1,354 abstracts).
v23 used v2/v3 corpora; pubmed_alt is NEW content.

Edge sources:
- edges_anatomical_alt.jsonl: location|symptom format (694 edges)
- edges_patient_focused_alt.jsonl: lay vocab phens

Both resolve to CUIs via MRCONSO and add to v23 SOTA KG with tuned boost.
"""
from __future__ import annotations
import sys, json, math, pickle, re, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v23_sota.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v25_alt.pkl"
ALT_ANAT = Path("pilot/data/pubmed_ddx_extra/edges_anatomical_alt.jsonl")
ALT_PF = Path("pilot/data/pubmed_ddx_extra/edges_patient_focused_alt.jsonl")

PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM"]
SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}


def normalize(text):
    t = text.lower().strip()
    t = re.sub(r'[()\[\]{}]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def lemma_word(w):
    if len(w) <= 3: return w
    for suffix in ["'s", "ies", "sses", "ches", "ses", "es", "ed", "ing", "s"]:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            base = w[:-len(suffix)]
            if suffix == "ies": return base + "y"
            return base
    return w


def lemmatize(t): return " ".join(lemma_word(w) for w in t.split())


def load_edges_jsonl(path, text_field):
    edges = []
    if not path.exists(): return edges
    with path.open() as f:
        for line in f:
            try: d = json.loads(line)
            except: continue
            if d.get("umls_cui") and d.get(text_field):
                edges.append((d["umls_cui"], d[text_field]))
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_in", default=str(GRAPH_IN))
    ap.add_argument("--graph_out", default=str(GRAPH_OUT))
    ap.add_argument("--anat_boost", type=float, default=2.7, help="boost for alt anatomical edges (mimic v23 anat)")
    ap.add_argument("--pf_boost", type=float, default=0.3, help="boost for alt patient-focused edges")
    args = ap.parse_args()

    print(f"Loading v23 SOTA from {args.graph_in}")
    G = pickle.load(open(args.graph_in, "rb"))
    print(f"  v23: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    anat_edges = load_edges_jsonl(ALT_ANAT, "phenotype")
    pf_edges = load_edges_jsonl(ALT_PF, "phenotype")
    print(f"  alt anatomical edges: {len(anat_edges):,}")
    print(f"  alt patient-focused edges: {len(pf_edges):,}")

    # Combined target set
    targets = set()
    for _, t in anat_edges + pf_edges:
        n = normalize(t)
        targets.add(n); targets.add(lemmatize(n))
    print(f"  Unique phen variants: {len(targets):,}")

    # MRCONSO resolution
    str2cui = {}
    t0 = time.time()
    print("Loading MRCONSO...")
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text)
            lem = lemmatize(norm)
            for variant in (norm, lem):
                if variant not in targets: continue
                cui = parts[0]; sab = parts[11]
                prio = SAB_PRIORITY.get(sab, 99)
                existing = str2cui.get(variant)
                if existing is None or prio < existing[1]:
                    str2cui[variant] = (cui, prio)
    print(f"  Resolved {len(str2cui):,} strings ({time.time()-t0:.0f}s)")

    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    # IDF from full KG
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1

    def add_edges(edges, boost, source_name):
        nonlocal G
        pair_count = defaultdict(int)
        for dis, text in edges:
            t = normalize(text)
            target_cui = str2cui.get(t, str2cui.get(lemmatize(t), (None, 99)))[0]
            if not target_cui or target_cui == dis: continue
            pair_count[(dis, target_cui)] += 1
        for (dis, p), co in pair_count.items():
            full_freq[p] = full_freq.get(p, 0) + co
        N = max(20000, len({v for _,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE"}))
        idf = {p: math.log(N / max(c, 1)) for p, c in full_freq.items()}

        added = 0; merged = 0; new_phens = 0
        for (dis, p), co in pair_count.items():
            w = math.log1p(co) * boost * idf.get(p, 1.0)
            if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
            if p not in G:
                G.add_node(p, ntype="Phenotype", name=p, source=source_name)
                new_phens += 1
            if (dis, p) in existing_pairs:
                cur = G[dis][p]
                if cur.get("etype") == "HAS_PHENOTYPE":
                    cur["weight"] = cur.get("weight", 0) + w
                    merged += 1
                    continue
            G.add_edge(dis, p, etype="HAS_PHENOTYPE", weight=w, source=source_name)
            existing_pairs.add((dis, p))
            added += 1
        print(f"  [{source_name}] added {added:,} new, merged {merged:,}, new phen nodes {new_phens:,}")

    add_edges(anat_edges, args.anat_boost, "alt_anatomical")
    add_edges(pf_edges, args.pf_boost, "alt_patient_focused")

    print(f"\nFinal v25: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"\nSaving to {args.graph_out}")
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
