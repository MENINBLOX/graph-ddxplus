#!/usr/bin/env python3
"""v26: v23 SOTA + alt-corpus + Statpearls IE edges.

New sources:
- pubmed_alt PF + anatomical (1,354 abstracts)
- Statpearls PF + anatomical (208 chunks from 52 disease-file pairs)

Strategy: add edges with low boost to avoid IDF dilution at @1.
"""
from __future__ import annotations
import sys, json, math, pickle, re, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v23_sota.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v26_combined.pkl"

SOURCES = [
    ("alt_pf", Path("pilot/data/pubmed_ddx_extra/edges_patient_focused_alt.jsonl"), "phenotype", 0.3),
    ("alt_anat", Path("pilot/data/pubmed_ddx_extra/edges_anatomical_alt.jsonl"), "phenotype", 0.3),
    ("sp_pf", Path("pilot/data/pubmed_ddx_extra/edges_patient_focused_statpearls.jsonl"), "phenotype", 0.5),
    ("sp_anat", Path("pilot/data/pubmed_ddx_extra/edges_anatomical_statpearls.jsonl"), "phenotype", 0.5),
]

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_in", default=str(GRAPH_IN))
    ap.add_argument("--graph_out", default=str(GRAPH_OUT))
    ap.add_argument("--only_new_phen", action="store_true",
                    help="only add edge if phen is new to v23 (avoid IDF dilution)")
    ap.add_argument("--no_merge_existing", action="store_true",
                    help="skip edges that already exist (no weight inflation)")
    args = ap.parse_args()

    print(f"Loading v23 SOTA from {args.graph_in}")
    G = pickle.load(open(args.graph_in, "rb"))
    print(f"  v23: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    existing_pairs = set()
    existing_phens = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))
            existing_phens.add(v)
    print(f"  existing (disease, phen) pairs: {len(existing_pairs):,}")
    print(f"  existing phen nodes: {len(existing_phens):,}")

    # Collect all targets
    all_edges = []  # (source_name, disease_cui, text, boost)
    for src, path, field, boost in SOURCES:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        edges = []
        with path.open() as f:
            for line in f:
                try: d = json.loads(line)
                except: continue
                if d.get("umls_cui") and d.get(field):
                    edges.append((src, d["umls_cui"], d[field], boost))
        all_edges.extend(edges)
        print(f"  [{src}] {len(edges):,} edges (boost={boost})")

    targets = set()
    for _, _, t, _ in all_edges:
        n = normalize(t); targets.add(n); targets.add(lemmatize(n))
    print(f"  Unique text variants: {len(targets):,}")

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
    print(f"  Resolved {len(str2cui):,} via MRCONSO ({time.time()-t0:.0f}s)")

    # Fallback: scispaCy phen-link cache
    scispacy_links = {}
    sp_path = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"
    if sp_path.exists():
        scispacy_links = json.load(open(sp_path))
        print(f"  scispaCy fallback: {len(scispacy_links):,} cached phens")

    # IDF baseline
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1

    # Aggregate per (source, dis, phen_cui)
    per_src = defaultdict(lambda: defaultdict(int))
    n_resolved = 0; n_via_scispacy = 0
    for src, dis, text, boost in all_edges:
        t = normalize(text)
        cuis = set()
        # MRCONSO direct match
        info = str2cui.get(t, str2cui.get(lemmatize(t)))
        if info: cuis.add(info[0])
        # scispaCy fallback (top-1 by score)
        sp = scispacy_links.get(text, scispacy_links.get(t))
        if sp:
            # Only take top match by score
            best = max(sp, key=lambda x: x[1]) if sp else None
            if best and best[1] > 0.85:
                cuis.add(best[0])
                if not info: n_via_scispacy += 1
        if not cuis: continue
        for target_cui in cuis:
            if target_cui == dis: continue
            per_src[src][(dis, target_cui, boost)] += 1
        n_resolved += 1
    print(f"  Resolved {n_resolved:,}/{len(all_edges):,} edge texts ({n_via_scispacy:,} via scispaCy)")

    # Add edges
    n_added = 0; n_merged = 0; n_skip = 0; n_new_phen = 0
    for src, edges_dict in per_src.items():
        s_added = 0; s_merged = 0; s_skip = 0
        for (dis, p, boost), co in edges_dict.items():
            full_freq[p] = full_freq.get(p, 0) + co
            w = math.log1p(co) * boost * (math.log(20000 / max(full_freq[p], 1)))
            if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
            is_new_phen = p not in existing_phens
            if is_new_phen:
                G.add_node(p, ntype="Phenotype", name=p, source=src)
                existing_phens.add(p)
                n_new_phen += 1
            if (dis, p) in existing_pairs:
                if args.no_merge_existing:
                    s_skip += 1; continue
                # merge weight
                cur = G[dis][p]
                if cur.get("etype") == "HAS_PHENOTYPE":
                    cur["weight"] = cur.get("weight", 0) + w
                    s_merged += 1
                    continue
            if args.only_new_phen and not is_new_phen:
                s_skip += 1
                continue
            G.add_edge(dis, p, etype="HAS_PHENOTYPE", weight=w, source=src)
            existing_pairs.add((dis, p))
            s_added += 1
        print(f"  [{src}] added {s_added:,} new, merged {s_merged:,}, skipped {s_skip:,}")
        n_added += s_added; n_merged += s_merged; n_skip += s_skip

    print(f"\nTotal: added {n_added:,} new edges, merged {n_merged:,}, skipped {n_skip:,}")
    print(f"  New phen nodes: {n_new_phen:,}")
    print(f"Final v26: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {args.graph_out}")
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
