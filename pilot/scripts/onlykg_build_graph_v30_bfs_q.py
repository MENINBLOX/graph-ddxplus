#!/usr/bin/env python3
"""v30: v28 UMLS Q-expand + BFS edges (Q-filtered).

BFS edges go through:
1. Resolve dst_text → CUI via MRCONSO + scispaCy fallback
2. Filter: ONLY add edges where target_cui is in Q (questionnaire universe)
3. Weight: log(1+co) * boost / sqrt(depth+1)
4. Only add to 49 DDXPlus diseases (avoid universal expansion)
"""
from __future__ import annotations
import sys, json, math, pickle, re, time, argparse
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v28_qexpand.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v30_bfs_q.pkl"
BFS_EDGES = Path("pilot/data/exhaustive_ie/edges_exhaustive_ie.jsonl")
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
SCISPACY_LINKS = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"

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


def weight_at_depth(d):
    return 1.0 / math.sqrt(d + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_in", default=str(GRAPH_IN))
    ap.add_argument("--graph_out", default=str(GRAPH_OUT))
    ap.add_argument("--boost", type=float, default=1.0)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_freq", type=int, default=2, help="min (disease, dst_cui) co-occurrence")
    ap.add_argument("--q_only", action="store_true", default=True, help="only add edges to Q-CUIs")
    ap.add_argument("--use_scispacy", action="store_true", default=True)
    args = ap.parse_args()

    print(f"Loading v28 from {args.graph_in}")
    G = pickle.load(open(args.graph_in, "rb"))
    print(f"  v28: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    Q = set()
    value_cuis = json.load(open(VALUE_CUIS))
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)
    print(f"  Q: {len(Q):,}")

    # Aggregate BFS edges
    pair_freq = defaultdict(int)
    pair_min_depth = {}
    n_raw = 0
    with open(args.bfs_edges if hasattr(args, 'bfs_edges') else BFS_EDGES) as f:
        for line in f:
            try: d = json.loads(line)
            except: continue
            n_raw += 1
            if d['depth'] > args.max_depth: continue
            t = normalize(d['dst_text'])
            if not t or len(t) > 80: continue
            key = (d['disease'], t)
            pair_freq[key] += 1
            cur = pair_min_depth.get(key)
            if cur is None or d['depth'] < cur:
                pair_min_depth[key] = d['depth']
    print(f"  BFS raw edges: {n_raw:,}")
    print(f"  Unique (disease, text) pairs: {len(pair_freq):,}")

    # Filter by min_freq
    filtered_pairs = {k: v for k, v in pair_freq.items() if v >= args.min_freq}
    print(f"  After min_freq={args.min_freq}: {len(filtered_pairs):,}")

    # Targets for resolution
    targets = set()
    for (d, t) in filtered_pairs:
        targets.add(t); targets.add(lemmatize(t))
    print(f"  Resolution targets: {len(targets):,}")

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
    print(f"  MRCONSO resolved: {len(str2cui):,} ({time.time()-t0:.0f}s)")

    # scispaCy fallback
    scispacy = json.load(open(SCISPACY_LINKS)) if SCISPACY_LINKS.exists() and args.use_scispacy else {}
    print(f"  scispaCy fallback: {len(scispacy):,}")

    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    # Resolve and aggregate
    resolved = defaultdict(lambda: {"co": 0, "min_depth": 999})
    n_q = 0; n_non_q = 0
    for (dis, t), co in filtered_pairs.items():
        cui = None
        info = str2cui.get(t, str2cui.get(lemmatize(t)))
        if info: cui = info[0]
        if not cui:
            sp = scispacy.get(t, scispacy.get(lemmatize(t)))
            if sp:
                best = max(sp, key=lambda x: x[1])
                if best[1] > 0.85: cui = best[0]
        if not cui: continue
        if cui == dis: continue
        if args.q_only and cui not in Q:
            n_non_q += 1
            continue
        n_q += 1
        key = (dis, cui)
        resolved[key]["co"] += co
        resolved[key]["min_depth"] = min(resolved[key]["min_depth"], pair_min_depth[(dis, t)])

    print(f"  Q-resolved: {n_q:,}, skipped non-Q: {n_non_q:,}")

    # IDF baseline
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1
    for (dis, p), info in resolved.items():
        full_freq[p] += info["co"]

    N = max(20000, len({v for _,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE"}))
    idf = {p: math.log(N / max(c, 1)) for p, c in full_freq.items()}

    added = 0; merged = 0; new_phen = 0
    for (dis, p), info in resolved.items():
        co = info["co"]; d = info["min_depth"]
        w = math.log1p(co) * weight_at_depth(d) * idf.get(p, 1.0) * args.boost
        if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
        if p not in G:
            G.add_node(p, ntype="Phenotype", name=p, source="v30_bfs")
            new_phen += 1
        if (dis, p) in existing_pairs:
            cur = G[dis][p]
            if cur.get("etype") == "HAS_PHENOTYPE":
                cur["weight"] = cur.get("weight", 0) + w
                merged += 1
                continue
        G.add_edge(dis, p, etype="HAS_PHENOTYPE", weight=w, source="bfs_q")
        existing_pairs.add((dis, p))
        added += 1

    print(f"\nAdded {added:,} new edges, merged {merged:,}, new phen nodes {new_phen:,}")
    print(f"Final v30: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"\nSaving to {args.graph_out}")
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
