#!/usr/bin/env python3
"""v33: v28 + BFS edges with UMLS Q-expansion.

For each BFS (disease, dst_text, depth) edge:
1. Resolve dst_text → CUI (MRCONSO + scispaCy)
2. If CUI in Q: add direct (disease, CUI) edge
3. Else: expand via UMLS to Q-CUIs, add (disease, Q') edges (decay further)
"""
from __future__ import annotations
import sys, json, math, pickle, re, time, argparse
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v28_qexpand.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v33_bfs_qexpand.pkl"
BFS_EDGES = Path("pilot/data/exhaustive_ie/edges_exhaustive_ie.jsonl")
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
SCISPACY_LINKS = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"
PHEN_TO_Q = Path("pilot/data/phen_to_q_umls_broad.json")  # broader: 5916 phens

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
    ap.add_argument("--boost", type=float, default=0.3)
    ap.add_argument("--expand_decay", type=float, default=0.5)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--min_freq", type=int, default=3)
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

    # Phen→Q expansion map
    phen_to_q = {k: set(v) for k, v in json.load(open(PHEN_TO_Q)).items()}
    print(f"  phen→Q map: {len(phen_to_q):,}")

    # Aggregate BFS edges per (disease, text)
    pair_freq = defaultdict(int)
    pair_min_depth = {}
    n_raw = 0
    with open(BFS_EDGES) as f:
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
    print(f"  BFS raw: {n_raw:,} edges, {len(pair_freq):,} unique pairs")

    filtered = {k: v for k, v in pair_freq.items() if v >= args.min_freq}
    print(f"  After min_freq={args.min_freq}: {len(filtered):,}")

    targets = set()
    for (d, t) in filtered:
        targets.add(t); targets.add(lemmatize(t))
    print(f"  Resolution targets: {len(targets):,}")

    # MRCONSO
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
    print(f"  MRCONSO: {len(str2cui):,} ({time.time()-t0:.0f}s)")

    scispacy = json.load(open(SCISPACY_LINKS)) if SCISPACY_LINKS.exists() else {}
    print(f"  scispaCy: {len(scispacy):,}")

    existing = set((u,v) for u,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")

    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1

    n_direct = 0; n_expand = 0; merged = 0; n_unresolved = 0
    for (dis, t), co in filtered.items():
        d = pair_min_depth[(dis, t)]
        # Resolve
        cui = None
        info = str2cui.get(t, str2cui.get(lemmatize(t)))
        if info: cui = info[0]
        if not cui:
            sp = scispacy.get(t, scispacy.get(lemmatize(t)))
            if sp:
                best = max(sp, key=lambda x: x[1])
                if best[1] > 0.85: cui = best[0]
        if not cui or cui == dis:
            n_unresolved += 1
            continue

        base_w = math.log1p(co) * weight_at_depth(d) * args.boost

        if cui in Q:
            # Direct
            target_cui = cui
            w = base_w * math.log(20000 / max(full_freq.get(target_cui, 1), 1))
            full_freq[target_cui] += co
            if (dis, target_cui) in existing:
                cur = G[dis][target_cui]
                if cur.get("etype") == "HAS_PHENOTYPE":
                    cur["weight"] = cur.get("weight", 0) + w
                    merged += 1
                    continue
            if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
            if target_cui not in G: G.add_node(target_cui, ntype="Phenotype", name=target_cui, source="v33_bfs")
            G.add_edge(dis, target_cui, etype="HAS_PHENOTYPE", weight=w, source="bfs_direct")
            existing.add((dis, target_cui))
            n_direct += 1
        else:
            # Expand via UMLS
            related_q = phen_to_q.get(cui, set())
            for q_cui in related_q:
                w = base_w * args.expand_decay * math.log(20000 / max(full_freq.get(q_cui, 1), 1))
                full_freq[q_cui] += co
                if (dis, q_cui) in existing:
                    cur = G[dis][q_cui]
                    if cur.get("etype") == "HAS_PHENOTYPE":
                        cur["weight"] = cur.get("weight", 0) + w
                        merged += 1
                        continue
                if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
                if q_cui not in G: G.add_node(q_cui, ntype="Phenotype", name=q_cui, source="v33_bfs_qexpand")
                G.add_edge(dis, q_cui, etype="HAS_PHENOTYPE", weight=w, source="bfs_qexpand")
                existing.add((dis, q_cui))
                n_expand += 1

    print(f"\nResolved: direct {n_direct:,}, qexpand {n_expand:,}, merged {merged:,}, unresolved {n_unresolved:,}")
    print(f"\nFinal v33: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Saving to {args.graph_out}")
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
