#!/usr/bin/env python3
"""v24: integrate Phase 3 BFS exhaustive edges into v23 SOTA.

Input: pilot/data/exhaustive_ie/edges_exhaustive_ie.jsonl
  records = {"disease": <cui>, "src": <cui>, "dst_text": <str>, "depth": <int>, "pmid": <str>}

Algorithm:
1. Load v23 SOTA KG
2. Aggregate per-(disease, dst_text) co-occurrence frequency and min_depth
3. Resolve dst_text → CUI via MRCONSO (English, preferred SAB)
4. Add edges with weight = log1p(co) * weight_at_depth(min_depth) * idf
   weight_at_depth(d) = 1 / sqrt(d + 1)  -- deeper = lower contribution
5. Save as onlykg_graph_v24_bfs.pkl
"""
from __future__ import annotations
import sys, json, math, pickle, re, time
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v23_sota.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v24_bfs.pkl"
BFS_EDGES = Path("pilot/data/exhaustive_ie/edges_exhaustive_ie.jsonl")

PREFERRED_SABS = ["HPO", "SNOMEDCT_US", "MSH", "MEDCIN", "NCI", "ICD10CM"]
SAB_PRIORITY = {s: i for i, s in enumerate(PREFERRED_SABS)}

# Accept only medical semantic types (DISO, FNDG, SOSY, BODY)
ACCEPT_TUI = {
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T047",  # Disease or Syndrome
    "T046",  # Pathologic Function
    "T037",  # Injury or Poisoning
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction
    "T023",  # Body Part, Organ, or Organ Component
    "T190",  # Anatomical Abnormality
    "T191",  # Neoplastic Process
}


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfs_edges", default=str(BFS_EDGES))
    ap.add_argument("--graph_in", default=str(GRAPH_IN))
    ap.add_argument("--graph_out", default=str(GRAPH_OUT))
    ap.add_argument("--boost", type=float, default=1.0, help="overall weight boost for BFS edges")
    ap.add_argument("--max_depth", type=int, default=8, help="cap edges by depth")
    ap.add_argument("--min_freq", type=int, default=2, help="minimum (disease, dst) co-occurrence to include")
    ap.add_argument("--tui_filter", action="store_true", help="restrict dst to medical TUIs")
    args = ap.parse_args()

    print(f"Loading v23 SOTA from {args.graph_in}")
    G = pickle.load(open(args.graph_in, "rb"))
    print(f"  v23: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nLoading BFS edges from {args.bfs_edges}")
    pair_freq = defaultdict(int)  # (disease, dst_text_norm) → count
    pair_min_depth = {}            # (disease, dst_text_norm) → min depth
    n_raw = 0
    with open(args.bfs_edges) as f:
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
    print(f"  Raw edges: {n_raw:,}")
    print(f"  Unique (disease, text) pairs: {len(pair_freq):,}")

    targets = set()
    for (d, t) in pair_freq.keys():
        if pair_freq[(d, t)] < args.min_freq: continue
        targets.add(t); targets.add(lemmatize(t))
    print(f"  Resolution targets: {len(targets):,}")

    # MRCONSO resolution
    str2cui = {}
    cui2tui = {}
    if args.tui_filter:
        print("\nLoading MRSTY for TUI filter...")
        t0 = time.time()
        with open(UMLS_DIR / "MRSTY.RRF") as f:
            for line in f:
                parts = line.split("|")
                if len(parts) < 4: continue
                tui = parts[1]
                if tui in ACCEPT_TUI:
                    cui2tui[parts[0]] = tui
        print(f"  MRSTY: {len(cui2tui):,} CUIs in target TUIs ({time.time()-t0:.0f}s)")

    print("\nLoading MRCONSO index...")
    t0 = time.time()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text)
            if norm not in targets:
                lem = lemmatize(norm)
                if lem not in targets: continue
                norm = lem
            cui = parts[0]; sab = parts[11]
            if args.tui_filter and cui not in cui2tui: continue
            prio = SAB_PRIORITY.get(sab, 99)
            existing = str2cui.get(norm)
            if existing is None or prio < existing[1]:
                str2cui[norm] = (cui, prio)
    print(f"  Resolved {len(str2cui):,}/{len(targets):,} strings ({time.time()-t0:.0f}s)")

    existing_pairs = set()
    for u, v, e in G.edges(data=True):
        if e.get("etype") == "HAS_PHENOTYPE":
            existing_pairs.add((u, v))

    # IDF based on combined phen frequency
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1

    # Resolve and add edges
    resolved_pairs = defaultdict(lambda: {"co": 0, "min_depth": 999})
    for (dis, t), co in pair_freq.items():
        if co < args.min_freq: continue
        target_cui = None
        if t in str2cui: target_cui = str2cui[t][0]
        elif lemmatize(t) in str2cui: target_cui = str2cui[lemmatize(t)][0]
        if not target_cui: continue
        if target_cui == dis: continue
        key = (dis, target_cui)
        resolved_pairs[key]["co"] += co
        resolved_pairs[key]["min_depth"] = min(resolved_pairs[key]["min_depth"], pair_min_depth[(dis, t)])
        full_freq[target_cui] += co

    print(f"  Resolved CUI pairs: {len(resolved_pairs):,}")

    # IDF
    N = max(20000, len({v for u,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE"}))
    idf = {p: math.log(N / max(c, 1)) for p, c in full_freq.items()}

    added = 0; merged = 0; new_phens_added = 0
    for (dis, target_cui), info in resolved_pairs.items():
        co = info["co"]
        min_depth = info["min_depth"]
        w = math.log1p(co) * weight_at_depth(min_depth) * idf.get(target_cui, 1.0) * args.boost

        if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
        if target_cui not in G:
            G.add_node(target_cui, ntype="Phenotype", name=target_cui, source="v24_bfs")
            new_phens_added += 1

        if (dis, target_cui) in existing_pairs:
            # Merge weight with existing
            cur = G[dis][target_cui]
            if cur.get("etype") == "HAS_PHENOTYPE":
                cur["weight"] = cur.get("weight", 0) + w
                merged += 1
                continue
        G.add_edge(dis, target_cui, etype="HAS_PHENOTYPE", weight=w, source="bfs_exhaustive")
        existing_pairs.add((dis, target_cui))
        added += 1

    print(f"\nAdded {added:,} new edges, merged {merged:,} into existing")
    print(f"  New phen nodes: {new_phens_added:,}")
    print(f"Final v24: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print(f"\nSaving to {args.graph_out}")
    with open(args.graph_out, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()
