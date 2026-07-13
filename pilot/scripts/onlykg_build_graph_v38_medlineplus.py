#!/usr/bin/env python3
"""v38: v28 + MedlinePlus patient-focused IE edges with UMLS Q-expansion.

Oracle test showed +10%p ceiling possible from KG content.
MedlinePlus has patient-oriented disease descriptions including risk factors,
history, common symptoms — different from PubMed academic abstracts.
"""
from __future__ import annotations
import sys, json, math, pickle, re, time, argparse
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from medkg_paths import MEDKG_ROOT, UMLS_DIR

GRAPH_IN = MEDKG_ROOT / "kg" / "onlykg_graph_v28_qexpand.pkl"
GRAPH_OUT = MEDKG_ROOT / "kg" / "onlykg_graph_v38_mlp.pkl"
VALUE_CUIS = MEDKG_ROOT / "kg" / "ddxplus_evidence_value_cuis.json"
SCISPACY_LINKS = MEDKG_ROOT / "kg" / "phenotype_scispacy_links.json"
PHEN_TO_Q = Path("pilot/data/phen_to_q_umls.json")

SOURCES = [
    ("mlp_pf", Path("pilot/data/pubmed_ddx_extra/edges_patient_focused_medlineplus.jsonl")),
    ("mlp_anat", Path("pilot/data/pubmed_ddx_extra/edges_anatomical_medlineplus.jsonl")),
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
    ap.add_argument("--boost", type=float, default=0.5)
    ap.add_argument("--expand_decay", type=float, default=0.5)
    args = ap.parse_args()

    print(f"Loading {args.graph_in}")
    G = pickle.load(open(args.graph_in, "rb"))
    print(f"  v28: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    Q = set()
    value_cuis = json.load(open(VALUE_CUIS))
    for ev_name, mapping in value_cuis.items():
        if isinstance(mapping, dict):
            for v in mapping.values():
                if isinstance(v, list): Q.update(v)

    phen_to_q = {k: set(v) for k, v in json.load(open(PHEN_TO_Q)).items()}

    # Collect (disease, text) edges from MedlinePlus IE outputs
    all_edges = []
    for src, path in SOURCES:
        if not path.exists():
            print(f"  [skip] {path}")
            continue
        cnt = 0
        with path.open() as f:
            for line in f:
                try: d = json.loads(line)
                except: continue
                if d.get("umls_cui") and d.get("phenotype"):
                    all_edges.append((src, d["umls_cui"], d["phenotype"]))
                    cnt += 1
        print(f"  [{src}] {cnt:,} edges")

    if not all_edges:
        print("No edges to add"); return

    targets = set()
    for _, _, t in all_edges:
        n = normalize(t); targets.add(n); targets.add(lemmatize(n))
    print(f"  Targets: {len(targets):,}")

    str2cui = {}
    print("Loading MRCONSO...")
    t0 = time.time()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15 or parts[1] != "ENG": continue
            text = parts[14]
            norm = normalize(text); lem = lemmatize(norm)
            for variant in (norm, lem):
                if variant not in targets: continue
                cui = parts[0]; sab = parts[11]
                prio = SAB_PRIORITY.get(sab, 99)
                existing = str2cui.get(variant)
                if existing is None or prio < existing[1]:
                    str2cui[variant] = (cui, prio)
    print(f"  MRCONSO: {len(str2cui):,} ({time.time()-t0:.0f}s)")

    scispacy = json.load(open(SCISPACY_LINKS)) if SCISPACY_LINKS.exists() else {}

    pair_count = defaultdict(int)
    for src, dis, text in all_edges:
        t = normalize(text)
        cui = None
        info = str2cui.get(t, str2cui.get(lemmatize(t)))
        if info: cui = info[0]
        if not cui:
            sp = scispacy.get(t, scispacy.get(lemmatize(t)))
            if sp:
                best = max(sp, key=lambda x: x[1])
                if best[1] > 0.85: cui = best[0]
        if not cui or cui == dis: continue
        pair_count[(src, dis, cui)] += 1

    existing = set((u,v) for u,v,e in G.edges(data=True) if e.get("etype")=="HAS_PHENOTYPE")
    full_freq = Counter()
    for u, v, ed in G.edges(data=True):
        if ed.get("etype") == "HAS_PHENOTYPE":
            full_freq[v] += 1

    n_direct = 0; n_expand = 0; n_merge = 0
    for (src, dis, p), co in pair_count.items():
        if p in Q:
            w = math.log1p(co) * args.boost * math.log(20000 / max(full_freq.get(p, 1), 1))
            full_freq[p] += co
            if (dis, p) in existing:
                G[dis][p][next(iter(G[dis][p]))]['weight'] = G[dis][p][next(iter(G[dis][p]))].get('weight', 0) + w
                n_merge += 1
                continue
            if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
            if p not in G: G.add_node(p, ntype="Phenotype", name=p, source=src)
            G.add_edge(dis, p, etype="HAS_PHENOTYPE", weight=w, source=src)
            existing.add((dis, p))
            n_direct += 1
        else:
            for q_cui in phen_to_q.get(p, set()):
                w = math.log1p(co) * args.boost * args.expand_decay * math.log(20000 / max(full_freq.get(q_cui, 1), 1))
                full_freq[q_cui] += co
                if (dis, q_cui) in existing:
                    G[dis][q_cui][next(iter(G[dis][q_cui]))]['weight'] = G[dis][q_cui][next(iter(G[dis][q_cui]))].get('weight', 0) + w
                    n_merge += 1
                    continue
                if dis not in G: G.add_node(dis, ntype="Disease", name=dis)
                if q_cui not in G: G.add_node(q_cui, ntype="Phenotype", name=q_cui, source=f"{src}_qexp")
                G.add_edge(dis, q_cui, etype="HAS_PHENOTYPE", weight=w, source=f"{src}_qexp")
                existing.add((dis, q_cui))
                n_expand += 1

    print(f"\nAdded: direct {n_direct:,}, qexpand {n_expand:,}, merged {n_merge:,}")
    print(f"Final v38: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    pickle.dump(G, open(args.graph_out, "wb"))
    print(f"Saved → {args.graph_out}")


if __name__ == "__main__":
    main()
