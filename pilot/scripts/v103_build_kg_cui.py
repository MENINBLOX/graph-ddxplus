#!/usr/bin/env python3
"""v103 CUI-grounded property graph KG builder.

v103_build_kg.py와 차이: phenotype NAME 노드 대신 UMLS CUI 노드 사용.
환자 evidence가 CUI이므로 매칭하려면 KG phenotype도 CUI여야 함.

Pipeline:
1. v103 per-disease aggregated phenotypes (name + attribute distribution)
2. Phenotype name → UMLS CUI 매핑 (v92 enhanced multi-substring strategy)
3. Disease CUI → Phenotype CUI edges with attribute distributions
4. Property graph (networkx MultiDiGraph) 저장

학술적 grounding: phenotype name은 UMLS standard CUI로 정규화 (reproducible).
"""
from __future__ import annotations
import json, pickle, argparse, glob, re
from pathlib import Path
from collections import defaultdict
import networkx as nx


# v92-style qualifier stripping
PREFIX_QUAL = re.compile(
    r"^(severe|mild|moderate|acute|chronic|sudden onset of|sudden|recurrent|"
    r"persistent|intermittent|episodic|progressive|transient|early|late|"
    r"unexplained|new onset|worsening|generalized|localized|diffuse|focal|"
    r"left|right|bilateral|unilateral|a |an |the )\s+", re.IGNORECASE)
UNCLOSED_PAREN = re.compile(r"\s*\([^)]*$")


def normalize_phen(s):
    s = s.lower().strip().rstrip(".,;:")
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)  # closed paren
    s = UNCLOSED_PAREN.sub("", s)            # unclosed paren
    s = re.sub(r"\b(e\.g\.|i\.e\.|etc\.).*$", "", s)
    if ',' in s:
        first = s.split(',')[0].strip()
        if len(first) >= 3: s = first
    s = re.sub(r"\s+", " ", s).strip()
    return s


def gen_candidates(name):
    base = normalize_phen(name)
    cands = [base]
    prev, cur = None, base
    while cur != prev:
        prev = cur
        cur = PREFIX_QUAL.sub("", cur).strip()
        if cur and cur != prev: cands.append(cur)
    # NOTE: contiguous n-gram sub-phrase matching was tested (mapping 47%->57%)
    # but it REGRESSED DDXPlus @1 (34%->28%): recovered phrases are generic,
    # low-IDF (e.g. "lymph nodes") and dilute top-1 discrimination. Per principle
    # #8 (GTPA@1 only) the exact + qualifier-stripped mapping is retained.
    # The @1 bottleneck is discrimination, not mapping recall (@10 already 79%).
    seen = set(); out = []
    for c in cands:
        if c and len(c) >= 3 and c not in seen:
            seen.add(c); out.append(c)
    return out


def load_umls_phen_strings(mrconso, mrsty):
    phen_tuis = {"T033","T184","T046","T037","T048","T049","T191","T190",
                 "T022","T023","T029","T030","T047","T019","T020"}
    phen_cuis = set()
    with open(mrsty) as f:
        for line in f:
            p = line.split("|")
            if len(p) >= 2 and p[1] in phen_tuis: phen_cuis.add(p[0])
    print(f"  Phen/disease CUIs: {len(phen_cuis):,}", flush=True)
    str2cui = {}
    n = 0
    with open(mrconso) as f:
        for line in f:
            n += 1
            p = line.split("|")
            if len(p) < 15 or p[1] != "ENG": continue
            c = p[0]
            if c not in phen_cuis: continue
            s = p[14].strip().lower()
            if not s or len(s) < 3 or len(s) > 80: continue
            s = re.sub(r"\s+", " ", s)
            if s not in str2cui:  # first (preferred) wins
                str2cui[s] = c
            if n % 3_000_000 == 0:
                print(f"    {n//1_000_000}M lines, {len(str2cui):,} strings", flush=True)
    print(f"  Done: {len(str2cui):,} strings", flush=True)
    return str2cui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="pilot/data/cache/v103strict_per_disease")
    ap.add_argument("--out", default="pilot/data/onlykg_graph_v103strict.pkl")
    ap.add_argument("--mrconso", default="/windows/data/umls_subset/MRCONSO.RRF")
    ap.add_argument("--mrsty", default="/windows/data/umls_subset/MRSTY.RRF")
    args = ap.parse_args()

    print("Loading UMLS phenotype strings...", flush=True)
    str2cui = load_umls_phen_strings(args.mrconso, args.mrsty)

    G = nx.MultiDiGraph()
    n_dis = 0; n_edges = 0; n_mapped = 0; n_total_phen = 0
    files = glob.glob(f"{args.in_dir}/*.json")
    print(f"Processing {len(files)} disease files...", flush=True)

    for path in files:
        try:
            data = json.load(open(path))
        except: continue
        dcui = data["cui"]; dname = data["disease"]
        agg = data.get("aggregated", {})
        if not agg: continue
        G.add_node(dcui, ntype="disease", name=dname)
        n_dis += 1
        for phen_name, pd in agg.items():
            n_total_phen += 1
            # Map phenotype name → CUI
            pcui = None
            for cand in gen_candidates(phen_name):
                if cand in str2cui:
                    pcui = str2cui[cand]; break
            if not pcui: continue  # unmappable
            if pcui == dcui: continue  # self-loop
            n_mapped += 1
            if pcui not in G:
                G.add_node(pcui, ntype="phenotype", name=phen_name)
            G.add_edge(dcui, pcui, etype="HAS_PHENOTYPE",
                       n_mentions=pd["n_mentions"],
                       frequency=pd["frequency_in_abstracts"],
                       location_dist=pd["location_dist"],
                       severity_dist=pd["severity_dist"],
                       onset_dist=pd["onset_dist"],
                       character_dist=pd["character_dist"],
                       phen_name=phen_name)
            n_edges += 1

    print(f"\nv103-strict KG: {n_dis} diseases, {n_edges} edges, {G.number_of_nodes()} nodes", flush=True)
    print(f"  Phenotype name → CUI mapped: {n_mapped}/{n_total_phen} ({100*n_mapped/max(n_total_phen,1):.1f}%)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(G, open(args.out, "wb"))
    print(f"  Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
