#!/usr/bin/env python3
"""Build a UMLS-only KG from MRREL semantic relations (noise-cleaned).

Disease→symptom edges from RELA fields:
  has_manifestation, manifestation_of (symmetric)
  may_be_finding_of_disease
  associated_with
  due_to / cause_of
  has_finding_site (anatomical)

Output: /mnt/medkg/kg/onlykg_graph_v50_umls_only.pkl

This is a quick empirical baseline to compare against PubMed-IE-derived KGs.
"""
from __future__ import annotations
import pickle, time, sys
from collections import defaultdict, Counter
from pathlib import Path
import networkx as nx

MRREL = "/windows/data/umls_subset/MRREL.RRF"
MRSTY = "/windows/data/umls_subset/MRSTY.RRF"
OUT = "/mnt/medkg/kg/onlykg_graph_v50_umls_only.pkl"

# Disease-symptom relations (RELA values).
# Direction: edge will be added as disease→symptom regardless of RELA orientation.
# {RELA: direction} where direction = "C1_is_disease" means C1 is the disease.
RELA_DIRECTIONS = {
    # disease has symptom/finding/morphology
    "has_manifestation": "C1_is_disease",          # disease has-manifestation symptom
    "manifestation_of": "C2_is_disease",            # symptom manifestation-of disease
    "may_be_finding_of_disease": "C2_is_disease",   # finding may-be-finding-of disease
    "has_associated_morphology": "C1_is_disease",
    "associated_morphology_of": "C2_is_disease",
    "has_pathological_process": "C1_is_disease",
    "associated_with": None,  # bidirectional, include both directions
    "associated_finding_of": "C2_is_disease",
    "due_to": "C1_is_disease",         # disease due-to symptom/condition (use disease side)
    "cause_of": "C2_is_disease",       # finding cause-of disease
    "has_clinical_course": "C1_is_disease",
    "has_finding_site": "C1_is_disease",  # disease has-finding-site location
    "finding_site_of": "C2_is_disease",
}


def main():
    t0 = time.time()

    # Disease-like TUIs (broader than just T047)
    DISEASE_TUIS = {
        "T047",  # Disease or Syndrome
        "T191",  # Neoplastic Process
        "T046",  # Pathologic Function
        "T019",  # Congenital Abnormality
        "T020",  # Acquired Abnormality
        "T037",  # Injury or Poisoning
        "T048",  # Mental or Behavioral Dysfunction
        "T049",  # Cell or Molecular Dysfunction
        "T190",  # Anatomical Abnormality
    }

    # Symptom-like TUIs (findings, signs, observations)
    SYMPTOM_TUIS = {
        "T184",  # Sign or Symptom
        "T033",  # Finding
        "T034",  # Laboratory or Test Result
        "T040",  # Organism Function (sometimes used for vitals)
        "T023",  # Body Part, Organ, or Organ Component (for finding_site)
        "T029",  # Body Location or Region
        "T030",  # Body Space or Junction
    }

    # Load TUIs
    print("Loading MRSTY...", flush=True)
    cui_tuis = defaultdict(set)
    with open(MRSTY) as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 4: continue
            cui_tuis[parts[0]].add(parts[1])

    disease_cuis = {c for c, t in cui_tuis.items() if t & DISEASE_TUIS}
    symptom_cuis = {c for c, t in cui_tuis.items() if t & SYMPTOM_TUIS}
    print(f"  diseases: {len(disease_cuis):,}, symptoms: {len(symptom_cuis):,}", flush=True)

    # Parse MRREL
    print("Building KG from MRREL...", flush=True)
    G = nx.MultiDiGraph()
    edges_added = 0
    rela_stats = Counter()
    n = 0
    with open(MRREL) as f:
        for line in f:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n//1_000_000}M rows, edges added: {edges_added:,}", flush=True)
            parts = line.split("|")
            if len(parts) < 8: continue
            c1, rel, c2, rela = parts[0], parts[3], parts[4], parts[7]
            if c1 == c2: continue
            if rela not in RELA_DIRECTIONS: continue

            # Determine which side is disease
            direction = RELA_DIRECTIONS[rela]
            if direction == "C1_is_disease":
                d_cui, s_cui = c1, c2
            elif direction == "C2_is_disease":
                d_cui, s_cui = c2, c1
            else:  # bidirectional
                # For bidirectional, add both as disease→symptom if applicable
                pairs = []
                if c1 in disease_cuis: pairs.append((c1, c2))
                if c2 in disease_cuis: pairs.append((c2, c1))
                for d_cui, s_cui in pairs:
                    G.add_edge(d_cui, s_cui, etype="HAS_PHENOTYPE",
                               weight=1.0, source="umls_mrrel", rela=rela)
                    edges_added += 1
                    rela_stats[rela] += 1
                continue

            # Ensure disease side is actually a disease (filter noise)
            if d_cui not in disease_cuis: continue

            G.add_edge(d_cui, s_cui, etype="HAS_PHENOTYPE",
                       weight=1.0, source="umls_mrrel", rela=rela)
            edges_added += 1
            rela_stats[rela] += 1

    print(f"\n  total rows scanned: {n:,}", flush=True)
    print(f"  edges added: {edges_added:,}", flush=True)
    print(f"  unique nodes: {G.number_of_nodes():,}", flush=True)
    print(f"  unique edges: {G.number_of_edges():,}", flush=True)
    print(f"  edges by RELA:", flush=True)
    for r, c in rela_stats.most_common():
        print(f"    {r}: {c:,}", flush=True)

    print(f"\nSaving: {OUT}", flush=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(G, f)
    print(f"Done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
