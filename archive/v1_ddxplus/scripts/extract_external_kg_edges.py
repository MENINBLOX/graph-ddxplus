from __future__ import annotations

"""Extract disease-symptom edges from Hetionet, PrimeKG, and MedGen,
mapping all IDs to UMLS CUIs for DDXPlus benchmarking.

Mapping strategies:
- Hetionet: DOID->CUI via HumanDO.obo xrefs + name fallback, MeSH->CUI via MRCONSO
- PrimeKG: disease name->CUI via MRCONSO name matching, HPO->CUI via MRCONSO
- MedGen: Already has CUIs, filter for DDXPlus overlap
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("/home/max/Graph-DDXPlus")
DATA = BASE / "data"


def load_ddxplus_cuis() -> tuple:
    """Load DDXPlus disease and symptom CUI sets.

    Returns:
        (dis_cuis, sym_cuis, dis_cui_to_name, sym_cui_to_name)
    """
    with open(DATA / "ddxplus/disease_umls_mapping.json") as f:
        disease_map = json.load(f)

    dis_cuis: set = set()
    dis_cui_to_name: dict = {}
    for name, info in disease_map["mapping"].items():
        cui = info["umls_cui"]
        dis_cuis.add(cui)
        dis_cui_to_name[cui] = name

    with open(DATA / "ddxplus/umls_mapping.json") as f:
        sym_map = json.load(f)

    sym_cuis: set = set()
    sym_cui_to_name: dict = {}
    for key, info in sym_map["mapping"].items():
        cui = info["cui"]
        sym_cuis.add(cui)
        sym_cui_to_name[cui] = info["name"]

    print(f"DDXPlus: {len(dis_cuis)} disease CUIs, {len(sym_cuis)} symptom CUIs")
    return dis_cuis, sym_cuis, dis_cui_to_name, sym_cui_to_name


def build_doid_to_cui_and_names() -> tuple:
    """Parse HumanDO.obo to extract DOID -> UMLS CUI mappings and DOID -> name."""
    obo_path = DATA / "external_kg/HumanDO.obo"
    doid_to_cuis: dict = defaultdict(set)
    doid_to_name: dict = {}

    current_id = None
    with open(obo_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("id: DOID:"):
                current_id = line.split("id: ")[1]
            elif line.startswith("name: ") and current_id:
                doid_to_name[current_id] = line.split("name: ")[1].lower().strip()
            elif line.startswith("xref: UMLS_CUI:") and current_id:
                cui = line.split("xref: UMLS_CUI:")[1].strip()
                doid_to_cuis[current_id].add(cui)
            elif line == "[Term]" or line == "[Typedef]":
                current_id = None

    print(f"DOID->CUI: {len(doid_to_cuis)} DOIDs mapped to CUIs")
    print(f"DOID->name: {len(doid_to_name)} DOIDs with names")
    return doid_to_cuis, doid_to_name


def build_mrconso_mappings(dis_cuis: set) -> tuple:
    """Single pass over MRCONSO to build all needed mappings.

    Returns:
        (mesh_to_cuis, hpo_to_cuis, dis_cui_names)
    """
    mesh_to_cuis: dict = defaultdict(set)
    hpo_to_cuis: dict = defaultdict(set)
    dis_cui_names: dict = defaultdict(set)  # CUI -> set of name variants

    with open(DATA / "umls_subset/MRCONSO.RRF") as f:
        for line in f:
            fields = line.strip().split("|")
            if len(fields) < 15:
                continue
            cui = fields[0]
            sab = fields[11]
            code = fields[13]
            name = fields[14]

            if sab == "MSH" and code.startswith("D"):
                mesh_to_cuis[code].add(cui)
            elif sab == "HPO":
                hpo_to_cuis[code].add(cui)

            if cui in dis_cuis:
                dis_cui_names[cui].add(name.lower().strip())

    print(f"MRCONSO single pass: MeSH={len(mesh_to_cuis)}, HPO={len(hpo_to_cuis)}, "
          f"disease name sets={len(dis_cui_names)}")
    return mesh_to_cuis, hpo_to_cuis, dis_cui_names


def build_name_to_cui(dis_cuis: set, dis_cui_to_name: dict, dis_cui_names: dict) -> dict:
    """Build disease name -> DDXPlus CUI lookup from MRCONSO names + DDXPlus names."""
    # Add DDXPlus disease names
    for cui, name in dis_cui_to_name.items():
        dis_cui_names[cui].add(name.lower().strip())

    name_to_cui: dict = {}
    for cui, names in dis_cui_names.items():
        for name in names:
            name_to_cui[name] = cui

    print(f"Name->CUI: {len(name_to_cui)} disease name variants for {len(dis_cuis)} DDXPlus diseases")
    return name_to_cui


def extract_hetionet(
    dis_cuis: set,
    sym_cuis: set,
    dis_cui_to_name: dict,
    sym_cui_to_name: dict,
    doid_to_cuis: dict,
    doid_to_name: dict,
    mesh_to_cuis: dict,
    name_to_cui: dict,
) -> list:
    """Extract disease-symptom edges from Hetionet.

    Uses two strategies for disease mapping:
    1. DOID -> CUI via HumanDO.obo xrefs
    2. DOID name -> DDXPlus disease name (fallback)
    """
    tsv_path = DATA / "external_kg/hetionet_disease_symptom.tsv"
    total = 0
    mapped = 0
    ddx_edges = []
    unmapped_diseases: set = set()
    unmapped_symptoms: set = set()
    matched_via_cui = 0
    matched_via_name = 0

    with open(tsv_path) as f:
        for line in f:
            total += 1
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            dis_str = parts[0]
            doid = dis_str.replace("Disease::", "")
            sym_str = parts[2]
            mesh_id = sym_str.replace("Symptom::", "")

            # Strategy 1: DOID -> CUI via OBO xrefs
            disease_cuis = doid_to_cuis.get(doid, set())
            dis_matches = disease_cuis & dis_cuis

            # Strategy 2: DOID name -> DDXPlus CUI (fallback)
            if not dis_matches:
                doid_name = doid_to_name.get(doid, "")
                if doid_name and doid_name in name_to_cui:
                    dis_matches = {name_to_cui[doid_name]}

            # MeSH -> CUI
            symptom_cuis = mesh_to_cuis.get(mesh_id, set())
            sym_matches = symptom_cuis & sym_cuis

            if dis_matches and sym_matches:
                for dc in dis_matches:
                    for sc in sym_matches:
                        ddx_edges.append((dc, sc))
                mapped += 1
                if disease_cuis & dis_cuis:
                    matched_via_cui += 1
                else:
                    matched_via_name += 1
            else:
                if not dis_matches:
                    unmapped_diseases.add(doid)
                if not sym_matches:
                    unmapped_symptoms.add(mesh_id)

    print(f"\n=== Hetionet ===")
    print(f"Total edges: {total}")
    print(f"Edges with both disease & symptom in DDXPlus: {mapped}")
    print(f"  - matched via CUI xref: {matched_via_cui}")
    print(f"  - matched via name:     {matched_via_name}")
    print(f"Unique DDXPlus edge pairs: {len(set(ddx_edges))}")
    print(f"Unmapped diseases (not in DDXPlus): {len(unmapped_diseases)}")
    print(f"Unmapped symptoms (not in DDXPlus): {len(unmapped_symptoms)}")

    matched_dis = {dc for dc, sc in ddx_edges}
    matched_sym = {sc for dc, sc in ddx_edges}
    print(f"Matched DDXPlus diseases: {len(matched_dis)}")
    for cui in sorted(matched_dis):
        print(f"  {cui}: {dis_cui_to_name.get(cui, '?')}")
    print(f"Matched DDXPlus symptoms: {len(matched_sym)}")

    return ddx_edges


def extract_primekg(
    dis_cuis: set,
    sym_cuis: set,
    dis_cui_to_name: dict,
    sym_cui_to_name: dict,
    hpo_to_cuis: dict,
    name_to_cui: dict,
) -> list:
    """Extract disease-symptom edges from PrimeKG using name matching."""
    csv_path = DATA / "external_kg/primekg.csv"
    total = 0
    mapped = 0
    ddx_edges = []
    matched_disease_names: set = set()
    unmatched_disease_names: set = set()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["display_relation"] != "phenotype present":
                continue
            total += 1

            x_type = row["x_type"]
            y_type = row["y_type"]

            if x_type == "disease":
                dis_name = row["x_name"].lower().strip()
                hpo_num = row["y_id"]
            elif y_type == "disease":
                dis_name = row["y_name"].lower().strip()
                hpo_num = row["x_id"]
            else:
                continue

            dis_cui = name_to_cui.get(dis_name)

            # HPO numeric ID -> full HPO ID -> CUI
            # Grouped IDs (with underscores) are skipped
            if "_" in hpo_num:
                continue
            try:
                hpo_id = f"HP:{int(hpo_num):07d}"
            except ValueError:
                continue
            symptom_cuis = hpo_to_cuis.get(hpo_id, set())
            sym_matches = symptom_cuis & sym_cuis

            if dis_cui and sym_matches:
                for sc in sym_matches:
                    ddx_edges.append((dis_cui, sc))
                mapped += 1
                matched_disease_names.add(dis_name)
            else:
                if not dis_cui:
                    unmatched_disease_names.add(dis_name)

    print(f"\n=== PrimeKG ===")
    print(f"Total phenotype-present edges: {total}")
    print(f"Edges with both disease & symptom in DDXPlus: {mapped}")
    print(f"Unique DDXPlus edge pairs: {len(set(ddx_edges))}")
    print(f"Matched disease names: {len(matched_disease_names)}")
    for n in sorted(matched_disease_names):
        cui = name_to_cui[n]
        print(f"  {n} -> {cui} ({dis_cui_to_name.get(cui, '?')})")
    print(f"Unmatched disease names (not in DDXPlus): {len(unmatched_disease_names)}")

    matched_sym = {sc for dc, sc in ddx_edges}
    print(f"Matched DDXPlus symptoms: {len(matched_sym)}")

    return ddx_edges


def extract_medgen(
    dis_cuis: set,
    sym_cuis: set,
    dis_cui_to_name: dict,
    sym_cui_to_name: dict,
) -> list:
    """Extract disease-symptom edges from MedGen HPO-OMIM mapping.

    MedGen uses OMIM-specific CUIs for diseases, which rarely overlap
    with DDXPlus general disease CUIs. This is a known limitation.
    """
    medgen_path = DATA / "external_kg/MedGen_HPO_OMIM_Mapping.txt"

    total = 0
    mapped = 0
    ddx_edges = []

    omim_cuis_seen: set = set()
    hpo_cuis_seen: set = set()
    omim_in_ddx_dis: set = set()
    hpo_in_ddx_sym: set = set()

    with open(medgen_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            total += 1
            fields = line.strip().split("|")
            if len(fields) < 6:
                continue

            omim_cui = fields[0]
            hpo_cui = fields[4]

            omim_cuis_seen.add(omim_cui)
            hpo_cuis_seen.add(hpo_cui)

            if omim_cui in dis_cuis:
                omim_in_ddx_dis.add(omim_cui)
            if hpo_cui in sym_cuis:
                hpo_in_ddx_sym.add(hpo_cui)

            if omim_cui in dis_cuis and hpo_cui in sym_cuis:
                ddx_edges.append((omim_cui, hpo_cui))
                mapped += 1

    print(f"\n=== MedGen HPO-OMIM ===")
    print(f"Total edges: {total}")
    print(f"Unique OMIM CUIs: {len(omim_cuis_seen)}")
    print(f"Unique HPO CUIs: {len(hpo_cuis_seen)}")
    print(f"OMIM CUIs overlapping DDXPlus diseases: {len(omim_in_ddx_dis)}")
    for cui in sorted(omim_in_ddx_dis):
        print(f"  {cui}: {dis_cui_to_name.get(cui, '?')}")
    print(f"HPO CUIs overlapping DDXPlus symptoms: {len(hpo_in_ddx_sym)}")
    print(f"Edges with both in DDXPlus: {mapped}")
    print(f"Unique DDXPlus edge pairs: {len(set(ddx_edges))}")
    print(f"\nNote: MedGen OMIM_CUI are OMIM-specific CUIs (rare Mendelian diseases),")
    print(f"  which rarely match DDXPlus common clinical disease CUIs.")

    return ddx_edges


def extract_phenotype_hpoa(
    dis_cuis: set,
    sym_cuis: set,
    dis_cui_to_name: dict,
    sym_cui_to_name: dict,
    hpo_to_cuis: dict,
    name_to_cui: dict,
) -> list:
    """Extract disease-symptom edges from phenotype.hpoa (HPO annotations).

    Format: DatabaseID, DiseaseName, Qualifier, HPO_ID, Reference, ...
    Tab-separated with comment lines starting with #.
    """
    hpoa_path = DATA / "external_kg/phenotype.hpoa"
    if not hpoa_path.exists():
        print("\n=== phenotype.hpoa === File not found, skipping.")
        return []

    total = 0
    mapped = 0
    ddx_edges = []
    matched_disease_names: set = set()

    with open(hpoa_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            total += 1
            fields = line.strip().split("\t")
            if len(fields) < 4:
                continue

            dis_name = fields[1].lower().strip()
            qualifier = fields[2]  # "NOT" means negated
            hpo_id = fields[3]     # e.g., HP:0000001

            # Skip negated phenotypes
            if qualifier == "NOT":
                continue

            dis_cui = name_to_cui.get(dis_name)
            symptom_cuis = hpo_to_cuis.get(hpo_id, set())
            sym_matches = symptom_cuis & sym_cuis

            if dis_cui and sym_matches:
                for sc in sym_matches:
                    ddx_edges.append((dis_cui, sc))
                mapped += 1
                matched_disease_names.add(dis_name)

    print(f"\n=== phenotype.hpoa ===")
    print(f"Total annotation rows: {total}")
    print(f"Edges with both disease & symptom in DDXPlus: {mapped}")
    print(f"Unique DDXPlus edge pairs: {len(set(ddx_edges))}")
    print(f"Matched disease names: {len(matched_disease_names)}")
    for n in sorted(matched_disease_names):
        cui = name_to_cui[n]
        print(f"  {n} -> {cui} ({dis_cui_to_name.get(cui, '?')})")

    return ddx_edges


def main() -> None:
    dis_cuis, sym_cuis, dis_cui_to_name, sym_cui_to_name = load_ddxplus_cuis()

    # Build all mappings
    doid_to_cuis, doid_to_name = build_doid_to_cui_and_names()
    mesh_to_cuis, hpo_to_cuis, dis_cui_names = build_mrconso_mappings(dis_cuis)
    name_to_cui = build_name_to_cui(dis_cuis, dis_cui_to_name, dis_cui_names)

    # Extract edges from each source
    hetio_edges = extract_hetionet(
        dis_cuis, sym_cuis, dis_cui_to_name, sym_cui_to_name,
        doid_to_cuis, doid_to_name, mesh_to_cuis, name_to_cui,
    )
    primekg_edges = extract_primekg(
        dis_cuis, sym_cuis, dis_cui_to_name, sym_cui_to_name,
        hpo_to_cuis, name_to_cui,
    )
    medgen_edges = extract_medgen(dis_cuis, sym_cuis, dis_cui_to_name, sym_cui_to_name)
    hpoa_edges = extract_phenotype_hpoa(
        dis_cuis, sym_cuis, dis_cui_to_name, sym_cui_to_name,
        hpo_to_cuis, name_to_cui,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    hetio_unique = set(hetio_edges)
    prime_unique = set(primekg_edges)
    medgen_unique = set(medgen_edges)
    hpoa_unique = set(hpoa_edges)

    print(f"Hetionet unique edges in DDXPlus: {len(hetio_unique)}")
    print(f"PrimeKG unique edges in DDXPlus:  {len(prime_unique)}")
    print(f"MedGen unique edges in DDXPlus:   {len(medgen_unique)}")
    print(f"HPOA unique edges in DDXPlus:     {len(hpoa_unique)}")

    all_edges = hetio_unique | prime_unique | medgen_unique | hpoa_unique
    print(f"\nCombined unique edges: {len(all_edges)}")

    # Overlap analysis
    hp = hetio_unique & prime_unique
    hh = hetio_unique & hpoa_unique
    ph = prime_unique & hpoa_unique
    print(f"Overlap Hetionet & PrimeKG: {len(hp)}")
    print(f"Overlap Hetionet & HPOA:    {len(hh)}")
    print(f"Overlap PrimeKG & HPOA:     {len(ph)}")

    # Disease and symptom coverage
    all_dis = {dc for dc, sc in all_edges}
    all_sym = {sc for dc, sc in all_edges}
    print(f"\nDDXPlus diseases covered: {len(all_dis)}/{len(dis_cuis)}")
    print(f"DDXPlus symptoms covered: {len(all_sym)}/{len(sym_cuis)}")

    uncovered_dis = dis_cuis - all_dis
    if uncovered_dis:
        print(f"\nUncovered diseases ({len(uncovered_dis)}):")
        for cui in sorted(uncovered_dis):
            print(f"  {cui}: {dis_cui_to_name.get(cui, '?')}")

    # Per-disease edge counts
    print(f"\nPer-disease edge counts:")
    dis_edge_counts: dict = defaultdict(int)
    for dc, sc in all_edges:
        dis_edge_counts[dc] += 1
    for cui in sorted(dis_edge_counts, key=dis_edge_counts.get, reverse=True):
        print(f"  {dis_cui_to_name.get(cui, '?'):40s} ({cui}): {dis_edge_counts[cui]} edges")

    # Save edges to JSON
    output = {
        "hetionet": [{"disease_cui": d, "symptom_cui": s} for d, s in sorted(hetio_unique)],
        "primekg": [{"disease_cui": d, "symptom_cui": s} for d, s in sorted(prime_unique)],
        "medgen": [{"disease_cui": d, "symptom_cui": s} for d, s in sorted(medgen_unique)],
        "hpoa": [{"disease_cui": d, "symptom_cui": s} for d, s in sorted(hpoa_unique)],
        "combined": [{"disease_cui": d, "symptom_cui": s} for d, s in sorted(all_edges)],
        "statistics": {
            "hetionet_edges": len(hetio_unique),
            "primekg_edges": len(prime_unique),
            "medgen_edges": len(medgen_unique),
            "hpoa_edges": len(hpoa_unique),
            "combined_edges": len(all_edges),
            "diseases_covered": len(all_dis),
            "symptoms_covered": len(all_sym),
            "total_ddx_diseases": len(dis_cuis),
            "total_ddx_symptoms": len(sym_cuis),
        },
    }

    out_path = DATA / "external_kg/external_kg_ddxplus_edges.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
