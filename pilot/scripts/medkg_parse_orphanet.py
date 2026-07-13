#!/usr/bin/env python3
"""Parse Orphanet en_product4.xml → disease–phenotype edges with frequency.

Output: /home/max/Graph-DDXPlus/data/medkg/processed/orphanet_edges.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path
from lxml import etree

XML_PATH = Path("/home/max/Graph-DDXPlus/data/medkg/orphanet/en_product4.xml")
OUT = Path("/home/max/Graph-DDXPlus/data/medkg/processed/orphanet_edges.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

FREQUENCY_MAP = {
    "Very frequent (99-80%)": 0.95,
    "Frequent (79-30%)": 0.55,
    "Occasional (29-5%)": 0.17,
    "Very rare (<4-1%)": 0.025,
    "Excluded (0%)": 0.0,
    "Obligate (100%)": 1.0,
}


def main():
    print(f"Parsing {XML_PATH}...")
    tree = etree.parse(str(XML_PATH))
    n_edges = 0
    n_diseases = 0
    with OUT.open("w") as out:
        for d in tree.iter("Disorder"):
            orpha_code = d.findtext("OrphaCode")
            disease_name = d.findtext("Name")
            if not orpha_code or not disease_name: continue
            assoc = d.findall(".//HPODisorderAssociation")
            n_diseases += 1
            for a in assoc:
                hpo_id = a.findtext(".//HPOId")
                hpo_term = a.findtext(".//HPOTerm")
                freq_name = a.findtext(".//HPOFrequency/Name")
                freq_score = FREQUENCY_MAP.get(freq_name, 0.5)
                edge = {
                    "disease": disease_name,
                    "disease_id": f"ORPHA:{orpha_code}",
                    "phenotype": hpo_term,
                    "phenotype_id": hpo_id,
                    "frequency": freq_name,
                    "frequency_score": freq_score,
                    "source": "orphanet",
                    "provenance": {
                        "orpha_code": orpha_code,
                        "url": f"https://www.orpha.net/consor/cgi-bin/OC_Exp.php?lng=en&Expert={orpha_code}",
                    },
                }
                out.write(json.dumps(edge, ensure_ascii=False) + "\n")
                n_edges += 1
    print(f"  Diseases: {n_diseases}")
    print(f"  Edges: {n_edges}")
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    main()
