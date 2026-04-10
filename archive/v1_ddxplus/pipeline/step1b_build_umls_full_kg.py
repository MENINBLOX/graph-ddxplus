#!/usr/bin/env python3
"""Step 1b: UMLS 전체 symptom-disease KG 구축.

UMLS MRREL에서 모든 symptom→disease 관계를 추출하여
데이터셋 독립적인 범용 진단 KG를 Neo4j에 구축한다.

이 KG는 DDXPlus, DxBench, 실제 병원 데이터 등
어떤 데이터셋이든 매핑만 하면 사용 가능.

전제조건:
  - data/umls_extracted/MRCONSO.RRF (step1a 완료)
  - data/umls_extracted/MRREL.RRF
  - data/umls_extracted/MRSTY.RRF
  - Neo4j 실행 중

사용법:
  uv run python pipeline/step1b_build_umls_full_kg.py [--port 7695]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

UMLS_DIR = Path("data/umls_extracted")

# Semantic types
SYMPTOM_STYS = {
    "T184",  # Sign or Symptom
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T048",  # Mental or Behavioral Dysfunction
    "T037",  # Injury or Poisoning
}
DISEASE_STYS = {
    "T047",  # Disease or Syndrome
    "T019",  # Congenital Abnormality
    "T191",  # Neoplastic Process
    "T046",  # Pathologic Function
}


def load_semantic_types() -> dict[str, set[str]]:
    print("[1/5] MRSTY 로드...")
    cui_stys: dict[str, set[str]] = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui_stys[p[0]].add(p[1])
    n_sym = sum(1 for stys in cui_stys.values() if stys & SYMPTOM_STYS)
    n_dis = sum(1 for stys in cui_stys.values() if stys & DISEASE_STYS)
    print(f"  CUIs: {len(cui_stys):,} (symptom STY: {n_sym:,}, disease STY: {n_dis:,})")
    return dict(cui_stys)


def load_concept_names() -> dict[str, str]:
    print("[2/5] MRCONSO 로드 (English names)...")
    cui_names: dict[str, str] = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui, lang, ts, name = p[0], p[1], p[2], p[14]
            if lang != "ENG":
                continue
            if cui not in cui_names or ts == "P":
                cui_names[cui] = name
    print(f"  {len(cui_names):,} CUI names")
    return cui_names


def extract_symptom_disease_edges(
    cui_stys: dict[str, set[str]],
) -> set[tuple[str, str]]:
    print("[3/5] MRREL에서 전체 symptom→disease 관계 추출...")
    edges: set[tuple[str, str]] = set()

    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui1, cui2 = p[0], p[4]
            s1 = cui_stys.get(cui1, set())
            s2 = cui_stys.get(cui2, set())

            if (s1 & SYMPTOM_STYS) and (s2 & DISEASE_STYS):
                edges.add((cui1, cui2))
            elif (s1 & DISEASE_STYS) and (s2 & SYMPTOM_STYS):
                edges.add((cui2, cui1))

    sym_cuis = set(e[0] for e in edges)
    dis_cuis = set(e[1] for e in edges)
    print(f"  {len(sym_cuis):,} symptoms → {len(dis_cuis):,} diseases, {len(edges):,} edges")
    return edges


def load_to_neo4j(
    edges: set[tuple[str, str]],
    cui_names: dict[str, str],
    neo4j_uri: str,
) -> None:
    print(f"[4/5] Neo4j ({neo4j_uri}) 로드...")
    sym_cuis = set(e[0] for e in edges)
    dis_cuis = set(e[1] for e in edges)

    driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", "password123"))
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run("CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.cui)")
        s.run("CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.cui)")

        # Disease nodes (batch)
        print(f"  Disease nodes: {len(dis_cuis):,}...")
        batch = [{"cui": c, "name": cui_names.get(c, c)} for c in dis_cuis]
        for i in range(0, len(batch), 1000):
            s.run(
                "UNWIND $batch AS row CREATE (d:Disease {cui: row.cui, name: row.name})",
                batch=batch[i:i+1000],
            )

        # Symptom nodes (batch)
        print(f"  Symptom nodes: {len(sym_cuis):,}...")
        batch = [{"cui": c, "name": cui_names.get(c, c), "ante": False} for c in sym_cuis]
        for i in range(0, len(batch), 1000):
            s.run(
                "UNWIND $batch AS row CREATE (s:Symptom {cui: row.cui, name: row.name, is_antecedent: row.ante})",
                batch=batch[i:i+1000],
            )

        # INDICATES edges (batch)
        print(f"  INDICATES edges: {len(edges):,}...")
        edge_list = [{"s": sc, "d": dc} for sc, dc in edges]
        for i in range(0, len(edge_list), 1000):
            s.run(
                """UNWIND $batch AS row
                   MATCH (s:Symptom {cui: row.s})
                   MATCH (d:Disease {cui: row.d})
                   CREATE (s)-[:INDICATES]->(d)""",
                batch=edge_list[i:i+1000],
            )
            if (i + 1000) % 10000 < 1000:
                print(f"    {min(i+1000, len(edge_list)):,}/{len(edge_list):,}")

        # 검증
        r = s.run(
            "MATCH (s:Symptom) WITH count(s) AS sc "
            "MATCH (d:Disease) WITH sc, count(d) AS dc "
            "MATCH ()-[r:INDICATES]->() RETURN sc, dc, count(r) AS rc"
        ).single()
        print(f"  완료: {r['sc']:,} symptoms, {r['dc']:,} diseases, {r['rc']:,} edges")

    driver.close()


def verify_ddxplus_coverage(edges: set[tuple[str, str]]) -> None:
    """DDXPlus 49개 질병이 UMLS 전체 KG에서 얼마나 커버되는지 확인."""
    print("[5/5] DDXPlus coverage 확인...")

    disease_to_symptoms = defaultdict(set)
    for sc, dc in edges:
        disease_to_symptoms[dc].add(sc)

    with open("data/ddxplus/disease_umls_mapping.json") as f:
        ddx_dis = json.load(f)["mapping"]
    with open("data/ddxplus/umls_mapping.json") as f:
        ddx_sym = json.load(f)["mapping"]

    ddx_sym_cuis = {v["cui"] for v in ddx_sym.values() if v.get("cui")}

    dis_cuis = set(e[1] for e in edges)
    total_in_kg = 0
    total_syms = 0

    for name, info in sorted(ddx_dis.items()):
        cui = info.get("umls_cui")
        if not cui:
            continue
        in_kg = cui in dis_cuis
        n_syms = len(disease_to_symptoms.get(cui, set()))
        n_overlap = len(disease_to_symptoms.get(cui, set()) & ddx_sym_cuis)
        if in_kg:
            total_in_kg += 1
        total_syms += n_syms
        print(f"    {'✓' if in_kg else '✗'} {name}: UMLS_syms={n_syms}, DDXPlus_overlap={n_overlap}")

    avg = total_syms / max(total_in_kg, 1)
    print(f"\n  DDXPlus 질병 in KG: {total_in_kg}/49")
    print(f"  평균 UMLS 증상/질병: {avg:.1f}")
    print(f"  (참고: DDXPlus conditions.json 평균: 18.1)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7695)
    args = parser.parse_args()

    neo4j_uri = f"bolt://localhost:{args.port}"

    print("=" * 60)
    print("Step 1b: UMLS 전체 symptom-disease KG 구축")
    print("=" * 60)

    cui_stys = load_semantic_types()
    cui_names = load_concept_names()
    edges = extract_symptom_disease_edges(cui_stys)
    load_to_neo4j(edges, cui_names, neo4j_uri)
    verify_ddxplus_coverage(edges)

    print("\n완료!")


if __name__ == "__main__":
    main()
