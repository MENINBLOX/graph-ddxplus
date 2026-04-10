#!/usr/bin/env python3
"""Step 1c: 다중 소스 disease-symptom KG 구축.

7개 독립 소스에서 disease-symptom 관계를 추출하여
Neo4j에 통합 KG를 구축한다.

소스:
  1. DDXPlus conditions.json - 벤치마크 공식 사양 (Mila, 2022)
  2. SemMedDB v43 - PubMed 문헌 NLP 추출 (NLM, 2024)
  3. UMLS MRREL 2025AB - 메타시소러스 관계 (NLM)
  4. Columbia KB - 임상 퇴원요약서 NLP (Columbia Univ.)
  5. HSDN - PubMed MeSH co-occurrence (Zhou et al., 2014)
  6. Wikidata - 커뮤니티 큐레이션
  7. Hetionet + PrimeKG + MedGen + HPOA - 통합 생의학 KG

전제조건:
  - data/ddxplus/ (DDXPlus 데이터)
  - data/umls_subset/MRCONSO.RRF, MRREL.RRF (UMLS MetamorphoSys)
  - data/semmeddb/ (SemMedDB)
  - data/hsdn/ (HSDN)
  - data/external_kg/ (Hetionet, PrimeKG, HPO annotations)
  - Neo4j 실행 중

사용법:
  uv run python pipeline/step1c_build_multi_source_kg.py [--port 7688]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

# ── 경로 ──
DATA = Path("data")
DDX = DATA / "ddxplus"
UMLS = DATA / "umls_subset"
EXT = DATA / "external_kg"
SEMMED = DATA / "semmeddb"
HSDN_DIR = DATA / "hsdn"

# ── SemMedDB predicate 필터 ──
SEMMED_PREDICATES = {
    "PROCESS_OF", "CAUSES", "MANIFESTATION_OF", "ASSOCIATED_WITH",
    "AFFECTS", "OCCURS_IN", "PREDISPOSES", "COMPLICATES",
}

# ── figshare URLs ──
FIGSHARE_EVIDENCES_EN = "https://ndownloader.figshare.com/files/40278013"
FIGSHARE_CONDITIONS_EN = "https://ndownloader.figshare.com/files/62561569"


def ensure_english_ddxplus() -> None:
    """English DDXPlus evidences/conditions 다운로드 (E_XX 매핑용)."""
    for fname, url in [
        ("release_evidences_en.json", FIGSHARE_EVIDENCES_EN),
        ("release_conditions_en.json", FIGSHARE_CONDITIONS_EN),
    ]:
        path = DDX / fname
        if path.exists():
            continue
        print(f"  다운로드: {fname}...")
        urllib.request.urlretrieve(url, str(path))
        print(f"  저장: {path} ({path.stat().st_size:,} bytes)")


def build_exx_to_french_mapping() -> dict[str, str]:
    """English E_XX 코드 → French evidence name 매핑 구축.

    DDXPlus conditions.json (English)의 키는 E_XX 형식이고,
    umls_mapping.json의 키는 French evidence name이다.
    question_en 텍스트 매칭으로 변환 테이블을 구축한다.
    """
    with open(DDX / "release_evidences_en.json") as f:
        evs_en = json.load(f)
    with open(DDX / "release_evidences.json") as f:
        evs_fr = json.load(f)

    # question_en 기준 매칭
    en_to_fr: dict[str, str] = {}
    for e_key, e_info in evs_en.items():
        q_en = e_info.get("question_en", "")
        for f_key, f_info in evs_fr.items():
            if f_info.get("question_en", "") == q_en:
                en_to_fr[e_key] = f_key
                break

    return en_to_fr


def load_ddxplus_mappings() -> tuple[
    dict[str, str], set[str], set[str], set[str],
]:
    """DDXPlus 매핑 파일 로드."""
    with open(DDX / "umls_mapping.json") as f:
        sym_map = json.load(f)["mapping"]
    with open(DDX / "disease_umls_mapping.json") as f:
        dis_map = json.load(f)["mapping"]
    with open(DDX / "release_evidences.json") as f:
        evs = json.load(f)

    sym_cuis = {v["cui"] for v in sym_map.values() if v.get("cui")}
    dis_cuis = {v["umls_cui"] for v in dis_map.values() if v.get("umls_cui")}

    antecedent_cuis: set[str] = set()
    for code, info in evs.items():
        cui = sym_map.get(code, {}).get("cui")
        if cui and info.get("is_antecedent", False):
            antecedent_cuis.add(cui)

    dis_name_to_cui = {
        name: v["umls_cui"] for name, v in dis_map.items() if v.get("umls_cui")
    }

    return dis_name_to_cui, sym_cuis, dis_cuis, antecedent_cuis


def extract_ddxplus(
    dis_name_to_cui: dict[str, str],
    exx_to_fr: dict[str, str],
) -> set[tuple[str, str]]:
    """DDXPlus conditions.json (English) + E_XX→French 매핑으로 엣지 추출."""
    with open(DDX / "release_conditions_en.json") as f:
        conds = json.load(f)
    with open(DDX / "umls_mapping.json") as f:
        sym_map = json.load(f)["mapping"]

    edges: set[tuple[str, str]] = set()
    for dname, cond in conds.items():
        dcui = dis_name_to_cui.get(dname)
        if not dcui:
            continue
        all_codes = list(cond.get("symptoms", {}).keys()) + list(
            cond.get("antecedents", {}).keys()
        )
        for ecode in all_codes:
            fr_name = exx_to_fr.get(ecode)
            if fr_name:
                scui = sym_map.get(fr_name, {}).get("cui")
                if scui:
                    edges.add((scui, dcui))
    return edges


def extract_semmeddb(
    sym_cuis: set[str], dis_cuis: set[str],
) -> set[tuple[str, str]]:
    """SemMedDB에서 DDXPlus CUI 쌍 엣지 추출."""
    gz_path = SEMMED / "semmedVER43_2024_R_PREDICATION.csv.gz"
    if not gz_path.exists():
        print("    [skip] SemMedDB 파일 없음")
        return set()

    edges: set[tuple[str, str]] = set()
    with gzip.open(gz_path, "rt", encoding="latin-1", errors="replace") as f:
        for row in csv.reader(f, quotechar='"'):
            if len(row) < 12 or row[3] not in SEMMED_PREDICATES:
                continue
            s_cui, o_cui = row[4], row[8]
            if s_cui in sym_cuis and o_cui in dis_cuis:
                edges.add((s_cui, o_cui))
            elif o_cui in sym_cuis and s_cui in dis_cuis:
                edges.add((o_cui, s_cui))
    return edges


def extract_umls_mrrel(
    sym_cuis: set[str], dis_cuis: set[str],
) -> set[tuple[str, str]]:
    """UMLS MRREL에서 DDXPlus CUI 쌍 엣지 추출."""
    if not (UMLS / "MRREL.RRF").exists():
        print("    [skip] UMLS MRREL 없음")
        return set()

    edges: set[tuple[str, str]] = set()
    with open(UMLS / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            c1, c2 = p[0], p[4]
            if not c1 or not c2:
                continue
            if c1 in sym_cuis and c2 in dis_cuis:
                edges.add((c1, c2))
            elif c2 in sym_cuis and c1 in dis_cuis:
                edges.add((c2, c1))
    return edges


def extract_columbia(
    sym_cuis: set[str], dis_cuis: set[str],
) -> set[tuple[str, str]]:
    """Columbia Disease-Symptom KB에서 엣지 추출."""
    path = HSDN_DIR / "columbia_disease_symptom_kb.csv"
    if not path.exists():
        print("    [skip] Columbia KB 없음")
        return set()

    edges: set[tuple[str, str]] = set()
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            dm = re.match(r"UMLS:(C\d+)", row[0])
            sm = re.match(r"UMLS:(C\d+)", row[2])
            if dm and sm:
                dc, sc = dm.group(1), sm.group(1)
                if sc in sym_cuis and dc in dis_cuis:
                    edges.add((sc, dc))
                elif dc in sym_cuis and sc in dis_cuis:
                    edges.add((dc, sc))
    return edges


def extract_hsdn(
    sym_cuis: set[str], dis_cuis: set[str],
    mesh_to_cui: dict[str, set[str]],
) -> set[tuple[str, str]]:
    """HSDN에서 MeSH→CUI 매핑 후 엣지 추출."""
    path = HSDN_DIR / "dhimmel_hsdn" / "data" / "symptoms-DO.tsv"
    if not path.exists():
        print("    [skip] HSDN 없음")
        return set()

    edges: set[tuple[str, str]] = set()
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 6:
                continue
            for sc in mesh_to_cui.get(p[5], set()):
                for dc in mesh_to_cui.get(p[4], set()):
                    if sc in sym_cuis and dc in dis_cuis:
                        edges.add((sc, dc))
                    elif dc in sym_cuis and sc in dis_cuis:
                        edges.add((dc, sc))
    return edges


def extract_wikidata(
    sym_cuis: set[str], dis_cuis: set[str],
    mesh_to_cui: dict[str, set[str]],
) -> set[tuple[str, str]]:
    """Wikidata disease-symptom 엣지 추출."""
    path = EXT / "wikidata_disease_symptom.csv"
    if not path.exists():
        print("    [skip] Wikidata 없음")
        return set()

    edges: set[tuple[str, str]] = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_umls = row.get("disease_umls", "").strip()
            s_umls = row.get("symptom_umls", "").strip()
            if d_umls and s_umls:
                if s_umls in sym_cuis and d_umls in dis_cuis:
                    edges.add((s_umls, d_umls))
                elif d_umls in sym_cuis and s_umls in dis_cuis:
                    edges.add((d_umls, s_umls))
            d_mesh = row.get("disease_mesh", "").strip()
            s_mesh = row.get("symptom_mesh", "").strip()
            if d_mesh and s_mesh:
                for dc in mesh_to_cui.get(d_mesh, set()):
                    for sc in mesh_to_cui.get(s_mesh, set()):
                        if sc in sym_cuis and dc in dis_cuis:
                            edges.add((sc, dc))
    return edges


def extract_external_kg(
    sym_cuis: set[str], dis_cuis: set[str],
) -> set[tuple[str, str]]:
    """Hetionet, PrimeKG, MedGen, HPOA 엣지 로드."""
    path = EXT / "external_kg_ddxplus_edges.json"
    if not path.exists():
        print("    [skip] external_kg_ddxplus_edges.json 없음")
        return set()

    with open(path) as f:
        data = json.load(f)

    edges: set[tuple[str, str]] = set()
    for source in ["hetionet", "primekg", "medgen", "hpoa", "combined"]:
        for pair in data.get(source, []):
            if isinstance(pair, dict):
                sc = pair.get("symptom_cui", "")
                dc = pair.get("disease_cui", "")
            elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                sc, dc = pair[0], pair[1]
            else:
                continue
            if sc in sym_cuis and dc in dis_cuis:
                edges.add((sc, dc))
    return edges


def load_to_neo4j(
    all_edges: set[tuple[str, str]],
    dis_cuis: set[str],
    sym_cuis: set[str],
    antecedent_cuis: set[str],
    neo4j_uri: str,
) -> None:
    """Neo4j에 KG 로드."""
    cui_names: dict[str, str] = {}
    if (UMLS / "MRCONSO.RRF").exists():
        with open(UMLS / "MRCONSO.RRF") as f:
            for line in f:
                p = line.strip().split("|")
                if p[1] == "ENG" and (p[0] not in cui_names or p[2] == "P"):
                    cui_names[p[0]] = p[14]

    driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", "password123"))
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run("CREATE INDEX IF NOT EXISTS FOR (s:Symptom) ON (s.cui)")
        s.run("CREATE INDEX IF NOT EXISTS FOR (d:Disease) ON (d.cui)")

        for c in dis_cuis:
            s.run("CREATE (d:Disease {cui: $c, name: $n})",
                  c=c, n=cui_names.get(c, c))
        for c in sym_cuis:
            s.run("CREATE (s:Symptom {cui: $c, name: $n, is_antecedent: $a})",
                  c=c, n=cui_names.get(c, c), a=c in antecedent_cuis)

        batch = [{"s": sc, "d": dc} for sc, dc in all_edges]
        for i in range(0, len(batch), 1000):
            s.run("""UNWIND $b AS r
                     MATCH (s:Symptom {cui: r.s})
                     MATCH (d:Disease {cui: r.d})
                     CREATE (s)-[:INDICATES]->(d)""",
                  b=batch[i:i + 1000])

        r = s.run(
            """MATCH (s:Symptom) WITH count(s) AS sc
               MATCH (d:Disease) WITH sc, count(d) AS dc
               MATCH ()-[r:INDICATES]->() RETURN sc, dc, count(r) AS rc"""
        ).single()
        print(f"  Neo4j: {r['sc']} symptoms, {r['dc']} diseases, {r['rc']} edges")

    driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="다중 소스 disease-symptom KG 구축")
    parser.add_argument("--port", type=int, default=7688)
    args = parser.parse_args()

    neo4j_uri = f"bolt://localhost:{args.port}"

    print("=" * 60)
    print("Step 1c: 다중 소스 Disease-Symptom KG 구축")
    print("=" * 60)

    # ── English DDXPlus 확보 ──
    print("\n[0/8] English DDXPlus 확보...")
    ensure_english_ddxplus()

    # ── 매핑 로드 ──
    print("[0/8] DDXPlus 매핑 로드...")
    dis_name_to_cui, sym_cuis, dis_cuis, ante_cuis = load_ddxplus_mappings()
    exx_to_fr = build_exx_to_french_mapping()
    print(f"  증상 CUI: {len(sym_cuis)}, 질환 CUI: {len(dis_cuis)}")
    print(f"  E_XX→FR 매핑: {len(exx_to_fr)}/223")

    # MeSH→CUI
    mesh_to_cui: dict[str, set[str]] = defaultdict(set)
    if (UMLS / "MRCONSO.RRF").exists():
        with open(UMLS / "MRCONSO.RRF") as f:
            for line in f:
                p = line.strip().split("|")
                if p[11] == "MSH":
                    mesh_to_cui[p[13]].add(p[0])

    # ── 소스별 추출 ──
    source_edges: dict[str, set[tuple[str, str]]] = {}

    print("\n[1/7] DDXPlus conditions.json (English, E_XX→FR 매핑)...")
    source_edges["ddxplus"] = extract_ddxplus(dis_name_to_cui, exx_to_fr)
    gt_edges = source_edges["ddxplus"]

    print("[2/7] SemMedDB v43...")
    source_edges["semmeddb"] = extract_semmeddb(sym_cuis, dis_cuis)

    print("[3/7] UMLS MRREL 2025AB...")
    source_edges["umls_mrrel"] = extract_umls_mrrel(sym_cuis, dis_cuis)

    print("[4/7] Columbia KB...")
    source_edges["columbia"] = extract_columbia(sym_cuis, dis_cuis)

    print("[5/7] HSDN (dhimmel)...")
    source_edges["hsdn"] = extract_hsdn(sym_cuis, dis_cuis, mesh_to_cui)

    print("[6/7] Wikidata...")
    source_edges["wikidata"] = extract_wikidata(sym_cuis, dis_cuis, mesh_to_cui)

    print("[7/7] Hetionet + PrimeKG + MedGen + HPOA...")
    source_edges["external"] = extract_external_kg(sym_cuis, dis_cuis)

    # ── 통합 ──
    print("\n통합...")
    all_edges: set[tuple[str, str]] = set()
    for edges in source_edges.values():
        all_edges |= edges

    # ── 통계 ──
    print(f"\n{'='*60}")
    print("소스별 기여:")
    stats: dict[str, dict] = {}
    for name, edges in sorted(source_edges.items(), key=lambda x: -len(x[1])):
        gt_match = len(edges & gt_edges)
        unique = len(edges - (all_edges - edges))
        print(f"  {name:15s}: {len(edges):>5} edges (GT일치: {gt_match}, 고유: {unique})")
        stats[name] = {"edges": len(edges), "gt_match": gt_match, "unique_contribution": unique}

    sym_c = len({e[0] for e in all_edges})
    dis_c = len({e[1] for e in all_edges})
    gt_tp = len(all_edges & gt_edges)
    print(f"\n통합 KG:")
    print(f"  엣지: {len(all_edges)}")
    print(f"  증상: {sym_c}/{len(sym_cuis)}")
    print(f"  질환: {dis_c}/{len(dis_cuis)}")
    print(f"  DDXPlus GT recall: {gt_tp}/{len(gt_edges)} ({gt_tp/len(gt_edges)*100:.1f}%)")

    # ── Neo4j 로드 ──
    print(f"\nNeo4j ({neo4j_uri}) 로드...")
    load_to_neo4j(all_edges, dis_cuis, sym_cuis, ante_cuis, neo4j_uri)

    # ── 결과 저장 ──
    result = {
        "description": "Multi-source disease-symptom KG",
        "sources": {
            "ddxplus": "DDXPlus conditions.json via E_XX→FR mapping (Mila, 2022)",
            "semmeddb": "SemMedDB v43 2024 (NLM, PubMed NLP)",
            "umls_mrrel": "UMLS MRREL 2025AB (NLM MetamorphoSys)",
            "columbia": "Columbia Disease-Symptom KB (clinical NLP)",
            "hsdn": "HSDN (Zhou et al., Nature Comm. 2014)",
            "wikidata": "Wikidata SPARQL (CC0)",
            "external": "Hetionet v1.0 + PrimeKG + MedGen + HPOA",
        },
        "statistics": stats,
        "combined_edges": len(all_edges),
        "symptom_coverage": sym_c,
        "disease_coverage": dis_c,
        "gt_recall": gt_tp / len(gt_edges) if gt_edges else 0,
    }
    output = Path("results/multi_source_kg_stats.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {output}")
    print(f"\n완료!")
    print(f"\n다음 단계:")
    print(f"  uv run python scripts/benchmark_umls_kg.py -n 1000 --port {args.port}")


if __name__ == "__main__":
    main()
