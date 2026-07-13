#!/usr/bin/env python3
"""텍스트 매칭 vs MeSH vs NER: CUI 추출 비교.

CUI의 UMLS 영문 이름으로 초록 텍스트를 직접 검색하여
MeSH/NER 없이 CUI를 추출하는 방법의 효과를 측정한다.

Aho-Corasick 알고리즘으로 수십만 패턴을 O(n) 매칭.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import ahocorasick

UMLS_DIR = Path("data/umls_extracted")
DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
ABSTRACTS_PER_DISEASE = 20

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}


def load_cui_stys():
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def build_aho_automaton():
    """DISO CUI의 모든 영문 이름으로 Aho-Corasick automaton 구축."""
    cui_stys = load_cui_stys()
    diso_cuis = {cui for cui, stys in cui_stys.items() if stys & ALLOWED_STYS}
    diso_cuis -= BLACKLIST

    # CUI별 영문 이름 수집
    cui_preferred = {}
    name_to_cui = {}  # lowercase name -> cui
    total_names = 0

    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui = p[0]
            if cui not in diso_cuis or p[1] != "ENG":
                continue
            name = p[14].strip()
            lower = name.lower()
            if len(lower) >= 4:  # 4글자 미만 무시
                name_to_cui[lower] = cui
                total_names += 1
            if p[2] == "P" and cui not in cui_preferred:
                cui_preferred[cui] = name

    # Aho-Corasick automaton 구축
    A = ahocorasick.Automaton()
    for name, cui in name_to_cui.items():
        A.add_word(name, (name, cui))
    A.make_automaton()

    print(f"  DISO CUI: {len(diso_cuis):,}, 이름 패턴: {total_names:,}")
    return A, cui_preferred, cui_stys


def text_match_aho(text_lower: str, automaton, disease_cui: str) -> set:
    """Aho-Corasick으로 초록에서 CUI 매칭."""
    matched = set()
    for end_idx, (name, cui) in automaton.iter(text_lower):
        if cui == disease_cui:
            continue
        # 단어 경계 확인
        start_idx = end_idx - len(name) + 1
        if start_idx > 0 and text_lower[start_idx - 1].isalpha():
            continue
        if end_idx + 1 < len(text_lower) and text_lower[end_idx + 1].isalpha():
            continue
        matched.add(cui)
    return matched


def load_mesh_to_cui():
    m = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D"):
                if p[13] not in m:
                    m[p[13]] = p[0]
    return m


def load_parent_map():
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parents[p[0]].add(p[4])
    return dict(parents)


def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f:
        conditions = json.load(f)
    with open("data/ddxplus/release_evidences_en.json") as f:
        ev_en = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)
    with open("data/ddxplus/umls_mapping.json") as f:
        umap = json.load(f)["mapping"]
    with open("data/ddxplus/disease_umls_mapping.json") as f:
        disease_map = json.load(f)["mapping"]

    eid_to_fr = {}
    for eid, en_info in ev_en.items():
        for fr_name, fr_info in ev_fr.items():
            if en_info.get("question_en") == fr_info.get("question_en") and en_info.get("question_en"):
                eid_to_fr[eid] = fr_name
                break

    disease_gold = {}
    disease_cuis = {}
    for disease_name, info in conditions.items():
        d_cui = disease_map.get(disease_name, {}).get("umls_cui")
        d_name = disease_map.get(disease_name, {}).get("umls_name", disease_name)
        if not d_cui:
            continue
        disease_cuis[disease_name] = {"cui": d_cui, "umls_name": d_name}
        symptom_cuis = set()
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False):
                continue
            fr_name = eid_to_fr.get(eid)
            if fr_name and fr_name in umap:
                cui = umap[fr_name].get("cui")
                if cui:
                    symptom_cuis.add(cui)
        if symptom_cuis:
            disease_gold[d_cui] = symptom_cuis

    return disease_gold, disease_cuis


def expand_cuis_1level(cuis, parent_map):
    expanded = set(cuis)
    for cui in cuis:
        expanded.update(parent_map.get(cui, set()))
    return expanded


def count_gold_hits(extracted_cuis, gold_cuis, parent_map):
    expanded = expand_cuis_1level(extracted_cuis, parent_map)
    gold_expanded = expand_cuis_1level(gold_cuis, parent_map)
    return len(expanded & gold_expanded)


def main():
    print("=" * 80)
    print("텍스트 매칭 vs MeSH: CUI 추출 비교 (Aho-Corasick)")
    print("=" * 80)

    print("\n[1] Aho-Corasick automaton 구축...")
    t0 = time.time()
    automaton, cui_preferred, cui_stys = build_aho_automaton()
    print(f"  구축 시간: {time.time()-t0:.1f}초")

    parent_map = load_parent_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        cui_to_mesh[cui].add(mesh)

    disease_gold, disease_cuis = prepare_gold()

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    print(f"\n[2] 질환별 비교 (각 {ABSTRACTS_PER_DISEASE}편)...")
    print(f"{'질환':<35} {'Gold':>4} | {'MeSH':>5} {'Hit':>3} {'%':>5} | {'Text':>5} {'Hit':>3} {'%':>5} | {'ms':>5}")
    print("-" * 90)

    totals_mesh = {"cuis": 0, "hits": 0, "gold": 0}
    totals_text = {"cuis": 0, "hits": 0, "gold": 0}
    n_diseases = 0

    for disease_name, dinfo in sorted(disease_cuis.items()):
        dc = dinfo["cui"]
        dn = dinfo["umls_name"]
        if dc not in disease_gold:
            continue
        gold_cuis = disease_gold[dc]

        mesh_uids = cui_to_mesh.get(dc, set())
        if mesh_uids:
            mesh_cond = " OR ".join(f"mesh_terms LIKE '%{m}%'" for m in mesh_uids)
            c.execute(f"""
                SELECT pmid, mesh_terms, abstract FROM abstracts
                WHERE ({mesh_cond})
                AND abstract IS NOT NULL AND length(abstract) > 200
                ORDER BY RANDOM() LIMIT ?
            """, (ABSTRACTS_PER_DISEASE,))
        else:
            keywords = re.sub(r'\(.*?\)', '', dn)
            keywords = re.sub(r'\b(NOS|unspecified|disease)\b', '', keywords, flags=re.IGNORECASE).strip()
            if len(keywords) < 4:
                continue
            c.execute("""
                SELECT pmid, mesh_terms, abstract FROM abstracts
                WHERE (title LIKE ? OR abstract LIKE ?)
                AND abstract IS NOT NULL AND length(abstract) > 200
                ORDER BY RANDOM() LIMIT ?
            """, (f"%{keywords}%", f"%{keywords}%", ABSTRACTS_PER_DISEASE))

        rows = c.fetchall()
        if not rows:
            continue

        # MeSH
        mesh_cuis_all = set()
        for _, mesh_json, _ in rows:
            try:
                mesh_list = json.loads(mesh_json) if mesh_json else []
            except Exception:
                mesh_list = []
            for mid in mesh_list:
                cui = mesh_to_cui.get(mid)
                if cui and cui != dc and (cui_stys.get(cui, set()) & ALLOWED_STYS) and cui not in BLACKLIST:
                    mesh_cuis_all.add(cui)
        mesh_hits = count_gold_hits(mesh_cuis_all, gold_cuis, parent_map)

        # 텍스트 매칭
        text_cuis_all = set()
        t0 = time.time()
        for _, _, abstract in rows:
            matched = text_match_aho(abstract.lower(), automaton, dc)
            text_cuis_all.update(matched)
        text_ms = (time.time() - t0) * 1000
        text_hits = count_gold_hits(text_cuis_all, gold_cuis, parent_map)

        mesh_pct = 100 * mesh_hits / len(gold_cuis) if gold_cuis else 0
        text_pct = 100 * text_hits / len(gold_cuis) if gold_cuis else 0

        short_name = disease_name[:33]
        print(f"{short_name:<35} {len(gold_cuis):>4} | "
              f"{len(mesh_cuis_all):>5} {mesh_hits:>3} {mesh_pct:>4.0f}% | "
              f"{len(text_cuis_all):>5} {text_hits:>3} {text_pct:>4.0f}% | "
              f"{text_ms:>4.0f}ms")

        n_diseases += 1
        totals_mesh["cuis"] += len(mesh_cuis_all)
        totals_mesh["hits"] += mesh_hits
        totals_mesh["gold"] += len(gold_cuis)
        totals_text["cuis"] += len(text_cuis_all)
        totals_text["hits"] += text_hits
        totals_text["gold"] += len(gold_cuis)

    conn.close()

    print("-" * 90)
    mesh_r = 100 * totals_mesh["hits"] / totals_mesh["gold"] if totals_mesh["gold"] else 0
    text_r = 100 * totals_text["hits"] / totals_text["gold"] if totals_text["gold"] else 0
    print(f"\n요약 ({n_diseases}개 질환)")
    print(f"  MeSH:      평균 {totals_mesh['cuis']/n_diseases:.0f} CUI/질환, "
          f"total hits {totals_mesh['hits']}/{totals_mesh['gold']}, recall {mesh_r:.1f}%")
    print(f"  텍스트매칭: 평균 {totals_text['cuis']/n_diseases:.0f} CUI/질환, "
          f"total hits {totals_text['hits']}/{totals_text['gold']}, recall {text_r:.1f}%")
    print(f"\n  텍스트매칭/MeSH recall 비율: {text_r/mesh_r:.1f}x" if mesh_r > 0 else "")


if __name__ == "__main__":
    main()
