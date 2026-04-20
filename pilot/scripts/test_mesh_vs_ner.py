#!/usr/bin/env python3
"""MeSH 태그 vs scispaCy NER: CUI 추출 비교 테스트.

동일 초록에서 두 방법으로 CUI를 추출하고
DDXPlus gold standard 대비 커버리지를 비교한다.

테스트 대상: DDXPlus 49개 질환 중 MeSH 매핑된 질환에서 각 20편
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

UMLS_DIR = Path("data/umls_extracted")
DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
ABSTRACTS_PER_DISEASE = 20

# ============================================================
# UMLS 로드
# ============================================================

def load_mesh_to_cui():
    m = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D"):
                if p[13] not in m:
                    m[p[13]] = p[0]
    return m


def load_cui_stys():
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def load_cui_names():
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and (p[0] not in names or p[2] == "P"):
                names[p[0]] = p[14]
    return names


def load_parent_map():
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parents[p[0]].add(p[4])
    return dict(parents)


# ============================================================
# Gold Standard
# ============================================================

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

    # 질환별 gold symptom CUI 셋
    disease_gold = {}  # disease_cui -> set of symptom CUIs
    disease_cuis = {}  # disease_name -> {cui, umls_name}
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


# ============================================================
# CUI 매칭 (1-level 전파 포함)
# ============================================================

def expand_cuis_1level(cuis: set, parent_map: dict) -> set:
    """CUI 셋에 1-level parent 추가."""
    expanded = set(cuis)
    for cui in cuis:
        expanded.update(parent_map.get(cui, set()))
    return expanded


def count_gold_hits(extracted_cuis: set, gold_cuis: set, parent_map: dict) -> int:
    """추출된 CUI 중 gold CUI와 매칭되는 수 (1-level 전파 포함)."""
    expanded = expand_cuis_1level(extracted_cuis, parent_map)
    gold_expanded = expand_cuis_1level(gold_cuis, parent_map)
    return len(expanded & gold_expanded)


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 80)
    print("MeSH 태그 vs scispaCy NER: CUI 추출 비교")
    print("=" * 80)

    # UMLS 로드
    print("\n[1] UMLS 로드...")
    mesh_to_cui = load_mesh_to_cui()
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    parent_map = load_parent_map()

    ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
    BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        cui_to_mesh[cui].add(mesh)

    disease_gold, disease_cuis = prepare_gold()
    print(f"  질환: {len(disease_cuis)}개, Gold 있는 질환: {len(disease_gold)}개")

    # scispaCy 로드
    print("\n[2] scispaCy 로드...")
    import spacy
    import scispacy  # noqa
    from scispacy.linking import EntityLinker  # noqa

    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True, "linker_name": "umls",
    })
    print("  완료")

    # DB 연결
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # 질환별 비교
    print(f"\n[3] 질환별 비교 (각 {ABSTRACTS_PER_DISEASE}편)...")
    print(f"{'질환':<35} {'Gold':>4} | {'MeSH CUI':>8} {'Hit':>3} {'%':>5} | {'NER CUI':>7} {'Hit':>3} {'%':>5} | {'시간':>5}")
    print("-" * 100)

    totals = {
        "diseases": 0,
        "mesh_cuis": 0, "mesh_hits": 0, "mesh_gold": 0,
        "ner_cuis": 0, "ner_hits": 0, "ner_gold": 0,
    }

    for disease_name, dinfo in sorted(disease_cuis.items()):
        dc = dinfo["cui"]
        dn = dinfo["umls_name"]

        if dc not in disease_gold:
            continue

        gold_cuis = disease_gold[dc]

        # MeSH 매핑 확인
        mesh_uids = cui_to_mesh.get(dc, set())
        if not mesh_uids:
            # 키워드 검색 fallback
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
        else:
            mesh_cond = " OR ".join(f"mesh_terms LIKE '%{m}%'" for m in mesh_uids)
            c.execute(f"""
                SELECT pmid, mesh_terms, abstract FROM abstracts
                WHERE ({mesh_cond})
                AND abstract IS NOT NULL AND length(abstract) > 200
                ORDER BY RANDOM() LIMIT ?
            """, (ABSTRACTS_PER_DISEASE,))

        rows = c.fetchall()
        if not rows:
            continue

        # MeSH 방식: mesh_terms 필드에서 CUI 추출
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

        # NER 방식: 초록 본문에서 scispaCy 추출
        ner_cuis_all = set()
        t0 = time.time()
        for _, _, abstract in rows:
            doc = nlp(abstract[:5000])
            for ent in doc.ents:
                for cui, score in ent._.kb_ents:
                    if score < 0.85:
                        break
                    stys = cui_stys.get(cui, set())
                    if stys & ALLOWED_STYS and cui not in BLACKLIST and cui != dc:
                        ner_cuis_all.add(cui)
        ner_time = time.time() - t0

        ner_hits = count_gold_hits(ner_cuis_all, gold_cuis, parent_map)

        # 출력
        mesh_pct = 100 * mesh_hits / len(gold_cuis) if gold_cuis else 0
        ner_pct = 100 * ner_hits / len(gold_cuis) if gold_cuis else 0

        short_name = disease_name[:33]
        print(f"{short_name:<35} {len(gold_cuis):>4} | "
              f"{len(mesh_cuis_all):>8} {mesh_hits:>3} {mesh_pct:>4.0f}% | "
              f"{len(ner_cuis_all):>7} {ner_hits:>3} {ner_pct:>4.0f}% | "
              f"{ner_time:>4.1f}s")

        totals["diseases"] += 1
        totals["mesh_cuis"] += len(mesh_cuis_all)
        totals["mesh_hits"] += mesh_hits
        totals["mesh_gold"] += len(gold_cuis)
        totals["ner_cuis"] += len(ner_cuis_all)
        totals["ner_hits"] += ner_hits
        totals["ner_gold"] += len(gold_cuis)

    conn.close()

    # 요약
    print("-" * 100)
    mesh_recall = 100 * totals["mesh_hits"] / totals["mesh_gold"] if totals["mesh_gold"] else 0
    ner_recall = 100 * totals["ner_hits"] / totals["ner_gold"] if totals["ner_gold"] else 0
    print(f"\n{'요약':<35} {'':>4} | "
          f"{totals['mesh_cuis']:>8} {totals['mesh_hits']:>3} {mesh_recall:>4.1f}% | "
          f"{totals['ner_cuis']:>7} {totals['ner_hits']:>3} {ner_recall:>4.1f}% |")
    print(f"\n  테스트 질환: {totals['diseases']}개")
    print(f"  MeSH: 평균 {totals['mesh_cuis']/totals['diseases']:.0f} CUI/질환, gold recall {mesh_recall:.1f}%")
    print(f"  NER:  평균 {totals['ner_cuis']/totals['diseases']:.0f} CUI/질환, gold recall {ner_recall:.1f}%")


if __name__ == "__main__":
    main()
