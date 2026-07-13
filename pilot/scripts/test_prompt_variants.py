#!/usr/bin/env python3
"""프롬프트 변형 비교 테스트.

3가지 프롬프트로 동일한 초록에서 관계를 추출하고 DDXPlus gold 대비 비교.

V1: 오픈엔디드 (모든 CUI 쌍 발견) — 질환 CUI 포함
V2: 질환 중심 ("이 질환과 관련된 CUI는?")
V3: 명시적 쌍 분류 (S2-J 스타일, 모든 쌍 제시)
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
import requests

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

TEST_DISEASES = ["Bronchitis", "Epiglottitis", "Influenza", "Pericarditis", "Pulmonary embolism"]
ABSTRACTS_PER_DISEASE = 10

# === 프롬프트 3가지 ===

PROMPT_V1 = """Abstract: {abstract}

Keywords found in this abstract:
{keywords}

From the abstract above, identify ALL pairs of keywords that have a medical relationship.
Medical relationships include: symptom-of, causes, complication-of, risk-factor-for, co-occurs-with, diagnostic-finding-of, manifestation-of.

Rules:
- ONLY report pairs where the abstract EXPLICITLY describes a relationship
- Do NOT infer relationships not stated in the text

JSON array only: [{{"cui_a":"...","cui_b":"...","relation":"..."}}]
If no pairs found: []"""

PROMPT_V2 = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Other medical concepts found in this abstract:
{keywords}

Which of the above concepts does the abstract describe as being related to {disease_name}?
For each related concept, classify the relationship type: symptom-of, causes, complication-of, risk-factor-for, diagnostic-finding-of, manifestation-of.

Rules:
- ONLY include concepts that the abstract EXPLICITLY links to {disease_name}
- Do NOT infer relationships not stated in the text

JSON array only: [{{"cui":"...","relation":"..."}}]
If none related: []"""

PROMPT_V3 = """Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {abstract}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""


def load_cui_stys():
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def load_parent_map():
    parents = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parents[p[0]].add(p[4])
    return dict(parents)


def load_mesh_to_cui():
    m = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D"):
                if p[13] not in m:
                    m[p[13]] = p[0]
    return m


def build_aho_automaton(cui_stys):
    diso_cuis = {cui for cui, stys in cui_stys.items() if stys & ALLOWED_STYS} - BLACKLIST
    cui_preferred = {}
    name_to_cuis = defaultdict(set)
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui = p[0]
            if cui not in diso_cuis or p[1] != "ENG":
                continue
            name = p[14].strip()
            lower = name.lower()
            if len(lower) >= 4:
                name_to_cuis[lower].add(cui)
            if p[2] == "P" and cui not in cui_preferred:
                cui_preferred[cui] = name
    A = ahocorasick.Automaton()
    for name, cuis in name_to_cuis.items():
        A.add_word(name, (name, next(iter(cuis))))
    A.make_automaton()
    return A, cui_preferred


def text_match_cuis(text_lower, automaton):
    """질환 CUI 제외하지 않음 — 모든 DISO CUI 반환."""
    matched = set()
    for end_idx, (name, cui) in automaton.iter(text_lower):
        start_idx = end_idx - len(name) + 1
        if start_idx > 0 and text_lower[start_idx - 1].isalpha():
            continue
        if end_idx + 1 < len(text_lower) and text_lower[end_idx + 1].isalpha():
            continue
        matched.add(cui)
    return matched


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

    gold_pairs = set()
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
                    gold_pairs.add(tuple(sorted([d_cui, cui])))
                    symptom_cuis.add(cui)
        disease_gold[d_cui] = symptom_cuis
    return gold_pairs, disease_cuis, disease_gold


def call_ollama(prompt):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }, timeout=300)
    return resp.json().get("response", "")


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return []


def cui_match(a, b, parent_map):
    if a == b:
        return True
    return b in parent_map.get(a, set()) or a in parent_map.get(b, set())


def evaluate(our_pairs, gold_pairs, parent_map):
    mg = set()
    mo = set()
    for op in our_pairs:
        for gp in gold_pairs:
            if ((cui_match(op[0], gp[0], parent_map) and cui_match(op[1], gp[1], parent_map)) or
                    (cui_match(op[0], gp[1], parent_map) and cui_match(op[1], gp[0], parent_map))):
                mg.add(gp)
                mo.add(op)
    p = len(mo) / len(our_pairs) if our_pairs else 0
    r = len(mg) / len(gold_pairs) if gold_pairs else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
            "matched": len(mg), "our": len(our_pairs), "gold": len(gold_pairs)}


def main():
    print("=" * 80)
    print("프롬프트 변형 비교: V1(오픈엔디드) vs V2(질환중심) vs V3(S2-J)")
    print(f"테스트: {len(TEST_DISEASES)}개 질환 × {ABSTRACTS_PER_DISEASE}편")
    print("=" * 80)

    print("\n[1] 데이터 로드...")
    cui_stys = load_cui_stys()
    parent_map = load_parent_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        cui_to_mesh[cui].add(mesh)
    automaton, cui_preferred = build_aho_automaton(cui_stys)
    gold_pairs, disease_cuis, disease_gold = prepare_gold()

    # 테스트 질환 초록 수집
    print("\n[2] 초록 수집 + CUI 추출...")
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    test_data = {}  # disease_name -> [{pmid, abstract, cuis, disease_cui}]
    for disease_name in TEST_DISEASES:
        dinfo = disease_cuis[disease_name]
        dc = dinfo["cui"]
        dn = dinfo["umls_name"]

        mesh_uids = cui_to_mesh.get(dc, set())
        if mesh_uids:
            mesh_cond = " OR ".join(f"mesh_terms LIKE '%{m}%'" for m in mesh_uids)
            c.execute(f"""
                SELECT pmid, abstract FROM abstracts
                WHERE ({mesh_cond}) AND abstract IS NOT NULL AND length(abstract) > 200
                ORDER BY RANDOM() LIMIT ?
            """, (ABSTRACTS_PER_DISEASE,))
        else:
            keywords = re.sub(r'\(.*?\)', '', dn).strip()
            c.execute("""
                SELECT pmid, abstract FROM abstracts
                WHERE (title LIKE ? OR abstract LIKE ?)
                AND abstract IS NOT NULL AND length(abstract) > 200
                ORDER BY RANDOM() LIMIT ?
            """, (f"%{keywords}%", f"%{keywords}%", ABSTRACTS_PER_DISEASE))

        rows = c.fetchall()
        docs = []
        for pmid, abstract in rows:
            cuis = text_match_cuis(abstract.lower(), automaton)
            if len(cuis) >= 2:
                docs.append({"pmid": pmid, "abstract": abstract, "cuis": sorted(cuis), "disease_cui": dc})
        test_data[disease_name] = docs
        print(f"  {disease_name}: {len(docs)}편, 평균CUI={sum(len(d['cuis']) for d in docs)/max(len(docs),1):.0f}")

    conn.close()

    # 테스트 gold 범위 (테스트 질환만)
    test_gold = set()
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        for gp in gold_pairs:
            if dc in gp:
                test_gold.add(gp)
    print(f"\n  테스트 Gold: {len(test_gold)}쌍")

    # === V1: 오픈엔디드 (질환 CUI 포함) ===
    print("\n[3] V1: 오픈엔디드 (모든 CUI 쌍 발견)...")
    v1_pairs = Counter()
    v1_time = time.time()
    v1_total = 0

    for disease_name, docs in test_data.items():
        dc = disease_cuis[disease_name]["cui"]
        for doc in docs:
            keywords_text = "\n".join(
                f"- {cui_preferred.get(cui, cui)} [{cui}]"
                for cui in doc["cuis"]
            )
            prompt = PROMPT_V1.format(abstract=doc["abstract"][:3000], keywords=keywords_text)
            response = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                a, b = item.get("cui_a", ""), item.get("cui_b", "")
                if a and b:
                    v1_pairs[tuple(sorted([a, b]))] += 1
            v1_total += 1
            if v1_total % 5 == 0:
                print(f"    {v1_total}/{sum(len(d) for d in test_data.values())} "
                      f"pairs={len(v1_pairs)}")

    v1_time = time.time() - v1_time
    v1_expanded = set(v1_pairs.keys())
    for (a, b) in list(v1_pairs.keys()):
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS:
                v1_expanded.add(tuple(sorted([pa, b])))
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS:
                v1_expanded.add(tuple(sorted([a, pb])))
    ev1 = evaluate(v1_expanded, test_gold, parent_map)
    print(f"  V1: {len(v1_pairs)} raw → {len(v1_expanded)} expanded | "
          f"P={ev1['P']:.3f} R={ev1['R']:.3f} F1={ev1['F1']:.3f} | {v1_time:.0f}s")

    # === V2: 질환 중심 ===
    print("\n[4] V2: 질환 중심 (질환과 관련된 CUI만)...")
    v2_pairs = Counter()
    v2_time = time.time()
    v2_total = 0

    for disease_name, docs in test_data.items():
        dc = disease_cuis[disease_name]["cui"]
        dn = disease_cuis[disease_name]["umls_name"]
        for doc in docs:
            other_cuis = [cui for cui in doc["cuis"] if cui != dc]
            if not other_cuis:
                continue
            keywords_text = "\n".join(
                f"- {cui_preferred.get(cui, cui)} [{cui}]"
                for cui in other_cuis
            )
            prompt = PROMPT_V2.format(
                abstract=doc["abstract"][:3000],
                disease_name=dn, disease_cui=dc,
                keywords=keywords_text
            )
            response = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                cui = item.get("cui", "")
                if cui:
                    v2_pairs[tuple(sorted([dc, cui]))] += 1
            v2_total += 1
            if v2_total % 5 == 0:
                print(f"    {v2_total}/{sum(len(d) for d in test_data.values())} "
                      f"pairs={len(v2_pairs)}")

    v2_time = time.time() - v2_time
    v2_expanded = set(v2_pairs.keys())
    for (a, b) in list(v2_pairs.keys()):
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS:
                v2_expanded.add(tuple(sorted([pa, b])))
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS:
                v2_expanded.add(tuple(sorted([a, pb])))
    ev2 = evaluate(v2_expanded, test_gold, parent_map)
    print(f"  V2: {len(v2_pairs)} raw → {len(v2_expanded)} expanded | "
          f"P={ev2['P']:.3f} R={ev2['R']:.3f} F1={ev2['F1']:.3f} | {v2_time:.0f}s")

    # === V3: S2-J (명시적 쌍, 질환-CUI 쌍만) ===
    print("\n[5] V3: S2-J (질환-CUI 쌍 명시적 분류)...")
    v3_pairs = Counter()
    v3_time = time.time()
    v3_total = 0

    for disease_name, docs in test_data.items():
        dc = disease_cuis[disease_name]["cui"]
        for doc in docs:
            other_cuis = [cui for cui in doc["cuis"] if cui != dc]
            if not other_cuis:
                continue
            # 질환-CUI 쌍만 생성 (최대 10개)
            pairs_to_check = [(dc, cui) for cui in other_cuis[:10]]
            pairs_text = "\n".join(
                f"- ({cui_preferred.get(a, a)[:40]}, {cui_preferred.get(b, b)[:40]}) [CUI: {a}, {b}]"
                for a, b in pairs_to_check
            )
            prompt = PROMPT_V3.format(abstract=doc["abstract"][:2500], pairs=pairs_text)
            response = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                cls = item.get("classification", "").lower().replace(" ", "_")
                if cls == "present":
                    a, b = item.get("cui_a", ""), item.get("cui_b", "")
                    if a and b:
                        v3_pairs[tuple(sorted([a, b]))] += 1
            v3_total += 1
            if v3_total % 5 == 0:
                print(f"    {v3_total}/{sum(len(d) for d in test_data.values())} "
                      f"pairs={len(v3_pairs)}")

    v3_time = time.time() - v3_time
    v3_expanded = set(v3_pairs.keys())
    for (a, b) in list(v3_pairs.keys()):
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS:
                v3_expanded.add(tuple(sorted([pa, b])))
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS:
                v3_expanded.add(tuple(sorted([a, pb])))
    ev3 = evaluate(v3_expanded, test_gold, parent_map)
    print(f"  V3: {len(v3_pairs)} raw → {len(v3_expanded)} expanded | "
          f"P={ev3['P']:.3f} R={ev3['R']:.3f} F1={ev3['F1']:.3f} | {v3_time:.0f}s")

    # === 요약 ===
    print("\n" + "=" * 80)
    print(f"{'방법':<45} {'P':>6} {'R':>6} {'F1':>6} {'Raw':>5} {'Exp':>6} {'Time':>5}")
    print("-" * 80)
    print(f"{'V1: 오픈엔디드 (모든 쌍 발견)':<45} {ev1['P']:>6.3f} {ev1['R']:>6.3f} {ev1['F1']:>6.3f} {len(v1_pairs):>5} {len(v1_expanded):>6} {v1_time:>4.0f}s")
    print(f"{'V2: 질환중심 (질환-CUI 관계만)':<45} {ev2['P']:>6.3f} {ev2['R']:>6.3f} {ev2['F1']:>6.3f} {len(v2_pairs):>5} {len(v2_expanded):>6} {v2_time:>4.0f}s")
    print(f"{'V3: S2-J (명시적 쌍 present/not_related)':<45} {ev3['P']:>6.3f} {ev3['R']:>6.3f} {ev3['F1']:>6.3f} {len(v3_pairs):>5} {len(v3_expanded):>6} {v3_time:>4.0f}s")
    print("=" * 80)

    # 샘플 출력
    print("\n=== V2 샘플 (질환중심) ===")
    names = cui_preferred
    for pair, cnt in v2_pairs.most_common(15):
        print(f"  {names.get(pair[0],pair[0])[:30]:<30} ↔ {names.get(pair[1],pair[1])[:30]:<30} (×{cnt})")


if __name__ == "__main__":
    main()
