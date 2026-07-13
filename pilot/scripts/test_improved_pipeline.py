#!/usr/bin/env python3
"""개선된 파이프라인 테스트 (5개 질환 × 20편).

개선 사항:
  1. 검색: 질환 CUI의 모든 UMLS 동의어로 키워드 확대
  2. CUI 추출: STY 확장 (C: +T033/T031/T040)
  3. 프롬프트: 동의어/상위개념 제외 규칙, 관계 유형 단순화
  4. 후처리: 동의어 필터 + manifestation-of 제거 + CUI 정규화
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

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049",
                "T033", "T031", "T040"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

TEST_DISEASES = ["Bronchitis", "Anemia", "Atrial fibrillation", "Anaphylaxis", "Pericarditis"]
ABSTRACTS_PER_DISEASE = 20

# 개선된 프롬프트: 동의어 제외 + 관계 유형 단순화
PROMPT_IMPROVED = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]
(Also known as: {disease_synonyms})

Medical concepts found in this abstract:
{keywords}

Task: Which of these concepts does the abstract describe as a SYMPTOM, CAUSE, COMPLICATION, or RISK FACTOR of {disease_name}?

Rules:
- ONLY include concepts EXPLICITLY linked to {disease_name} in the text
- EXCLUDE synonyms or alternate names of {disease_name} itself
- EXCLUDE concepts that are subtypes or parent categories of {disease_name}
- Classify each as: symptom-of, causes, complication-of, risk-factor-for, co-occurs-with

JSON only: [{{"cui":"...","relation":"..."}}]
If none: []"""


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


def load_synonym_map():
    syn = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] == "SY":
                syn[p[0]].add(p[4])
                syn[p[4]].add(p[0])
    return dict(syn)


def load_mesh_to_cui():
    m = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D"):
                if p[13] not in m:
                    m[p[13]] = p[0]
    return m


def load_cui_all_names():
    """CUI별 모든 영문 이름 (동의어 검색용)."""
    all_names = defaultdict(set)
    preferred = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] != "ENG":
                continue
            cui = p[0]
            name = p[14].strip()
            if len(name) >= 3:
                all_names[cui].add(name)
            if p[2] == "P" and cui not in preferred:
                preferred[cui] = name
    return dict(all_names), preferred


def build_aho_automaton(cui_stys, allowed):
    target = {cui for cui, stys in cui_stys.items() if stys & allowed} - BLACKLIST
    cui_preferred = {}
    A = ahocorasick.Automaton()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui = p[0]
            if cui not in target or p[1] != "ENG":
                continue
            lower = p[14].strip().lower()
            if len(lower) >= 4:
                try:
                    A.add_word(lower, (lower, cui))
                except Exception:
                    pass
            if p[2] == "P" and cui not in cui_preferred:
                cui_preferred[cui] = p[14].strip()
    A.make_automaton()
    return A, cui_preferred


def text_match(text_lower, automaton, exclude_cui=None):
    matched = set()
    for end_idx, (name, cui) in automaton.iter(text_lower):
        if cui == exclude_cui:
            continue
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
    disease_cuis = {}
    for disease_name, info in conditions.items():
        d_cui = disease_map.get(disease_name, {}).get("umls_cui")
        d_name = disease_map.get(disease_name, {}).get("umls_name", disease_name)
        if not d_cui:
            continue
        disease_cuis[disease_name] = {"cui": d_cui, "umls_name": d_name}
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False):
                continue
            fr_name = eid_to_fr.get(eid)
            if fr_name and fr_name in umap:
                cui = umap[fr_name].get("cui")
                if cui:
                    gold_pairs.add(tuple(sorted([d_cui, cui])))
    return gold_pairs, disease_cuis


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
    mg, mo = set(), set()
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


def search_with_synonyms(conn, disease_cui, disease_name, umls_name,
                         cui_to_mesh, cui_all_names, limit):
    """모든 동의어로 키워드 검색."""
    c = conn.cursor()
    rows = []
    seen = set()

    # 1) MeSH
    mesh_uids = cui_to_mesh.get(disease_cui, set())
    if mesh_uids:
        mesh_cond = " OR ".join("mesh_terms LIKE '%%%s%%'" % m for m in mesh_uids)
        c.execute(f"""SELECT pmid, abstract FROM abstracts
            WHERE ({mesh_cond}) AND abstract IS NOT NULL AND length(abstract)>200
            ORDER BY RANDOM() LIMIT ?""", (limit,))
        for pmid, abstract in c.fetchall():
            if pmid not in seen:
                seen.add(pmid)
                rows.append((pmid, abstract))

    # 2) 모든 동의어로 키워드 검색
    if len(rows) < limit:
        all_names = cui_all_names.get(disease_cui, set())
        # DDXPlus 이름도 추가
        all_names.add(disease_name)
        # 키워드 후보 생성
        keywords = set()
        for name in all_names:
            kw = re.sub(r'\(.*?\)', '', name)
            kw = re.sub(r'\b(NOS|unspecified|disease)\b', '', kw, flags=re.IGNORECASE).strip()
            kw = kw.strip(',').strip('/').strip()
            if len(kw) >= 4:
                keywords.add(kw)
            for part in name.split('/'):
                part = part.strip()
                if len(part) >= 4:
                    keywords.add(part)

        for kw in sorted(keywords, key=len, reverse=True)[:10]:
            if len(rows) >= limit:
                break
            remaining = limit - len(rows)
            exclude = ",".join("'%s'" % p for p in seen) if seen else "''"
            c.execute(f"""SELECT pmid, abstract FROM abstracts
                WHERE (title LIKE ? OR abstract LIKE ?)
                AND abstract IS NOT NULL AND length(abstract)>200
                AND pmid NOT IN ({exclude})
                ORDER BY RANDOM() LIMIT ?""",
                      (f"%{kw}%", f"%{kw}%", remaining))
            for pmid, abstract in c.fetchall():
                if pmid not in seen:
                    seen.add(pmid)
                    rows.append((pmid, abstract))

    return rows


def main():
    print("=" * 80)
    print("개선된 파이프라인 테스트")
    print("=" * 80)

    print("\n[1] 데이터 로드...")
    cui_stys = load_cui_stys()
    parent_map = load_parent_map()
    synonym_map = load_synonym_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        cui_to_mesh[cui].add(mesh)
    cui_all_names, cui_preferred = load_cui_all_names()
    automaton, _ = build_aho_automaton(cui_stys, ALLOWED_STYS)
    gold_pairs, disease_cuis = prepare_gold()

    test_gold = set()
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        for gp in gold_pairs:
            if dc in gp:
                test_gold.add(gp)
    print(f"  테스트 Gold: {len(test_gold)}쌍")

    # 초록 수집
    print("\n[2] 초록 수집 (동의어 확장 검색)...")
    conn = sqlite3.connect(str(DB_PATH))
    test_data = {}
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        umls_name = disease_cuis[dn]["umls_name"]
        rows = search_with_synonyms(conn, dc, dn, umls_name, cui_to_mesh,
                                    cui_all_names, ABSTRACTS_PER_DISEASE)
        docs = []
        for pmid, abstract in rows:
            cuis = text_match(abstract.lower(), automaton, exclude_cui=dc)
            if cuis:
                docs.append({"pmid": pmid, "abstract": abstract, "cuis": sorted(cuis)})
        test_data[dn] = docs
        n_syn = len(cui_all_names.get(dc, set()))
        print(f"  {dn}: {len(docs)}편 (동의어 {n_syn}개), 평균CUI={sum(len(d['cuis']) for d in docs)/max(len(docs),1):.0f}")
    conn.close()

    # ========== V2 Original (비교용) ==========
    print("\n[3] V2 Original 프롬프트...")
    PROMPT_V2_ORIG = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Other medical concepts found in this abstract:
{keywords}

Which of the above concepts does the abstract describe as being related to {disease_name}?
For each related concept, classify the relationship type:
symptom-of, causes, complication-of, risk-factor-for, diagnostic-finding-of, manifestation-of, co-occurs-with.

Rules:
- ONLY include concepts that the abstract EXPLICITLY links to {disease_name}
- Do NOT infer relationships not stated in the text

JSON array only: [{{"cui":"...","relation":"..."}}]
If none related: []"""

    v2_pairs = Counter()
    t0 = time.time()
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        d_name = disease_cuis[dn]["umls_name"]
        for doc in test_data[dn]:
            kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in doc["cuis"])
            prompt = PROMPT_V2_ORIG.format(
                abstract=doc["abstract"][:3000], disease_name=d_name,
                disease_cui=dc, keywords=kw)
            response = call_ollama(prompt)
            for item in parse_json(response):
                cui = item.get("cui", "")
                if cui:
                    v2_pairs[tuple(sorted([dc, cui]))] += 1
    v2_time = time.time() - t0

    v2_exp = set(v2_pairs.keys())
    for (a, b) in list(v2_pairs.keys()):
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS:
                v2_exp.add(tuple(sorted([pa, b])))
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS:
                v2_exp.add(tuple(sorted([a, pb])))
    ev_v2 = evaluate(v2_exp, test_gold, parent_map)
    print(f"  V2 orig: pairs={len(v2_pairs)} exp={len(v2_exp)} "
          f"P={ev_v2['P']:.3f} R={ev_v2['R']:.3f} F1={ev_v2['F1']:.3f} | {v2_time:.0f}s")

    # ========== Improved 프롬프트 ==========
    print("\n[4] Improved 프롬프트 (동의어 제외 규칙 + 관계 단순화)...")

    imp_pairs = Counter()
    t0 = time.time()
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        d_name = disease_cuis[dn]["umls_name"]
        d_syns = ", ".join(list(cui_all_names.get(dc, {d_name}))[:5])
        for doc in test_data[dn]:
            kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in doc["cuis"])
            prompt = PROMPT_IMPROVED.format(
                abstract=doc["abstract"][:3000], disease_name=d_name,
                disease_cui=dc, disease_synonyms=d_syns, keywords=kw)
            response = call_ollama(prompt)
            for item in parse_json(response):
                cui = item.get("cui", "")
                if cui:
                    imp_pairs[tuple(sorted([dc, cui]))] += 1
    imp_time = time.time() - t0

    imp_exp = set(imp_pairs.keys())
    for (a, b) in list(imp_pairs.keys()):
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS:
                imp_exp.add(tuple(sorted([pa, b])))
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS:
                imp_exp.add(tuple(sorted([a, pb])))
    ev_imp = evaluate(imp_exp, test_gold, parent_map)
    print(f"  Improved: pairs={len(imp_pairs)} exp={len(imp_exp)} "
          f"P={ev_imp['P']:.3f} R={ev_imp['R']:.3f} F1={ev_imp['F1']:.3f} | {imp_time:.0f}s")

    # ========== Improved + 후처리 ==========
    print("\n[5] Improved + 후처리 (동의어/부모 제거)...")

    imp_filtered = Counter()
    for pair, cnt in imp_pairs.items():
        a, b = pair
        if b in synonym_map.get(a, set()) or a in synonym_map.get(b, set()):
            continue
        if b in parent_map.get(a, set()) or a in parent_map.get(b, set()):
            continue
        imp_filtered[pair] = cnt

    imp_f_exp = set(imp_filtered.keys())
    for (a, b) in list(imp_filtered.keys()):
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS:
                imp_f_exp.add(tuple(sorted([pa, b])))
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS:
                imp_f_exp.add(tuple(sorted([a, pb])))
    ev_imp_f = evaluate(imp_f_exp, test_gold, parent_map)
    print(f"  Imp+filter: pairs={len(imp_filtered)} exp={len(imp_f_exp)} "
          f"P={ev_imp_f['P']:.3f} R={ev_imp_f['R']:.3f} F1={ev_imp_f['F1']:.3f}")

    # 질환별 상세
    print("\n[6] 질환별 상세 (Improved + 후처리)...")
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        d_pairs = {p for p in imp_f_exp if dc in p or
                   any(cui_match(dc, x, parent_map) for x in p)}
        d_gold = {gp for gp in test_gold if dc in gp}
        if d_gold:
            ev = evaluate(d_pairs, d_gold, parent_map)
            print(f"  {dn:<25} P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} "
                  f"match={ev['matched']}/{ev['gold']} pairs={len(d_pairs)}")

    # 요약
    print("\n" + "=" * 80)
    print(f"{'방법':<35} {'pairs':>6} {'exp':>6} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"{'-'*70}")
    print(f"{'V2 Original':<35} {len(v2_pairs):>6} {len(v2_exp):>6} {ev_v2['P']:>7.3f} {ev_v2['R']:>7.3f} {ev_v2['F1']:>7.3f}")
    print(f"{'Improved prompt':<35} {len(imp_pairs):>6} {len(imp_exp):>6} {ev_imp['P']:>7.3f} {ev_imp['R']:>7.3f} {ev_imp['F1']:>7.3f}")
    print(f"{'Improved + post-filter':<35} {len(imp_filtered):>6} {len(imp_f_exp):>6} {ev_imp_f['P']:>7.3f} {ev_imp_f['R']:>7.3f} {ev_imp_f['F1']:>7.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
