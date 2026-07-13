#!/usr/bin/env python3
"""Semantic Type 확장 테스트.

5가지 STY 조합으로 텍스트 매칭 CUI 추출 후 gold recall 비교.
이어서 최적 조합으로 V2 프롬프트 LLM 테스트 (5개 질환 × 10편).

조합:
  A: 현재 (DISO only, 10 types)
  B: +T033(Finding)
  C: +T033+T031(Body Substance)+T040(Organism Function)
  D: +T033+T031+T040+T201(Clinical Attribute)+T060(Diagnostic Procedure)
  E: 전체 (A+모든 후보)
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

BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

BASE_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}

COMBOS = {
    "A_base":     BASE_STYS,
    "B_+T033":    BASE_STYS | {"T033"},
    "C_+T033_31_40": BASE_STYS | {"T033", "T031", "T040"},
    "D_+all_symptom": BASE_STYS | {"T033", "T031", "T040", "T201"},
    "E_full":     BASE_STYS | {"T033", "T031", "T040", "T201", "T060", "T079"},
}

TEST_DISEASES = ["Bronchitis", "Anemia", "Atrial fibrillation", "Anaphylaxis", "Pericarditis"]
ABSTRACTS_PER_DISEASE = 20

PROMPT_V2 = """Abstract: {abstract}

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


def build_aho_for_stys(allowed_stys, cui_stys):
    """주어진 STY 조합으로 Aho-Corasick automaton 구축."""
    target_cuis = {cui for cui, stys in cui_stys.items() if stys & allowed_stys} - BLACKLIST
    cui_preferred = {}
    A = ahocorasick.Automaton()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui = p[0]
            if cui not in target_cuis or p[1] != "ENG":
                continue
            name = p[14].strip()
            lower = name.lower()
            if len(lower) >= 4:
                try:
                    A.add_word(lower, (lower, cui))
                except Exception:
                    pass
            if p[2] == "P" and cui not in cui_preferred:
                cui_preferred[cui] = name
    A.make_automaton()
    return A, cui_preferred, len(target_cuis)


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
    disease_gold = {}
    disease_cuis = {}
    for disease_name, info in conditions.items():
        d_cui = disease_map.get(disease_name, {}).get("umls_cui")
        d_name = disease_map.get(disease_name, {}).get("umls_name", disease_name)
        if not d_cui:
            continue
        disease_cuis[disease_name] = {"cui": d_cui, "umls_name": d_name}
        syms = set()
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False):
                continue
            fr_name = eid_to_fr.get(eid)
            if fr_name and fr_name in umap:
                cui = umap[fr_name].get("cui")
                if cui:
                    gold_pairs.add(tuple(sorted([d_cui, cui])))
                    syms.add(cui)
        disease_gold[d_cui] = syms
    return gold_pairs, disease_cuis, disease_gold


def expand_1level(cuis, parent_map):
    exp = set(cuis)
    for c in cuis:
        exp.update(parent_map.get(c, set()))
    return exp


def count_hits(extracted, gold, parent_map):
    return len(expand_1level(extracted, parent_map) & expand_1level(gold, parent_map))


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
    print("=" * 90)
    print("Semantic Type 확장 테스트")
    print("=" * 90)

    print("\n[1] 데이터 로드...")
    cui_stys = load_cui_stys()
    parent_map = load_parent_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        cui_to_mesh[cui].add(mesh)
    gold_pairs, disease_cuis, disease_gold = prepare_gold()

    # 테스트 gold
    test_gold = set()
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        for gp in gold_pairs:
            if dc in gp:
                test_gold.add(gp)

    # 초록 수집 (1회, 공유)
    print("\n[2] 초록 수집...")
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    test_abstracts = {}
    for dn in TEST_DISEASES:
        dc = disease_cuis[dn]["cui"]
        mesh_uids = cui_to_mesh.get(dc, set())
        if mesh_uids:
            mesh_cond = " OR ".join(f"mesh_terms LIKE '%{m}%'" for m in mesh_uids)
            c.execute(f"""SELECT pmid, abstract FROM abstracts
                WHERE ({mesh_cond}) AND abstract IS NOT NULL AND length(abstract)>200
                ORDER BY RANDOM() LIMIT ?""", (ABSTRACTS_PER_DISEASE,))
        else:
            kw = re.sub(r'\(.*?\)', '', disease_cuis[dn]["umls_name"]).strip()
            c.execute("""SELECT pmid, abstract FROM abstracts
                WHERE (title LIKE ? OR abstract LIKE ?) AND abstract IS NOT NULL AND length(abstract)>200
                ORDER BY RANDOM() LIMIT ?""", (f"%{kw}%", f"%{kw}%", ABSTRACTS_PER_DISEASE))
        test_abstracts[dn] = c.fetchall()
        print(f"  {dn}: {len(test_abstracts[dn])}편")
    conn.close()

    # ========== PART 1: CUI 추출 커버리지 비교 ==========
    print("\n" + "=" * 90)
    print("PART 1: STY 조합별 CUI 추출 커버리지 (텍스트 매칭)")
    print("=" * 90)

    best_combo = None
    best_recall = 0

    for combo_name, allowed in COMBOS.items():
        print(f"\n--- {combo_name} ({len(allowed)} STY) ---")
        t0 = time.time()
        automaton, cui_pref, n_cuis = build_aho_for_stys(allowed, cui_stys)
        build_time = time.time() - t0
        print(f"  Automaton: {n_cuis:,} CUI, 구축 {build_time:.1f}s")

        total_hits = 0
        total_gold = 0
        total_cuis = 0

        for dn in TEST_DISEASES:
            dc = disease_cuis[dn]["cui"]
            gold_syms = disease_gold.get(dc, set())
            if not gold_syms:
                continue

            all_cuis = set()
            for pmid, abstract in test_abstracts[dn]:
                matched = text_match(abstract.lower(), automaton, exclude_cui=dc)
                all_cuis.update(matched)

            hits = count_hits(all_cuis, gold_syms, parent_map)
            pct = 100 * hits / len(gold_syms) if gold_syms else 0
            total_hits += hits
            total_gold += len(gold_syms)
            total_cuis += len(all_cuis)

            print(f"  {dn:<30} gold={len(gold_syms):>2} CUI={len(all_cuis):>4} hit={hits:>2} ({pct:.0f}%)")

        recall = 100 * total_hits / total_gold if total_gold else 0
        print(f"  총합: CUI={total_cuis:,} hit={total_hits}/{total_gold} recall={recall:.1f}%")

        if recall > best_recall:
            best_recall = recall
            best_combo = combo_name

    print(f"\n최적 CUI 추출: {best_combo} (recall {best_recall:.1f}%)")

    # ========== PART 2: 최적 2개 + base로 LLM 테스트 ==========
    print("\n" + "=" * 90)
    print("PART 2: LLM 관계 분류 비교 (V2 프롬프트)")
    print("=" * 90)

    # base + 최적 + C(중간)로 LLM 비교
    llm_combos = ["A_base", "C_+T033_31_40", best_combo]
    llm_combos = list(dict.fromkeys(llm_combos))  # 중복 제거

    for combo_name in llm_combos:
        allowed = COMBOS[combo_name]
        print(f"\n--- LLM: {combo_name} ---")
        automaton, cui_pref, n_cuis = build_aho_for_stys(allowed, cui_stys)

        pair_counts = Counter()
        total_rels = 0
        t0 = time.time()

        for dn in TEST_DISEASES:
            dc = disease_cuis[dn]["cui"]
            d_name = disease_cuis[dn]["umls_name"]

            for pmid, abstract in test_abstracts[dn]:
                cuis = text_match(abstract.lower(), automaton, exclude_cui=dc)
                if not cuis:
                    continue

                keywords_text = "\n".join(
                    f"- {cui_pref.get(cui, cui)} [{cui}]"
                    for cui in sorted(cuis)
                )
                prompt = PROMPT_V2.format(
                    abstract=abstract[:3000],
                    disease_name=d_name, disease_cui=dc,
                    keywords=keywords_text,
                )
                response = call_ollama(prompt)
                parsed = parse_json(response)
                for item in parsed:
                    cui = item.get("cui", "")
                    if cui:
                        pair_counts[tuple(sorted([dc, cui]))] += 1
                        total_rels += 1

        llm_time = time.time() - t0

        # 평가 (MC=1, CUI 전파)
        expanded = set(pair_counts.keys())
        for (a, b) in list(pair_counts.keys()):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & allowed:
                    expanded.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & allowed:
                    expanded.add(tuple(sorted([a, pb])))

        ev = evaluate(expanded, test_gold, parent_map)
        print(f"  relations={total_rels} pairs={len(pair_counts)} expanded={len(expanded)}")
        print(f"  P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} match={ev['matched']}/{ev['gold']} | {llm_time:.0f}s")

        # 질환별 상세
        for dn in TEST_DISEASES:
            dc = disease_cuis[dn]["cui"]
            gold_syms = disease_gold.get(dc, set())
            disease_pairs = {p for p in expanded if dc in p or
                             any(cui_match(dc, x, parent_map) for x in p)}
            disease_ev = evaluate(disease_pairs, {gp for gp in test_gold if dc in gp}, parent_map)
            print(f"    {dn:<25} P={disease_ev['P']:.3f} R={disease_ev['R']:.3f} F1={disease_ev['F1']:.3f} "
                  f"match={disease_ev['matched']}/{disease_ev['gold']}")

    print("\n" + "=" * 90)
    print("완료")
    print("=" * 90)


if __name__ == "__main__":
    main()
