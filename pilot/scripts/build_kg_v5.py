#!/usr/bin/env python3
"""KG 구축 V5: 3가지 개선 동시 적용.

개선 1: 초록 500편/질환 (V4: 200편)
개선 2: CUI 매칭 정규화 강화 (SY + PAR 2-level)
개선 3: MC sweep 세밀화 (1,2,3,4,5,7,10,15,20)

V4 결과와 동일 조건(200편)도 함께 비교하여
각 개선이 얼마나 기여하는지 측정.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049",
                "T033", "T031", "T040"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT = """Abstract: {abstract}

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

MAX_ABSTRACTS = 500
RESULTS_FILE = RESULTS_DIR / "kg_v5_results.json"


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
    """SY + RQ 동의어 매핑."""
    syn = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("SY", "RQ"):
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
    all_names = defaultdict(set)
    preferred = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] != "ENG": continue
            cui = p[0]
            name = p[14].strip()
            if len(name) >= 3:
                all_names[cui].add(name)
            if p[2] == "P" and cui not in preferred:
                preferred[cui] = name
    return dict(all_names), preferred


def build_aho_automaton(cui_stys):
    target = {cui for cui, stys in cui_stys.items() if stys & ALLOWED_STYS} - BLACKLIST
    A = ahocorasick.Automaton()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[0] not in target or p[1] != "ENG": continue
            lower = p[14].strip().lower()
            if len(lower) >= 4:
                try: A.add_word(lower, (lower, p[0]))
                except: pass
    A.make_automaton()
    return A


def text_match_cuis(text_lower, automaton, exclude_cui=None):
    matched = set()
    for end_idx, (name, cui) in automaton.iter(text_lower):
        if cui == exclude_cui: continue
        start_idx = end_idx - len(name) + 1
        if start_idx > 0 and text_lower[start_idx-1].isalpha(): continue
        if end_idx+1 < len(text_lower) and text_lower[end_idx+1].isalpha(): continue
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
    for eid, en in ev_en.items():
        for fr_name, fr in ev_fr.items():
            if en.get("question_en") == fr.get("question_en") and en.get("question_en"):
                eid_to_fr[eid] = fr_name; break
    gold_pairs = set()
    disease_cuis = {}
    for dn, info in conditions.items():
        d_cui = disease_map.get(dn, {}).get("umls_cui")
        d_name = disease_map.get(dn, {}).get("umls_name", dn)
        if not d_cui: continue
        disease_cuis[dn] = {"cui": d_cui, "umls_name": d_name}
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False): continue
            fr_name = eid_to_fr.get(eid)
            if fr_name and fr_name in umap:
                cui = umap[fr_name].get("cui")
                if cui: gold_pairs.add(tuple(sorted([d_cui, cui])))
    return gold_pairs, disease_cuis


def search_abstracts(conn, dc, dn, umls_name, cui_to_mesh, cui_all_names, limit):
    c = conn.cursor()
    rows, seen = [], set()
    mesh_uids = cui_to_mesh.get(dc, set())
    if mesh_uids:
        mesh_cond = " OR ".join("mesh_terms LIKE '%%%s%%'" % m for m in mesh_uids)
        c.execute(f"SELECT pmid, abstract FROM abstracts WHERE ({mesh_cond}) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?", (limit,))
        for pmid, ab in c.fetchall():
            if pmid not in seen: seen.add(pmid); rows.append((pmid, ab))
    if len(rows) < limit:
        keywords = []
        for raw in [dn, umls_name]:
            kw = re.sub(r'\(.*?\)', '', raw).strip()
            kw = re.sub(r'\b(NOS|unspecified)\b', '', kw, flags=re.IGNORECASE).strip()
            if len(kw) >= 4: keywords.append(kw)
            for part in raw.split('/'):
                part = part.strip()
                if len(part) >= 4 and part not in keywords: keywords.append(part)
        syn_kws = set()
        for name in cui_all_names.get(dc, set()):
            kw = re.sub(r'\(.*?\)', '', name).strip()
            kw = re.sub(r'\b(NOS|unspecified|disease)\b', '', kw, flags=re.IGNORECASE).strip().strip(',').strip('/').strip()
            if len(kw) >= 4 and kw not in keywords: syn_kws.add(kw)
        keywords.extend(sorted(syn_kws, key=len)[:10])
        for kw in keywords:
            if len(rows) >= limit: break
            c.execute("SELECT pmid, abstract FROM abstracts WHERE (title LIKE ? OR abstract LIKE ?) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?",
                      (f"%{kw}%", f"%{kw}%", limit - len(rows)))
            for pmid, ab in c.fetchall():
                if pmid not in seen: seen.add(pmid); rows.append((pmid, ab))
    return rows


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return []


def build_ancestor_map(parent_map, max_depth=2):
    """2-level 조상 맵."""
    cache = {}
    def get(cui, depth=0):
        if cui in cache: return cache[cui]
        if depth >= max_depth or cui not in parent_map:
            cache[cui] = set(); return set()
        anc = set()
        for p in parent_map[cui]:
            anc.add(p)
            if depth + 1 < max_depth:
                anc |= get(p, depth + 1)
        cache[cui] = anc
        return anc
    return get


def evaluate_strict(our_pairs, gold_pairs, parent_map):
    """1-level 전파 매칭 (V4와 동일)."""
    def cm(a, b):
        if a == b: return True
        return b in parent_map.get(a, set()) or a in parent_map.get(b, set())
    mg, mo = set(), set()
    for op in our_pairs:
        for gp in gold_pairs:
            if ((cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0]))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our_pairs) if our_pairs else 0
    r = len(mg)/len(gold_pairs) if gold_pairs else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return {"P": round(p,4), "R": round(r,4), "F1": round(f1,4), "matched": len(mg)}


def evaluate_relaxed(our_pairs, gold_pairs, parent_map, synonym_map):
    """2-level 전파 + 동의어 매칭."""
    get_anc = build_ancestor_map(parent_map, 2)
    def cm(a, b):
        if a == b: return True
        if b in parent_map.get(a, set()) or a in parent_map.get(b, set()): return True
        # 동의어
        if b in synonym_map.get(a, set()) or a in synonym_map.get(b, set()): return True
        # 2-level 조상
        a_anc = get_anc(a)
        b_anc = get_anc(b)
        if b in a_anc or a in b_anc: return True
        if a_anc & b_anc: return True  # 공통 조상
        return False
    mg, mo = set(), set()
    for op in our_pairs:
        for gp in gold_pairs:
            if ((cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0]))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our_pairs) if our_pairs else 0
    r = len(mg)/len(gold_pairs) if gold_pairs else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return {"P": round(p,4), "R": round(r,4), "F1": round(f1,4), "matched": len(mg)}


def main():
    print("=" * 80)
    print(f"KG 구축 V5: 500편 + SY+PAR2 매칭 + MC sweep")
    print("=" * 80)

    print("\n[1/7] UMLS 로드...")
    cui_stys = load_cui_stys()
    parent_map = load_parent_map()
    synonym_map = load_synonym_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items(): cui_to_mesh[cui].add(mesh)
    cui_all_names, cui_preferred = load_cui_all_names()
    automaton = build_aho_automaton(cui_stys)
    gold_pairs, disease_cuis = prepare_gold()
    print(f"  질환: {len(disease_cuis)}개, Gold: {len(gold_pairs)}쌍")

    print(f"\n[2/7] 초록 수집 ({MAX_ABSTRACTS}편/질환)...")
    conn = sqlite3.connect(str(DB_PATH))
    all_tasks = []
    for idx, (dn, dinfo) in enumerate(sorted(disease_cuis.items())):
        dc, un = dinfo["cui"], dinfo["umls_name"]
        rows = search_abstracts(conn, dc, dn, un, cui_to_mesh, cui_all_names, MAX_ABSTRACTS)
        docs = []
        for pmid, ab in rows:
            cuis = text_match_cuis(ab.lower(), automaton, exclude_cui=dc)
            if cuis:
                docs.append({"pmid": pmid, "abstract": ab, "cuis": sorted(cuis)})
        print(f"  [{idx+1}/49] {dn}: {len(docs)}편")
        for doc in docs:
            kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in doc["cuis"])
            all_tasks.append({
                "prompt": PROMPT.format(abstract=doc["abstract"][:3000],
                    disease_name=cui_preferred.get(dc, dc), disease_cui=dc, keywords=kw),
                "disease_cui": dc, "pmid": doc["pmid"],
            })
    conn.close()
    print(f"\n  총: {len(all_tasks)}편")

    print("\n[3/7] vLLM 모델 로드...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=4096)

    print(f"\n[4/7] vLLM batch ({len(all_tasks)}편)...")
    convs = [[{"role": "user", "content": t["prompt"]}] for t in all_tasks]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

    print("\n[5/7] 파싱 + 후처리...")
    all_cls = []
    for task, out in zip(all_tasks, outputs):
        for item in parse_json(out.outputs[0].text):
            cui, rel = item.get("cui", ""), item.get("relation", "")
            if cui and rel:
                all_cls.append({"disease_cui": task["disease_cui"], "cui": cui, "relation": rel, "pmid": task["pmid"]})
    print(f"  Raw: {len(all_cls):,}")

    filtered = []
    for c in all_cls:
        if c["relation"] == "manifestation-of": continue
        dc, cui = c["disease_cui"], c["cui"]
        if cui in synonym_map.get(dc, set()) or dc in synonym_map.get(cui, set()): continue
        if cui in parent_map.get(dc, set()) or dc in parent_map.get(cui, set()): continue
        filtered.append(c)
    print(f"  Filtered: {len(filtered):,}")

    pair_counts = Counter()
    for c in filtered:
        pair_counts[tuple(sorted([c["disease_cui"], c["cui"]]))] += 1

    print(f"\n[6/7] 벤치마크...")

    def build_kg_prop(pc, mc):
        kg = {p for p, cnt in pc.items() if cnt >= mc}
        exp = set(kg)
        for (a, b) in list(kg):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a, pb])))
        return kg, exp

    def build_kg_noprop(pc, mc):
        kg = {p for p, cnt in pc.items() if cnt >= mc}
        return kg, kg

    print(f"\n  {'MC':>4} | {'edges':>7} {'P_1lv':>7} {'R_1lv':>7} {'F1_1lv':>7} | {'P_2lv':>7} {'R_2lv':>7} {'F1_2lv':>7}")
    print(f"  {'-'*70}")

    best_f1_1lv, best_mc_1lv = 0, 1
    best_f1_2lv, best_mc_2lv = 0, 1

    for mc in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        _, exp = build_kg_prop(pair_counts, mc)
        ev1 = evaluate_strict(exp, gold_pairs, parent_map)
        ev2 = evaluate_relaxed(exp, gold_pairs, parent_map, synonym_map)
        m1 = " ★" if ev1["F1"] > best_f1_1lv else ""
        m2 = " ★" if ev2["F1"] > best_f1_2lv else ""
        if ev1["F1"] > best_f1_1lv: best_f1_1lv, best_mc_1lv = ev1["F1"], mc
        if ev2["F1"] > best_f1_2lv: best_f1_2lv, best_mc_2lv = ev2["F1"], mc
        print(f"  MC={mc:>2} | {len(exp):>7,} {ev1['P']:>7.3f} {ev1['R']:>7.3f} {ev1['F1']:>7.3f}{m1} | {ev2['P']:>7.3f} {ev2['R']:>7.3f} {ev2['F1']:>7.3f}{m2}")

    print(f"\n  전파 없이:")
    print(f"  {'MC':>4} | {'edges':>7} {'P_1lv':>7} {'R_1lv':>7} {'F1_1lv':>7} | {'P_2lv':>7} {'R_2lv':>7} {'F1_2lv':>7}")
    print(f"  {'-'*70}")
    for mc in [1, 2, 3, 5, 10]:
        _, raw = build_kg_noprop(pair_counts, mc)
        ev1 = evaluate_strict(raw, gold_pairs, parent_map)
        ev2 = evaluate_relaxed(raw, gold_pairs, parent_map, synonym_map)
        print(f"  MC={mc:>2} | {len(raw):>7,} {ev1['P']:>7.3f} {ev1['R']:>7.3f} {ev1['F1']:>7.3f} | {ev2['P']:>7.3f} {ev2['R']:>7.3f} {ev2['F1']:>7.3f}")

    # [7] 저장
    print(f"\n[7/7] 저장...")
    _, best_exp = build_kg_prop(pair_counts, best_mc_2lv)
    best_ev = evaluate_relaxed(best_exp, gold_pairs, parent_map, synonym_map)

    output = {
        "config": {"model": "google/gemma-4-E4B-it", "engine": "vllm_batch_chat",
                   "max_abstracts": MAX_ABSTRACTS, "best_mc_1lv": best_mc_1lv, "best_mc_2lv": best_mc_2lv},
        "stats": {"total_abstracts": len(all_tasks), "raw": len(all_cls), "filtered": len(filtered),
                  "unique_pairs": len(pair_counts), "time_seconds": round(elapsed, 1)},
        "benchmark_1level": {"best_mc": best_mc_1lv, "best_f1": best_f1_1lv},
        "benchmark_2level_syn": {"best_mc": best_mc_2lv, "best_f1": best_f1_2lv,
                                  "best": best_ev},
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"V5 결과 ({len(all_tasks)}편, {elapsed:.0f}초):")
    print(f"  1-level 매칭: MC={best_mc_1lv}, F1={best_f1_1lv:.3f}")
    print(f"  2-level+SY 매칭: MC={best_mc_2lv}, F1={best_f1_2lv:.3f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
