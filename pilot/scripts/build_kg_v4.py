#!/usr/bin/env python3
"""KG 구축 V4: 동의어 검색 확장(200편) + vLLM batch + 후처리.

V3 대비 개선:
  - 질환당 200편 (V3: 100편)
  - vLLM batch (5분 vs Ollama 18시간)
  - chat template 적용
  - 후처리: manifestation-of 제거 + 동의어/부모 제거
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

MAX_ABSTRACTS_PER_DISEASE = 200
RESULTS_FILE = RESULTS_DIR / "kg_v4_results.json"


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


def build_aho_automaton(cui_stys):
    target = {cui for cui, stys in cui_stys.items() if stys & ALLOWED_STYS} - BLACKLIST
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
    A.make_automaton()
    return A


def text_match_cuis(text_lower, automaton, exclude_cui=None):
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


def search_with_synonyms(conn, disease_cui, disease_name, umls_name,
                         cui_to_mesh, cui_all_names, limit):
    c = conn.cursor()
    rows = []
    seen = set()

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

    if len(rows) < limit:
        all_names = cui_all_names.get(disease_cui, set())
        keywords = []
        # DDXPlus 원래 이름 최우선
        for raw in [disease_name, umls_name]:
            kw = re.sub(r'\(.*?\)', '', raw).strip()
            kw = re.sub(r'\b(NOS|unspecified)\b', '', kw, flags=re.IGNORECASE).strip()
            if len(kw) >= 4:
                keywords.append(kw)
            for part in raw.split('/'):
                part = part.strip()
                if len(part) >= 4 and part not in keywords:
                    keywords.append(part)
        # UMLS 동의어 (짧은 것 우선 — 검색 히트율 높음)
        syn_kws = set()
        for name in all_names:
            kw = re.sub(r'\(.*?\)', '', name).strip()
            kw = re.sub(r'\b(NOS|unspecified|disease)\b', '', kw, flags=re.IGNORECASE).strip()
            kw = kw.strip(',').strip('/').strip()
            if len(kw) >= 4 and kw not in keywords:
                syn_kws.add(kw)
        keywords.extend(sorted(syn_kws, key=len)[:10])

        for kw in keywords:
            if len(rows) >= limit:
                break
            remaining = limit - len(rows)
            c.execute("""SELECT pmid, abstract FROM abstracts
                WHERE (title LIKE ? OR abstract LIKE ?)
                AND abstract IS NOT NULL AND length(abstract)>200
                ORDER BY RANDOM() LIMIT ?""",
                      (f"%{kw}%", f"%{kw}%", remaining))
            for pmid, abstract in c.fetchall():
                if pmid not in seen:
                    seen.add(pmid)
                    rows.append((pmid, abstract))
    return rows


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


def main():
    print("=" * 80)
    print(f"KG 구축 V4: 동의어 검색({MAX_ABSTRACTS_PER_DISEASE}편) + vLLM batch + 후처리")
    print("=" * 80)

    # [1] UMLS 로드
    print("\n[1/7] UMLS 로드...")
    cui_stys = load_cui_stys()
    parent_map = load_parent_map()
    synonym_map = load_synonym_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        cui_to_mesh[cui].add(mesh)
    cui_all_names, cui_preferred = load_cui_all_names()
    automaton = build_aho_automaton(cui_stys)
    gold_pairs, disease_cuis = prepare_gold()
    print(f"  질환: {len(disease_cuis)}개, Gold: {len(gold_pairs)}쌍")

    # [2] 초록 수집 + CUI 추출
    print(f"\n[2/7] 초록 수집 (질환당 {MAX_ABSTRACTS_PER_DISEASE}편)...")
    conn = sqlite3.connect(str(DB_PATH))
    all_tasks = []

    for idx, (disease_name, dinfo) in enumerate(sorted(disease_cuis.items())):
        dc = dinfo["cui"]
        dn = dinfo["umls_name"]
        rows = search_with_synonyms(conn, dc, disease_name, dn,
                                    cui_to_mesh, cui_all_names, MAX_ABSTRACTS_PER_DISEASE)

        docs = []
        for pmid, abstract in rows:
            cuis = text_match_cuis(abstract.lower(), automaton, exclude_cui=dc)
            if cuis:
                docs.append({"pmid": pmid, "abstract": abstract, "cuis": sorted(cuis)})

        n_syn = len(cui_all_names.get(dc, set()))
        avg = sum(len(d["cuis"]) for d in docs) / max(len(docs), 1)
        print(f"  [{idx+1}/49] {disease_name}: {len(docs)}편 (동의어{n_syn}), 평균CUI={avg:.0f}")

        for doc in docs:
            kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in doc["cuis"])
            all_tasks.append({
                "prompt": PROMPT.format(
                    abstract=doc["abstract"][:3000],
                    disease_name=cui_preferred.get(dc, dc),
                    disease_cui=dc, keywords=kw),
                "disease_cui": dc,
                "pmid": doc["pmid"],
            })

    conn.close()
    print(f"\n  총 초록: {len(all_tasks)}편")

    # [3] vLLM 모델 로드
    print("\n[3/7] vLLM 모델 로드 (GPU0)...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(
        model="google/gemma-4-E4B-it",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    sampling = SamplingParams(temperature=0, max_tokens=4096)

    # [4] vLLM batch 추론 (chat template)
    print(f"\n[4/7] vLLM batch 추론 ({len(all_tasks)}편)...")
    conversations = [[{"role": "user", "content": t["prompt"]}] for t in all_tasks]

    t0 = time.time()
    outputs = llm.chat(conversations, sampling)
    elapsed = time.time() - t0
    print(f"  완료: {len(outputs)}건, {elapsed:.1f}초 ({len(outputs)/elapsed:.1f}/s)")

    # [5] 파싱 + 후처리
    print("\n[5/7] 파싱 + 후처리...")
    all_cls_raw = []
    for task, output in zip(all_tasks, outputs):
        text = output.outputs[0].text
        for item in parse_json(text):
            cui = item.get("cui", "")
            rel = item.get("relation", "")
            if cui and rel:
                all_cls_raw.append({
                    "pmid": task["pmid"],
                    "disease_cui": task["disease_cui"],
                    "cui": cui, "relation": rel,
                })

    print(f"  Raw 관계: {len(all_cls_raw):,}건")

    filtered = []
    removed_manif = 0
    removed_syn = 0
    for c in all_cls_raw:
        if c["relation"] == "manifestation-of":
            removed_manif += 1
            continue
        dc, cui = c["disease_cui"], c["cui"]
        if cui in synonym_map.get(dc, set()) or dc in synonym_map.get(cui, set()):
            removed_syn += 1
            continue
        if cui in parent_map.get(dc, set()) or dc in parent_map.get(cui, set()):
            removed_syn += 1
            continue
        filtered.append(c)

    print(f"  manifestation-of 제거: {removed_manif:,}")
    print(f"  동의어/부모 제거: {removed_syn:,}")
    print(f"  Filtered: {len(filtered):,}")

    # [6] 벤치마크
    print("\n[6/7] 벤치마크...")
    pair_counts_raw = Counter()
    pair_counts_filtered = Counter()
    pair_relations = defaultdict(Counter)

    for c in all_cls_raw:
        pair_counts_raw[tuple(sorted([c["disease_cui"], c["cui"]]))] += 1
    for c in filtered:
        pair = tuple(sorted([c["disease_cui"], c["cui"]]))
        pair_counts_filtered[pair] += 1
        pair_relations[pair][c["relation"]] += 1

    def build_kg(pc, mc, prop=True):
        kg = {p for p, cnt in pc.items() if cnt >= mc}
        if prop:
            exp = set(kg)
            for (a, b) in list(kg):
                for pa in parent_map.get(a, set()):
                    if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                        exp.add(tuple(sorted([pa, b])))
                for pb in parent_map.get(b, set()):
                    if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                        exp.add(tuple(sorted([a, pb])))
            return kg, exp
        return kg, kg

    print(f"\n  Raw (CUI 전파):")
    for mc in [1, 2, 3, 5, 10]:
        _, exp = build_kg(pair_counts_raw, mc)
        ev = evaluate(exp, gold_pairs, parent_map)
        print(f"    MC={mc:>2}: edges={len(exp):>6,} P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} match={ev['matched']}/{ev['gold']}")

    print(f"\n  Filtered (CUI 전파):")
    best_f1, best_mc, best_result = 0, 1, None
    for mc in [1, 2, 3, 5, 10]:
        _, exp = build_kg(pair_counts_filtered, mc)
        ev = evaluate(exp, gold_pairs, parent_map)
        marker = " ★" if ev["F1"] > best_f1 else ""
        print(f"    MC={mc:>2}: edges={len(exp):>6,} P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} match={ev['matched']}/{ev['gold']}{marker}")
        if ev["F1"] > best_f1:
            best_f1, best_mc, best_result = ev["F1"], mc, ev

    print(f"\n  Filtered (전파 없음):")
    for mc in [1, 2, 3, 5]:
        raw_kg, _ = build_kg(pair_counts_filtered, mc, False)
        ev = evaluate(raw_kg, gold_pairs, parent_map)
        print(f"    MC={mc:>2}: edges={len(raw_kg):>6,} P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} match={ev['matched']}/{ev['gold']}")

    # [7] 저장
    print("\n[7/7] 저장...")
    kg_raw, kg_exp = build_kg(pair_counts_filtered, best_mc)
    rel_dist = Counter(c["relation"] for c in filtered)

    output = {
        "config": {
            "model": "google/gemma-4-E4B-it", "engine": "vllm_batch_chat",
            "max_abstracts_per_disease": MAX_ABSTRACTS_PER_DISEASE,
            "post_filter": "manifestation-of + synonym/parent",
            "sty": "DISO+T033+T031+T040", "best_mc": best_mc,
        },
        "stats": {
            "total_abstracts": len(all_tasks),
            "raw_relations": len(all_cls_raw), "filtered_relations": len(filtered),
            "removed_manifestation": removed_manif, "removed_synonym": removed_syn,
            "unique_pairs_raw": len(pair_counts_raw),
            "unique_pairs_filtered": len(pair_counts_filtered),
            "vllm_time_seconds": round(elapsed, 1),
            "throughput": round(len(outputs) / elapsed, 1),
            "relation_distribution": dict(rel_dist),
        },
        "benchmark": {"best": best_result, "best_mc": best_mc},
        "kg_edges": [
            {"cui_a": p[0], "cui_b": p[1],
             "name_a": cui_preferred.get(p[0], p[0]),
             "name_b": cui_preferred.get(p[1], p[1]),
             "count": pair_counts_filtered[p],
             "relations": dict(pair_relations[p])}
            for p in sorted(kg_raw, key=lambda x: -pair_counts_filtered[x])
        ],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  저장: {RESULTS_FILE}")
    print(f"\n{'='*80}")
    print(f"V4 최종: MC={best_mc}, F1={best_f1:.3f} "
          f"(P={best_result['P']:.3f}, R={best_result['R']:.3f})")
    print(f"처리: {len(all_tasks)}편, {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
