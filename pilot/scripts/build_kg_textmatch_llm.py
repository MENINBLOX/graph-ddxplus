#!/usr/bin/env python3
"""KG 구축: 텍스트 매칭(Aho-Corasick) + LLM 관계 분류.

파이프라인:
  1. UMLS DISO CUI → 영문 이름 사전 + Aho-Corasick automaton 구축
  2. 질환 CUI → PubMed SQLite 검색 (MeSH + 키워드)
  3. 초록 텍스트에서 Aho-Corasick으로 DISO CUI 추출
  4. 초록 + CUI 리스트 → LLM 1회 호출 (관계 쌍 발견 + 분류)
  5. 통계 집계 (MC≥N) + CUI 1-level 전파 → KG
  6. DDXPlus 벤치마크
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
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT = """Abstract: {abstract}

Keywords found in this abstract:
{keywords}

From the abstract above, identify all pairs of keywords that have a medical relationship.
Medical relationships include: symptom-of, causes, complication-of, risk-factor-for, co-occurs-with, diagnostic-finding-of, manifestation-of.

Rules:
- ONLY report pairs where the abstract EXPLICITLY describes a relationship
- Do NOT infer relationships not stated in the text
- Use the CUI codes provided

Respond ONLY with a JSON array:
[{{"cui_a":"...","cui_b":"...","relation":"symptom-of|causes|complication-of|risk-factor-for|co-occurs-with|diagnostic-finding-of|manifestation-of"}}]
If no related pairs found, respond: []"""

MAX_ABSTRACTS_PER_DISEASE = 100
CKPT_FILE = DATA_DIR / "textmatch_llm_checkpoint.json"
RESULTS_FILE = RESULTS_DIR / "textmatch_llm_kg_results.json"


# ============================================================
# UMLS
# ============================================================

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
    """DISO CUI 영문 이름으로 Aho-Corasick automaton 구축."""
    diso_cuis = {cui for cui, stys in cui_stys.items() if stys & ALLOWED_STYS}
    diso_cuis -= BLACKLIST

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
        # 하나의 이름이 여러 CUI에 매핑될 수 있음 → 첫 번째 사용
        A.add_word(name, (name, next(iter(cuis))))
    A.make_automaton()

    return A, cui_preferred


def text_match_cuis(text_lower, automaton, exclude_cui=None):
    """Aho-Corasick으로 초록에서 CUI 매칭."""
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


# ============================================================
# LLM
# ============================================================

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


# ============================================================
# 평가
# ============================================================

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


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 80)
    print("KG 구축: 텍스트 매칭 + LLM 관계 분류")
    print("=" * 80)

    # 체크포인트
    ckpt = {"phase": "init", "docs": [], "classifications": [], "processed_pmids": []}
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            ckpt = json.load(f)
        print(f"  체크포인트: phase={ckpt['phase']}, "
              f"docs={len(ckpt.get('docs', []))}, "
              f"cls={len(ckpt.get('classifications', []))}, "
              f"processed={len(ckpt.get('processed_pmids', []))}")

    def save_ckpt():
        with open(CKPT_FILE, "w") as f:
            json.dump(ckpt, f, ensure_ascii=False)

    # [1] UMLS + Aho-Corasick
    print("\n[1/6] UMLS 로드 + Aho-Corasick 구축...")
    cui_stys = load_cui_stys()
    parent_map = load_parent_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        cui_to_mesh[cui].add(mesh)

    automaton, cui_preferred = build_aho_automaton(cui_stys)
    gold_pairs, disease_cuis = prepare_gold()
    print(f"  질환: {len(disease_cuis)}개, Gold: {len(gold_pairs)}쌍")

    # [2] 초록 수집 + 텍스트 매칭 CUI 추출
    if ckpt["phase"] == "init":
        print(f"\n[2/6] 초록 수집 + CUI 추출 (질환당 {MAX_ABSTRACTS_PER_DISEASE}편)...")
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        all_docs = []
        for idx, (disease_name, dinfo) in enumerate(sorted(disease_cuis.items())):
            dc = dinfo["cui"]
            dn = dinfo["umls_name"]

            # MeSH 검색
            mesh_uids = cui_to_mesh.get(dc, set())
            rows = []
            if mesh_uids:
                mesh_cond = " OR ".join(f"mesh_terms LIKE '%{m}%'" for m in mesh_uids)
                c.execute(f"""
                    SELECT pmid, abstract FROM abstracts
                    WHERE ({mesh_cond})
                    AND abstract IS NOT NULL AND length(abstract) > 200
                    ORDER BY RANDOM() LIMIT ?
                """, (MAX_ABSTRACTS_PER_DISEASE,))
                rows = c.fetchall()

            # 부족하면 키워드 검색 보충 (UMLS 이름 + DDXPlus 질환명)
            if len(rows) < MAX_ABSTRACTS_PER_DISEASE:
                # 후보 키워드: UMLS 이름, DDXPlus 이름, 약어 등
                candidates = set()
                for raw in [dn, disease_name]:
                    kw = re.sub(r'\(.*?\)', '', raw)
                    kw = re.sub(r'\b(NOS|unspecified|disease)\b', '', kw, flags=re.IGNORECASE).strip()
                    kw = kw.strip(',').strip('/').strip()
                    if len(kw) >= 4:
                        candidates.add(kw)
                    # "/" 구분된 이름 분리 (예: "COPD exacerbation / infection")
                    for part in raw.split('/'):
                        part = part.strip()
                        if len(part) >= 4:
                            candidates.add(part)

                existing_pmids = {r[0] for r in rows}
                for kw in candidates:
                    if len(rows) >= MAX_ABSTRACTS_PER_DISEASE:
                        break
                    remaining = MAX_ABSTRACTS_PER_DISEASE - len(rows)
                    exclude = ",".join(f"'{p}'" for p in existing_pmids) if existing_pmids else "''"
                    c.execute(f"""
                        SELECT pmid, abstract FROM abstracts
                        WHERE (title LIKE ? OR abstract LIKE ?)
                        AND abstract IS NOT NULL AND length(abstract) > 200
                        AND pmid NOT IN ({exclude})
                        ORDER BY RANDOM() LIMIT ?
                    """, (f"%{kw}%", f"%{kw}%", remaining))
                    new_rows = c.fetchall()
                    rows.extend(new_rows)
                    existing_pmids.update(r[0] for r in new_rows)

            if not rows:
                print(f"  [{idx+1}/49] {disease_name}: 초록 없음")
                continue

            # Aho-Corasick CUI 추출
            disease_docs = []
            for pmid, abstract in rows:
                cuis = text_match_cuis(abstract.lower(), automaton, exclude_cui=dc)
                if len(cuis) >= 2:
                    disease_docs.append({
                        "pmid": pmid,
                        "abstract": abstract,
                        "disease_cui": dc,
                        "cuis": sorted(cuis),
                    })

            all_docs.extend(disease_docs)
            print(f"  [{idx+1}/49] {disease_name}: "
                  f"검색={len(rows)}편, CUI≥2={len(disease_docs)}편, "
                  f"평균CUI={sum(len(d['cuis']) for d in disease_docs)/max(len(disease_docs),1):.0f}")

        conn.close()

        # 중복 pmid 제거 (여러 질환에서 같은 초록이 나올 수 있음)
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc["pmid"] not in seen:
                seen.add(doc["pmid"])
                unique_docs.append(doc)

        ckpt["docs"] = unique_docs
        ckpt["phase"] = "extracted"
        save_ckpt()
        print(f"\n  총 초록: {len(all_docs)}편 (중복 제거 후: {len(unique_docs)}편)")
    else:
        print(f"\n[2/6] CUI 추출 완료 (체크포인트): {len(ckpt['docs'])}편")

    # [3] LLM 분류
    print(f"\n[3/6] LLM 분류 ({len(ckpt['docs'])}편, 초록당 1회 호출)...")

    all_cls = ckpt.get("classifications", [])
    processed_pmids = set(ckpt.get("processed_pmids", []))

    start = time.time()
    new_count = 0

    for idx, doc in enumerate(ckpt["docs"]):
        pmid = doc["pmid"]
        if pmid in processed_pmids:
            continue

        abstract = doc["abstract"]
        cuis = doc["cuis"]

        # 키워드 목록 생성
        keywords_text = "\n".join(
            f"- {cui_preferred.get(cui, cui)} [{cui}]"
            for cui in cuis
        )

        prompt = PROMPT.format(abstract=abstract[:3000], keywords=keywords_text)

        try:
            response = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                cui_a = item.get("cui_a", "")
                cui_b = item.get("cui_b", "")
                relation = item.get("relation", "")
                if cui_a and cui_b and relation:
                    all_cls.append({
                        "pmid": pmid,
                        "cui_a": cui_a,
                        "cui_b": cui_b,
                        "relation": relation,
                    })
        except Exception as e:
            print(f"    LLM 오류 (pmid={pmid}): {e}")

        processed_pmids.add(pmid)
        new_count += 1

        if new_count % 10 == 0 and new_count > 0:
            elapsed = time.time() - start
            rate = new_count / elapsed
            remaining = len(ckpt["docs"]) - len(processed_pmids)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{len(processed_pmids):>5d}/{len(ckpt['docs'])}] "
                  f"relations={len(all_cls):,} "
                  f"{rate:.2f}/s ETA={eta/60:.0f}분")

            ckpt["classifications"] = all_cls
            ckpt["processed_pmids"] = sorted(processed_pmids)
            save_ckpt()

    ckpt["classifications"] = all_cls
    ckpt["processed_pmids"] = sorted(processed_pmids)
    ckpt["phase"] = "llm_done"
    save_ckpt()

    elapsed = time.time() - start
    rel_dist = Counter(c["relation"] for c in all_cls)
    print(f"\n  완료: {len(all_cls):,}건 ({elapsed/60:.1f}분)")
    print(f"  관계 분포:")
    for rel, cnt in rel_dist.most_common():
        print(f"    {rel}: {cnt:,}")

    # [4] KG 구축
    print(f"\n[4/6] KG 구축...")

    pair_counts = Counter()
    pair_relations = defaultdict(Counter)  # (cui_a, cui_b) -> {relation: count}
    for c in all_cls:
        pair = tuple(sorted([c["cui_a"], c["cui_b"]]))
        pair_counts[pair] += 1
        pair_relations[pair][c["relation"]] += 1

    print(f"  고유 관계 쌍: {len(pair_counts):,}")

    # [5] MC sweep + 벤치마크
    print(f"\n[5/6] 벤치마크 (MC sweep)...")

    best_f1 = 0
    best_mc = 1
    best_result = None

    for mc in [1, 2, 3, 5, 10]:
        # MC 필터
        kg_raw = {p for p, cnt in pair_counts.items() if cnt >= mc}

        # CUI 1-level 전파
        expanded = set(kg_raw)
        for (a, b) in list(kg_raw):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                    expanded.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                    expanded.add(tuple(sorted([a, pb])))

        ev = evaluate(expanded, gold_pairs, parent_map)
        marker = " ★" if ev["F1"] > best_f1 else ""
        print(f"  MC={mc:>2d}: raw={len(kg_raw):>6,} expanded={len(expanded):>6,} "
              f"P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} "
              f"match={ev['matched']}/{ev['gold']}{marker}")

        if ev["F1"] > best_f1:
            best_f1 = ev["F1"]
            best_mc = mc
            best_result = ev

    # CUI 전파 없이도 테스트
    print(f"\n  CUI 전파 없이:")
    for mc in [1, 2, 3]:
        kg_raw = {p for p, cnt in pair_counts.items() if cnt >= mc}
        ev = evaluate(kg_raw, gold_pairs, parent_map)
        print(f"  MC={mc:>2d} (no prop): raw={len(kg_raw):>6,} "
              f"P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} "
              f"match={ev['matched']}/{ev['gold']}")

    # [6] 저장
    print(f"\n[6/6] 결과 저장...")

    # 최적 MC로 최종 KG
    kg_raw = {p for p, cnt in pair_counts.items() if cnt >= best_mc}

    output = {
        "config": {
            "model": MODEL,
            "cui_extraction": "text_match_aho_corasick",
            "max_abstracts_per_disease": MAX_ABSTRACTS_PER_DISEASE,
            "best_mc": best_mc,
        },
        "stats": {
            "diseases": len(disease_cuis),
            "total_abstracts": len(ckpt["docs"]),
            "total_relations": len(all_cls),
            "unique_pairs": len(pair_counts),
            "kg_edges_raw": len(kg_raw),
            "relation_distribution": dict(rel_dist),
        },
        "benchmark": {
            "best": best_result,
            "best_mc": best_mc,
        },
        "kg_edges": [
            {"cui_a": p[0], "cui_b": p[1],
             "name_a": cui_preferred.get(p[0], p[0]),
             "name_b": cui_preferred.get(p[1], p[1]),
             "count": pair_counts[p],
             "relations": dict(pair_relations[p])}
            for p in sorted(kg_raw, key=lambda x: -pair_counts[x])
        ],
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  저장: {RESULTS_FILE}")
    print(f"\n{'='*80}")
    print(f"최종: MC={best_mc}, F1={best_f1:.3f} "
          f"(P={best_result['P']:.3f}, R={best_result['R']:.3f})")
    print(f"KG: {len(kg_raw):,}개 엣지, 관계 {len(all_cls):,}건")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
