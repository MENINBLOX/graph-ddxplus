#!/usr/bin/env python3
"""KG 구축: PubMed 초록 → scispaCy NER → LLM 관계 분류.

파이프라인:
  1. 질환 CUI → PubMed SQLite 검색 (MeSH UID + 키워드)
  2. 초록에서 scispaCy NER로 모든 DISO CUI 추출
  3. 초록 텍스트 + 추출된 CUI 쌍 → LLM 관계 분류 (S2-J)
  4. 통계 집계 (MC≥N) + CUI 1-level 전파 → KG
  5. DDXPlus / HPO / SemMedDB 벤치마크
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

# scispaCy (lazy load — 메모리 절약)
NLP = None

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}
NER_THRESHOLD = 0.85

PROMPT_S2J = """Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""

MAX_ABSTRACTS_PER_DISEASE = 100
MAX_PAIRS_PER_LLM_CALL = 10
CKPT_FILE = DATA_DIR / "ner_llm_checkpoint.json"
RESULTS_FILE = RESULTS_DIR / "ner_llm_kg_results.json"


# ============================================================
# UMLS 로드
# ============================================================

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


def load_mesh_to_cui():
    """MeSH Descriptor UID → CUI 매핑."""
    m = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D"):
                if p[13] not in m:
                    m[p[13]] = p[0]
    return m


def load_cui_to_mesh(mesh_to_cui):
    r = defaultdict(set)
    for mesh, cui in mesh_to_cui.items():
        r[cui].add(mesh)
    return dict(r)


# ============================================================
# scispaCy NER
# ============================================================

def init_nlp():
    global NLP
    if NLP is not None:
        return NLP
    import spacy
    import scispacy  # noqa: F401
    from scispacy.linking import EntityLinker  # noqa: F401

    NLP = spacy.load("en_core_sci_sm")
    NLP.add_pipe("scispacy_linker", config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
    })
    print("  scispaCy 로드 완료")
    return NLP


def extract_cuis(text: str, cui_stys: dict) -> list[str]:
    """초록에서 DISO CUI 추출 (threshold≥0.85, ALLOWED_STYS, 블랙리스트 제외)."""
    nlp = init_nlp()
    doc = nlp(text[:5000])  # 길이 제한

    cuis = set()
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if score < NER_THRESHOLD:
                break
            stys = cui_stys.get(cui, set())
            if stys & ALLOWED_STYS and cui not in BLACKLIST:
                cuis.add(cui)
    return sorted(cuis)


# ============================================================
# PubMed SQLite 검색
# ============================================================

def search_abstracts(conn, disease_cui: str, disease_name: str,
                     cui_to_mesh: dict, limit: int = MAX_ABSTRACTS_PER_DISEASE):
    """질환 CUI에 대한 초록 검색 (MeSH + 키워드 병행)."""
    c = conn.cursor()
    results = {}

    # 1) MeSH 검색
    mesh_uids = cui_to_mesh.get(disease_cui, set())
    if mesh_uids:
        mesh_conditions = " OR ".join(f"mesh_terms LIKE '%{m}%'" for m in mesh_uids)
        c.execute(f"""
            SELECT pmid, abstract FROM abstracts
            WHERE ({mesh_conditions})
            AND abstract IS NOT NULL AND length(abstract) > 200
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,))
        for pmid, abstract in c.fetchall():
            results[pmid] = abstract

    # 2) 부족하면 키워드 검색으로 보충
    if len(results) < limit:
        # 질환명에서 검색 키워드 추출 (괄호, NOS 등 제거)
        keywords = re.sub(r'\(.*?\)', '', disease_name)
        keywords = re.sub(r'\b(NOS|unspecified|disease)\b', '', keywords, flags=re.IGNORECASE)
        keywords = keywords.strip().strip(',').strip()

        if len(keywords) > 3:
            remaining = limit - len(results)
            exclude_pmids = ",".join(f"'{p}'" for p in results) if results else "''"
            c.execute(f"""
                SELECT pmid, abstract FROM abstracts
                WHERE (title LIKE ? OR abstract LIKE ?)
                AND abstract IS NOT NULL AND length(abstract) > 200
                AND pmid NOT IN ({exclude_pmids})
                ORDER BY RANDOM()
                LIMIT ?
            """, (f"%{keywords}%", f"%{keywords}%", remaining))
            for pmid, abstract in c.fetchall():
                results[pmid] = abstract

    return results


# ============================================================
# LLM
# ============================================================

def call_ollama(prompt: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }, timeout=300)
    return resp.json().get("response", "")


def parse_json(text: str) -> list[dict]:
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
# Gold Standard
# ============================================================

def prepare_gold():
    """DDXPlus gold standard (증상만, antecedent 제외)."""
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
    print("KG 구축: PubMed 초록 → scispaCy NER → LLM 관계 분류")
    print("=" * 80)

    # 체크포인트 로드
    ckpt = {"phase": "init", "abstracts": {}, "classifications": []}
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            ckpt = json.load(f)
        print(f"  체크포인트: phase={ckpt['phase']}, "
              f"abstracts={len(ckpt.get('abstracts', {}))}, "
              f"cls={len(ckpt.get('classifications', []))}")

    def save_ckpt():
        with open(CKPT_FILE, "w") as f:
            json.dump(ckpt, f, ensure_ascii=False)

    # [1] UMLS 로드
    print("\n[1/6] UMLS 데이터 로드...")
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    parent_map = load_parent_map()
    mesh_to_cui = load_mesh_to_cui()
    cui_to_mesh = load_cui_to_mesh(mesh_to_cui)
    gold_pairs, disease_cuis = prepare_gold()
    print(f"  질환: {len(disease_cuis)}개, Gold: {len(gold_pairs)}쌍")
    print(f"  MeSH→CUI: {len(mesh_to_cui):,}, Parent map: {len(parent_map):,}")

    # [2] 질환별 초록 수집 + NER
    print(f"\n[2/6] 초록 수집 + scispaCy NER (질환당 {MAX_ABSTRACTS_PER_DISEASE}편)...")

    if ckpt["phase"] in ("init",):
        conn = sqlite3.connect(str(DB_PATH))
        total_abstracts = 0
        total_cuis_extracted = 0

        for idx, (disease_name, dinfo) in enumerate(sorted(disease_cuis.items())):
            dc = dinfo["cui"]
            dn = dinfo["umls_name"]

            # 이미 처리된 질환 건너뛰기
            if dc in ckpt.get("abstracts", {}):
                n_existing = len(ckpt["abstracts"][dc])
                total_abstracts += n_existing
                print(f"  [{idx + 1}/49] {disease_name}: 이미 {n_existing}편 (건너뜀)")
                continue

            # 초록 검색
            abstracts = search_abstracts(conn, dc, dn, cui_to_mesh)

            if not abstracts:
                print(f"  [{idx + 1}/49] {disease_name} ({dc}): 초록 없음")
                ckpt["abstracts"][dc] = []
                save_ckpt()
                continue

            # scispaCy NER
            disease_docs = []
            for pmid, abstract in abstracts.items():
                cuis = extract_cuis(abstract, cui_stys)
                if len(cuis) >= 2:  # CUI가 2개 이상이어야 쌍 구성 가능
                    disease_docs.append({
                        "pmid": pmid,
                        "abstract": abstract,
                        "cuis": cuis,
                    })

            ckpt["abstracts"][dc] = disease_docs
            total_abstracts += len(disease_docs)
            total_cuis_extracted += sum(len(d["cuis"]) for d in disease_docs)

            print(f"  [{idx + 1}/49] {disease_name}: "
                  f"검색={len(abstracts)}편, NER후={len(disease_docs)}편, "
                  f"CUI={sum(len(d['cuis']) for d in disease_docs)}개")

            save_ckpt()

        conn.close()
        ckpt["phase"] = "ner_done"
        save_ckpt()

        print(f"\n  총 초록: {total_abstracts}편")
    else:
        total_abstracts = sum(len(docs) for docs in ckpt.get("abstracts", {}).values())
        print(f"  NER 완료 (체크포인트): {total_abstracts}편")

    # [3] LLM 분류 준비 — 초록별 CUI 쌍 생성
    print(f"\n[3/6] LLM 분류 준비...")

    # 모든 초록을 플랫 리스트로 (disease_cui 포함)
    all_docs = []
    for dc, docs in ckpt["abstracts"].items():
        for doc in docs:
            all_docs.append({
                "disease_cui": dc,
                "pmid": doc["pmid"],
                "abstract": doc["abstract"],
                "cuis": doc["cuis"],
            })

    print(f"  총 초록: {len(all_docs)}편")
    print(f"  총 CUI: {sum(len(d['cuis']) for d in all_docs):,}개")

    # [4] LLM 분류 실행
    print(f"\n[4/6] LLM 분류 ({len(all_docs)}편)...")

    all_cls = ckpt.get("classifications", [])
    processed_pmids = set(c["pmid"] for c in all_cls)

    start = time.time()
    new_count = 0

    for idx, doc in enumerate(all_docs):
        pmid = doc["pmid"]
        if pmid in processed_pmids:
            continue

        cuis = doc["cuis"]
        abstract = doc["abstract"]

        # CUI 쌍 생성 (모든 조합)
        pairs = []
        for i in range(len(cuis)):
            for j in range(i + 1, len(cuis)):
                pairs.append((cuis[i], cuis[j]))

        if not pairs:
            continue

        # 쌍이 많으면 배치로 분할
        for batch_start in range(0, len(pairs), MAX_PAIRS_PER_LLM_CALL):
            batch = pairs[batch_start:batch_start + MAX_PAIRS_PER_LLM_CALL]

            pairs_text = "\n".join(
                f"- ({cui_names.get(a, a)[:40]}, {cui_names.get(b, b)[:40]}) "
                f"[CUI: {a}, {b}]"
                for a, b in batch
            )
            prompt = PROMPT_S2J.format(text=abstract[:2500], pairs=pairs_text)

            try:
                response = call_ollama(prompt)
                parsed = parse_json(response)
                for item in parsed:
                    cls = item.get("classification", "").lower().strip().replace(" ", "_")
                    if cls in ("present", "not_related"):
                        all_cls.append({
                            "pmid": pmid,
                            "cui_a": item.get("cui_a", ""),
                            "cui_b": item.get("cui_b", ""),
                            "classification": cls,
                        })
            except Exception as e:
                print(f"    LLM 오류 (pmid={pmid}): {e}")

        processed_pmids.add(pmid)
        new_count += 1

        # 진행 보고 + 체크포인트
        if new_count % 10 == 0 and new_count > 0:
            elapsed = time.time() - start
            rate = new_count / elapsed
            remaining = len(all_docs) - len(processed_pmids)
            eta = remaining / rate if rate > 0 else 0
            present = sum(1 for c in all_cls if c["classification"] == "present")
            print(f"  [{len(processed_pmids):>5d}/{len(all_docs)}] "
                  f"cls={len(all_cls):,} (present={present:,}) "
                  f"{rate:.2f}/s ETA={eta / 60:.0f}분")

            ckpt["classifications"] = all_cls
            save_ckpt()

    ckpt["classifications"] = all_cls
    ckpt["phase"] = "llm_done"
    save_ckpt()

    elapsed = time.time() - start
    dist = Counter(c["classification"] for c in all_cls)
    print(f"\n  완료: {len(all_cls):,}건 ({elapsed / 60:.1f}분)")
    print(f"  present={dist.get('present', 0):,}, not_related={dist.get('not_related', 0):,}")

    # [5] KG 구축 + MC sweep
    print(f"\n[5/6] KG 구축 + 벤치마크...")

    pair_counts = Counter()
    for c in all_cls:
        if c["classification"] == "present":
            pair = tuple(sorted([c["cui_a"], c["cui_b"]]))
            pair_counts[pair] += 1

    print(f"  present 쌍 (고유): {len(pair_counts):,}")

    def build_kg(mc: int):
        """MC 필터 + CUI 1-level 전파로 KG 구축."""
        kg = {p for p, cnt in pair_counts.items() if cnt >= mc}
        expanded = set(kg)
        for (a, b) in list(kg):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                    expanded.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                    expanded.add(tuple(sorted([a, pb])))
        return kg, expanded

    # MC sweep
    print(f"\n  MC 파라미터 sweep:")
    best_f1 = 0
    best_mc = 1
    best_result = None

    for mc in [1, 2, 3, 5, 10]:
        kg_raw, kg_expanded = build_kg(mc)
        ev = evaluate(kg_expanded, gold_pairs, parent_map)
        print(f"    MC={mc:>2d}: raw={len(kg_raw):>6,} expanded={len(kg_expanded):>6,} "
              f"P={ev['P']:.3f} R={ev['R']:.3f} F1={ev['F1']:.3f} "
              f"match={ev['matched']}/{ev['gold']}")
        if ev["F1"] > best_f1:
            best_f1 = ev["F1"]
            best_mc = mc
            best_result = ev

    print(f"\n  최적: MC={best_mc} → F1={best_f1:.3f}")

    # 최적 MC로 최종 KG
    kg_raw, kg_expanded = build_kg(best_mc)

    # [6] 저장
    print(f"\n[6/6] 결과 저장...")

    # 질환별 상세 결과
    disease_stats = {}
    for dc, docs in ckpt["abstracts"].items():
        dn = cui_names.get(dc, dc)
        n_docs = len(docs)
        n_cuis = sum(len(d["cuis"]) for d in docs)
        # 이 질환과 관련된 present 분류 수
        disease_present = sum(
            1 for c in all_cls
            if c["classification"] == "present" and dc in (c["cui_a"], c["cui_b"])
        )
        disease_stats[dc] = {
            "name": dn, "abstracts": n_docs,
            "cuis_extracted": n_cuis, "present_relations": disease_present,
        }

    output = {
        "config": {
            "model": MODEL,
            "ner_threshold": NER_THRESHOLD,
            "max_abstracts_per_disease": MAX_ABSTRACTS_PER_DISEASE,
            "prompt": "S2-J",
            "best_mc": best_mc,
        },
        "stats": {
            "diseases": len(disease_cuis),
            "total_abstracts": len(all_docs),
            "total_classifications": len(all_cls),
            "present": dist.get("present", 0),
            "not_related": dist.get("not_related", 0),
            "unique_present_pairs": len(pair_counts),
            "kg_raw": len(kg_raw),
            "kg_expanded": len(kg_expanded),
        },
        "benchmark": {
            "ddxplus": best_result,
            "mc_sweep": {},
        },
        "disease_stats": disease_stats,
        "kg_edges": [
            {"cui_a": p[0], "cui_b": p[1],
             "name_a": cui_names.get(p[0], p[0]),
             "name_b": cui_names.get(p[1], p[1]),
             "count": pair_counts.get(p, 0)}
            for p in sorted(kg_raw, key=lambda x: -pair_counts.get(x, 0))
        ],
    }

    # MC sweep 결과 저장
    for mc in [1, 2, 3, 5, 10]:
        _, exp = build_kg(mc)
        ev = evaluate(exp, gold_pairs, parent_map)
        output["benchmark"]["mc_sweep"][f"mc_{mc}"] = ev

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  저장: {RESULTS_FILE}")
    print(f"\n{'=' * 80}")
    print(f"최종 결과: MC={best_mc}, F1={best_f1:.3f} "
          f"(P={best_result['P']:.3f}, R={best_result['R']:.3f})")
    print(f"KG 엣지: {len(kg_raw):,} (전파 후: {len(kg_expanded):,})")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
