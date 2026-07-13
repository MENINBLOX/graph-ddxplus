#!/usr/bin/env python3
"""KG 구축 V6: Ollama + S2-J + 텍스트 매칭 + 500편/질환.

파일럿 F1=0.793의 핵심 요소(Ollama + S2-J)를 대규모에 적용.
vLLM은 HF 모델 차이로 성능 저하 → Ollama 사용.
체크포인트 지원.
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
import requests

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049",
                "T033", "T031", "T040"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT_S2J = """Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""

MAX_ABSTRACTS = 500
MAX_PAIRS_PER_CALL = 15
CKPT_FILE = DATA_DIR / "kg_v6_checkpoint.json"
RESULTS_FILE = RESULTS_DIR / "kg_v6_results.json"


def load_umls():
    cui_stys = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set)
    synonym_map = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[3] in ("PAR", "RB"): parent_map[p[0]].add(p[4])
            if p[3] == "SY": synonym_map[p[0]].add(p[4]); synonym_map[p[4]].add(p[0])
    mesh_to_cui = {}
    cui_all_names = defaultdict(set)
    cui_preferred = {}
    target = {c for c, s in cui_stys.items() if s & ALLOWED_STYS} - BLACKLIST
    A = ahocorasick.Automaton()
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui:
                mesh_to_cui[p[13]] = p[0]
            if p[1] == "ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cui_preferred:
                    cui_preferred[p[0]] = p[14].strip()
            if p[0] in target and p[1] == "ENG":
                lower = p[14].strip().lower()
                if len(lower) >= 4:
                    try: A.add_word(lower, (lower, p[0]))
                    except: pass
    A.make_automaton()
    return dict(cui_stys), dict(parent_map), dict(synonym_map), mesh_to_cui, dict(cui_all_names), cui_preferred, A


def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences_en.json") as f: ev_en = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
    with open("data/ddxplus/umls_mapping.json") as f: umap = json.load(f)["mapping"]
    with open("data/ddxplus/disease_umls_mapping.json") as f: dm = json.load(f)["mapping"]
    eid_to_fr = {}
    for eid, en in ev_en.items():
        for fn, fr in ev_fr.items():
            if en.get("question_en") == fr.get("question_en") and en.get("question_en"):
                eid_to_fr[eid] = fn; break
    gold, dcuis = set(), {}
    for dn, info in cond.items():
        dc = dm.get(dn, {}).get("umls_cui"); un = dm.get(dn, {}).get("umls_name", dn)
        if not dc: continue
        dcuis[dn] = {"cui": dc, "umls_name": un}
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False): continue
            fn = eid_to_fr.get(eid)
            if fn and fn in umap:
                cui = umap[fn].get("cui")
                if cui: gold.add(tuple(sorted([dc, cui])))
    return gold, dcuis


def text_match(text_lower, A, exclude=None):
    matched = set()
    for ei, (n, c) in A.iter(text_lower):
        if c == exclude: continue
        si = ei - len(n) + 1
        if si > 0 and text_lower[si-1].isalpha(): continue
        if ei+1 < len(text_lower) and text_lower[ei+1].isalpha(): continue
        matched.add(c)
    return matched


def search_abs(conn, dc, dn, un, c2m, can, limit):
    c = conn.cursor(); rows, seen = [], set()
    muids = c2m.get(dc, set())
    if muids:
        mc = " OR ".join("mesh_terms LIKE '%%%s%%'" % m for m in muids)
        c.execute(f"SELECT pmid, abstract FROM abstracts WHERE ({mc}) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?", (limit,))
        for p, a in c.fetchall():
            if p not in seen: seen.add(p); rows.append((p, a))
    if len(rows) < limit:
        kws = []
        for raw in [dn, un]:
            kw = re.sub(r'\(.*?\)', '', raw).strip()
            kw = re.sub(r'\b(NOS|unspecified)\b', '', kw, flags=re.IGNORECASE).strip()
            if len(kw) >= 4: kws.append(kw)
            for part in raw.split('/'):
                part = part.strip()
                if len(part) >= 4 and part not in kws: kws.append(part)
        syns = set()
        for name in can.get(dc, set()):
            kw = re.sub(r'\(.*?\)', '', name).strip()
            kw = re.sub(r'\b(NOS|unspecified|disease)\b', '', kw, flags=re.IGNORECASE).strip().strip(',./').strip()
            if len(kw) >= 4 and kw not in kws: syns.add(kw)
        kws.extend(sorted(syns, key=len)[:10])
        for kw in kws:
            if len(rows) >= limit: break
            c.execute("SELECT pmid, abstract FROM abstracts WHERE (title LIKE ? OR abstract LIKE ?) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?",
                      (f"%{kw}%", f"%{kw}%", limit-len(rows)))
            for p, a in c.fetchall():
                if p not in seen: seen.add(p); rows.append((p, a))
    return rows


def call_ollama(prompt):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }, timeout=300)
    return resp.json().get("response", "")


def parse_json_r(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text); text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try: return json.loads(m.group())
        except: pass
    return []


def evaluate(our, gold, pm):
    def cm(a, b):
        if a == b: return True
        return b in pm.get(a, set()) or a in pm.get(b, set())
    mg, mo = set(), set()
    for op in our:
        for gp in gold:
            if ((cm(op[0],gp[0]) and cm(op[1],gp[1])) or (cm(op[0],gp[1]) and cm(op[1],gp[0]))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our) if our else 0; r = len(mg)/len(gold) if gold else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return round(p,4), round(r,4), round(f1,4), len(mg)


def main():
    print("=" * 80)
    print(f"KG 구축 V6: Ollama + S2-J + TextMatch ({MAX_ABSTRACTS}편/질환)")
    print("=" * 80)

    # 체크포인트
    ckpt = {"phase": "init", "docs": {}, "classifications": [], "processed_keys": []}
    if CKPT_FILE.exists():
        with open(CKPT_FILE) as f:
            ckpt = json.load(f)
        print(f"  체크포인트: phase={ckpt['phase']}, cls={len(ckpt.get('classifications', []))}, "
              f"processed={len(ckpt.get('processed_keys', []))}")

    def save_ckpt():
        with open(CKPT_FILE, "w") as f:
            json.dump(ckpt, f, ensure_ascii=False)

    print("\n[1] UMLS 로드...")
    cui_stys, parent_map, synonym_map, mesh_to_cui, cui_all_names, cui_preferred, automaton = load_umls()
    c2m = defaultdict(set)
    for m, c in mesh_to_cui.items(): c2m[c].add(m)
    gold, dcuis = prepare_gold()
    print(f"  Gold: {len(gold)}쌍")

    # 초록 수집
    if ckpt["phase"] == "init":
        print(f"\n[2] 초록 수집 ({MAX_ABSTRACTS}편/질환)...")
        conn = sqlite3.connect(str(DB_PATH))
        for idx, (dn, dinfo) in enumerate(sorted(dcuis.items())):
            dc, un = dinfo["cui"], dinfo["umls_name"]
            if dc in ckpt["docs"]: continue
            rows = search_abs(conn, dc, dn, un, c2m, cui_all_names, MAX_ABSTRACTS)
            docs = []
            for pmid, ab in rows:
                cuis = text_match(ab.lower(), automaton, exclude=dc)
                if cuis:
                    docs.append({"pmid": pmid, "abstract": ab, "cuis": sorted(cuis)})
            ckpt["docs"][dc] = docs
            print(f"  [{idx+1}/49] {dn}: {len(docs)}편")
            save_ckpt()
        conn.close()
        ckpt["phase"] = "extracted"
        save_ckpt()

    total = sum(len(d) for d in ckpt["docs"].values())
    print(f"  총: {total}편")

    # LLM 분류 (Ollama, S2-J)
    print(f"\n[3] Ollama S2-J 분류...")
    all_cls = ckpt.get("classifications", [])
    processed = set(ckpt.get("processed_keys", []))
    start = time.time()
    new_count = 0

    for dc, docs in ckpt["docs"].items():
        dn = cui_preferred.get(dc, dc)
        for doc in docs:
            key = f"{dc}_{doc['pmid']}"
            if key in processed: continue

            cuis = doc["cuis"][:MAX_PAIRS_PER_CALL]
            pairs_text = "\n".join(
                f"- ({dn[:40]}, {cui_preferred.get(c, c)[:40]}) [CUI: {dc}, {c}]"
                for c in cuis
            )
            prompt = PROMPT_S2J.format(text=doc["abstract"][:2500], pairs=pairs_text)

            try:
                response = call_ollama(prompt)
                for item in parse_json_r(response):
                    cls = item.get("classification", "").lower().replace(" ", "_")
                    if cls == "present":
                        a, b = item.get("cui_a", ""), item.get("cui_b", "")
                        if a and b:
                            other = b if a == dc else a
                            all_cls.append({"dc": dc, "cui": other, "pmid": doc["pmid"]})
            except Exception as e:
                print(f"    오류: {e}")

            processed.add(key)
            new_count += 1

            if new_count % 10 == 0:
                elapsed = time.time() - start
                rate = new_count / elapsed
                remaining = total - len(processed)
                eta = remaining / rate if rate > 0 else 0
                print(f"  [{len(processed):>5}/{total}] rels={len(all_cls):,} "
                      f"{rate:.2f}/s ETA={eta/60:.0f}분")
                ckpt["classifications"] = all_cls
                ckpt["processed_keys"] = sorted(processed)
                save_ckpt()

    ckpt["classifications"] = all_cls
    ckpt["processed_keys"] = sorted(processed)
    ckpt["phase"] = "done"
    save_ckpt()

    elapsed = time.time() - start
    print(f"\n  완료: {len(all_cls):,}건 ({elapsed/60:.1f}분)")

    # 후처리 + 평가
    print(f"\n[4] 후처리 + 벤치마크...")
    filtered = []
    for c in all_cls:
        if c["cui"] in synonym_map.get(c["dc"], set()): continue
        if c["cui"] in parent_map.get(c["dc"], set()) or c["dc"] in parent_map.get(c["cui"], set()): continue
        filtered.append(c)

    pc = Counter()
    for c in filtered:
        pc[tuple(sorted([c["dc"], c["cui"]]))] += 1

    print(f"  Raw: {len(all_cls):,}, Filtered: {len(filtered):,}, Pairs: {len(pc):,}")

    best_f1, best_mc = 0, 1
    for mc in [1, 2, 3, 5, 7, 10, 15, 20]:
        kg = {p for p, cnt in pc.items() if cnt >= mc}
        exp = set(kg)
        for (a, b) in list(kg):
            for pa in parent_map.get(a, set()):
                if cui_stys.get(pa, set()) & ALLOWED_STYS and pa not in BLACKLIST:
                    exp.add(tuple(sorted([pa, b])))
            for pb in parent_map.get(b, set()):
                if cui_stys.get(pb, set()) & ALLOWED_STYS and pb not in BLACKLIST:
                    exp.add(tuple(sorted([a, pb])))
        p, r, f1, m = evaluate(exp, gold, parent_map)
        marker = " ★" if f1 > best_f1 else ""
        if f1 > best_f1: best_f1, best_mc = f1, mc
        print(f"  MC={mc:>2} edges={len(exp):>6,} P={p:.3f} R={r:.3f} F1={f1:.3f} match={m}/{len(gold)}{marker}")

    print(f"\n{'='*80}")
    print(f"V6 최종: MC={best_mc}, F1={best_f1:.3f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
