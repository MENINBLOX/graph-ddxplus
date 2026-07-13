#!/usr/bin/env python3
"""KG 구축용 프롬프트만 준비 (CPU-only, GPU 미사용).

벤치마크별로 PubMed 검색 + 텍스트 매칭 + 프롬프트 생성을 병렬 실행 가능.
나중에 vLLM batch에서 한꺼번에 처리.
"""
from __future__ import annotations
import json, os, re, sqlite3, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
import multiprocessing as mp

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")

ALLOWED_STYS = {"T047","T184","T191","T046","T048","T037","T019","T020","T190","T049","T033","T031","T040"}
BLACKLIST = {"C1457887","C3257980","C0012634","C0699748","C3839861"}

PROMPT = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Clinical findings and symptoms found in this abstract:
{keywords}

From the abstract, identify which of the above findings are CLINICAL SYMPTOMS or SIGNS that a patient with {disease_name} would present with.

JSON only: [{{"cui":"...","relation":"symptom-of|sign-of"}}]
If none: []"""


def load_umls():
    print("[UMLS]", flush=True)
    cui_stys = defaultdict(set)
    with open(UMLS_DIR/"MRSTY.RRF") as f:
        for l in f: p = l.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set); synonym_map = defaultdict(set)
    with open(UMLS_DIR/"MRREL.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[3] in("PAR","RB"): parent_map[p[0]].add(p[4])
            if p[3]=="SY": synonym_map[p[0]].add(p[4]); synonym_map[p[4]].add(p[0])
    cui_preferred = {}; mesh_to_cui = {}; cui_all_names = defaultdict(set)
    target = {c for c, s in cui_stys.items() if s & ALLOWED_STYS} - BLACKLIST
    aho = ahocorasick.Automaton()
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[11]=="MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui:
                mesh_to_cui[p[13]] = p[0]
            if p[1]=="ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2]=="P" and p[0] not in cui_preferred: cui_preferred[p[0]] = p[14].strip()
            if p[0] in target and p[1]=="ENG":
                lo = p[14].strip().lower()
                if len(lo) >= 4:
                    try: aho.add_word(lo, (lo, p[0]))
                    except: pass
    aho.make_automaton()
    return aho, cui_preferred, mesh_to_cui, dict(parent_map), dict(synonym_map)


def search_disease(args):
    """단일 질환 PubMed 검색 + 텍스트 매칭 (병렬 worker)."""
    dn, dc, un, max_abs, mesh_to_cui_serial = args
    c2m = defaultdict(set)
    for m, cu in mesh_to_cui_serial.items(): c2m[cu].add(m)

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    rows, seen = [], set()
    muids = c2m.get(dc, set())
    if muids:
        mc = " OR ".join("mesh_terms LIKE '%%%s%%'" % m for m in muids)
        c.execute(f"SELECT pmid,abstract FROM abstracts WHERE ({mc}) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?", (max_abs,))
        for p,a in c.fetchall():
            if p not in seen: seen.add(p); rows.append((p,a))
    if len(rows) < max_abs:
        kw = re.sub(r"\(.*?\)", "", un).strip()
        if len(kw) >= 4:
            c.execute("SELECT pmid,abstract FROM abstracts WHERE (title LIKE ? OR abstract LIKE ?) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?",
                      (f"%{kw}%", f"%{kw}%", max_abs - len(rows)))
            for p,a in c.fetchall():
                if p not in seen: seen.add(p); rows.append((p,a))
    conn.close()
    return (dn, dc, rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: prepare_kg_prompts.py [symcat|rarebench] [max_abs]", flush=True)
        return
    bench = sys.argv[1]
    max_abs = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    output = RESULTS_DIR / f"kg_{bench}_prompts.json"
    if output.exists():
        print(f"이미 존재: {output}", flush=True); return

    # 질환 추출
    if bench == "symcat":
        with open("data/symcat/symcat_parsed.json") as f: sc = json.load(f)
        with open("data/symcat/disease_umls_mapping.json") as f: dm_raw = json.load(f)
        dm = dm_raw.get("mapping", dm_raw)
        diseases = {}
        for dn in sc["diseases"]:
            info = dm.get(dn, {})
            cui = info.get("cui") or info.get("umls_cui")
            if cui:
                diseases[dn] = {"cui": cui, "umls_name": info.get("umls_name", dn)}
    elif bench == "rarebench":
        diseases_codes = set()
        for f in ["HMS", "LIRICAL", "MME", "RAMEDIS"]:
            path = f"data/rarebench/data/{f}.jsonl"
            if not os.path.exists(path): continue
            with open(path) as fp:
                for line in fp:
                    d = json.loads(line)
                    for dx in d.get("RareDisease", []):
                        diseases_codes.add(dx)
        with open("data/rarebench/disease_umls_mapping.json") as f:
            dm = json.load(f)["mapping"]
        cui_to_d = {}
        for code in diseases_codes:
            if code in dm:
                cui = dm[code]["umls_cui"]
                name = dm[code].get("umls_name") or dm[code].get("disease_name")
                if not cui or not name: continue
                if cui not in cui_to_d:
                    cui_to_d[cui] = {"name": name, "code": code}
        diseases = {info["name"]: {"cui": cui, "umls_name": info["name"]} for cui, info in cui_to_d.items()}
    else:
        print(f"Unknown: {bench}", flush=True); return

    print(f"[{bench}] 질환 {len(diseases)}개", flush=True)

    # UMLS 로드
    aho, cui_preferred, mesh_to_cui, parent_map, synonym_map = load_umls()
    print(f"  CUIs: {len(cui_preferred):,}", flush=True)

    # 병렬 PubMed 검색
    print(f"\n[{bench}] PubMed 검색 (병렬, {max_abs}편/질환)...", flush=True)
    args_list = [(dn, info["cui"], info["umls_name"], max_abs, mesh_to_cui)
                 for dn, info in sorted(diseases.items(), key=lambda x: str(x[0])) if info["cui"]]

    t0 = time.time()
    with mp.Pool(processes=min(16, len(args_list))) as pool:
        all_rows = pool.map(search_disease, args_list)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # 텍스트 매칭 + 프롬프트 생성
    print(f"\n[{bench}] 텍스트 매칭 + 프롬프트...", flush=True)
    t0 = time.time()
    tasks = []
    for dn, dc, rows in all_rows:
        for pmid, ab in rows:
            ab_lower = ab.lower()
            cuis = set()
            for ei, (n, c2) in aho.iter(ab_lower):
                if c2 == dc: continue
                si = ei - len(n) + 1
                if si > 0 and ab_lower[si-1].isalpha(): continue
                if ei+1 < len(ab_lower) and ab_lower[ei+1].isalpha(): continue
                cuis.add(c2)
            if cuis:
                kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in sorted(cuis)[:50])
                tasks.append({"prompt": PROMPT.format(abstract=ab[:3000],
                    disease_name=cui_preferred.get(dc, dc), disease_cui=dc, keywords=kw),
                    "dc": dc})
    print(f"  완료: {time.time()-t0:.0f}초, 프롬프트: {len(tasks):,}", flush=True)

    # 저장
    save_data = {
        "bench": bench,
        "diseases": diseases,
        "tasks": tasks,
        "cui_preferred": cui_preferred,
        "parent_map": {k: list(v) for k, v in parent_map.items()},
        "synonym_map": {k: list(v) for k, v in synonym_map.items()},
    }
    with open(output, "w") as f:
        json.dump(save_data, f)
    print(f"  저장: {output}", flush=True)


if __name__ == "__main__":
    main()
