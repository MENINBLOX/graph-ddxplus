#!/usr/bin/env python3
"""Ablation 테스트 (최적화): 초록 수 vs F1.

2000편/질환 한 번 수집 → 서브셋으로 50/100/200/500/1000/2000 실험.
vLLM batch로 각 실험 실행.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import random
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049",
                "T033", "T031", "T040"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

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


def load_umls():
    cui_stys = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            cui_stys[p[0]].add(p[1])
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
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[11] == "MSH" and p[13].startswith("D") and p[13] not in mesh_to_cui:
                mesh_to_cui[p[13]] = p[0]
            if p[1] == "ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cui_preferred:
                    cui_preferred[p[0]] = p[14].strip()
    target = {c for c, s in cui_stys.items() if s & ALLOWED_STYS} - BLACKLIST
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
    return dict(cui_stys), dict(parent_map), dict(synonym_map), mesh_to_cui, dict(cui_all_names), cui_preferred, A


def prepare_gold():
    with open("data/ddxplus/release_conditions_en.json") as f: conditions = json.load(f)
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
    for dn, info in conditions.items():
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
                      (f"%{kw}%", f"%{kw}%", limit - len(rows)))
            for p, a in c.fetchall():
                if p not in seen: seen.add(p); rows.append((p, a))
    return rows


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
    print("Ablation: 초록 수 vs F1 (1-level 매칭)")
    print("=" * 80)

    print("\n[1] UMLS 로드...")
    cui_stys, parent_map, synonym_map, mesh_to_cui, cui_all_names, cui_preferred, automaton = load_umls()
    c2m = defaultdict(set)
    for m, c in mesh_to_cui.items(): c2m[c].add(m)
    gold, dcuis = prepare_gold()

    # 2000편/질환 한 번만 수집
    print("\n[2] 초록 수집 (2000편/질환, 1회)...")
    conn = sqlite3.connect(str(DB_PATH))
    all_docs = {}  # dc -> [(pmid, abstract, cuis), ...]

    for idx, (dn, dinfo) in enumerate(sorted(dcuis.items())):
        dc, un = dinfo["cui"], dinfo["umls_name"]
        rows = search_abs(conn, dc, dn, un, c2m, cui_all_names, 2000)
        docs = []
        for pmid, ab in rows:
            cuis = text_match(ab.lower(), automaton, exclude=dc)
            if cuis:
                docs.append({"pmid": pmid, "abstract": ab, "cuis": sorted(cuis)})
        all_docs[dc] = docs
        print(f"  [{idx+1}/49] {dn}: {len(docs)}편")
    conn.close()
    total = sum(len(d) for d in all_docs.values())
    print(f"\n  총: {total}편")

    # vLLM 로드
    print("\n[3] vLLM 로드...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=4096)

    # 실험: 서브셋 크기별
    results = {}

    for abs_limit in [50, 100, 200, 500, 1000, 2000]:
        print(f"\n{'='*60}")
        print(f"실험: {abs_limit}편/질환")
        print(f"{'='*60}")

        # 서브셋 선택 (앞에서부터 잘라서 — 랜덤 이미 섞여있음)
        tasks = []
        for dc, docs in all_docs.items():
            subset = docs[:abs_limit]
            dn = cui_preferred.get(dc, dc)
            for doc in subset:
                kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in doc["cuis"])
                tasks.append({
                    "prompt": PROMPT_V2.format(abstract=doc["abstract"][:3000],
                        disease_name=dn, disease_cui=dc, keywords=kw),
                    "disease_cui": dc,
                })
        print(f"  총 초록: {len(tasks)}")

        # vLLM batch
        convs = [[{"role": "user", "content": t["prompt"]}] for t in tasks]
        t0 = time.time()
        outputs = llm.chat(convs, sampling)
        elapsed = time.time() - t0
        print(f"  LLM: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)")

        # 파싱 + 후처리
        all_cls = []
        for task, out in zip(tasks, outputs):
            for item in parse_json_r(out.outputs[0].text):
                cui, rel = item.get("cui", ""), item.get("relation", "")
                if cui and rel:
                    all_cls.append({"dc": task["disease_cui"], "cui": cui, "rel": rel})

        filtered = []
        for c in all_cls:
            if c["rel"] == "manifestation-of": continue
            if c["cui"] in synonym_map.get(c["dc"], set()): continue
            if c["cui"] in parent_map.get(c["dc"], set()) or c["dc"] in parent_map.get(c["cui"], set()): continue
            filtered.append(c)

        print(f"  Raw: {len(all_cls)}, Filtered: {len(filtered)}")

        pc = Counter()
        for c in filtered:
            pc[tuple(sorted([c["dc"], c["cui"]]))] += 1

        # MC sweep
        best_f1, best_mc = 0, 1
        print(f"  {'MC':>4} {'edges':>7} {'P':>7} {'R':>7} {'F1':>7}")
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
            print(f"  MC={mc:>2} {len(exp):>7,} {p:>7.3f} {r:>7.3f} {f1:>7.3f}{marker}")

        results[abs_limit] = (best_f1, best_mc, len(tasks))

    # 요약
    print(f"\n{'='*80}")
    print(f"요약: 초록 수 vs F1 (1-level PAR 매칭)")
    print(f"{'='*80}")
    print(f"  {'편/질환':>8} {'총 초록':>8} {'Best MC':>8} {'F1':>8}")
    print(f"  {'-'*36}")
    for ac in [50, 100, 200, 500, 1000, 2000]:
        f1, mc, total = results[ac]
        print(f"  {ac:>8} {total:>8,} {mc:>8} {f1:>8.3f}")
    print(f"\n  참고: 파일럿 = 45편/질환, scispaCy NER + S2-J, F1=0.793")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
