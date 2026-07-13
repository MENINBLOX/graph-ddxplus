#!/usr/bin/env python3
"""SymCat 50개 질환 이름으로 PubMed KG 구축.

원칙: SymCat에서 질환 이름만 사용. 증상-질환 관계는 사용하지 않음.
"""
from __future__ import annotations
import json, os, re, sqlite3, time
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from vllm import LLM, SamplingParams

DB_PATH = Path("/home/max/pubmed_data/pubmed.db")
UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")
KG_CACHE = RESULTS_DIR / "kg_symcat_cache.json"

ALLOWED_STYS = {"T047","T184","T191","T046","T048","T037","T019","T020","T190","T049","T033","T031","T040"}
BLACKLIST = {"C1457887","C3257980","C0012634","C0699748","C3839861"}

PROMPT = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Clinical findings and symptoms found in this abstract:
{keywords}

From the abstract, identify which of the above findings are CLINICAL SYMPTOMS or SIGNS that a patient with {disease_name} would present with.

Include symptoms patients report and physical examination findings.
Exclude lab results, imaging, other diseases, and synonyms of {disease_name}.

JSON only: [{{"cui":"...","relation":"symptom-of|sign-of"}}]
If none: []"""

MAX_ABSTRACTS = 500


def main():
    print("="*80, flush=True)
    print("SymCat 50개 질환 PubMed KG 구축", flush=True)
    print("="*80, flush=True)

    # SymCat 질환 이름 + UMLS 매핑
    with open("data/symcat/symcat_parsed.json") as f:
        sc = json.load(f)
    with open("data/symcat/disease_umls_mapping.json") as f:
        dm_raw = json.load(f)

    # disease_umls_mapping 구조 확인
    print(f"DM keys: {list(dm_raw.keys())[:3]}", flush=True)
    if "mapping" in dm_raw:
        dm = dm_raw["mapping"]
    else:
        dm = dm_raw

    diseases = {}
    for dn in sc["diseases"]:
        info = dm.get(dn, {})
        cui = info.get("cui") or info.get("umls_cui")
        if cui:
            diseases[dn] = {"cui": cui, "umls_name": info.get("umls_name", info.get("name", dn))}
        else:
            diseases[dn] = {"cui": None, "umls_name": dn}
    mapped = sum(1 for d in diseases.values() if d["cui"])
    print(f"질환: {len(diseases)}, UMLS 매핑: {mapped}", flush=True)

    # UMLS load (Aho-Corasick + names)
    print("\nUMLS 로드...", flush=True)
    cui_stys = defaultdict(set)
    with open(UMLS_DIR/"MRSTY.RRF") as f:
        for l in f: p=l.strip().split("|"); cui_stys[p[0]].add(p[1])
    parent_map = defaultdict(set); synonym_map = defaultdict(set)
    with open(UMLS_DIR/"MRREL.RRF") as f:
        for l in f:
            p=l.strip().split("|")
            if p[3] in("PAR","RB"): parent_map[p[0]].add(p[4])
            if p[3]=="SY": synonym_map[p[0]].add(p[4]); synonym_map[p[4]].add(p[0])

    cui_preferred = {}
    mesh_to_cui = {}
    cui_all_names = defaultdict(set)
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
    print(f"  CUIs: {len(cui_preferred):,}", flush=True)

    c2m = defaultdict(set)
    for m,c in mesh_to_cui.items(): c2m[c].add(m)

    # PubMed 검색 + 텍스트 매칭
    print(f"\nPubMed 검색 ({MAX_ABSTRACTS}편/질환)...", flush=True)
    conn = sqlite3.connect(str(DB_PATH))
    tasks = []
    for idx, (dn, info) in enumerate(sorted(diseases.items())):
        dc = info["cui"]
        if not dc:
            print(f"  [{idx+1}/{len(diseases)}] {dn}: CUI 없음, 스킵", flush=True)
            continue
        un = info["umls_name"]
        c = conn.cursor()
        rows, seen = [], set()
        muids = c2m.get(dc, set())
        if muids:
            mc = " OR ".join("mesh_terms LIKE '%%%s%%'" % m for m in muids)
            c.execute(f"SELECT pmid,abstract FROM abstracts WHERE ({mc}) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?", (MAX_ABSTRACTS,))
            for p,a in c.fetchall():
                if p not in seen: seen.add(p); rows.append((p,a))
        if len(rows) < MAX_ABSTRACTS:
            kws = []
            for raw in [dn, un]:
                kw = re.sub(r"\(.*?\)", "", raw).strip()
                kw = re.sub(r"\b(NOS|unspecified)\b", "", kw, flags=re.IGNORECASE).strip()
                if len(kw) >= 4: kws.append(kw)
            for kw in kws:
                if len(rows) >= MAX_ABSTRACTS: break
                c.execute("SELECT pmid,abstract FROM abstracts WHERE (title LIKE ? OR abstract LIKE ?) AND abstract IS NOT NULL AND length(abstract)>200 ORDER BY RANDOM() LIMIT ?",
                          (f"%{kw}%", f"%{kw}%", MAX_ABSTRACTS - len(rows)))
                for p,a in c.fetchall():
                    if p not in seen: seen.add(p); rows.append((p,a))

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
                kw = "\n".join(f"- {cui_preferred.get(c, c)} [{c}]" for c in sorted(cuis))
                tasks.append({"prompt": PROMPT.format(abstract=ab[:3000],
                    disease_name=cui_preferred.get(dc, dc), disease_cui=dc, keywords=kw),
                    "dc": dc})
        print(f"  [{idx+1}/{len(diseases)}] {dn}: {len(rows)}편", flush=True)
    conn.close()

    print(f"\n총 프롬프트: {len(tasks):,}", flush=True)

    # vLLM
    print("\nvLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=4096)
    convs = [[{"role": "user", "content": t["prompt"]}] for t in tasks]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # Parse
    all_rels = []
    for task, out in zip(tasks, outputs):
        text = out.outputs[0].text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```json\s*", "", text); text = re.sub(r"```\s*$", "", text)
        m = re.search(r"\[[\s\S]*?\]", text)
        if not m: continue
        try: items = json.loads(m.group())
        except: continue
        for item in items:
            if not isinstance(item, dict): continue
            cui = item.get("cui", ""); rel = item.get("relation", "")
            if cui and rel and rel != "manifestation-of":
                dc = task["dc"]
                if cui not in synonym_map.get(dc, set()) and cui not in parent_map.get(dc, set()) and dc not in parent_map.get(cui, set()):
                    all_rels.append({"dc": dc, "cui": cui})

    pair_counts = Counter(tuple(sorted([r["dc"], r["cui"]])) for r in all_rels)
    print(f"관계: {len(all_rels):,}, 고유 쌍: {len(pair_counts):,}", flush=True)

    cache_data = {
        "pair_counts": [[list(k), v] for k, v in pair_counts.most_common()],
        "diseases": {dn: info for dn, info in diseases.items()},
        "stats": {"prompts": len(tasks), "n_diseases": len(diseases)},
    }
    with open(KG_CACHE, "w") as f:
        json.dump(cache_data, f)
    print(f"저장: {KG_CACHE}", flush=True)


if __name__ == "__main__":
    main()
