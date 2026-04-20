#!/usr/bin/env python3
"""전체 KG 구축: 2,217편 × 최적 설정 + 다중 벤치마크 평가.

최적 설정:
- NER: scispaCy 0.85, T033/T034 제외, 블랙리스트 5 CUI
- CUI 정규화: MRREL PAR/RB 1-level
- LLM: gemma4 S2-J (이진 + 관계 범위 명시)
- 통계: MC=3, FDR 없음
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
import scipy.stats as stats

DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
UMLS_DIR = Path("data/umls_extracted")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"
MAX_PAIRS_PER_DOC = 15

ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT = """Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""


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
        try: return json.loads(m.group())
        except: pass
    return []

def dunning_g2(a, b, c, d):
    n = a+b+c+d
    if n == 0: return 0
    def g(o, e): return o*math.log(o/e) if o>0 and e>0 else 0
    ea=(a+b)*(a+c)/n; eb=(a+b)*(b+d)/n; ec=(c+d)*(a+c)/n; ed=(c+d)*(b+d)/n
    return 2*(g(a,ea)+g(b,eb)+g(c,ec)+g(d,ed))

def bh_fdr(pvals):
    n=len(pvals)
    if not n: return []
    idx=sorted(range(n),key=lambda i:pvals[i])
    q=[0.0]*n; prev=1.0
    for i in range(n-1,-1,-1):
        j=idx[i]; q[j]=min(prev,pvals[j]*n/(i+1)); prev=q[j]
    return q

def build_ancestor_map(parent_map, max_depth=1):
    cache = {}
    def get(cui, d=0):
        if cui in cache: return cache[cui]
        if d >= max_depth or cui not in parent_map:
            cache[cui] = set(); return set()
        anc = set()
        for p in parent_map[cui]:
            anc.add(p)
            if d + 1 < max_depth:
                anc |= get(p, d+1)
        cache[cui] = anc
        return anc
    return get

def cui_match(a, b, get_ancestors):
    if a == b: return True
    aa = get_ancestors(a)
    ba = get_ancestors(b)
    return b in aa or a in ba or bool(aa & ba)

def evaluate(our_pairs, gold_pairs, get_ancestors):
    mg = set(); mo = set()
    for op in our_pairs:
        for gp in gold_pairs:
            if ((cui_match(op[0],gp[0],get_ancestors) and cui_match(op[1],gp[1],get_ancestors)) or
                (cui_match(op[0],gp[1],get_ancestors) and cui_match(op[1],gp[0],get_ancestors))):
                mg.add(gp); mo.add(op)
    p = len(mo)/len(our_pairs) if our_pairs else 0
    r = len(mg)/len(gold_pairs) if gold_pairs else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    return {"precision":round(p,4),"recall":round(r,4),"f1":round(f1,4),
            "n_our":len(our_pairs),"n_gold":len(gold_pairs),"matched":len(mg)}


def main():
    print("=" * 80)
    print("전체 KG 구축: 2,217편 × 최적 설정")
    print("=" * 80)

    # 로드
    print("\n[1/6] 데이터 로드...")
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    parent_map = load_parent_map()
    get_ancestors = build_ancestor_map(parent_map, max_depth=1)

    with open(DATA_DIR / "exp_documents.json") as f:
        all_docs = json.load(f)["documents"]
    print(f"  문서: {len(all_docs)}")

    # ============================================================
    # [2/6] LLM 분류 (체크포인트 지원)
    # ============================================================
    print(f"\n[2/6] LLM 분류 (S2-J, {len(all_docs)}편)...")

    ckpt_file = DATA_DIR / "build_kg_checkpoint.json"
    all_cls = []
    processed = set()
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = json.load(f)
            all_cls = ckpt.get("classifications", [])
            processed = set(c["pmid"] for c in all_cls)
        print(f"  체크포인트: {len(processed)}편 완료")

    start = time.time()
    actual = 0
    errors = 0

    for idx, doc in enumerate(all_docs):
        if doc["pmid"] in processed:
            continue

        cuis = [c for c in doc["cuis"] if (cui_stys.get(c, set()) & ALLOWED_STYS) and c not in BLACKLIST]
        if len(cuis) < 2:
            continue

        pairs = []
        for i in range(min(len(cuis), MAX_PAIRS_PER_DOC)):
            for j in range(i+1, min(len(cuis), MAX_PAIRS_PER_DOC)):
                pairs.append({"cui_a": min(cuis[i],cuis[j]), "cui_b": max(cuis[i],cuis[j])})
        pairs = pairs[:MAX_PAIRS_PER_DOC]
        if not pairs: continue

        pairs_text = "\n".join(
            f"- ({cui_names.get(p['cui_a'],p['cui_a'])[:40]}, "
            f"{cui_names.get(p['cui_b'],p['cui_b'])[:40]}) "
            f"[CUI: {p['cui_a']}, {p['cui_b']}]"
            for p in pairs
        )
        prompt = PROMPT.format(text=doc["text"][:2500], pairs=pairs_text)

        try:
            response = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                cls = item.get("classification","").lower().strip().replace(" ","_")
                if cls in ("present","not_related"):
                    all_cls.append({
                        "pmid": doc["pmid"], "cui_a": item.get("cui_a",""),
                        "cui_b": item.get("cui_b",""), "classification": cls,
                        "seed_disease": doc.get("seed_disease",""),
                    })
            actual += 1
        except:
            errors += 1

        if actual % 20 == 0 and actual > 0:
            elapsed = time.time() - start
            rate = actual / elapsed
            remaining = (len(all_docs) - len(processed) - actual) / rate if rate > 0 else 0
            print(f"  [{len(processed)+actual:4d}/{len(all_docs)}] "
                  f"cls={len(all_cls):,} err={errors} "
                  f"{rate:.2f}/s ETA={remaining/60:.0f}분")
            with open(ckpt_file, "w") as f:
                json.dump({"classifications": all_cls}, f)

    # 최종 저장
    with open(ckpt_file, "w") as f:
        json.dump({"classifications": all_cls}, f)

    elapsed = time.time() - start
    dist = Counter(c["classification"] for c in all_cls)
    print(f"\n  LLM 완료: {len(all_cls):,}건 ({elapsed/60:.1f}분)")
    print(f"  present={dist.get('present',0)}, not_related={dist.get('not_related',0)}")

    # ============================================================
    # [3/6] KG 구축 (CUI 전파 + MC=3)
    # ============================================================
    print(f"\n[3/6] KG 구축...")

    pair_counts = Counter()
    pair_pmids = defaultdict(set)
    cui_doc_freq = Counter()

    for c in all_cls:
        if c["classification"] == "present":
            pair = tuple(sorted([c["cui_a"], c["cui_b"]]))
            pair_counts[pair] += 1
            pair_pmids[pair].add(c["pmid"])
        cui_doc_freq[c["cui_a"]] += 1
        cui_doc_freq[c["cui_b"]] += 1

    # CUI 1-level 전파
    expanded = Counter()
    expanded_pmids = defaultdict(set)
    for (a, b), cnt in pair_counts.items():
        expanded[(a, b)] += cnt
        expanded_pmids[(a, b)] |= pair_pmids[(a, b)]
        for pa in parent_map.get(a, set()):
            if cui_stys.get(pa, set()) & ALLOWED_STYS:
                key = tuple(sorted([pa, b]))
                expanded[key] += cnt
                expanded_pmids[key] |= pair_pmids[(a, b)]
        for pb in parent_map.get(b, set()):
            if cui_stys.get(pb, set()) & ALLOWED_STYS:
                key = tuple(sorted([a, pb]))
                expanded[key] += cnt
                expanded_pmids[key] |= pair_pmids[(a, b)]

    # MC=3 필터
    kg_edges = []
    for (a, b), cnt in expanded.items():
        if cnt < 3: continue
        c_a = max(cui_doc_freq.get(a, 1), 1)
        c_b = max(cui_doc_freq.get(b, 1), 1)
        total = len(all_docs)
        oe = (cnt * total) / (c_a * c_b) if c_a*c_b > 0 else 0
        jensen = (cnt**0.6) * (oe**0.4) if cnt > 0 and oe > 0 else 0

        ga = cnt; gb = max(c_a-cnt, 0); gc = max(c_b-cnt, 0); gd = max(total-ga-gb-gc, 0)
        g2 = dunning_g2(ga, gb, gc, gd)
        pv = 1 - stats.chi2.cdf(g2, df=1) if g2 > 0 else 1.0

        kg_edges.append({
            "cui_a": a, "cui_b": b, "polarity": "present",
            "n_present": cnt, "jensen_score": round(jensen, 3),
            "g2": round(g2, 2), "p_value": pv,
            "pmids": sorted(expanded_pmids[(a, b)]),
        })

    kg_edges.sort(key=lambda e: -e["jensen_score"])

    unique_cuis = set()
    for e in kg_edges:
        unique_cuis.add(e["cui_a"])
        unique_cuis.add(e["cui_b"])

    print(f"  KG: {len(unique_cuis)} 노드, {len(kg_edges)} 엣지")

    # Semantic type 분포
    sty_dist = Counter()
    for cui in unique_cuis:
        for s in cui_stys.get(cui, set()) & ALLOWED_STYS:
            sty_dist[s] += 1
    sty_names = {"T047":"Disease","T184":"Sign/Symptom","T191":"Neoplastic","T046":"PathFunc",
                 "T048":"Mental","T037":"Injury","T019":"Congenital","T020":"Acquired",
                 "T190":"AnatAbnorm","T049":"CellDys"}
    print(f"  Semantic type 분포:")
    for s, cnt in sty_dist.most_common():
        print(f"    {sty_names.get(s,s)}: {cnt}")

    # 상위 엣지
    print(f"\n  상위 20 엣지:")
    for e in kg_edges[:20]:
        a = cui_names.get(e["cui_a"], e["cui_a"])[:30]
        b = cui_names.get(e["cui_b"], e["cui_b"])[:30]
        print(f"    {a:30s} - {b:30s} n={e['n_present']:>3d} J={e['jensen_score']:.2f}")

    # ============================================================
    # [4/6] DDXPlus 벤치마크
    # ============================================================
    print(f"\n[4/6] DDXPlus 벤치마크...")

    with open(DATA_DIR / "gold_standard.json") as f:
        gold = json.load(f)

    # Symptom only gold
    import sys; sys.path.insert(0, ".")
    from pilot.scripts.exp_full_test import prepare_gold_symptom_only
    gold_sym = prepare_gold_symptom_only()
    gold_full = set(tuple(p) for p in gold["ddxplus"]["pairs"])

    our_pairs = set((e["cui_a"], e["cui_b"]) for e in kg_edges)
    eval_sym = evaluate(our_pairs, gold_sym, get_ancestors)
    eval_full = evaluate(our_pairs, gold_full, get_ancestors)

    print(f"  DDXPlus (증상만 {len(gold_sym)}쌍): P={eval_sym['precision']:.3f} R={eval_sym['recall']:.3f} F1={eval_sym['f1']:.3f} ({eval_sym['matched']}/{len(gold_sym)})")
    print(f"  DDXPlus (전체 {len(gold_full)}쌍):   P={eval_full['precision']:.3f} R={eval_full['recall']:.3f} F1={eval_full['f1']:.3f} ({eval_full['matched']}/{len(gold_full)})")

    # 질환별 recall
    disease_cuis = gold["ddxplus"]["disease_cuis"]
    disease_sym_map = gold["ddxplus"].get("disease_symptom_map", {})
    print(f"\n  질환별 recall (상위/하위 5):")
    disease_recalls = []
    for dname, dcui in disease_cuis.items():
        d_gold = set()
        for gp in gold_sym:
            if dcui in gp:
                d_gold.add(gp)
        if not d_gold: continue
        d_matched = 0
        for op in our_pairs:
            for gp in d_gold:
                if ((cui_match(op[0],gp[0],get_ancestors) and cui_match(op[1],gp[1],get_ancestors)) or
                    (cui_match(op[0],gp[1],get_ancestors) and cui_match(op[1],gp[0],get_ancestors))):
                    d_matched += 1
                    break
        rec = d_matched / len(d_gold) if d_gold else 0
        disease_recalls.append((dname, rec, d_matched, len(d_gold)))

    disease_recalls.sort(key=lambda x: -x[1])
    for name, rec, m, t in disease_recalls[:5]:
        print(f"    {name:40s} recall={rec:.1%} ({m}/{t})")
    print(f"    ...")
    for name, rec, m, t in disease_recalls[-5:]:
        print(f"    {name:40s} recall={rec:.1%} ({m}/{t})")

    # ============================================================
    # [5/6] HPO 벤치마크
    # ============================================================
    print(f"\n[5/6] HPO 벤치마크...")
    hpo_gold = set(tuple(p) for p in gold["hpo"]["pairs"])
    if hpo_gold:
        eval_hpo = evaluate(our_pairs, hpo_gold, get_ancestors)
        print(f"  HPO ({len(hpo_gold)}쌍): P={eval_hpo['precision']:.3f} R={eval_hpo['recall']:.3f} F1={eval_hpo['f1']:.3f} ({eval_hpo['matched']}/{len(hpo_gold)})")
    else:
        print(f"  HPO gold 데이터 없음")

    # ============================================================
    # [6/6] SemMedDB 비교
    # ============================================================
    print(f"\n[6/6] SemMedDB 비교...")
    semmed_file = DATA_DIR / "semmed_baseline_pairs.json"
    if semmed_file.exists():
        with open(semmed_file) as f:
            semmed_pairs = set(tuple(p) for p in json.load(f))
        # 우리 KG 엣지가 SemMedDB에 있는 비율
        in_semmed = sum(1 for op in our_pairs if op in semmed_pairs)
        print(f"  SemMedDB ({len(semmed_pairs):,}쌍)")
        print(f"  우리 KG → SemMedDB 겹침: {in_semmed}/{len(our_pairs)} ({in_semmed/len(our_pairs):.1%})")
        # SemMedDB → 우리 KG
        semmed_in_ours = sum(1 for sp in semmed_pairs if sp in our_pairs)
        print(f"  SemMedDB → 우리 KG 겹침: {semmed_in_ours}/{len(semmed_pairs)} ({semmed_in_ours/len(semmed_pairs):.1%})")
    else:
        print(f"  SemMedDB baseline 파일 없음")

    # 저장
    output = {
        "config": {
            "model": MODEL, "prompt": "S2-J", "ner_threshold": 0.85,
            "allowed_stys": sorted(ALLOWED_STYS), "blacklist": sorted(BLACKLIST),
            "cui_propagation": "1-level PAR/RB", "min_cooccurrence": 3,
        },
        "kg_stats": {
            "n_documents": len(all_docs), "n_classifications": len(all_cls),
            "n_nodes": len(unique_cuis), "n_edges": len(kg_edges),
            "sty_distribution": dict(sty_dist),
        },
        "benchmarks": {
            "ddxplus_symptom": eval_sym,
            "ddxplus_full": eval_full,
            "hpo": evaluate(our_pairs, hpo_gold, get_ancestors) if hpo_gold else {},
        },
        "edges": kg_edges,
    }
    with open(DATA_DIR / "final_kg.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {DATA_DIR / 'final_kg.json'}")
    print("완료!")


if __name__ == "__main__":
    main()
