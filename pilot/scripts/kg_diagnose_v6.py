#!/usr/bin/env python3
"""진단 v6: 텍스트 매칭 + UMLS parent/child CUI 확장.

v3c(29.3%) → parent CUI 확장으로 매칭 범위 확대.
매칭된 증상 CUI의 UMLS 계층(PAR/RB, CHD/RN)을 따라 확장하여
KG 엣지와의 교집합을 넓힘.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick

UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")
KG_CACHE = RESULTS_DIR / "kg_v3_cache.json"

NOISE_CUIS = {
    "C0150312", "C0442743", "C0039082", "C0221423", "C1457887",
    "C0205390", "C0442804", "C3839861", "C0332157", "C1457868",
    "C0445223", "C1272751", "C0015663", "C0277814", "C5202885",
    "C0153933", "C0585362",
}

STOPWORDS = {
    "does", "have", "your", "you", "the", "and", "for", "are", "with",
    "that", "this", "from", "been", "were", "being", "which", "their",
    "than", "other", "about", "into", "over", "some", "only", "very",
    "also", "just", "more", "most", "such", "much", "will", "would",
    "could", "should", "make", "like", "time", "when", "what", "where",
    "how", "who", "all", "each", "every", "both", "few", "any", "not",
    "can", "may", "her", "his", "its", "our", "they", "them", "then",
    "had", "has", "him", "but", "one", "two", "way", "day", "did",
    "get", "got", "let", "say", "she", "too", "use", "yes", "yet",
    "now", "new", "old", "see", "own", "why", "try", "ask", "set",
    "related", "reason", "consulting", "significant", "measured",
    "thermometer", "either", "believe", "racing", "missing", "beat",
    "fast", "irregularly", "problems", "situation", "associated",
    "inability", "speak", "trouble", "keeping", "opening", "raising",
    "annoying", "else", "body", "somewhere", "anywhere", "nowhere",
    "recently", "currently", "usually", "often", "sometimes",
    "worse", "better",
}


def load_umls():
    print("  MRCONSO...", end="", flush=True)
    can = defaultdict(set)
    cp = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cp:
                    cp[p[0]] = p[14].strip()
    print(f" {len(can):,}", flush=True)

    print("  MRREL...", end="", flush=True)
    parent_map = defaultdict(set)
    child_map = defaultdict(set)
    syn_map = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parent_map[p[0]].add(p[4])
                child_map[p[4]].add(p[0])
            if p[3] == "SY":
                syn_map[p[0]].add(p[4])
                syn_map[p[4]].add(p[0])
    print(f" {len(parent_map):,} parents, {len(child_map):,} children", flush=True)

    return dict(can), cp, dict(parent_map), dict(child_map), dict(syn_map)


def load_ddxplus():
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f:
        cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)

    diseases = {}
    fr2cui = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]
        diseases[dn] = {"cui": dc, "umls_name": icd_map[dn]["umls_name"],
                        "fr": info.get("cond-name-fr", "")}
        fr2cui[info.get("cond-name-fr", "")] = dc

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {
            "question_en": info.get("question_en", ""),
            "is_antecedent": info.get("is_antecedent", False),
            "value_en": {},
        }
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"):
                    ev_info[eid]["value_en"][k] = v["en"]
    return diseases, fr2cui, ev_info


def load_kg():
    with open(KG_CACHE) as f:
        cache = json.load(f)
    pc = Counter()
    for k, v in cache["pair_counts"]:
        pc[tuple(k)] = v
    return pc


def build_ds(pc, dcs, mc=1):
    ds = defaultdict(dict)
    scuis = set()
    for (a, b), cnt in pc.items():
        if cnt < mc: continue
        if a in NOISE_CUIS or b in NOISE_CUIS: continue
        if a in dcs: ds[a][b] = cnt; scuis.add(b)
        if b in dcs: ds[b][a] = cnt; scuis.add(a)
    return dict(ds), scuis


def build_aho(scuis, can):
    aho = ahocorasick.Automaton()
    n = 0
    for cui in scuis:
        for name in can.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) < 4 or lo in STOPWORDS: continue
            try: aho.add_word(lo, (lo, cui)); n += 1
            except: pass
    aho.make_automaton()
    return aho, n


def text_match_patient(evidences, ev_info, aho):
    """환자 evidence → 텍스트 매칭으로 증상 CUI."""
    cuis = set()
    for ev in evidences:
        parts = ev.split("_@_")
        base = parts[0]
        value = parts[1] if len(parts) > 1 else None
        info = ev_info.get(base, {})
        if info.get("is_antecedent"): continue

        terms = []
        bc = re.sub(r"_.*", "", base)
        bc = re.sub(r"xx$", "", bc)
        if len(bc) >= 3 and bc not in STOPWORDS:
            terms.append(bc)

        q = info.get("question_en", "")
        if q:
            text = re.sub(r"\(.*?\)", "", q)
            text = re.sub(r"[?.,;:!]", "", text)
            terms.extend(w.lower() for w in text.split()
                         if w.lower() not in STOPWORDS and len(w) >= 3)
            ql = q.lower()
            for phrase in ["chest pain", "sore throat", "shortness of breath",
                           "difficulty breathing", "weight loss", "weight gain",
                           "loss of consciousness", "muscle pain", "muscle spasm",
                           "nasal congestion", "runny nose", "skin lesion",
                           "skin rash", "black stool", "bloody stool",
                           "heart palpitation", "double vision", "swollen"]:
                if phrase in ql: terms.append(phrase)

        if value:
            val_en = info.get("value_en", {}).get(value, "")
            if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                vc = re.sub(r"\([rl]\)", "", val_en.lower()).strip()
                if vc and vc not in STOPWORDS:
                    terms.append(vc)
                    if "pain" in q.lower():
                        terms.append(f"{vc} pain")
                        for part in vc.split():
                            if part not in STOPWORDS and len(part) >= 4:
                                terms.append(f"{part} pain")

        pt = " . ".join(terms)
        for ei, (n, cui) in aho.iter(pt):
            si = ei - len(n) + 1
            if si > 0 and pt[si - 1].isalpha(): continue
            if ei + 1 < len(pt) and pt[ei + 1].isalpha(): continue
            cuis.add(cui)

    return cuis


def expand_cuis(cuis, parent_map, child_map, syn_map, scuis, levels=1):
    """CUI를 UMLS 계층으로 확장 (KG 증상 범위 내에서만)."""
    expanded = set(cuis)
    frontier = set(cuis)
    for _ in range(levels):
        next_frontier = set()
        for c in frontier:
            # Parents
            for p in parent_map.get(c, set()):
                if p in scuis and p not in expanded:
                    next_frontier.add(p)
            # Children
            for ch in child_map.get(c, set()):
                if ch in scuis and ch not in expanded:
                    next_frontier.add(ch)
            # Synonyms
            for s in syn_map.get(c, set()):
                if s in scuis and s not in expanded:
                    next_frontier.add(s)
        expanded.update(next_frontier)
        frontier = next_frontier
    return expanded


# ─── 알고리즘 ───

def d_bayesian(ps, ds, dcs, all_s):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = -1e6; continue
        tw = sum(s.values()) + len(all_s) * 0.1
        sc[dc] = sum(math.log((s[x] + 0.1) / tw + 1e-10) if x in s
                     else math.log(0.1 / tw + 1e-10) for x in ps)
    return sorted(sc.items(), key=lambda x: -x[1])

def d_v15(ps, ds, dcs):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = 0; continue
        cf = sum(1 for x in s if x in ps)
        dn = sum(1 for x in s if x not in ps)
        sc[dc] = cf / (cf + dn + 1) * cf if cf else 0
    return sorted(sc.items(), key=lambda x: -x[1])

def d_idf(ps, ds, dcs, sdf, nd):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = 0; continue
        sc[dc] = sum((math.log(nd / (sdf[x] + 1)) + 1) * c
                     for x, c in s.items() if x in ps)
    return sorted(sc.items(), key=lambda x: -x[1])


def evaluate(patients, ds, aho, ev_info, fr2cui, dcs, scuis,
             parent_map, child_map, syn_map, expand_levels, algo,
             sdf, nd, all_s):
    t1 = t3 = t5 = t10 = n = nm = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1

        ps = text_match_patient(p["evidences"], ev_info, aho)
        if expand_levels > 0:
            ps = expand_cuis(ps, parent_map, child_map, syn_map, scuis, expand_levels)
        if not ps: nm += 1; continue

        if algo == "bayesian": r = d_bayesian(ps, ds, dcs, all_s)
        elif algo == "v15": r = d_v15(ps, ds, dcs)
        elif algo == "idf": r = d_idf(ps, ds, dcs, sdf, nd)

        rd = [d for d, _ in r]
        if rd and rd[0] == tdc: t1 += 1
        if tdc in rd[:3]: t3 += 1
        if tdc in rd[:5]: t5 += 1
        if tdc in rd[:10]: t10 += 1

    return {
        "n": n, "nm": nm,
        "gtpa1": round(100 * t1 / n, 2) if n else 0,
        "gtpa3": round(100 * t3 / n, 2) if n else 0,
        "gtpa5": round(100 * t5 / n, 2) if n else 0,
        "gtpa10": round(100 * t10 / n, 2) if n else 0,
    }


def main():
    print("=" * 80, flush=True)
    print("진단 v6: 텍스트 매칭 + UMLS CUI 확장", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] 로드...", flush=True)
    can, cp, parent_map, child_map, syn_map = load_umls()
    diseases, fr2cui, ev_info = load_ddxplus()
    dcs = {v["cui"] for v in diseases.values()}
    pc = load_kg()
    print(f"  KG: {len(pc):,} 쌍", flush=True)

    print("\n[2] 테스트 데이터...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "pathology": row["PATHOLOGY"],
            })
    print(f"  {len(patients):,}명", flush=True)

    best = 0
    best_cfg = ""
    algos = ["bayesian", "v15", "idf"]

    for mc in [1, 2, 3]:
        ds, scuis = build_ds(pc, dcs, mc=mc)
        aho, nn = build_aho(scuis, can)
        ndw = sum(1 for d in dcs if d in ds and ds[d])

        sdf = Counter()
        for syms in ds.values():
            for s in syms: sdf[s] += 1
        nd = max(len(ds), 1)
        all_s = set()
        for syms in ds.values(): all_s.update(syms.keys())

        for expand in [0, 1, 2]:
            print(f"\n  MC={mc} expand={expand}: 증상={len(scuis)}, 질환={ndw}", flush=True)

            for algo in algos:
                t0 = time.time()
                r = evaluate(patients, ds, aho, ev_info, fr2cui, dcs, scuis,
                             parent_map, child_map, syn_map, expand, algo,
                             sdf, nd, all_s)
                el = time.time() - t0
                m = ""
                if r["gtpa1"] > best:
                    best = r["gtpa1"]
                    best_cfg = f"MC={mc} exp={expand} {algo}"
                    m = " ★"
                print(
                    f"    {algo:<10}: @1={r['gtpa1']:>5.1f}% @3={r['gtpa3']:>5.1f}% "
                    f"@5={r['gtpa5']:>5.1f}% @10={r['gtpa10']:>5.1f}% "
                    f"(nm={r['nm']:,}, {el:.0f}s){m}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"최고 GTPA@1 = {best:.1f}% ({best_cfg})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
