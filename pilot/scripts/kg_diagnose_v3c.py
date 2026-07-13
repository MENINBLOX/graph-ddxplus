#!/usr/bin/env python3
"""진단 v3c: 캐시된 KG + evidence 이름 포함 텍스트 매칭.

v3b 문제점: evidence 질문의 구어체와 KG 의학 용어 간 어휘 불일치.
개선:
  1. evidence 프랑스어 이름을 검색 텍스트에 포함 (melena=Melena, dyspn⊂Dyspnea)
  2. pain + location 합성어 생성 강화
  3. 질문에서 핵심 의학 구절 추출
  4. min_len=4 (pain 매칭 필수)
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


def load_umls_names():
    print("  MRCONSO...", end="", flush=True)
    cui_all_names = defaultdict(set)
    cui_preferred = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cui_preferred:
                    cui_preferred[p[0]] = p[14].strip()
    print(f" {len(cui_all_names):,}", flush=True)
    return dict(cui_all_names), cui_preferred


def load_ddxplus():
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f:
        cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)

    diseases = {}
    disease_fr_to_cui = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]
        fr = info.get("cond-name-fr", "")
        diseases[dn] = {"cui": dc, "umls_name": icd_map[dn]["umls_name"], "fr": fr}
        disease_fr_to_cui[fr] = dc

    ev_text_info = {}
    for eid, info in ev_fr.items():
        ev_text_info[eid] = {
            "question_en": info.get("question_en", ""),
            "is_antecedent": info.get("is_antecedent", False),
            "value_en": {},
        }
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"):
                    ev_text_info[eid]["value_en"][k] = v["en"]

    return diseases, disease_fr_to_cui, ev_text_info


def load_kg_cache():
    with open(KG_CACHE) as f:
        cache = json.load(f)
    pair_counts = Counter()
    for k, v in cache["pair_counts"]:
        pair_counts[tuple(k)] = v
    print(f"  KG: {len(pair_counts):,} 쌍", flush=True)
    return pair_counts


def build_symptom_aho(symptom_cuis, cui_all_names, min_len=4):
    aho = ahocorasick.Automaton()
    n = 0
    for cui in symptom_cuis:
        for name in cui_all_names.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) < min_len: continue
            words = lo.split()
            if len(words) == 1 and lo in STOPWORDS: continue
            try: aho.add_word(lo, (lo, cui)); n += 1
            except: pass
    aho.make_automaton()
    return aho, n


def patient_evidence_to_text(evidences, ev_text_info):
    """환자 evidence → 풍부한 텍스트 (evidence 이름 포함)."""
    terms = []
    for ev in evidences:
        parts = ev.split("_@_")
        base = parts[0]
        value = parts[1] if len(parts) > 1 else None

        info = ev_text_info.get(base, {})
        if info.get("is_antecedent"):
            continue

        # 1. Evidence 프랑스어 이름 그대로 추가 (의학 용어와 동일한 경우 많음)
        # melena → melena, dyspn → dyspn(ea), palpit → palpit(ations)
        base_clean = re.sub(r"_.*", "", base)  # douleurxx_endroitducorps → douleurxx
        base_clean = re.sub(r"xx$", "", base_clean)  # douleurxx → douleur
        if len(base_clean) >= 3 and base_clean not in STOPWORDS:
            terms.append(base_clean)

        # 2. 질문에서 의학 키워드 추출
        q = info.get("question_en", "")
        if q:
            text = re.sub(r"\(.*?\)", "", q)
            text = re.sub(r"[?.,;:!]", "", text)
            words = [w.lower() for w in text.split()
                     if w.lower() not in STOPWORDS and len(w) >= 3]
            terms.extend(words)

            # 핵심 의학 구절 추출 (2-3 word phrases)
            q_lower = q.lower()
            for phrase in [
                "chest pain", "sore throat", "shortness of breath",
                "difficulty breathing", "weight loss", "weight gain",
                "loss of consciousness", "muscle pain", "muscle spasm",
                "nasal congestion", "runny nose", "skin lesion",
                "skin rash", "black stool", "bloody stool",
                "heart palpitation", "double vision",
                "high pitched sound", "swollen",
            ]:
                if phrase in q_lower:
                    terms.append(phrase)

        # 3. Value 영문 추가
        if value and info.get("value_en"):
            val_en = info["value_en"].get(value, "")
            if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                val_clean = re.sub(r"\([rl]\)", "", val_en.lower()).strip()
                if val_clean and val_clean not in STOPWORDS:
                    terms.append(val_clean)
                    # pain/lesion + location compound
                    if "pain" in q.lower():
                        terms.append(f"{val_clean} pain")
                        for part in val_clean.split():
                            if part not in STOPWORDS and len(part) >= 4:
                                terms.append(f"{part} pain")
                    if "lesion" in q.lower() or "rash" in q.lower() or "skin" in q.lower():
                        terms.append(f"{val_clean} rash")
                        terms.append(f"{val_clean} lesion")

    return " . ".join(terms)


def match_text(text, aho):
    matched = set()
    for ei, (n, cui) in aho.iter(text):
        si = ei - len(n) + 1
        if si > 0 and text[si - 1].isalpha(): continue
        if ei + 1 < len(text) and text[ei + 1].isalpha(): continue
        matched.add(cui)
    return matched


# ─── 진단 알고리즘 ───

def d_coverage(ps, ds, dcs):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = 0; continue
        sc[dc] = sum(1 for x in s if x in ps) / len(s)
    return sorted(sc.items(), key=lambda x: -x[1])

def d_idf(ps, ds, dcs, sdf, nd):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = 0; continue
        v = 0
        for x, c in s.items():
            if x in ps: v += (math.log(nd / (sdf[x] + 1)) + 1) * c
        sc[dc] = v
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

def d_bayesian(ps, ds, dcs, all_s):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = -1e6; continue
        tw = sum(s.values()) + len(all_s) * 0.1
        ls = 0
        for x in ps:
            p = (s[x] + 0.1) / tw if x in s else 0.1 / tw
            ls += math.log(p + 1e-10)
        sc[dc] = ls
    return sorted(sc.items(), key=lambda x: -x[1])

def d_idf_neg(ps, ds, dcs, sdf, nd):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = 0; continue
        v = 0
        for x, c in s.items():
            idf = math.log(nd / (sdf[x] + 1)) + 1
            if x in ps: v += idf * c
            else: v -= idf * 0.5
        sc[dc] = v
    return sorted(sc.items(), key=lambda x: -x[1])

def d_weighted(ps, ds, dcs):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = 0; continue
        t = sum(s.values())
        sc[dc] = sum(c for x, c in s.items() if x in ps) / t
    return sorted(sc.items(), key=lambda x: -x[1])


def evaluate(patients, ds, aho, ev_info, fr2cui, dcs, algo, sdf, nd, all_s):
    t1 = t3 = t5 = t10 = n = nm = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        pt = patient_evidence_to_text(p["evidences"], ev_info)
        ps = match_text(pt, aho)
        if not ps: nm += 1; continue

        if algo == "coverage": r = d_coverage(ps, ds, dcs)
        elif algo == "weighted": r = d_weighted(ps, ds, dcs)
        elif algo == "idf": r = d_idf(ps, ds, dcs, sdf, nd)
        elif algo == "v15_ratio": r = d_v15(ps, ds, dcs)
        elif algo == "bayesian": r = d_bayesian(ps, ds, dcs, all_s)
        elif algo == "idf_neg": r = d_idf_neg(ps, ds, dcs, sdf, nd)

        rd = [d for d, _ in r]
        if rd and rd[0] == tdc: t1 += 1
        if tdc in rd[:3]: t3 += 1
        if tdc in rd[:5]: t5 += 1
        if tdc in rd[:10]: t10 += 1

    return {
        "n": n, "no_match": nm,
        "gtpa1": round(100 * t1 / n, 2) if n else 0,
        "gtpa3": round(100 * t3 / n, 2) if n else 0,
        "gtpa5": round(100 * t5 / n, 2) if n else 0,
        "gtpa10": round(100 * t10 / n, 2) if n else 0,
    }


def main():
    print("=" * 80, flush=True)
    print("진단 v3c: evidence 이름 + 의학구절 포함 텍스트 매칭", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] 로드...", flush=True)
    cui_all_names, cui_preferred = load_umls_names()
    diseases, fr2cui, ev_info = load_ddxplus()
    dcs = {v["cui"] for v in diseases.values()}
    pc = load_kg_cache()

    print("\n[2] 테스트 데이터...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "pathology": row["PATHOLOGY"],
            })
    print(f"  {len(patients):,}명", flush=True)

    # Debug: 매칭 확인
    print("\n[DEBUG] 매칭 검증...", flush=True)
    for mc in [1]:
        scuis = set()
        ds = defaultdict(dict)
        for (a, b), cnt in pc.items():
            if cnt < mc: continue
            if a in dcs: ds[a][b] = cnt; scuis.add(b)
            if b in dcs: ds[b][a] = cnt; scuis.add(a)
        ds = dict(ds)
        aho, nn = build_symptom_aho(scuis, cui_all_names, min_len=4)

        for pi in range(3):
            p = patients[pi]
            pt = patient_evidence_to_text(p["evidences"], ev_info)
            ps = match_text(pt, aho)
            matched_names = []
            for ei, (n, cui) in aho.iter(pt):
                si = ei - len(n) + 1
                if si > 0 and pt[si - 1].isalpha(): continue
                if ei + 1 < len(pt) and pt[ei + 1].isalpha(): continue
                matched_names.append((n, cui_preferred.get(cui, "?")))
            print(f"  P{pi} ({p['pathology']}): {len(ps)} CUIs", flush=True)
            print(f"    Text: {pt[:200]}...", flush=True)
            for n, pref in sorted(set(matched_names)):
                print(f"    {n:<35} → {pref}", flush=True)
            print(flush=True)

    # 평가
    print("[3] 평가...", flush=True)
    algos = ["coverage", "weighted", "idf", "v15_ratio", "bayesian", "idf_neg"]
    best = 0
    best_cfg = ""
    results = []

    for mc in [1, 2, 3, 5]:
        scuis = set()
        ds = defaultdict(dict)
        for (a, b), cnt in pc.items():
            if cnt < mc: continue
            if a in dcs: ds[a][b] = cnt; scuis.add(b)
            if b in dcs: ds[b][a] = cnt; scuis.add(a)
        ds = dict(ds)
        aho, nn = build_symptom_aho(scuis, cui_all_names, min_len=4)
        ndw = sum(1 for d in dcs if d in ds and ds[d])

        sdf = Counter()
        for syms in ds.values():
            for s in syms: sdf[s] += 1
        nd = max(len(ds), 1)
        all_s = set()
        for syms in ds.values(): all_s.update(syms.keys())

        print(f"\n  MC={mc}: 증상={len(scuis)}, 이름={nn:,}, 질환={ndw}", flush=True)
        for algo in algos:
            t0 = time.time()
            r = evaluate(patients, ds, aho, ev_info, fr2cui, dcs, algo, sdf, nd, all_s)
            el = time.time() - t0
            m = ""
            if r["gtpa1"] > best:
                best = r["gtpa1"]
                best_cfg = f"MC={mc} {algo}"
                m = " ★"
            print(
                f"    {algo:<12}: GTPA@1={r['gtpa1']:>5.1f}% "
                f"@3={r['gtpa3']:>5.1f}% @5={r['gtpa5']:>5.1f}% "
                f"@10={r['gtpa10']:>5.1f}% "
                f"(nm={r['no_match']:,}, {el:.0f}s){m}", flush=True)
            results.append({"mc": mc, "algo": algo, **r})

    with open(RESULTS_DIR / "kg_diagnose_v3c_results.json", "w") as f:
        json.dump({"best": best, "config": best_cfg, "results": results}, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print(f"최고 GTPA@1 = {best:.1f}% ({best_cfg})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
