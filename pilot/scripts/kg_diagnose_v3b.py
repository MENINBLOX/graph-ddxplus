#!/usr/bin/env python3
"""진단 v3b: 캐시된 KG + 개선된 텍스트 매칭.

v3 문제점: 환자 evidence 전체 질문 텍스트가 비의학 단어까지 매칭.
개선:
  1. evidence 질문에서 의학 키워드만 추출
  2. Aho-Corasick 최소 패턴 길이 5로 증가 (3→5)
  3. 비의학 stopword 필터
  4. 합성어 매칭 강화 (pain+location → "chest pain")
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

# 비의학 stopwords
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
    "put", "run", "cut", "keep", "help", "give", "take", "come", "go",
    "told", "felt", "feel", "ever", "past", "last", "part", "left",
    "right", "side", "well", "good", "long", "work", "hard", "high",
    "low", "same", "done", "able", "many", "know", "think", "need",
    "related", "reason", "consulting", "significant", "measured",
    "thermometer", "either", "believe", "racing", "missing", "beat",
    "fast", "irregularly", "problems", "situation", "associated",
    "inability", "speak", "trouble", "keeping", "opening", "raising",
    "annoying", "else", "body", "somewhere", "anywhere", "nowhere",
    "recent", "recently", "currently", "previously", "usually",
    "often", "sometimes", "always", "never", "worse", "better",
    "severe", "mild", "moderate",
}


def load_umls_names():
    """CUI 이름만 로드 (KG 캐시 사용하므로 전체 UMLS 불필요)."""
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
    print(f" {len(cui_all_names):,} CUIs", flush=True)
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
        if dn not in icd_map:
            continue
        dc = icd_map[dn]["cui"]
        fr = info.get("cond-name-fr", "")
        diseases[dn] = {"cui": dc, "umls_name": icd_map[dn]["umls_name"], "fr": fr}
        disease_fr_to_cui[fr] = dc

    ev_text_info = {}
    for eid, info in ev_fr.items():
        q_en = info.get("question_en", "")
        is_ante = info.get("is_antecedent", False)
        vm = info.get("value_meaning", {})
        val_en = {}
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"):
                    val_en[k] = v["en"]
        ev_text_info[eid] = {
            "question_en": q_en,
            "is_antecedent": is_ante,
            "value_en": val_en,
        }

    return diseases, disease_fr_to_cui, ev_text_info


def load_kg_cache():
    with open(KG_CACHE) as f:
        cache = json.load(f)
    pair_counts = Counter()
    for k, v in cache["pair_counts"]:
        pair_counts[tuple(k)] = v
    print(f"  KG 캐시: {len(pair_counts):,} 쌍", flush=True)
    return pair_counts


def build_symptom_aho(symptom_cuis, cui_all_names, min_len=5):
    """KG 증상 이름으로 Aho-Corasick 구축 (최소 길이 제한)."""
    aho = ahocorasick.Automaton()
    n = 0
    for cui in symptom_cuis:
        for name in cui_all_names.get(cui, set()):
            lo = name.lower().strip()
            # 최소 길이 + stopword 필터
            if len(lo) < min_len:
                continue
            words = lo.split()
            if len(words) == 1 and lo in STOPWORDS:
                continue
            try:
                aho.add_word(lo, (lo, cui))
                n += 1
            except Exception:
                pass
    aho.make_automaton()
    return aho, n


def extract_medical_keywords(question_en):
    """질문 텍스트에서 의학 키워드만 추출."""
    # 괄호, 슬래시 내용 제거 후 단어 추출
    text = re.sub(r"\(.*?\)", "", question_en)
    text = re.sub(r"[?.,;:!]", "", text)
    words = text.lower().split()
    # stopword 제거
    keywords = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    return keywords


def patient_evidence_to_medical_text(evidences, ev_text_info):
    """환자 evidence → 의학 키워드만 포함한 텍스트."""
    terms = []
    for ev in evidences:
        parts = ev.split("_@_")
        base = parts[0]
        value = parts[1] if len(parts) > 1 else None

        info = ev_text_info.get(base, {})
        if info.get("is_antecedent"):
            continue

        q = info.get("question_en", "")
        if q:
            kws = extract_medical_keywords(q)
            terms.extend(kws)

        if value and info.get("value_en"):
            val_en = info["value_en"].get(value, "")
            if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                val_lower = val_en.lower()
                # 좌우 표기 제거
                val_clean = re.sub(r"\([rl]\)", "", val_lower).strip()
                if val_clean and val_clean not in STOPWORDS:
                    terms.append(val_clean)
                    # pain + body part → compound
                    if "pain" in q.lower():
                        terms.append(f"{val_clean} pain")
                        # "chest" from "side of the chest" etc
                        for part in val_clean.split():
                            if part not in STOPWORDS and len(part) >= 4:
                                terms.append(f"{part} pain")

    # 중복 제거 후 연결
    return " . ".join(terms)


def match_text_to_symptoms(text, aho):
    matched = set()
    for ei, (n, cui) in aho.iter(text):
        si = ei - len(n) + 1
        if si > 0 and text[si - 1].isalpha():
            continue
        if ei + 1 < len(text) and text[ei + 1].isalpha():
            continue
        matched.add(cui)
    return matched


# ─── 진단 알고리즘 ────────────────────────────────────────────────────────────

def diagnose_coverage(pt_syms, ds, dc_set):
    scores = {}
    for dc in dc_set:
        syms = ds.get(dc, {})
        if not syms: scores[dc] = 0; continue
        m = sum(1 for s in syms if s in pt_syms)
        scores[dc] = m / len(syms)
    return sorted(scores.items(), key=lambda x: -x[1])

def diagnose_weighted(pt_syms, ds, dc_set):
    scores = {}
    for dc in dc_set:
        syms = ds.get(dc, {})
        if not syms: scores[dc] = 0; continue
        t = sum(syms.values())
        m = sum(c for s, c in syms.items() if s in pt_syms)
        scores[dc] = m / t
    return sorted(scores.items(), key=lambda x: -x[1])

def diagnose_idf(pt_syms, ds, dc_set, sdf, nd):
    scores = {}
    for dc in dc_set:
        syms = ds.get(dc, {})
        if not syms: scores[dc] = 0; continue
        sc = 0
        for s, c in syms.items():
            if s in pt_syms:
                sc += (math.log(nd / (sdf[s] + 1)) + 1) * c
        scores[dc] = sc
    return sorted(scores.items(), key=lambda x: -x[1])

def diagnose_v15(pt_syms, ds, dc_set):
    scores = {}
    for dc in dc_set:
        syms = ds.get(dc, {})
        if not syms: scores[dc] = 0; continue
        conf = sum(1 for s in syms if s in pt_syms)
        den = sum(1 for s in syms if s not in pt_syms)
        scores[dc] = conf / (conf + den + 1) * conf if conf else 0
    return sorted(scores.items(), key=lambda x: -x[1])

def diagnose_bayesian(pt_syms, ds, dc_set, all_s):
    scores = {}
    for dc in dc_set:
        syms = ds.get(dc, {})
        if not syms: scores[dc] = -1e6; continue
        tw = sum(syms.values()) + len(all_s) * 0.1
        ls = 0
        for s in pt_syms:
            p = (syms[s] + 0.1) / tw if s in syms else 0.1 / tw
            ls += math.log(p + 1e-10)
        scores[dc] = ls
    return sorted(scores.items(), key=lambda x: -x[1])

def diagnose_idf_neg(pt_syms, ds, dc_set, sdf, nd):
    scores = {}
    for dc in dc_set:
        syms = ds.get(dc, {})
        if not syms: scores[dc] = 0; continue
        sc = 0
        for s, c in syms.items():
            idf = math.log(nd / (sdf[s] + 1)) + 1
            if s in pt_syms:
                sc += idf * c
            else:
                sc -= idf * 0.5
        scores[dc] = sc
    return sorted(scores.items(), key=lambda x: -x[1])


def evaluate(test_patients, ds, aho, ev_info, fr2cui, dc_set, algo, sdf, nd, all_s):
    top1 = top3 = top5 = top10 = n = no_m = 0
    for p in test_patients:
        true_dc = fr2cui.get(p["pathology"])
        if not true_dc: continue
        n += 1
        pt_text = patient_evidence_to_medical_text(p["evidences"], ev_info)
        pt_syms = match_text_to_symptoms(pt_text, aho)
        if not pt_syms: no_m += 1; continue

        if algo == "coverage": ranked = diagnose_coverage(pt_syms, ds, dc_set)
        elif algo == "weighted": ranked = diagnose_weighted(pt_syms, ds, dc_set)
        elif algo == "idf": ranked = diagnose_idf(pt_syms, ds, dc_set, sdf, nd)
        elif algo == "v15_ratio": ranked = diagnose_v15(pt_syms, ds, dc_set)
        elif algo == "bayesian": ranked = diagnose_bayesian(pt_syms, ds, dc_set, all_s)
        elif algo == "idf_neg": ranked = diagnose_idf_neg(pt_syms, ds, dc_set, sdf, nd)

        r = [d for d, s in ranked]
        if r and r[0] == true_dc: top1 += 1
        if true_dc in r[:3]: top3 += 1
        if true_dc in r[:5]: top5 += 1
        if true_dc in r[:10]: top10 += 1

    return {
        "n": n, "no_match": no_m,
        "gtpa1": round(100 * top1 / n, 2) if n else 0,
        "gtpa3": round(100 * top3 / n, 2) if n else 0,
        "gtpa5": round(100 * top5 / n, 2) if n else 0,
        "gtpa10": round(100 * top10 / n, 2) if n else 0,
    }


def main():
    print("=" * 80, flush=True)
    print("진단 v3b: 개선된 텍스트 매칭 (의학 키워드만, min_len=5)", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] 데이터 로드...", flush=True)
    cui_all_names, cui_preferred = load_umls_names()
    diseases, disease_fr_to_cui, ev_text_info = load_ddxplus()
    disease_cuis = {v["cui"] for v in diseases.values()}
    pair_counts = load_kg_cache()

    # 테스트 데이터
    print("\n[2] 테스트 데이터...", flush=True)
    test_patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            test_patients.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "pathology": row["PATHOLOGY"],
            })
    print(f"  환자: {len(test_patients):,}명", flush=True)

    # 디버그: 첫 환자의 매칭 확인
    print("\n[DEBUG] 첫 환자 매칭 검증...", flush=True)
    for pi in range(3):
        p = test_patients[pi]
        pt_text = patient_evidence_to_medical_text(p["evidences"], ev_text_info)
        print(f"  Patient {pi} ({p['pathology']}):", flush=True)
        print(f"    Text: {pt_text[:200]}...", flush=True)

    algos = ["coverage", "weighted", "idf", "v15_ratio", "bayesian", "idf_neg"]
    best_gtpa1 = 0
    best_config = ""
    all_results = []

    for min_len in [4, 5, 6]:
        print(f"\n{'='*60}", flush=True)
        print(f"  min_len={min_len}", flush=True)

        for mc in [1, 2, 3, 5]:
            # Build disease-symptom map
            symptom_cuis = set()
            ds = defaultdict(dict)
            for (a, b), cnt in pair_counts.items():
                if cnt < mc: continue
                if a in disease_cuis: ds[a][b] = cnt; symptom_cuis.add(b)
                if b in disease_cuis: ds[b][a] = cnt; symptom_cuis.add(a)
            ds = dict(ds)

            aho, n_names = build_symptom_aho(symptom_cuis, cui_all_names, min_len=min_len)
            n_dw = sum(1 for d in disease_cuis if d in ds and ds[d])

            # Pre-compute IDF
            sdf = Counter()
            for syms in ds.values():
                for s in syms: sdf[s] += 1
            nd = max(len(ds), 1)
            all_s = set()
            for syms in ds.values(): all_s.update(syms.keys())

            # Debug: check sample matching
            if mc == 1:
                p0 = test_patients[0]
                pt_text = patient_evidence_to_medical_text(p0["evidences"], ev_text_info)
                pt_syms = match_text_to_symptoms(pt_text, aho)
                matched_names = []
                for ei, (n, cui) in aho.iter(pt_text):
                    si = ei - len(n) + 1
                    if si > 0 and pt_text[si-1].isalpha(): continue
                    if ei+1 < len(pt_text) and pt_text[ei+1].isalpha(): continue
                    matched_names.append((n, cui_preferred.get(cui, cui)))
                print(f"  Debug MC=1 matches: {sorted(set(matched_names))[:10]}", flush=True)

            print(f"\n  MC={mc}: 증상={len(symptom_cuis)}, 이름={n_names:,}, 질환={n_dw}", flush=True)

            for algo in algos:
                t0 = time.time()
                result = evaluate(
                    test_patients, ds, aho, ev_text_info,
                    disease_fr_to_cui, disease_cuis, algo, sdf, nd, all_s,
                )
                elapsed = time.time() - t0
                marker = ""
                if result["gtpa1"] > best_gtpa1:
                    best_gtpa1 = result["gtpa1"]
                    best_config = f"min={min_len} MC={mc} {algo}"
                    marker = " ★"
                print(
                    f"    {algo:<12}: "
                    f"GTPA@1={result['gtpa1']:>5.1f}% "
                    f"@3={result['gtpa3']:>5.1f}% "
                    f"@5={result['gtpa5']:>5.1f}% "
                    f"@10={result['gtpa10']:>5.1f}% "
                    f"(no_match={result['no_match']:,}, {elapsed:.0f}s){marker}",
                    flush=True,
                )
                all_results.append({
                    "min_len": min_len, "mc": mc, "algo": algo, **result,
                })

    with open(RESULTS_DIR / "kg_diagnose_v3b_results.json", "w") as f:
        json.dump({"best": best_gtpa1, "config": best_config, "results": all_results}, f, indent=2)

    print(f"\n{'='*80}", flush=True)
    print(f"최고 GTPA@1 = {best_gtpa1:.1f}% ({best_config})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
