#!/usr/bin/env python3
"""전체 실험: Phase A + B + C 통합 실행.

1. CUI 정규화 (MRREL hierarchy) 적용
2. Gold standard 개선 (antecedent 제외, CUI 정규화)
3. Phase A: NER 10개 변형 평가 (기존 LLM 결과 재사용)
4. Phase B: 프롬프트 10개 변형 × 상위 3 NER (LLM 실행)
5. Phase C: 통계 파라미터 sweep
6. 최종 상위 10개 선별 + 보고서
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from itertools import product

import requests
import scipy.stats as stats

DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
UMLS_DIR = Path("data/umls_extracted")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"

DISO_ALL = {"T047", "T184", "T033", "T034", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
ALLOWED_BASE = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}


# ============================================================
# 유틸리티
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
            if p[3] in ("PAR", "RB"):  # parent or broader
                child, parent = p[0], p[4]
                parents[child].add(parent)
    return dict(parents)

def build_ancestor_map(parent_map, max_depth=3):
    """각 CUI의 조상 CUI 셋 (max_depth까지)."""
    ancestor_cache = {}
    def get_ancestors(cui, depth=0):
        if cui in ancestor_cache:
            return ancestor_cache[cui]
        if depth >= max_depth or cui not in parent_map:
            ancestor_cache[cui] = set()
            return set()
        ancestors = set()
        for p in parent_map[cui]:
            ancestors.add(p)
            ancestors |= get_ancestors(p, depth + 1)
        ancestor_cache[cui] = ancestors
        return ancestors
    return get_ancestors

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
    n = a + b + c + d
    if n == 0: return 0.0
    def g(o, e): return o * math.log(o / e) if o > 0 and e > 0 else 0
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


# ============================================================
# CUI 정규화 매칭
# ============================================================

def cui_match_normalized(cui_a, cui_b, get_ancestors):
    """두 CUI가 같거나 조상 관계에 있으면 매칭."""
    if cui_a == cui_b:
        return True
    a_anc = get_ancestors(cui_a)
    b_anc = get_ancestors(cui_b)
    if cui_b in a_anc or cui_a in b_anc:
        return True
    # 공통 조상이 있으면 매칭 (같은 개념의 다른 CUI)
    if a_anc & b_anc:
        return True
    return False

def evaluate_normalized(our_pairs, gold_pairs, get_ancestors):
    """CUI 정규화를 적용한 recall/precision/F1."""
    matched_gold = set()
    matched_our = set()

    for op in our_pairs:
        for gp in gold_pairs:
            # our (a,b) vs gold (c,d): a~c and b~d, or a~d and b~c
            if ((cui_match_normalized(op[0], gp[0], get_ancestors) and
                 cui_match_normalized(op[1], gp[1], get_ancestors)) or
                (cui_match_normalized(op[0], gp[1], get_ancestors) and
                 cui_match_normalized(op[1], gp[0], get_ancestors))):
                matched_gold.add(gp)
                matched_our.add(op)

    precision = len(matched_our) / len(our_pairs) if our_pairs else 0
    recall = len(matched_gold) / len(gold_pairs) if gold_pairs else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "n_our": len(our_pairs), "n_gold": len(gold_pairs),
        "n_matched_gold": len(matched_gold), "n_matched_our": len(matched_our),
    }


# ============================================================
# Gold Standard 준비 (증상만, antecedent 제외)
# ============================================================

def prepare_gold_symptom_only():
    """DDXPlus에서 symptom만 추출 (antecedent 제외)."""
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
    for disease_name, info in conditions.items():
        d_cui = disease_map.get(disease_name, {}).get("umls_cui")
        if not d_cui: continue
        for eid in info.get("symptoms", {}):
            if ev_en.get(eid, {}).get("is_antecedent", False):
                continue  # antecedent 제외
            fr_name = eid_to_fr.get(eid)
            if fr_name and fr_name in umap:
                cui = umap[fr_name].get("cui")
                if cui:
                    gold_pairs.add(tuple(sorted([d_cui, cui])))
    return gold_pairs


# ============================================================
# 프롬프트 정의 (10개 변형)
# ============================================================

PROMPTS = {}

PROMPTS["S2-A"] = """You are a biomedical relation extractor. Analyze this PubMed abstract and classify concept pair relationships.

RULES:
1. "present" = text EXPLICITLY states a positive relationship
2. "absent" = text EXPLICITLY states a NEGATIVE relationship
3. "not_related" = both concepts appear but NO explicit relationship
4. Do NOT infer relationships not explicitly stated

Text: {text}
Pairs: {pairs}
Respond ONLY with JSON array: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-B"] = """Classify each concept pair relationship from the text.

Definitions:
- "present": A medical relationship exists between the two concepts (e.g., one is a symptom, cause, complication, or manifestation of the other)
- "absent": The text explicitly states that NO relationship exists between them
- "not_related": Both concepts appear but no relationship is stated

Text: {text}
Pairs: {pairs}
JSON array only: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-C"] = """You are a biomedical relation extractor. Classify concept pair relationships.

Examples:
Text: "Patients with pneumonia typically present with fever and cough."
- (Pneumonia, Fever): present
- (Pneumonia, Cough): present
- (Fever, Cough): not_related

Text: "Heart failure is not typically associated with skin rash."
- (Heart failure, Skin rash): absent

Now classify:
Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-D"] = """You are a biomedical relation extractor.

Examples:
Text: "Pneumonia causes fever, cough, and dyspnea. It is not associated with joint pain."
- (C0032285 Pneumonia, C0015967 Fever): present (symptom of)
- (C0032285 Pneumonia, C0010200 Cough): present (symptom of)
- (C0032285 Pneumonia, C0013404 Dyspnea): present (symptom of)
- (C0032285 Pneumonia, C0003862 Joint pain): absent (explicitly excluded)
- (C0015967 Fever, C0010200 Cough): not_related (co-occur but no direct relationship)

Text: "GERD presents with heartburn and regurgitation. Chest pain may mimic cardiac disease."
- (C0017168 GERD, C0018834 Heartburn): present
- (C0017168 GERD, C0008031 Chest pain): present
- (C0018834 Heartburn, C0008031 Chest pain): not_related

Now classify these pairs from the text below. Consider ONLY what the text explicitly states.

Text: {text}
Pairs: {pairs}
JSON only: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-E"] = """You are a biomedical relation extractor. Think step by step.

For each pair:
Step 1: Find where each concept is mentioned in the text.
Step 2: Determine if the text states a relationship between them.
Step 3: Classify as present (relationship stated), absent (no relationship stated explicitly), or not_related (no connection described).

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-F"] = """You are a biomedical expert. Given the text and concept pairs, determine relationships.

For each pair, follow these steps:
1. Identify the sentence(s) where both concepts appear or are discussed
2. Check if the text describes a medical relationship (symptom, cause, complication, risk factor)
3. Check for negation cues (not, no, without, absence, rules out)
4. Classify: present (positive relationship), absent (negative/excluded), not_related (no connection)

Example:
Text: "Patients with pneumonia present with fever and productive cough."
(Pneumonia, Fever) -> Both in same sentence, "present with" indicates symptom -> present
(Pneumonia, Cough) -> Same sentence, symptom relationship -> present

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-G"] = """Classify biomedical concept pair relationships from text.

Text: {text}

For each pair below, respond with ONLY "present" or "not_related":
- "present" = the text states these concepts are medically related
- "not_related" = no relationship stated

{pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""

PROMPTS["S2-H"] = """You are a precise biomedical relation extractor. Extract ONLY relationships that are EXPLICITLY and CLEARLY stated in the text. When in doubt, classify as not_related.

Do NOT extract:
- Implied relationships
- General knowledge not stated in text
- Relationships between concepts that merely co-occur

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-I"] = """You are a comprehensive biomedical relation extractor. Extract ALL relationships between concept pairs, including:
- Directly stated relationships
- Clearly implied relationships from context
- Relationships where one concept is discussed as a feature, characteristic, or aspect of another

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|absent|not_related"}}]"""

PROMPTS["S2-J"] = """Extract medical relationships from text. For each concept pair, classify as:
- "present": These concepts have a medical relationship (symptom-disease, cause-effect, complication, co-occurrence, risk factor, treatment indication, diagnostic finding)
- "not_related": No medical relationship described in the text

Text: {text}
Pairs: {pairs}
JSON: [{{"cui_a":"...","cui_b":"...","classification":"present|not_related"}}]"""


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 80)
    print("전체 실험: Phase A + B + C")
    print("=" * 80)

    # 로드
    print("\n[1/7] 데이터 로드...")
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()
    parent_map = load_parent_map()
    get_ancestors = build_ancestor_map(parent_map, max_depth=3)

    with open(DATA_DIR / "exp_documents.json") as f:
        all_docs = json.load(f)["documents"]

    # Gold standard (symptom only)
    gold_sym = prepare_gold_symptom_only()
    # Gold standard (전체)
    with open(DATA_DIR / "gold_standard.json") as f:
        gold_full = set(tuple(p) for p in json.load(f)["ddxplus"]["pairs"])

    print(f"  문서: {len(all_docs)}, Gold(증상만): {len(gold_sym)}, Gold(전체): {len(gold_full)}")
    print(f"  PAR/RB 관계: {sum(len(v) for v in parent_map.values()):,}")

    # 200편 서브셋
    subset = []
    dc = Counter()
    for doc in all_docs:
        d = doc["seed_disease"]
        if dc[d] < 4:
            subset.append(doc)
            dc[d] += 1
        if len(subset) >= 200: break
    print(f"  서브셋: {len(subset)}편")

    # ============================================================
    # [2/7] Phase B: 10개 프롬프트 변형 LLM 실행
    # ============================================================
    print(f"\n[2/7] Phase B: 10개 프롬프트 × {len(subset)}편 LLM 분류...")

    # 체크포인트
    ckpt_file = DATA_DIR / "exp_full_checkpoint.json"
    all_prompt_results = {}  # prompt_id -> [classifications]
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            all_prompt_results = json.load(f)
        print(f"  체크포인트: {list(all_prompt_results.keys())} 완료")

    prompt_ids = list(PROMPTS.keys())

    for pid in prompt_ids:
        if pid in all_prompt_results:
            print(f"  {pid}: 이미 완료 ({len(all_prompt_results[pid])}건)")
            continue

        prompt_template = PROMPTS[pid]
        classifications = []
        start = time.time()

        for idx, doc in enumerate(subset):
            cuis = [c for c in doc["cuis"] if (cui_stys.get(c, set()) & ALLOWED_BASE) and c not in BLACKLIST]
            if len(cuis) < 2: continue

            pairs = []
            for i in range(min(len(cuis), 15)):
                for j in range(i+1, min(len(cuis), 15)):
                    pairs.append({"cui_a": min(cuis[i],cuis[j]), "cui_b": max(cuis[i],cuis[j])})
            pairs = pairs[:15]
            if not pairs: continue

            pairs_text = "\n".join(
                f"- ({cui_names.get(p['cui_a'],p['cui_a'])[:40]}, "
                f"{cui_names.get(p['cui_b'],p['cui_b'])[:40]}) "
                f"[CUI: {p['cui_a']}, {p['cui_b']}]"
                for p in pairs
            )
            prompt = prompt_template.format(text=doc["text"][:2500], pairs=pairs_text)

            try:
                response = call_ollama(prompt)
                parsed = parse_json(response)
                for item in parsed:
                    cls = item.get("classification","").lower().strip().replace(" ","_")
                    if cls in ("present","absent","not_related"):
                        classifications.append({
                            "pmid": doc["pmid"], "cui_a": item.get("cui_a",""),
                            "cui_b": item.get("cui_b",""), "classification": cls,
                        })
            except: pass

            if (idx+1) % 20 == 0:
                elapsed = time.time()-start
                rate = (idx+1)/elapsed
                eta = (len(subset)-idx-1)/rate
                print(f"    {pid} [{idx+1:3d}/{len(subset)}] cls={len(classifications):,} {rate:.2f}/s ETA={eta/60:.0f}분")

        all_prompt_results[pid] = classifications
        elapsed = time.time()-start
        dist = Counter(c["classification"] for c in classifications)
        print(f"  {pid} 완료: {len(classifications)}건 ({elapsed/60:.1f}분) present={dist.get('present',0)} nr={dist.get('not_related',0)}")

        # 체크포인트 저장
        with open(ckpt_file, "w") as f:
            json.dump(all_prompt_results, f)

    # ============================================================
    # [3/7] NER 필터 변형 정의
    # ============================================================
    print(f"\n[3/7] NER 필터 변형 정의...")

    # 간소화: 핵심 3개만 (Phase A에서 차이 없었으므로)
    ner_filters = {
        "NER-base": {"stys": ALLOWED_BASE, "blacklist": BLACKLIST, "propagate": 0},
        "NER-prop1": {"stys": ALLOWED_BASE, "blacklist": BLACKLIST, "propagate": 1},
        "NER-prop2": {"stys": ALLOWED_BASE, "blacklist": BLACKLIST, "propagate": 2},
        "NER-full": {"stys": DISO_ALL, "blacklist": set(), "propagate": 0},
    }

    # ============================================================
    # [4/7] Phase C: 통계 파라미터 sweep
    # ============================================================
    print(f"\n[4/7] Phase C: 통계 sweep + 평가...")

    alphas = [0.3, 0.5, 0.6, 0.7]
    fdrs = [0.01, 0.05, 0.10, 1.0]
    min_coocs = [1, 2, 3]

    total_docs = len(subset)
    results = []

    for pid in prompt_ids:
        cls_list = all_prompt_results.get(pid, [])
        if not cls_list: continue

        for ner_id, ner_cfg in ner_filters.items():
            allowed_stys = ner_cfg["stys"]
            bl = ner_cfg["blacklist"]
            prop_depth = ner_cfg["propagate"]

            # CUI 필터 적용
            filtered_cls = []
            for c in cls_list:
                a_ok = (cui_stys.get(c["cui_a"], set()) & allowed_stys) and c["cui_a"] not in bl
                b_ok = (cui_stys.get(c["cui_b"], set()) & allowed_stys) and c["cui_b"] not in bl
                if a_ok and b_ok:
                    filtered_cls.append(c)

            # present 쌍 집계
            pair_counts = Counter()
            pair_pmids = defaultdict(set)
            cui_doc_freq = Counter()

            for c in filtered_cls:
                pair = tuple(sorted([c["cui_a"], c["cui_b"]]))
                if c["classification"] == "present":
                    pair_counts[pair] += 1
                pair_pmids[pair].add(c["pmid"])
                cui_doc_freq[c["cui_a"]] += 1
                cui_doc_freq[c["cui_b"]] += 1

            # CUI 계층 전파
            if prop_depth > 0:
                expanded = Counter()
                for (a, b), cnt in pair_counts.items():
                    expanded[(a, b)] += cnt
                    a_anc = get_ancestors(a) if prop_depth >= 1 else set()
                    b_anc = get_ancestors(b) if prop_depth >= 1 else set()
                    for pa in a_anc:
                        if cui_stys.get(pa, set()) & allowed_stys:
                            expanded[tuple(sorted([pa, b]))] += cnt
                    for pb in b_anc:
                        if cui_stys.get(pb, set()) & allowed_stys:
                            expanded[tuple(sorted([a, pb]))] += cnt
                pair_counts = expanded

            # 통계 sweep
            for alpha, fdr_thresh, min_co in product(alphas, fdrs, min_coocs):
                # 필터: 최소 공출현
                candidates = {p: c for p, c in pair_counts.items() if c >= min_co}
                if not candidates: continue

                # Jensen score + G²
                edges = []
                for (ca, cb), np_ in candidates.items():
                    c_a = max(cui_doc_freq.get(ca, 1), 1)
                    c_b = max(cui_doc_freq.get(cb, 1), 1)
                    oe = (np_ * total_docs) / (c_a * c_b) if c_a*c_b > 0 else 0
                    jensen = (np_**alpha) * (oe**(1-alpha)) if np_>0 and oe>0 else 0

                    a = np_
                    b = max(c_a - np_, 0)
                    c = max(c_b - np_, 0)
                    d = max(total_docs - a - b - c, 0)
                    g2 = dunning_g2(a, b, c, d)
                    pv = 1 - stats.chi2.cdf(g2, df=1) if g2 > 0 else 1.0
                    edges.append({"cui_a": ca, "cui_b": cb, "p_value": pv, "n": np_})

                # FDR
                if fdr_thresh < 1.0:
                    pvals = [e["p_value"] for e in edges]
                    qvals = bh_fdr(pvals)
                    sig_edges = [e for e, q in zip(edges, qvals) if q < fdr_thresh]
                else:
                    sig_edges = edges

                our_pairs = set(tuple(sorted([e["cui_a"], e["cui_b"]])) for e in sig_edges)

                if not our_pairs: continue

                # DDXPlus 평가 (CUI 정규화 적용)
                eval_sym = evaluate_normalized(our_pairs, gold_sym, get_ancestors)
                eval_full = evaluate_normalized(our_pairs, gold_full, get_ancestors)

                config_id = f"{pid}|{ner_id}|a={alpha}|fdr={fdr_thresh}|mc={min_co}"
                results.append({
                    "config_id": config_id,
                    "prompt": pid, "ner": ner_id,
                    "alpha": alpha, "fdr": fdr_thresh, "min_cooc": min_co,
                    "n_edges": len(sig_edges),
                    "sym_precision": eval_sym["precision"],
                    "sym_recall": eval_sym["recall"],
                    "sym_f1": eval_sym["f1"],
                    "sym_matched": eval_sym["n_matched_gold"],
                    "full_precision": eval_full["precision"],
                    "full_recall": eval_full["recall"],
                    "full_f1": eval_full["f1"],
                    "full_matched": eval_full["n_matched_gold"],
                })

    print(f"  총 조합: {len(results):,}")

    # ============================================================
    # [5/7] 결과 정렬 + 상위 10 선별
    # ============================================================
    print(f"\n[5/7] 결과 정렬...")
    results.sort(key=lambda x: -x["sym_f1"])

    print(f"\n{'='*100}")
    print(f"상위 20 (DDXPlus symptom F1 기준)")
    print(f"{'='*100}")
    print(f"{'Rank':>4} {'Prompt':>6} {'NER':>10} {'α':>4} {'FDR':>5} {'MC':>3} {'Edges':>6} {'P':>6} {'R':>6} {'F1':>6} {'Match':>6}")
    print(f"{'-'*100}")
    for i, r in enumerate(results[:20]):
        print(f"{i+1:4d} {r['prompt']:>6} {r['ner']:>10} {r['alpha']:>4} {r['fdr']:>5} {r['min_cooc']:>3} "
              f"{r['n_edges']:>6} {r['sym_precision']:>6.3f} {r['sym_recall']:>6.3f} {r['sym_f1']:>6.3f} "
              f"{r['sym_matched']:>6}")

    # ============================================================
    # [6/7] 프롬프트별 최고 성능
    # ============================================================
    print(f"\n{'='*100}")
    print(f"프롬프트별 최고 F1")
    print(f"{'='*100}")
    prompt_best = {}
    for r in results:
        pid = r["prompt"]
        if pid not in prompt_best or r["sym_f1"] > prompt_best[pid]["sym_f1"]:
            prompt_best[pid] = r
    for pid in prompt_ids:
        if pid in prompt_best:
            r = prompt_best[pid]
            print(f"  {pid}: F1={r['sym_f1']:.3f} (P={r['sym_precision']:.3f} R={r['sym_recall']:.3f}) "
                  f"edges={r['n_edges']} ner={r['ner']} α={r['alpha']} fdr={r['fdr']} mc={r['min_cooc']}")

    # ============================================================
    # [7/7] 저장
    # ============================================================
    output = {
        "n_combinations": len(results),
        "top10": results[:10],
        "top20": results[:20],
        "prompt_best": prompt_best,
        "all_results": results[:100],  # 상위 100개만 저장
    }
    with open(RESULTS_DIR / "experiment_full_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {RESULTS_DIR / 'experiment_full_results.json'}")
    print("완료!")


if __name__ == "__main__":
    main()
