#!/usr/bin/env python3
"""진단 v7: Bayesian 1차 후보 → LLM 2차 re-ranking.

핵심 인사이트:
  - Bayesian @10=80.4% (상위 10에 정답 80%)
  - Bayesian @1=29.6% (1위 정확도 30%)
  - 갭: 상위 10 내에서 1위 변별력 부족

전략:
  1차: Bayesian으로 상위 10개 후보 선정 (KG 기반)
  2차: LLM에 환자 증상 + 10개 후보 제시 → 최종 1위 선택

LLM이 환자 증상 패턴과 질환 후보를 직접 비교하여 re-ranking.
LLM 호출: 134K × 1회 = 134K 프롬프트 (vLLM batch로 처리)
"""
from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

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

RERANK_PROMPT = """A patient presents with the following symptoms and findings:
{symptoms}

Which of these diseases is MOST LIKELY?
{candidates}

Answer with ONLY the number (1-{n}) of the most likely disease."""


def load_umls_names():
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
    return dict(can), cp


def load_ddxplus():
    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f:
        icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f:
        cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f:
        ev_fr = json.load(f)

    diseases = {}
    fr2cui = {}
    cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]
        diseases[dn] = {"cui": dc, "umls_name": icd_map[dn]["umls_name"],
                        "fr": info.get("cond-name-fr", "")}
        fr2cui[info.get("cond-name-fr", "")] = dc
        cui2name[dc] = dn

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
    return diseases, fr2cui, ev_info, cui2name


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
    for cui in scuis:
        for name in can.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) < 4 or lo in STOPWORDS: continue
            try: aho.add_word(lo, (lo, cui))
            except: pass
    aho.make_automaton()
    return aho


def text_match_patient(evidences, ev_info, aho):
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
        if len(bc) >= 3 and bc not in STOPWORDS: terms.append(bc)

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


def d_bayesian(ps, ds, dcs, all_s):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = -1e6; continue
        tw = sum(s.values()) + len(all_s) * 0.1
        sc[dc] = sum(math.log((s[x] + 0.1) / tw + 1e-10) if x in s
                     else math.log(0.1 / tw + 1e-10) for x in ps)
    return sorted(sc.items(), key=lambda x: -x[1])


def patient_symptoms_text(evidences, ev_info):
    """환자 증상을 영문 텍스트로 요약 (LLM re-ranking용)."""
    symptoms = []
    for ev in evidences:
        parts = ev.split("_@_")
        base = parts[0]
        value = parts[1] if len(parts) > 1 else None
        info = ev_info.get(base, {})
        if info.get("is_antecedent"): continue

        q = info.get("question_en", "")
        if not q: continue

        # 질문을 간결한 증상으로 변환
        q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your |Does the ", "", q)
        q_clean = re.sub(r"\?$", "", q_clean).strip()

        if value:
            val_en = info.get("value_en", {}).get(value, "")
            if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                symptoms.append(f"{q_clean}: {val_en}")
            else:
                symptoms.append(q_clean)
        else:
            symptoms.append(q_clean)

    return "\n".join(f"- {s}" for s in symptoms[:20])  # 최대 20개


def main():
    print("=" * 80, flush=True)
    print("진단 v7b: Bayesian top-20 → LLM re-ranking", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] 로드...", flush=True)
    can, cp = load_umls_names()
    diseases, fr2cui, ev_info, cui2name = load_ddxplus()
    dcs = {v["cui"] for v in diseases.values()}
    pc = load_kg()
    print(f"  KG: {len(pc):,} 쌍", flush=True)

    # Build KG (MC=1, best setting)
    ds, scuis = build_ds(pc, dcs, mc=1)
    aho = build_aho(scuis, can)

    sdf = Counter()
    for syms in ds.values():
        for s in syms: sdf[s] += 1
    nd = max(len(ds), 1)
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    print("\n[2] 테스트 데이터...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "pathology": row["PATHOLOGY"],
            })
    print(f"  {len(patients):,}명", flush=True)

    # [3] 1차 Bayesian: 모든 환자 → 상위 10 후보
    print("\n[3] 1차 Bayesian...", flush=True)
    t0 = time.time()
    patient_candidates = []
    top1_baseline = 0
    n = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        ps = text_match_patient(p["evidences"], ev_info, aho)
        ranked = d_bayesian(ps, ds, dcs, all_s)
        top10 = [dc for dc, _ in ranked[:20]]
        patient_candidates.append({
            "patient": p,
            "true_dc": tdc,
            "top10": top10,
            "top10_names": [cui2name.get(dc, cp.get(dc, dc)) for dc in top10],  # top20 actually
        })
        if top10 and top10[0] == tdc:
            top1_baseline += 1

    print(f"  Baseline @1={100*top1_baseline/n:.1f}% ({time.time()-t0:.0f}s)", flush=True)

    # 정답이 top10에 있는 환자만 re-ranking 대상
    in_top10 = sum(1 for pc in patient_candidates if pc["true_dc"] in pc["top10"])
    print(f"  정답 in top10: {in_top10:,}/{n:,} ({100*in_top10/n:.1f}%)", flush=True)

    # [4] 2차 LLM re-ranking
    print("\n[4] LLM re-ranking 프롬프트 생성...", flush=True)
    prompts = []
    for pc in patient_candidates:
        sym_text = patient_symptoms_text(pc["patient"]["evidences"], ev_info)
        cands = "\n".join(f"{i+1}. {name}" for i, name in enumerate(pc["top10_names"]))
        prompt = RERANK_PROMPT.format(
            symptoms=sym_text,
            candidates=cands,
            n=len(pc["top10"]),
        )
        prompts.append(prompt)

    print(f"  프롬프트: {len(prompts):,}개", flush=True)

    # vLLM batch
    print("\n[5] vLLM batch re-ranking...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(
        model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.95, enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    sampling = SamplingParams(temperature=0, max_tokens=16)
    convs = [[{"role": "user", "content": p}] for p in prompts]

    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)", flush=True)

    # [6] 결과 파싱 + 평가
    print("\n[6] 평가...", flush=True)
    t1_rerank = t3_rerank = t5_rerank = 0
    parse_fail = 0

    for pc_item, out in zip(patient_candidates, outputs):
        answer = out.outputs[0].text.strip()
        # 숫자 추출
        m = re.search(r"(\d+)", answer)
        if m:
            idx = int(m.group(1)) - 1  # 0-based
            if 0 <= idx < len(pc_item["top10"]):
                reranked = list(pc_item["top10"])
                chosen = reranked.pop(idx)
                reranked.insert(0, chosen)
            else:
                reranked = pc_item["top10"]
                parse_fail += 1
        else:
            reranked = pc_item["top10"]
            parse_fail += 1

        tdc = pc_item["true_dc"]
        if reranked and reranked[0] == tdc: t1_rerank += 1
        if tdc in reranked[:3]: t3_rerank += 1
        if tdc in reranked[:5]: t5_rerank += 1

    n_total = len(patient_candidates)
    print(f"\n  Baseline (Bayesian only):", flush=True)
    print(f"    @1={100*top1_baseline/n_total:.1f}%", flush=True)
    print(f"\n  Re-ranked (Bayesian + LLM):", flush=True)
    print(f"    @1={100*t1_rerank/n_total:.1f}% @3={100*t3_rerank/n_total:.1f}% "
          f"@5={100*t5_rerank/n_total:.1f}%", flush=True)
    print(f"    parse_fail={parse_fail:,}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"Baseline @1={100*top1_baseline/n_total:.1f}% → Re-ranked @1={100*t1_rerank/n_total:.1f}%", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
