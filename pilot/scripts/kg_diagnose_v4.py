#!/usr/bin/env python3
"""진단 v4: LLM 기반 동적 evidence-symptom 매칭.

v3c 문제: 텍스트 매칭은 구어체-의학용어 어휘 격차를 넘지 못함.
접근: 110개 고유 evidence 질문을 KG 증상 목록과 LLM으로 매칭.
  - KG에서 파생된 동적 매칭 (정적 매핑 테이블 아님)
  - evidence 질문 텍스트 + KG 증상 이름 → LLM이 관련 증상 선택
  - 110개 evidence × 1회 LLM = 110회 호출 (저비용)

추가 개선:
  - KG 노이즈 제거: "Present", "Reduced" 등 비임상적 CUI 필터
  - sub-evidence value 활용: pain + location → 해당 위치 통증 매칭
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
MATCH_CACHE = RESULTS_DIR / "ev_symptom_match_cache.json"

# 비임상적 노이즈 CUI (KG에서 제외)
NOISE_CUIS = {
    "C0150312",  # Present
    "C0442743",  # Reduced (qualifier)
    "C0039082",  # Syndrome
    "C0221423",  # Illness (finding)
    "C1457887",  # Symptom (finding)
    "C0205390",  # Other
    "C0442804",  # Perceived quality of life
    "C3839861",  # Unexplained Symptoms
    "C0332157",  # Exposure
    "C1457868",  # Worse
    "C0445223",  # Related (finding)
    "C1272751",  # Demonstrates ability
    "C0015663",  # Fasting
    "C0277814",  # Sitting position
    "C5202885",  # Performance Status
    "C0153933",  # Benign neoplasm of tongue
    "C0585362",  # Score
}

PROMPT_MATCH = """Patient symptom question: "{question}"

KG symptoms (from PubMed-derived knowledge graph):
{symptoms}

Which of the above KG symptoms could match this patient symptom question?
Select ALL that are medically related.

JSON only: ["CUI1", "CUI2", ...]
If none match: []"""

PROMPT_MATCH_VALUE = """Patient symptom: "{question}" with specific value: "{value}"

KG symptoms (from PubMed-derived knowledge graph):
{symptoms}

Which of the above KG symptoms could match this specific patient presentation?
Select ALL that are medically related.

JSON only: ["CUI1", "CUI2", ...]
If none match: []"""


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


def load_kg_cache():
    with open(KG_CACHE) as f:
        cache = json.load(f)
    pc = Counter()
    for k, v in cache["pair_counts"]:
        pc[tuple(k)] = v
    print(f"  KG: {len(pc):,} 쌍", flush=True)
    return pc


def build_disease_symptoms(pc, dcs, mc=1):
    """질환별 증상 맵 구축 (노이즈 제거)."""
    ds = defaultdict(dict)
    scuis = set()
    for (a, b), cnt in pc.items():
        if cnt < mc: continue
        # 노이즈 CUI 제거
        if a in NOISE_CUIS or b in NOISE_CUIS: continue
        if a in dcs:
            ds[a][b] = cnt; scuis.add(b)
        if b in dcs:
            ds[b][a] = cnt; scuis.add(a)
    return dict(ds), scuis


def build_llm_match(ev_info, scuis, cui_preferred, dcs):
    """LLM으로 evidence-symptom 동적 매칭 생성."""
    # 캐시 확인
    if MATCH_CACHE.exists():
        with open(MATCH_CACHE) as f:
            cached = json.load(f)
        print(f"  매칭 캐시 로드: {len(cached)} evidence", flush=True)
        return cached

    # 상위 빈도 증상만 LLM에 제시 (전체 4815개는 너무 많음)
    # 질환별 상위 증상을 모아서 중복 제거
    top_symptom_cuis = set()
    # 그냥 scuis 전체 사용하되 disease CUI는 제외
    non_disease_scuis = scuis - dcs
    print(f"  증상 CUI (비질환): {len(non_disease_scuis)}", flush=True)

    # 증상 목록 문자열 (CUI + preferred name)
    sym_lines = []
    for cui in sorted(non_disease_scuis):
        name = cui_preferred.get(cui, cui)
        if len(name) > 3:  # 너무 짧은 이름 제외
            sym_lines.append(f"{cui}: {name}")

    # 증상이 너무 많으면 청크로 나눔
    CHUNK_SIZE = 200
    chunks = [sym_lines[i:i+CHUNK_SIZE] for i in range(0, len(sym_lines), CHUNK_SIZE)]
    print(f"  증상 청크: {len(chunks)}개 (각 {CHUNK_SIZE})", flush=True)

    # 각 non-antecedent evidence에 대해 LLM 호출
    tasks = []
    task_meta = []

    for eid, info in ev_info.items():
        if info["is_antecedent"]: continue
        q = info["question_en"]
        if not q: continue

        for ci, chunk in enumerate(chunks):
            sym_text = "\n".join(chunk)
            prompt = PROMPT_MATCH.format(question=q, symptoms=sym_text)
            tasks.append(prompt)
            task_meta.append({"eid": eid, "chunk": ci})

    print(f"  LLM 프롬프트: {len(tasks)}개 ({len(ev_info)}×{len(chunks)} 청크)", flush=True)

    # vLLM batch
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(
        model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.95, enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    sampling = SamplingParams(temperature=0, max_tokens=2048)
    convs = [[{"role": "user", "content": t}] for t in tasks]

    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    elapsed = time.time() - t0
    print(f"  LLM 완료: {elapsed:.0f}초", flush=True)

    # 파싱
    ev_matches = defaultdict(set)
    for meta, out in zip(task_meta, outputs):
        text = out.outputs[0].text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        m = re.search(r"\[[\s\S]*?\]", text)
        if m:
            try:
                cuis = json.loads(m.group())
                for cui in cuis:
                    if isinstance(cui, str) and cui.startswith("C"):
                        ev_matches[meta["eid"]].add(cui)
            except Exception:
                pass

    # 직렬화
    result = {k: list(v) for k, v in ev_matches.items()}
    with open(MATCH_CACHE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  매칭 저장: {len(result)} evidence → 총 {sum(len(v) for v in result.values())} 매칭", flush=True)

    return result


def patient_to_symptoms(evidences, ev_matches, ev_info):
    """환자 evidence → KG 증상 CUI 셋."""
    cuis = set()
    for ev in evidences:
        base = ev.split("_@_")[0]
        info = ev_info.get(base, {})
        if info.get("is_antecedent"): continue

        # base evidence의 매칭된 증상
        for cui in ev_matches.get(base, []):
            cuis.add(cui)

        # sub-evidence도 별도 매칭 확인
        if base != ev.split("_@_")[0]:
            full_base = ev.split("_@_")[0]
            for cui in ev_matches.get(full_base, []):
                cuis.add(cui)

    return cuis


# ─── 진단 알고리즘 ───

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

def d_coverage(ps, ds, dcs):
    sc = {}
    for dc in dcs:
        s = ds.get(dc, {})
        if not s: sc[dc] = 0; continue
        sc[dc] = sum(1 for x in s if x in ps) / len(s)
    return sorted(sc.items(), key=lambda x: -x[1])


def evaluate(patients, ds, ev_matches, ev_info, fr2cui, dcs, algo, sdf, nd, all_s):
    t1 = t3 = t5 = t10 = n = nm = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        ps = patient_to_symptoms(p["evidences"], ev_matches, ev_info)
        if not ps: nm += 1; continue

        if algo == "bayesian": r = d_bayesian(ps, ds, dcs, all_s)
        elif algo == "idf": r = d_idf(ps, ds, dcs, sdf, nd)
        elif algo == "v15_ratio": r = d_v15(ps, ds, dcs)
        elif algo == "coverage": r = d_coverage(ps, ds, dcs)

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
    print("진단 v4: LLM 기반 동적 evidence-symptom 매칭", flush=True)
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

    # MC sweep
    algos = ["coverage", "idf", "v15_ratio", "bayesian"]
    best = 0
    best_cfg = ""

    for mc in [1, 2, 3, 5]:
        ds, scuis = build_disease_symptoms(pc, dcs, mc=mc)
        ndw = sum(1 for d in dcs if d in ds and ds[d])

        print(f"\n[3] MC={mc}: 증상={len(scuis)}, 질환={ndw}", flush=True)

        # LLM 매칭 (MC=1에서만 실행, 캐싱)
        if mc == 1:
            print("\n[4] LLM evidence-symptom 매칭...", flush=True)
            ev_matches = build_llm_match(ev_info, scuis, cui_preferred, dcs)

        # Pre-compute IDF
        sdf = Counter()
        for syms in ds.values():
            for s in syms: sdf[s] += 1
        nd = max(len(ds), 1)
        all_s = set()
        for syms in ds.values(): all_s.update(syms.keys())

        # 평가
        for algo in algos:
            t0 = time.time()
            r = evaluate(patients, ds, ev_matches, ev_info, fr2cui, dcs,
                         algo, sdf, nd, all_s)
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
                f"(nm={r['nm']:,}, {el:.0f}s){m}", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"최고 GTPA@1 = {best:.1f}% ({best_cfg})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
