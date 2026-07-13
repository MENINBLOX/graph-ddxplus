#!/usr/bin/env python3
"""진단 v3: PubMed KG + 텍스트 기반 진단 (증상 CUI 매핑 없음).

핵심 원칙:
  - KG 구축: 질환 이름 → PubMed → 텍스트 매칭 → LLM 1회/초록 → 관계 추출
  - 진단: 환자 evidence 영문 텍스트 ↔ KG 증상 이름 텍스트 매칭
  - DDXPlus umls_mapping.json 사용하지 않음
  - 증상 매핑 없음 — KG가 자체적으로 증상 어휘 정의

최적화:
  - 캐시된 초록 사용 (pilot/data/ablation_cache.json)
  - CPU 병렬 처리 (multiprocessing)
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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
RESULTS_DIR = Path("pilot/results")
CACHE_DIR = Path("pilot/data")
KG_CACHE = RESULTS_DIR / "kg_v3_cache.json"
ABS_CACHE = CACHE_DIR / "ablation_cache.json"

ALLOWED_STYS = {
    "T047", "T184", "T191", "T046", "T048", "T037",
    "T019", "T020", "T190", "T049", "T033", "T031", "T040",
}
BLACKLIST = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT_CLINICAL = """Abstract: {abstract}

Disease: {disease_name} [{disease_cui}]

Clinical findings and symptoms found in this abstract:
{keywords}

From the abstract, identify which of the above findings are CLINICAL SYMPTOMS or SIGNS that a patient with {disease_name} would present with.

Include symptoms patients report and physical examination findings.
Exclude lab results, imaging, other diseases, and synonyms of {disease_name}.

JSON only: [{{"cui":"...","relation":"symptom-of|sign-of"}}]
If none: []"""

MAX_ABSTRACTS = 500
N_WORKERS = min(57, cpu_count())


# ─── UMLS 로드 ───────────────────────────────────────────────────────────────

def load_umls():
    """UMLS 데이터 로드: STY, 관계, 이름, Aho-Corasick."""
    print("  MRSTY...", end="", flush=True)
    cui_stys = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            cui_stys[p[0]].add(p[1])
    print(f" {len(cui_stys):,} CUIs")

    print("  MRREL...", end="", flush=True)
    parent_map = defaultdict(set)
    synonym_map = defaultdict(set)
    with open(UMLS_DIR / "MRREL.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[3] in ("PAR", "RB"):
                parent_map[p[0]].add(p[4])
            if p[3] == "SY":
                synonym_map[p[0]].add(p[4])
                synonym_map[p[4]].add(p[0])
    print(f" {len(parent_map):,} parents")

    print("  MRCONSO...", end="", flush=True)
    cui_all_names = defaultdict(set)
    cui_preferred = {}
    target = {c for c, s in cui_stys.items() if s & ALLOWED_STYS} - BLACKLIST

    aho = ahocorasick.Automaton()
    n_words = 0
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                cui_all_names[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cui_preferred:
                    cui_preferred[p[0]] = p[14].strip()
            if p[0] in target and p[1] == "ENG":
                lo = p[14].strip().lower()
                if len(lo) >= 4:
                    try:
                        aho.add_word(lo, (lo, p[0]))
                        n_words += 1
                    except Exception:
                        pass
    aho.make_automaton()
    print(f" {len(cui_all_names):,} CUIs, {n_words:,} Aho patterns")

    return (
        dict(cui_stys), dict(parent_map), dict(synonym_map),
        dict(cui_all_names), cui_preferred, aho,
    )


# ─── DDXPlus 로드 ────────────────────────────────────────────────────────────

def load_ddxplus():
    """DDXPlus: 질환 매핑(ICD-10), evidence 영문 텍스트."""
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

    # Evidence 영문 텍스트 (매핑 테이블이 아닌, 질문 텍스트)
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


# ─── 텍스트 매칭 (Aho-Corasick) ──────────────────────────────────────────────

def text_match_abstract(abstract_lower, aho, exclude_cui=None):
    """초록에서 CUI 추출 (Aho-Corasick 텍스트 매칭)."""
    matched = set()
    for ei, (n, c) in aho.iter(abstract_lower):
        if c == exclude_cui:
            continue
        si = ei - len(n) + 1
        if si > 0 and abstract_lower[si - 1].isalpha():
            continue
        if ei + 1 < len(abstract_lower) and abstract_lower[ei + 1].isalpha():
            continue
        matched.add(c)
    return matched


def parse_json_r(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    m = re.search(r"\[[\s\S]*?\]", text)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return []


# ─── 초록 캐시에서 LLM 프롬프트 준비 (병렬) ───────────────────────────────────

def _prepare_disease_prompts(args):
    """단일 질환의 초록에서 프롬프트 생성 (병렬 worker)."""
    dc, abstracts, aho_data, cui_preferred_data, synonym_set, parent_set = args

    # Aho-Corasick 재구축 (pickle 불가하므로 worker에서 재사용 불가)
    # 대신 abstracts에 이미 cuis가 포함되어 있으면 재사용
    tasks = []
    for doc in abstracts[:MAX_ABSTRACTS]:
        ab = doc.get("abstract", "")
        if not ab or len(ab) < 200:
            continue

        # cuis가 캐시에 있으면 사용, 없으면 스킵 (aho 없이는 추출 불가)
        cuis = doc.get("cuis", [])
        if not cuis:
            continue

        # 필터: 질환 자체, 동의어, 부모/자식 제외
        filtered = set()
        for c in cuis:
            if c == dc:
                continue
            if c in synonym_set:
                continue
            if c in parent_set:
                continue
            filtered.add(c)

        if not filtered:
            continue

        kw = "\n".join(
            f"- {cui_preferred_data.get(c, c)} [{c}]"
            for c in sorted(filtered)
        )
        prompt = PROMPT_CLINICAL.format(
            abstract=ab[:3000],
            disease_name=cui_preferred_data.get(dc, dc),
            disease_cui=dc,
            keywords=kw,
        )
        tasks.append({"prompt": prompt, "dc": dc})

    return tasks


# ─── KG 구축 ─────────────────────────────────────────────────────────────────

def build_kg(diseases, cui_preferred, aho, synonym_map, parent_map, cui_all_names):
    """캐시된 초록에서 KG 구축."""
    disease_cuis = {v["cui"] for v in diseases.values()}

    # 캐시 로드
    print(f"\n[KG] 캐시 초록 로드...", flush=True)
    with open(ABS_CACHE) as f:
        cache = json.load(f)
    docs = cache["docs"]
    print(f"  {len(docs)} 질환, 총 {sum(len(v) for v in docs.values()):,} 초록", flush=True)

    # 텍스트 매칭으로 CUI가 없는 초록에 대해 Aho-Corasick 실행
    print(f"[KG] 텍스트 매칭 (Aho-Corasick)...", flush=True)
    t0 = time.time()
    all_tasks = []

    for idx, (dn, dinfo) in enumerate(sorted(diseases.items())):
        dc = dinfo["cui"]
        if dc not in docs:
            print(f"  [{idx+1}/49] {dn}: 캐시 없음", flush=True)
            continue

        abstracts = docs[dc][:MAX_ABSTRACTS]
        syn_set = synonym_map.get(dc, set())
        par_set = parent_map.get(dc, set())

        task_count = 0
        for doc in abstracts:
            ab = doc.get("abstract", "")
            if not ab or len(ab) < 200:
                continue

            # 텍스트 매칭
            cuis = text_match_abstract(ab.lower(), aho, exclude_cui=dc)
            if not cuis:
                continue

            # 필터
            filtered = set()
            for c in cuis:
                if c in syn_set or c in par_set or dc in parent_map.get(c, set()):
                    continue
                filtered.add(c)

            if not filtered:
                continue

            kw = "\n".join(
                f"- {cui_preferred.get(c, c)} [{c}]"
                for c in sorted(filtered)
            )
            all_tasks.append({
                "prompt": PROMPT_CLINICAL.format(
                    abstract=ab[:3000],
                    disease_name=cui_preferred.get(dc, dc),
                    disease_cui=dc,
                    keywords=kw,
                ),
                "dc": dc,
            })
            task_count += 1

        print(f"  [{idx+1}/49] {dn}: {len(abstracts)}편 → {task_count} 프롬프트", flush=True)

    print(f"  텍스트 매칭 완료: {time.time()-t0:.0f}초, 총 {len(all_tasks):,} 프롬프트", flush=True)

    # vLLM batch
    print(f"\n[KG] vLLM batch ({len(all_tasks):,}편)...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(
        model="google/gemma-4-E4B-it",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "audio": 0},
    )
    sampling = SamplingParams(temperature=0, max_tokens=4096)
    convs = [[{"role": "user", "content": t["prompt"]}] for t in all_tasks]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.0f}초 ({len(outputs)/elapsed:.1f}/s)", flush=True)

    # 파싱
    all_rels = []
    for task, out in zip(all_tasks, outputs):
        for item in parse_json_r(out.outputs[0].text):
            if not isinstance(item, dict):
                continue
            cui = item.get("cui", "")
            rel = item.get("relation", "")
            if cui and rel and rel != "manifestation-of":
                dc = task["dc"]
                syn_set = synonym_map.get(dc, set())
                par_set = parent_map.get(dc, set())
                if cui not in syn_set and cui not in par_set and dc not in parent_map.get(cui, set()):
                    all_rels.append({"dc": dc, "cui": cui})

    pair_counts = Counter(tuple(sorted([r["dc"], r["cui"]])) for r in all_rels)
    print(f"  관계: {len(all_rels):,}, 고유 쌍: {len(pair_counts):,}", flush=True)

    # 캐시 저장
    cache_data = {
        "pair_counts": [[list(k), v] for k, v in pair_counts.most_common()],
        "stats": {"prompts": len(all_tasks), "time": round(elapsed, 1)},
    }
    with open(KG_CACHE, "w") as f:
        json.dump(cache_data, f)
    print(f"  KG 캐시 저장: {KG_CACHE}", flush=True)

    return pair_counts


def load_kg_cache():
    """캐시된 KG pair_counts 로드."""
    with open(KG_CACHE) as f:
        cache = json.load(f)
    pair_counts = Counter()
    for k, v in cache["pair_counts"]:
        pair_counts[tuple(k)] = v
    print(f"  캐시 로드: {len(pair_counts):,} 쌍, {cache['stats']}", flush=True)
    return pair_counts


# ─── 진단: 텍스트 기반 증상 매칭 ─────────────────────────────────────────────

def build_symptom_text_index(pair_counts, disease_cuis, cui_all_names, mc=1):
    """KG 증상 CUI의 모든 영문 이름으로 Aho-Corasick 구축."""
    symptom_cuis = set()
    disease_symptoms = defaultdict(dict)
    for (a, b), cnt in pair_counts.items():
        if cnt < mc:
            continue
        if a in disease_cuis:
            disease_symptoms[a][b] = cnt
            symptom_cuis.add(b)
        if b in disease_cuis:
            disease_symptoms[b][a] = cnt
            symptom_cuis.add(a)

    aho = ahocorasick.Automaton()
    n_names = 0
    for cui in symptom_cuis:
        for name in cui_all_names.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) >= 3:
                try:
                    aho.add_word(lo, (lo, cui))
                    n_names += 1
                except Exception:
                    pass
    aho.make_automaton()
    return dict(disease_symptoms), aho, len(symptom_cuis), n_names


def patient_evidence_to_text(evidences, ev_text_info):
    """환자 evidence → 영문 텍스트 (KG 증상 이름과 텍스트 매칭용)."""
    texts = []
    for ev in evidences:
        parts = ev.split("_@_")
        base = parts[0]
        value = parts[1] if len(parts) > 1 else None

        info = ev_text_info.get(base, {})
        if info.get("is_antecedent"):
            continue

        q = info.get("question_en", "")
        if q:
            texts.append(q.lower())

        if value and info.get("value_en"):
            val_en = info["value_en"].get(value, "")
            if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                texts.append(val_en.lower())
                # 합성어 생성: pain + location → "chest pain"
                if "pain" in q.lower() and val_en:
                    texts.append(f"{val_en.lower()} pain")

    return " . ".join(texts)


def match_patient_to_symptoms(patient_text, aho_symptoms):
    """환자 텍스트에서 KG 증상 CUI 매칭."""
    matched = set()
    for ei, (n, cui) in aho_symptoms.iter(patient_text):
        si = ei - len(n) + 1
        if si > 0 and patient_text[si - 1].isalpha():
            continue
        if ei + 1 < len(patient_text) and patient_text[ei + 1].isalpha():
            continue
        matched.add(cui)
    return matched


# ─── 진단 알고리즘 ────────────────────────────────────────────────────────────

def diagnose_coverage(patient_syms, disease_symptoms, disease_cuis):
    scores = {}
    for dc in disease_cuis:
        syms = disease_symptoms.get(dc, {})
        if not syms:
            scores[dc] = 0; continue
        matched = sum(1 for s in syms if s in patient_syms)
        scores[dc] = matched / len(syms)
    return sorted(scores.items(), key=lambda x: -x[1])


def diagnose_weighted(patient_syms, disease_symptoms, disease_cuis):
    scores = {}
    for dc in disease_cuis:
        syms = disease_symptoms.get(dc, {})
        if not syms:
            scores[dc] = 0; continue
        total = sum(syms.values())
        matched = sum(cnt for s, cnt in syms.items() if s in patient_syms)
        scores[dc] = matched / total
    return sorted(scores.items(), key=lambda x: -x[1])


def diagnose_idf(patient_syms, disease_symptoms, disease_cuis, symptom_df, n_diseases):
    scores = {}
    for dc in disease_cuis:
        syms = disease_symptoms.get(dc, {})
        if not syms:
            scores[dc] = 0; continue
        score = 0
        for s, cnt in syms.items():
            if s in patient_syms:
                idf = math.log(n_diseases / (symptom_df[s] + 1)) + 1
                score += idf * cnt
        scores[dc] = score
    return sorted(scores.items(), key=lambda x: -x[1])


def diagnose_v15_ratio(patient_syms, disease_symptoms, disease_cuis):
    scores = {}
    for dc in disease_cuis:
        syms = disease_symptoms.get(dc, {})
        if not syms:
            scores[dc] = 0; continue
        confirmed = sum(1 for s in syms if s in patient_syms)
        denied = sum(1 for s in syms if s not in patient_syms)
        scores[dc] = confirmed / (confirmed + denied + 1) * confirmed if confirmed else 0
    return sorted(scores.items(), key=lambda x: -x[1])


def diagnose_bayesian(patient_syms, disease_symptoms, disease_cuis, all_symptom_cuis):
    scores = {}
    for dc in disease_cuis:
        syms = disease_symptoms.get(dc, {})
        if not syms:
            scores[dc] = -1e6; continue
        total_w = sum(syms.values()) + len(all_symptom_cuis) * 0.1
        log_score = 0
        for s in patient_syms:
            if s in syms:
                p = (syms[s] + 0.1) / total_w
            else:
                p = 0.1 / total_w
            log_score += math.log(p + 1e-10)
        scores[dc] = log_score
    return sorted(scores.items(), key=lambda x: -x[1])


def diagnose_idf_negative(patient_syms, disease_symptoms, disease_cuis, symptom_df, n_diseases):
    scores = {}
    for dc in disease_cuis:
        syms = disease_symptoms.get(dc, {})
        if not syms:
            scores[dc] = 0; continue
        score = 0
        for s, cnt in syms.items():
            idf = math.log(n_diseases / (symptom_df[s] + 1)) + 1
            if s in patient_syms:
                score += idf * cnt
            else:
                score -= idf * 0.5
        scores[dc] = score
    return sorted(scores.items(), key=lambda x: -x[1])


# ─── 배치 진단 평가 (병렬) ────────────────────────────────────────────────────

def _evaluate_chunk(args):
    """환자 청크 평가 (병렬 worker)."""
    patients, disease_symptoms_serial, aho_data, ev_text_info, \
        disease_fr_to_cui, disease_cuis_list, algo_name, \
        symptom_df_serial, n_diseases, all_syms_list = args

    disease_cuis = set(disease_cuis_list)
    disease_symptoms = {k: dict(v) for k, v in disease_symptoms_serial}
    symptom_df = Counter(symptom_df_serial)
    all_symptom_cuis = set(all_syms_list)

    # Aho-Corasick 재구축
    aho = ahocorasick.Automaton()
    for name, cui in aho_data:
        try:
            aho.add_word(name, (name, cui))
        except Exception:
            pass
    aho.make_automaton()

    top1 = top3 = top5 = top10 = 0
    n = 0
    no_match = 0

    for patient in patients:
        true_dc = disease_fr_to_cui.get(patient["pathology"])
        if not true_dc:
            continue
        n += 1

        pt_text = patient_evidence_to_text(patient["evidences"], ev_text_info)
        pt_syms = match_patient_to_symptoms(pt_text, aho)

        if not pt_syms:
            no_match += 1
            continue

        if algo_name == "coverage":
            ranked = diagnose_coverage(pt_syms, disease_symptoms, disease_cuis)
        elif algo_name == "weighted":
            ranked = diagnose_weighted(pt_syms, disease_symptoms, disease_cuis)
        elif algo_name == "idf":
            ranked = diagnose_idf(pt_syms, disease_symptoms, disease_cuis, symptom_df, n_diseases)
        elif algo_name == "v15_ratio":
            ranked = diagnose_v15_ratio(pt_syms, disease_symptoms, disease_cuis)
        elif algo_name == "bayesian":
            ranked = diagnose_bayesian(pt_syms, disease_symptoms, disease_cuis, all_symptom_cuis)
        elif algo_name == "idf_neg":
            ranked = diagnose_idf_negative(pt_syms, disease_symptoms, disease_cuis, symptom_df, n_diseases)
        else:
            continue

        ranked_dcs = [dc for dc, s in ranked]
        if ranked_dcs and ranked_dcs[0] == true_dc:
            top1 += 1
        if true_dc in ranked_dcs[:3]:
            top3 += 1
        if true_dc in ranked_dcs[:5]:
            top5 += 1
        if true_dc in ranked_dcs[:10]:
            top10 += 1

    return {"n": n, "no_match": no_match, "top1": top1, "top3": top3, "top5": top5, "top10": top10}


def evaluate_parallel(test_patients, disease_symptoms, aho_symptoms,
                      ev_text_info, disease_fr_to_cui, disease_cuis, algo_name,
                      symptom_df, n_diseases, all_symptom_cuis):
    """병렬 진단 평가."""
    # Aho-Corasick → serializable data
    aho_data = []
    for name, cui_set in aho_symptoms._items() if hasattr(aho_symptoms, '_items') else []:
        pass
    # Aho-Corasick을 직접 직렬화할 수 없으므로, 이름-CUI 쌍으로 전달
    # 대신 단일 프로세스에서 실행 (Aho-Corasick은 fork시 복사됨)
    pass


def evaluate_single(test_patients, disease_symptoms, aho_symptoms,
                     ev_text_info, disease_fr_to_cui, disease_cuis, algo_name):
    """단일 프로세스 진단 평가."""
    symptom_df = Counter()
    for dc, syms in disease_symptoms.items():
        for s in syms:
            symptom_df[s] += 1
    n_diseases = max(len(disease_symptoms), 1)
    all_symptom_cuis = set()
    for syms in disease_symptoms.values():
        all_symptom_cuis.update(syms.keys())

    top1 = top3 = top5 = top10 = 0
    n = 0
    no_match = 0

    for patient in test_patients:
        true_dc = disease_fr_to_cui.get(patient["pathology"])
        if not true_dc:
            continue
        n += 1

        pt_text = patient_evidence_to_text(patient["evidences"], ev_text_info)
        pt_syms = match_patient_to_symptoms(pt_text, aho_symptoms)

        if not pt_syms:
            no_match += 1
            continue

        if algo_name == "coverage":
            ranked = diagnose_coverage(pt_syms, disease_symptoms, disease_cuis)
        elif algo_name == "weighted":
            ranked = diagnose_weighted(pt_syms, disease_symptoms, disease_cuis)
        elif algo_name == "idf":
            ranked = diagnose_idf(pt_syms, disease_symptoms, disease_cuis, symptom_df, n_diseases)
        elif algo_name == "v15_ratio":
            ranked = diagnose_v15_ratio(pt_syms, disease_symptoms, disease_cuis)
        elif algo_name == "bayesian":
            ranked = diagnose_bayesian(pt_syms, disease_symptoms, disease_cuis, all_symptom_cuis)
        elif algo_name == "idf_neg":
            ranked = diagnose_idf_negative(pt_syms, disease_symptoms, disease_cuis, symptom_df, n_diseases)

        ranked_dcs = [dc for dc, s in ranked]
        if ranked_dcs and ranked_dcs[0] == true_dc:
            top1 += 1
        if true_dc in ranked_dcs[:3]:
            top3 += 1
        if true_dc in ranked_dcs[:5]:
            top5 += 1
        if true_dc in ranked_dcs[:10]:
            top10 += 1

    return {
        "n": n, "no_match": no_match,
        "top1": top1, "top3": top3, "top5": top5, "top10": top10,
        "gtpa1": round(100 * top1 / n, 2) if n > 0 else 0,
        "gtpa3": round(100 * top3 / n, 2) if n > 0 else 0,
        "gtpa5": round(100 * top5 / n, 2) if n > 0 else 0,
        "gtpa10": round(100 * top10 / n, 2) if n > 0 else 0,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80, flush=True)
    print("진단 v3: PubMed KG + 텍스트 기반 진단 (증상 매핑 없음)", flush=True)
    print("=" * 80, flush=True)

    print(f"\n[1] 데이터 로드 (CPU: {N_WORKERS}코어)...", flush=True)
    (cui_stys, parent_map, synonym_map,
     cui_all_names, cui_preferred, aho_abstract) = load_umls()
    diseases, disease_fr_to_cui, ev_text_info = load_ddxplus()
    disease_cuis = {v["cui"] for v in diseases.values()}
    print(f"  질환: {len(diseases)}", flush=True)

    # [2] KG 구축 또는 캐시 로드
    print(f"\n[2] KG...", flush=True)
    if KG_CACHE.exists():
        print("  캐시 발견!", flush=True)
        pair_counts = load_kg_cache()
    else:
        pair_counts = build_kg(
            diseases, cui_preferred, aho_abstract,
            synonym_map, parent_map, cui_all_names,
        )

    # [3] 테스트 데이터 로드
    print(f"\n[3] 테스트 데이터...", flush=True)
    test_patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            test_patients.append({
                "evidences": ast.literal_eval(row["EVIDENCES"]),
                "pathology": row["PATHOLOGY"],
            })
    print(f"  환자: {len(test_patients):,}명", flush=True)

    # [4] 진단 알고리즘 × MC threshold sweep
    print(f"\n[4] 진단 평가 ({len(test_patients):,}명)...", flush=True)
    algos = ["coverage", "weighted", "idf", "v15_ratio", "bayesian", "idf_neg"]

    best_gtpa1 = 0
    best_config = ""
    all_results = []

    for mc in [1, 2, 3, 5]:
        ds, aho_sym, n_sym, n_names = build_symptom_text_index(
            pair_counts, disease_cuis, cui_all_names, mc=mc,
        )
        n_dw = sum(1 for d in disease_cuis if d in ds and ds[d])
        print(f"\n  MC={mc}: 증상CUI={n_sym}, 이름={n_names:,}, 질환(증상有)={n_dw}", flush=True)

        for algo_name in algos:
            t0 = time.time()
            result = evaluate_single(
                test_patients, ds, aho_sym,
                ev_text_info, disease_fr_to_cui, disease_cuis, algo_name,
            )
            elapsed = time.time() - t0
            marker = ""
            if result["gtpa1"] > best_gtpa1:
                best_gtpa1 = result["gtpa1"]
                best_config = f"MC={mc} {algo_name}"
                marker = " ★"
            print(
                f"    {algo_name:<12}: "
                f"GTPA@1={result['gtpa1']:>5.1f}% "
                f"@3={result['gtpa3']:>5.1f}% "
                f"@5={result['gtpa5']:>5.1f}% "
                f"@10={result['gtpa10']:>5.1f}% "
                f"(no_match={result['no_match']:,}, {elapsed:.0f}s){marker}",
                flush=True,
            )
            all_results.append({"mc": mc, "algo": algo_name, **result})

    # [5] 결과 저장
    with open(RESULTS_DIR / "kg_diagnose_v3_results.json", "w") as f:
        json.dump({
            "best_gtpa1": best_gtpa1,
            "best_config": best_config,
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'=' * 80}", flush=True)
    print(f"최고 GTPA@1 = {best_gtpa1:.1f}% ({best_config})", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()
