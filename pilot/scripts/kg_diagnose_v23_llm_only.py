#!/usr/bin/env python3
"""진단 v23: LLM only (KG 미사용) - KG 가치 측정용 baseline.

환자 증상 + 49개 질환 이름 전체 → LLM이 직접 선택 (KG 없이).
이 결과가 v17(56.2%)보다 낮으면 KG가 가치를 추가하는 것.
"""
from __future__ import annotations
import ast, csv, json, os, re, time
from pathlib import Path
from vllm import LLM, SamplingParams

PROMPT = """Patient: {age}yo {sex}
Chief complaint: {chief}

Symptoms:
{profile}

Which of these 49 diseases is MOST LIKELY?
{candidates}

Answer with ONLY the number (1-49)."""


def main():
    print("="*80, flush=True)
    print("진단 v23: LLM only (KG 미사용)", flush=True)
    print("="*80, flush=True)

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)

    diseases = []  # 49 disease names in order
    fr2idx = {}
    for dn in sorted(cond):
        if dn not in icd_map: continue
        diseases.append(dn)
        fr2idx[cond[dn].get("cond-name-fr", "")] = len(diseases) - 1

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]

    def patient_profile(evidences):
        from collections import defaultdict
        sec = defaultdict(list)
        for ev in evidences:
            parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
            info = ev_info.get(base, {})
            q = info.get("question_en", "")
            val_en = info.get("value_en", {}).get(value, "") if value else ""
            if val_en and val_en.lower() in ("na", "nowhere", "n"): val_en = ""
            q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
            entry = q_clean + (f": {val_en}" if val_en else "")
            if info.get("is_antecedent"): sec["history"].append(entry); continue
            if "pain" in q.lower(): sec["pain"].append(entry)
            elif "skin" in q.lower(): sec["skin"].append(entry)
            elif any(k in q.lower() for k in ["breath","cough","wheez","dyspn"]): sec["resp"].append(entry)
            elif any(k in q.lower() for k in ["heart","palpit","chest"]): sec["cardiac"].append(entry)
            else: sec["other"].append(entry)
        out = []
        for k in ["pain","skin","resp","cardiac","other","history"]:
            if sec[k]: out.append(f"{k.upper()}: " + "; ".join(sec[k]))
        return "\n".join(out)

    print("\n테스트 데이터...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})
    print(f"  {len(patients):,}명", flush=True)

    # 49 후보는 모든 환자에게 동일
    cands = "\n".join(f"{i+1}. {dn}" for i, dn in enumerate(diseases))

    print("\n프롬프트...", flush=True)
    prompts = []
    truth = []
    for p in patients:
        idx = fr2idx.get(p["pathology"], -1)
        if idx < 0: continue
        truth.append(idx)
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        prompts.append(PROMPT.format(
            age=p["age"], sex="Male" if p["sex"] == "M" else "Female",
            chief=chief, profile=profile, candidates=cands,
        ))
    print(f"  프롬프트: {len(prompts):,}", flush=True)

    print("\nvLLM batch...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=16)
    convs = [[{"role": "user", "content": p}] for p in prompts]
    t0 = time.time()
    outputs = llm.chat(convs, sampling)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # 평가
    correct = 0; pf = 0
    for t, out in zip(truth, outputs):
        text = out.outputs[0].text.strip()
        m = re.search(r"(\d+)", text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(diseases):
                if idx == t: correct += 1
            else: pf += 1
        else: pf += 1

    n = len(truth)
    print(f"\nLLM only @1 = {100*correct/n:.1f}%", flush=True)
    print(f"parse_fail = {pf}", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v23 LLM-only GTPA@1 = {100*correct/n:.1f}%", flush=True)
    print(f"v17 (KG+LLM) = 56.2% — KG 기여: +{56.2-100*correct/n:.1f}%p", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
