#!/usr/bin/env python3
"""진단 v77: Generative differential diagnosis from closed set.

LLM에게 49 disease 명단 주고 "환자에게 가장 가능성 높은 진단 top-3 순서대로 적어"
요청. 응답에서 disease 이름 매칭 → ranked output.
"""
from __future__ import annotations
import ast, csv, json, math, os, re, time, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")

TRANSLATION_FIX = {
    "haunting": "stabbing", "tugging": "pulling", "sensitive": "tender",
    "a knife stroke": "stabbing", "a cramp": "cramping", "haunted": "stabbing",
    "sickening": "nauseating", "tedious": "tiresome", "scary": "frightening",
    "violent": "severe",
}
DISEASE_NAME_FIX = {
    "URTI": "Upper respiratory tract infection (URTI)",
    "Larygospasm": "Laryngospasm",
    "GERD": "Gastroesophageal reflux disease (GERD)",
    "PSVT": "Paroxysmal supraventricular tachycardia (PSVT)",
    "SLE": "Systemic lupus erythematosus (SLE)",
    "Boerhaave": "Boerhaave syndrome (esophageal rupture)",
    "Possible NSTEMI / STEMI": "NSTEMI or STEMI (myocardial infarction)",
    "HIV (initial infection)": "Acute HIV infection (acute retroviral syndrome)",
    "Localized edema": "Localized edema",
    "Bronchospasm / acute asthma exacerbation": "Acute asthma exacerbation",
    "Acute COPD exacerbation / infection": "Acute COPD exacerbation",
    "Pulmonary embolism": "Pulmonary embolism (PE)",
    "Atrial fibrillation": "Atrial fibrillation (AFib)",
    "Whooping cough": "Whooping cough (pertussis)",
    "Pulmonary neoplasm": "Pulmonary neoplasm (lung cancer)",
    "Pancreatic neoplasm": "Pancreatic neoplasm (pancreatic cancer)",
    "Influenza": "Influenza (flu)",
    "Tuberculosis": "Tuberculosis (TB)",
    "Allergic sinusitis": "Allergic rhinitis/sinusitis",
}


def fix_translation(s):
    if not s: return s
    for bad, good in TRANSLATION_FIX.items():
        s = s.replace(bad, good)
    return s

def disease_full_name(short):
    return DISEASE_NAME_FIX.get(short, short)


def main():
    print("="*80, flush=True)
    print("진단 v77: Generative DDx from closed set", flush=True)
    print("="*80, flush=True)

    cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG" and p[2] == "P" and p[0] not in cp:
                cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)

    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; diseases[dn] = {"cui": dc}
        fr2cui[info.get("cond-name-fr", "")] = dc; cui2name[dc] = dn
    dcs_list = sorted(set(d["cui"] for d in diseases.values()))

    # Build name → CUI lookup with fuzzy match
    name_to_cui = {}
    name_keys_per_cui = {}
    for dc in dcs_list:
        name = disease_full_name(cui2name.get(dc, dc))
        keys = set()
        keys.add(name.lower())
        # Without parenthetical part
        no_paren = re.sub(r"\s*\(.*?\)", "", name).lower().strip()
        if no_paren: keys.add(no_paren)
        # Just the parenthetical
        m = re.search(r"\(([^)]+)\)", name)
        if m: keys.add(m.group(1).lower().strip())
        # Short form (cui2name original)
        short = cui2name.get(dc, dc).lower()
        keys.add(short)
        for k in keys:
            if k: name_to_cui[k] = dc
        name_keys_per_cui[dc] = keys

    # Disease list as numbered enum for prompt
    disease_lines = []
    for i, dc in enumerate(dcs_list):
        name = disease_full_name(cui2name.get(dc, dc))
        disease_lines.append(f"{i+1}. {name}")
    disease_enum = "\n".join(disease_lines)

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]

    def patient_profile(evidences):
        pain_chars=[]; pain_locs=[]; pain_radiations=[]
        pain_intens=None; pain_speed=None; pain_present=False
        symptoms=[]; history=[]
        for ev in evidences:
            parts = ev.split("_@_"); base=parts[0]; value=parts[1] if len(parts)>1 else None
            info = ev_info.get(base, {}); q = info.get("question_en", "")
            val_en = info.get("value_en", {}).get(value, "") if value else ""
            if val_en and val_en.lower() in ("na","nowhere","n"): val_en=""
            val_en = fix_translation(val_en)
            if info.get("is_antecedent"):
                q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
                history.append(q_clean + (f": {val_en}" if val_en else "")); continue
            if "douleur" in base or "_dlr" in base:
                if "carac" in base and val_en: pain_chars.append(val_en)
                elif "endroitducorps" in base and val_en: pain_locs.append(val_en)
                elif "irrad" in base and val_en: pain_radiations.append(val_en)
                elif "intens" in base and value: pain_intens=value
                elif "soudain" in base and value: pain_speed=value
                elif base=="douleurxx": pain_present=True
                continue
            q_clean = re.sub(r"Do you |Are you |Have you |Did you |Is your ", "", q).rstrip("?").strip() if q else ""
            symptoms.append(q_clean + (f": {val_en}" if val_en else ""))
        out=[]
        if pain_present or pain_chars:
            pp=[]
            if pain_chars: pp.append(f"character ({', '.join(pain_chars)})")
            if pain_locs: pp.append(f"location ({', '.join(pain_locs)})")
            if pain_radiations: pp.append(f"radiation to ({', '.join(pain_radiations)})")
            if pain_intens: pp.append(f"intensity {pain_intens}/10")
            if pain_speed: pp.append(f"onset {pain_speed}/10")
            out.append("PAIN: " + "; ".join(pp))
        if symptoms: out.append("OTHER: " + "; ".join(symptoms))
        if history: out.append("HISTORY: " + "; ".join(history))
        return "\n".join(out)

    SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    print(f"\n[2] 테스트 (SUBSET={SUBSET})...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= SUBSET: break
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})
    print(f"  {len(patients):,}명", flush=True)

    candidates = []; n = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        candidates.append({"patient": p, "true_dc": tdc})

    print(f"\n[3] vLLM init...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    print("\n[4] Generative DDx prompts...", flush=True)
    sampling = SamplingParams(temperature=0, max_tokens=80)
    convs = []
    for c in candidates:
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        age_sex = f"{p['age']}yo {'Male' if p['sex']=='M' else 'Female'}"
        prompt = f"""Patient: {age_sex}
Chief complaint: {chief}
{profile}

From the following list of 49 possible diagnoses, identify the TOP 5 most likely (most likely first):
{disease_enum}

Output format: comma-separated list of 5 diagnosis numbers, e.g. "5, 12, 7, 23, 1". Output ONLY the list, nothing else."""
        convs.append([{"role": "user", "content": prompt}])

    print(f"  Total LLM calls: {len(convs)}", flush=True)
    CHUNK = 5000
    rankings = []
    t0 = time.time()
    for chunk_start in range(0, len(convs), CHUNK):
        chunk = convs[chunk_start:chunk_start+CHUNK]
        outs = llm.chat(chunk, sampling)
        for out in outs:
            text = out.outputs[0].text.strip()
            # Parse numbers
            nums = re.findall(r"\b(\d{1,2})\b", text)
            picks = []
            for x in nums:
                idx = int(x) - 1
                if 0 <= idx < len(dcs_list):
                    dc = dcs_list[idx]
                    if dc not in picks:
                        picks.append(dc)
                if len(picks) >= 5: break
            rankings.append(picks)
        print(f"  진행: {chunk_start+len(chunk)}/{len(convs)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  완료: {time.time()-t0:.0f}초", flush=True)

    # Show samples
    if outs: print(f"  샘플 응답: {outs[0].outputs[0].text.strip()[:200]}", flush=True)

    t1 = t3 = t5 = 0
    no_pick = 0
    for c_idx, c in enumerate(candidates):
        picks = rankings[c_idx]
        if not picks:
            no_pick += 1
            continue
        if picks[0] == c["true_dc"]: t1 += 1
        if c["true_dc"] in picks[:3]: t3 += 1
        if c["true_dc"] in picks[:5]: t5 += 1

    print(f"\n  v77 (gen DDx) @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @5={100*t5/n:.1f}% (no_pick={no_pick})", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v77 GTPA@1 = {100*t1/n:.1f}% (SUBSET={n})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
