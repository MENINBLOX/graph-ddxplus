#!/usr/bin/env python3
"""진단 v69: v59→top10→Yes/No logprob re-score.

가설: 0-100 score는 calibration 어려움. Yes/No 토큰 logprob이 더 잘 calibrated.
P(Yes) = exp(logprob_yes) / (exp(logprob_yes) + exp(logprob_no))
"""
from __future__ import annotations
import ast, csv, json, math, os, re, time, sys
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick
from vllm import LLM, SamplingParams

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'does','have','your','you','the','and','for','are','with','that','this','from','been','were','being','which','their','than','other','about','into','over','some','only','very','also','just','more','most','such','much','will','would','could','should','make','like','time','when','what','where','how','who','all','each','every','both','few','any','not','can','may','her','his','its','our','they','them','then','had','has','him','but','one','two','way','day','did','get','got','let','say','she','too','use','yes','yet','now','new','old','see','own','why','try','ask','set','related','reason','consulting','significant','measured','thermometer','either','believe','racing','missing','beat','fast','irregularly','problems','situation','associated','inability','speak','trouble','keeping','opening','raising','annoying','else','body','somewhere','anywhere','nowhere','recently','currently','usually','often','sometimes','worse','better'}

EVIDENCE_MEDTERM = {
    "diplopie": ["diplopia", "double vision"], "flushing": ["flushing", "facial flushing", "erythema"],
    "gain_poids": ["weight gain"], "perte_poids": ["weight loss"],
    "impression_mort": ["impending doom", "anxiety"],
    "lesions_peau_desquame": ["desquamation", "skin peeling"],
    "lesions_peau_couleur": ["skin discoloration", "rash"],
    "melena": ["melena", "tarry stool", "black stool"],
    "pdc": ["syncope", "loss of consciousness", "fainting"],
    "protu_langue": ["tongue protrusion", "tongue thrusting"],
    "psy_depers": ["depersonalization", "derealization"],
    "ptose": ["ptosis", "blepharoptosis", "eyelid drooping"],
    "ww_dd": ["orthopnea", "worse lying down"],
    "ww_nuit": ["nocturnal symptoms", "night-time symptoms"],
    "anxiete_s": ["anxiety"], "diaph": ["diaphoresis", "sweating"],
    "fatig_mod": ["fatigue", "tiredness"], "pale": ["pallor", "pale skin"],
    "stridor": ["stridor"], "wheezing": ["wheezing"],
    "convulsion": ["convulsion", "seizure"], "confusion": ["confusion", "disorientation"],
    "apnee": ["apnea", "sleep apnea"], "laryngospasme": ["laryngospasm"],
    "tachycardie": ["tachycardia"], "bradycardie": ["bradycardia"],
    "hemoptysie": ["hemoptysis", "coughing up blood"],
    "nausee": ["nausea", "vomiting"], "diarrhee": ["diarrhea"],
    "vomi_sg": ["hematemesis", "blood vomiting"], "douleurxx": ["pain"],
}
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
    print("진단 v69: Yes/No logprob re-score", flush=True)
    print("="*80, flush=True)

    can = defaultdict(set); cp = {}
    with open(UMLS_DIR/"MRCONSO.RRF") as f:
        for l in f:
            p = l.strip().split("|")
            if p[1] == "ENG":
                can[p[0]].add(p[14].strip())
                if p[2] == "P" and p[0] not in cp: cp[p[0]] = p[14].strip()

    with open("data/ddxplus/disease_icd10_cui_mapping.json") as f: icd_map = json.load(f)
    with open("data/ddxplus/release_conditions_en.json") as f: cond = json.load(f)
    with open("data/ddxplus/release_evidences.json") as f: ev_fr = json.load(f)
    with open(KG_CACHE) as f: cache = json.load(f)

    pc = Counter()
    for k, v in cache["pair_counts"]: pc[tuple(k)] = v

    diseases = {}; fr2cui = {}; cui2name = {}
    for dn, info in cond.items():
        if dn not in icd_map: continue
        dc = icd_map[dn]["cui"]; diseases[dn] = {"cui": dc}
        fr2cui[info.get("cond-name-fr", "")] = dc; cui2name[dc] = dn
    dcs = set(d["cui"] for d in diseases.values())

    ev_info = {}
    for eid, info in ev_fr.items():
        ev_info[eid] = {"question_en": info.get("question_en", ""), "is_antecedent": info.get("is_antecedent", False), "value_en": {}}
        vm = info.get("value_meaning", {})
        if isinstance(vm, dict):
            for k, v in vm.items():
                if isinstance(v, dict) and v.get("en"): ev_info[eid]["value_en"][k] = v["en"]

    ds = defaultdict(dict); scuis = set()
    for (a, b), cnt in pc.items():
        if a in NOISE or b in NOISE: continue
        if a in dcs: ds[a][b] = cnt; scuis.add(b)
        if b in dcs: ds[b][a] = cnt; scuis.add(a)
    ds = dict(ds)

    aho = ahocorasick.Automaton()
    for cui in scuis:
        for name in can.get(cui, set()):
            lo = name.lower().strip()
            if len(lo) < 4 or lo in STOPWORDS: continue
            try: aho.add_word(lo, (lo, cui))
            except: pass
    aho.make_automaton()
    all_s = set()
    for syms in ds.values(): all_s.update(syms.keys())

    def text_match(evidences):
        cuis = set()
        for ev in evidences:
            parts = ev.split("_@_"); base = parts[0]; value = parts[1] if len(parts) > 1 else None
            info = ev_info.get(base, {})
            if info.get("is_antecedent"): continue
            terms = []
            if base in EVIDENCE_MEDTERM: terms.extend(EVIDENCE_MEDTERM[base])
            bc = re.sub(r"_.*", "", base); bc = re.sub(r"xx$", "", bc)
            if len(bc) >= 3 and bc not in STOPWORDS: terms.append(bc)
            q = info.get("question_en", "")
            if q:
                text = re.sub(r"\(.*?\)", "", q); text = re.sub(r"[?.,;:!]", "", text)
                terms.extend(w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) >= 3)
                ql = q.lower()
                for ph in ["chest pain","sore throat","shortness of breath","difficulty breathing","weight loss","weight gain","loss of consciousness","muscle pain","muscle spasm","nasal congestion","runny nose","skin lesion","skin rash","black stool","bloody stool","heart palpitation","double vision","swollen"]:
                    if ph in ql: terms.append(ph)
            if value:
                val_en = info.get("value_en", {}).get(value, "")
                if val_en and val_en.lower() not in ("na", "nowhere", "n"):
                    vc = re.sub(r"\([rl]\)", "", val_en.lower()).strip()
                    if vc and vc not in STOPWORDS:
                        terms.append(vc)
                        if "pain" in q.lower(): terms.append(f"{vc} pain")
            pt = " . ".join(terms)
            for ei, (n, cui) in aho.iter(pt):
                si = ei - len(n) + 1
                if si > 0 and pt[si-1].isalpha(): continue
                if ei+1 < len(pt) and pt[ei+1].isalpha(): continue
                cuis.add(cui)
        return cuis

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

    age_sex_disease = defaultdict(Counter)
    with open("data/ddxplus/release_train_patients.csv") as f:
        for row in csv.DictReader(f):
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            age_bin = min(int(row.get("AGE", 0)) // 10 * 10, 80)
            age_sex_disease[(age_bin, row.get("SEX", "M"))][tdc] += 1

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

    print("\n[3] Bayesian baseline (for reference)...", flush=True)
    candidates = []; baseline = 0; n = 0
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        n += 1
        ps = text_match(p["evidences"])
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            ll = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in ps)
            age_bin = min(int(p["age"]) // 10 * 10, 80)
            prior_c = age_sex_disease[(age_bin, p["sex"])]
            prior_t = sum(prior_c.values())
            prior = (prior_c.get(dc, 0) + 1) / (prior_t + len(dcs)) if prior_t > 0 else 1.0/len(dcs)
            sc[dc] = ll + math.log(prior)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        # use ALL 49 disease candidates (no KG top-K filter)
        topk = list(dcs)
        candidates.append({"patient": p, "true_dc": tdc, "topk": topk, "bay_top1": ranked[0][0]})
        if ranked[0][0] == tdc: baseline += 1
    print(f"  Bayesian @1={100*baseline/n:.1f}%", flush=True)

    print("\n[4] vLLM init...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})

    # Stage 1: per-candidate scoring (49 candidates per patient) - same as v59
    print("\n[5] Stage 1: Per-candidate 0-100 scoring (all 49)...", flush=True)
    sampling = SamplingParams(temperature=0, max_tokens=8)

    convs = []
    pair_meta = []  # (c_idx, d_idx)
    for c_idx, c in enumerate(candidates):
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        age_sex = f"{p['age']}yo {'Male' if p['sex']=='M' else 'Female'}"

        for d_idx, dc in enumerate(c["topk"]):
            name = disease_full_name(cui2name.get(dc, dc))
            prompt = f"""Patient: {age_sex}
Chief complaint: {chief}
{profile}

Diagnosis hypothesis: {name}

How likely is this hypothesis given the patient's presentation? Reply with ONLY a percentage 0-100 (0=impossible, 100=certain)."""
            convs.append([{"role": "user", "content": prompt}])
            pair_meta.append((c_idx, d_idx))

    print(f"  Total LLM calls: {len(convs)}", flush=True)

    CHUNK = 5000
    all_scores = {}
    t0 = time.time()
    for chunk_start in range(0, len(convs), CHUNK):
        chunk = convs[chunk_start:chunk_start+CHUNK]
        outs = llm.chat(chunk, sampling)
        for i, out in enumerate(outs):
            c_idx, d_idx = pair_meta[chunk_start + i]
            text = out.outputs[0].text.strip()
            m = re.search(r"(\d+)", text)
            score = int(m.group(1)) if m else 0
            score = min(score, 100)
            all_scores[(c_idx, d_idx)] = score
        print(f"  진행: {chunk_start+len(chunk)}/{len(convs)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  Stage 1 완료: {time.time()-t0:.0f}초", flush=True)

    # Stage 1 ranking → top-10
    print("\n[6] Stage 1 → top-10", flush=True)
    stage1_top10 = []
    t1_v59 = 0
    in_top10_v59 = 0
    for c_idx, c in enumerate(candidates):
        scored = [(d_idx, all_scores.get((c_idx, d_idx), 0)) for d_idx in range(len(c["topk"]))]
        scored.sort(key=lambda x: -x[1])
        top10 = [c["topk"][d_idx] for d_idx, _ in scored[:10]]
        stage1_top10.append(top10)
        tdc = c["true_dc"]
        if top10[0] == tdc: t1_v59 += 1
        if tdc in top10: in_top10_v59 += 1
    nt = len(candidates)
    print(f"  v59 stage1 @1={100*t1_v59/nt:.1f}%, top10={100*in_top10_v59/nt:.1f}%", flush=True)

    # Stage 2: Yes/No logprob scoring on top-10
    print("\n[7] Stage 2: Yes/No logprob re-score", flush=True)
    sampling2 = SamplingParams(temperature=0, max_tokens=1, logprobs=20)

    convs2 = []
    pair_meta2 = []
    for c_idx, c in enumerate(candidates):
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()
        age_sex = f"{p['age']}yo {'Male' if p['sex']=='M' else 'Female'}"
        for d_idx, dc in enumerate(stage1_top10[c_idx]):
            name = disease_full_name(cui2name.get(dc, dc))
            prompt = f"""Patient: {age_sex}
Chief complaint: {chief}
{profile}

Question: Is this patient's most likely diagnosis "{name}"? Answer with ONLY "Yes" or "No"."""
            convs2.append([{"role": "user", "content": prompt}])
            pair_meta2.append((c_idx, d_idx))

    print(f"  Stage 2 calls: {len(convs2):,}", flush=True)
    yes_probs = {}
    t0 = time.time()
    for chunk_start in range(0, len(convs2), CHUNK):
        chunk = convs2[chunk_start:chunk_start+CHUNK]
        outs = llm.chat(chunk, sampling2)
        for i, out in enumerate(outs):
            c_idx, d_idx = pair_meta2[chunk_start + i]
            o = out.outputs[0]
            # logprobs is a list of dicts {token_id: Logprob}
            yes_lp = -1e6
            no_lp = -1e6
            if o.logprobs and len(o.logprobs) > 0:
                first_token_lps = o.logprobs[0]
                for tok_id, lp_obj in first_token_lps.items():
                    tok = lp_obj.decoded_token.strip().lower() if hasattr(lp_obj, 'decoded_token') else ""
                    lp = lp_obj.logprob if hasattr(lp_obj, 'logprob') else lp_obj
                    if tok in ("yes", "y", "true"):
                        yes_lp = max(yes_lp, lp)
                    elif tok in ("no", "n", "false"):
                        no_lp = max(no_lp, lp)
            # P(Yes | Yes or No) — softmax over the two
            if yes_lp == -1e6 and no_lp == -1e6:
                p_yes = 0.0
            elif yes_lp == -1e6:
                p_yes = 0.0
            elif no_lp == -1e6:
                p_yes = 1.0
            else:
                # softmax(yes_lp, no_lp)
                m = max(yes_lp, no_lp)
                p_yes = math.exp(yes_lp - m) / (math.exp(yes_lp - m) + math.exp(no_lp - m))
            yes_probs[(c_idx, d_idx)] = p_yes
        print(f"  Stage 2 진행: {chunk_start+len(chunk)}/{len(convs2)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  Stage 2 완료: {time.time()-t0:.0f}초", flush=True)

    t1 = t3 = t5 = 0
    for c_idx, c in enumerate(candidates):
        scored = [(d_idx, yes_probs.get((c_idx, d_idx), 0.0)) for d_idx in range(len(stage1_top10[c_idx]))]
        scored.sort(key=lambda x: -x[1])
        reranked = [stage1_top10[c_idx][d_idx] for d_idx, _ in scored]
        tdc = c["true_dc"]
        if reranked[0] == tdc: t1 += 1
        if tdc in reranked[:3]: t3 += 1
        if tdc in reranked[:5]: t5 += 1

    print(f"\n  Bayesian @1={100*baseline/nt:.1f}%", flush=True)
    print(f"  v59 stage1 @1={100*t1_v59/nt:.1f}%, top10={100*in_top10_v59/nt:.1f}%", flush=True)
    print(f"  v69 yes/no @1={100*t1/nt:.1f}% @3={100*t3/nt:.1f}% @5={100*t5/nt:.1f}%", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"v69 GTPA@1 = {100*t1/nt:.1f}% (SUBSET={nt})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
