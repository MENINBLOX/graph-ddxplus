#!/usr/bin/env python3
"""진단 v32: 멀티 프롬프트 LLM 앙상블 + KG signature.

가설: 단일 프롬프트는 LLM이 한 view만 사용. 다양한 framing으로 reasoning.
3개 프롬프트로 LLM 호출 → 다수결 (또는 가중 합)

Prompts:
  P1 - which is most likely (basic)
  P2 - match symptoms to features (KG sig 활용)
  P3 - rule out + final pick (differential)

각 프롬프트마다 LLM 호출 → 3개 답 → majority vote
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
    "Localized edema": "Localized edema (e.g., bite, allergic, mechanical)",
    "Bronchospasm / acute asthma exacerbation": "Acute asthma exacerbation",
    "Acute COPD exacerbation / infection": "Acute COPD exacerbation",
    "Pulmonary embolism": "Pulmonary embolism (PE)",
    "Atrial fibrillation": "Atrial fibrillation (AFib)",
    "Whooping cough": "Whooping cough (pertussis)",
    "Croup": "Croup",
    "Pulmonary neoplasm": "Pulmonary neoplasm (lung cancer)",
    "Pancreatic neoplasm": "Pancreatic neoplasm (pancreatic cancer)",
    "Acute rhinosinusitis": "Acute rhinosinusitis",
    "Influenza": "Influenza (flu)",
    "Tuberculosis": "Tuberculosis (TB)",
    "Panic attack": "Panic attack",
    "Scombroid food poisoning": "Scombroid food poisoning (histamine fish poisoning)",
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
    print("진단 v32: 3-prompt ensemble + majority vote", flush=True)
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

    sym_df = Counter()
    for d, syms in ds.items():
        for s in syms: sym_df[s] += 1
    N_d = len(dcs)
    idf = {s: math.log(N_d / df) for s, df in sym_df.items()}

    DF_GATE = 30
    disease_signatures = {}
    for dc, syms in ds.items():
        scored = [(s, c * idf.get(s, 0)) for s, c in syms.items() if sym_df.get(s, 99) <= DF_GATE]
        scored.sort(key=lambda x: -x[1])
        sig = []
        for s, _ in scored[:8]:
            name = cp.get(s, "")
            if name and len(name) > 3:
                sig.append(name)
            if len(sig) >= 5: break
        disease_signatures[dc] = sig

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
        pain_intens=None; pain_speed=None; pain_precision=None; pain_present=False
        skin_locs=[]; skin_colors=[]; skin_props=[]; skin_present=False
        edema_locs=[]; edema_present=False
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
                elif "precis" in base and value: pain_precision=value
                elif base=="douleurxx": pain_present=True
                continue
            if "lesions_peau" in base:
                if "endroitducorps" in base and val_en: skin_locs.append(val_en)
                elif "couleur" in base and val_en: skin_colors.append(val_en)
                elif "desquame" in base: skin_props.append("peeling")
                elif "elevee" in base: skin_props.append("elevated")
                elif "plusqu1cm" in base: skin_props.append(">1cm")
                elif "prurit" in base: skin_props.append("pruritic")
                elif base=="lesions_peau": skin_present=True
                continue
            if "oedeme" in base:
                if "endroitducorps" in base and val_en: edema_locs.append(val_en)
                elif base=="oedeme": edema_present=True
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
            if pain_precision: pp.append(f"precision {pain_precision}/10")
            out.append("PAIN: " + "; ".join(pp))
        if skin_present or skin_locs:
            sp=[]
            if skin_locs: sp.append(f"location ({', '.join(skin_locs)})")
            if skin_colors: sp.append(f"color ({', '.join(skin_colors)})")
            if skin_props: sp.append(", ".join(skin_props))
            out.append("SKIN: " + "; ".join(sp))
        if edema_present or edema_locs:
            out.append("EDEMA: " + (", ".join(edema_locs) if edema_locs else "present"))
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

    SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"\n[2] 테스트 (SUBSET={SUBSET}, K={K})...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= SUBSET: break
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M"), "initial": row.get("INITIAL_EVIDENCE", "")})
    print(f"  {len(patients):,}명", flush=True)

    print("\n[3] Bayesian + prior top-K...", flush=True)
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
        topk = [dc for dc, _ in ranked[:K]]
        candidates.append({"patient": p, "true_dc": tdc, "topk": topk})
        if topk and topk[0] == tdc: baseline += 1
    in_topk = sum(1 for c in candidates if c["true_dc"] in c["topk"])
    print(f"  Baseline @1={100*baseline/n:.1f}%, top{K}={100*in_topk/n:.1f}%", flush=True)

    # Build 3 prompts per patient
    print("\n[4] 3-prompt ensemble 생성...", flush=True)
    prompts_p1 = []  # basic
    prompts_p2 = []  # symptom matching with KG sigs
    prompts_p3 = []  # rule out / differential
    for c in candidates:
        p = c["patient"]
        profile = patient_profile(p["evidences"])
        ie = p.get("initial", "")
        chief = ev_info.get(ie, {}).get("question_en", ie) if ie else "—"
        chief = re.sub(r"Do you |Have you |Are you ", "", chief).rstrip("?").strip()

        # Candidates with sigs
        cands_sig = []; cands_simple = []
        for i, dc in enumerate(c["topk"]):
            name = disease_full_name(cui2name.get(dc, dc))
            sig = disease_signatures.get(dc, [])
            sig_str = ", ".join(sig[:5]) if sig else ""
            cands_sig.append(f"{i+1}. {name} - typical features: {sig_str}")
            cands_simple.append(f"{i+1}. {name}")
        cands_sig_s = "\n".join(cands_sig)
        cands_simple_s = "\n".join(cands_simple)

        age_sex = f"{p['age']}yo {'Male' if p['sex']=='M' else 'Female'}"

        # P1: basic
        p1 = f"""Patient: {age_sex}
Chief complaint: {chief}

{profile}

Which diagnosis is MOST LIKELY?
{cands_simple_s}

Answer with ONLY the number (1-{len(c['topk'])})."""

        # P2: KG sigs
        p2 = f"""Patient: {age_sex}
Chief complaint: {chief}

{profile}

Compare the patient's symptoms to each candidate's typical features. Which best matches?

{cands_sig_s}

Answer with ONLY the number (1-{len(c['topk'])})."""

        # P3: differential
        p3 = f"""You are a clinician doing differential diagnosis for a {age_sex} patient.

Chief complaint: {chief}
{profile}

Candidates:
{cands_sig_s}

First eliminate candidates incompatible with the patient's symptoms. Then choose the BEST FIT.

Answer with ONLY the number (1-{len(c['topk'])})."""
        prompts_p1.append(p1); prompts_p2.append(p2); prompts_p3.append(p3)

    print("\n[5] vLLM 3 batches...", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.95, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    sampling = SamplingParams(temperature=0, max_tokens=16)

    all_picks = []
    for i, prompts in enumerate([prompts_p1, prompts_p2, prompts_p3]):
        print(f"\n  Batch {i+1}: {len(prompts)} prompts...", flush=True)
        convs = [[{"role": "user", "content": p}] for p in prompts]
        t0 = time.time()
        outputs = llm.chat(convs, sampling)
        print(f"    완료: {time.time()-t0:.0f}초", flush=True)
        picks = []
        for c, out in zip(candidates, outputs):
            m = re.search(r"(\d+)", out.outputs[0].text.strip())
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(c["topk"]):
                    picks.append(c["topk"][idx])
                    continue
            picks.append(c["topk"][0])
        all_picks.append(picks)

    print("\n[6] 평가 (개별 + majority vote)...", flush=True)
    nt = len(candidates)
    for i, picks in enumerate(all_picks):
        correct = sum(1 for c, p in zip(candidates, picks) if p == c["true_dc"])
        print(f"  P{i+1} alone: {100*correct/nt:.1f}%", flush=True)

    # Majority vote
    correct_maj = 0
    for i, c in enumerate(candidates):
        votes = Counter([all_picks[0][i], all_picks[1][i], all_picks[2][i]])
        winner = votes.most_common(1)[0][0]
        if winner == c["true_dc"]: correct_maj += 1
    print(f"\n  Majority vote: {100*correct_maj/nt:.1f}%", flush=True)

    print(f"\n{'='*80}", flush=True)
    print(f"v32 GTPA@1 = {100*correct_maj/nt:.1f}% (SUBSET={nt})", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
