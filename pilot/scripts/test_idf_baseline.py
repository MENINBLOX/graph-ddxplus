#!/usr/bin/env python3
"""IDF/PMI 가중 베이지안 - LLM 없이 빠른 평가.

목표: top10 84.7% → 95%, baseline @1 30% → ?

IDF: log(N / DF(s)) — 모든 질환에 흔한 증상 무시
PMI: log P(s|d) - log P(s) — 질환별 특이성
TF-IDF: 위 두 개 결합
"""
from __future__ import annotations
import ast, csv, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path
import ahocorasick

UMLS_DIR = Path("data/umls_extracted")
KG_CACHE = Path("pilot/results/kg_v3_cache.json")
NOISE = {'C0150312','C0442743','C0039082','C0221423','C1457887','C0205390','C0442804','C3839861','C0332157','C1457868','C0445223','C1272751','C0015663','C0277814','C5202885','C0153933','C0585362'}
STOPWORDS = {'does','have','your','you','the','and','for','are','with','that','this','from','been','were','being','which','their','than','other','about','into','over','some','only','very','also','just','more','most','such','much','will','would','could','should','make','like','time','when','what','where','how','who','all','each','every','both','few','any','not','can','may','her','his','its','our','they','them','then','had','has','him','but','one','two','way','day','did','get','got','let','say','she','too','use','yes','yet','now','new','old','see','own','why','try','ask','set','related','reason','consulting','significant','measured','thermometer','either','believe','racing','missing','beat','fast','irregularly','problems','situation','associated','inability','speak','trouble','keeping','opening','raising','annoying','else','body','somewhere','anywhere','nowhere','recently','currently','usually','often','sometimes','worse','better'}

EVIDENCE_MEDTERM = {
    "diplopie": ["diplopia", "double vision"],
    "flushing": ["flushing", "facial flushing", "erythema"],
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


def main():
    print("=" * 80, flush=True)
    print("IDF/PMI 가중 베이지안 빠른 평가 (LLM 없이)", flush=True)
    print("=" * 80, flush=True)

    # 로드
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

    # IDF: 각 증상이 몇 개의 질환에 등장하는지
    sym_df = Counter()  # symptom CUI → number of diseases
    for d, syms in ds.items():
        for s in syms:
            sym_df[s] += 1
    N_diseases = len(dcs)
    idf = {s: math.log(N_diseases / df) for s, df in sym_df.items()}

    # P(s) marginal: 전체 disease 중 s의 가중평균
    sym_total = Counter()  # 모든 d의 c(d, s) 합
    disease_total = Counter()  # 각 d의 sum of counts
    for d, syms in ds.items():
        for s, c in syms.items():
            sym_total[s] += c
            disease_total[d] += c
    grand = sum(sym_total.values())
    p_sym = {s: c / grand for s, c in sym_total.items()} if grand > 0 else {}

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

    # 테스트 데이터 (10K)
    print("\n[1] 10K 테스트 로드...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 10000: break
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M")})

    # 증상 매칭 (한 번만)
    matched = []
    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        ps = text_match(p["evidences"])
        matched.append({"true_dc": tdc, "ps": ps, "age": p["age"], "sex": p["sex"]})
    print(f"  {len(matched):,}명", flush=True)

    # === Strategy 0: 기존 Bayesian (baseline) ===
    print("\n[Strategy 0] 기존 Bayesian (baseline)", flush=True)
    t1 = t3 = t10 = 0
    for m in matched:
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in m["ps"])
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    n = len(matched)
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    # === Strategy 1: IDF weighting ===
    print("\n[Strategy 1] IDF-weighted log-likelihood", flush=True)
    t1 = t3 = t10 = 0
    for m in matched:
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            ll = 0
            for x in m["ps"]:
                w = idf.get(x, 1.0)  # IDF weight
                if x in s:
                    ll += w * math.log((s[x]+0.1)/tw+1e-10)
                else:
                    ll += w * math.log(0.1/tw+1e-10)
            sc[dc] = ll
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    # === Strategy 2: PMI (P(s|d) / P(s)) — 양수만 ===
    print("\n[Strategy 2] PMI (log P(s|d)/P(s))", flush=True)
    t1 = t3 = t10 = 0
    for m in matched:
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = disease_total[dc] + len(all_s) * 0.1
            ll = 0
            for x in m["ps"]:
                pmi = 0
                if x in s and x in p_sym:
                    p_sd = (s[x] + 0.1) / tw
                    pmi = math.log(p_sd / (p_sym[x] + 1e-10) + 1e-10)
                ll += pmi
            sc[dc] = ll
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    # === Strategy 3: PPMI ===
    print("\n[Strategy 3] PPMI (only positive)", flush=True)
    t1 = t3 = t10 = 0
    for m in matched:
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = disease_total[dc]
            ll = 0
            for x in m["ps"]:
                if x in s and x in p_sym and tw > 0:
                    p_sd = s[x] / tw
                    pmi = math.log(p_sd / p_sym[x] + 1e-10)
                    if pmi > 0: ll += pmi
            sc[dc] = ll
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    # === Strategy 4: TF-IDF cosine similarity ===
    print("\n[Strategy 4] TF-IDF cosine similarity", flush=True)
    t1 = t3 = t10 = 0
    # Pre-compute disease TF-IDF vectors
    d_vec = {}
    for dc, syms in ds.items():
        td = sum(syms.values()) or 1
        v = {}
        for s, c in syms.items():
            tf = c / td
            v[s] = tf * idf.get(s, 0)
        norm = math.sqrt(sum(x*x for x in v.values())) or 1
        d_vec[dc] = {s: x/norm for s, x in v.items()}

    for m in matched:
        if not m["ps"]:
            sc = {dc: -1e6 for dc in dcs}
        else:
            # patient TF-IDF
            tf_p = 1.0 / len(m["ps"])
            p_vec = {s: tf_p * idf.get(s, 0) for s in m["ps"]}
            p_norm = math.sqrt(sum(x*x for x in p_vec.values())) or 1
            p_vec = {s: x/p_norm for s, x in p_vec.items()}
            sc = {}
            for dc in dcs:
                v = d_vec.get(dc, {})
                cos = sum(p_vec[s] * v.get(s, 0) for s in p_vec)
                sc[dc] = cos
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    # === Strategy 5: Bayesian + IDF gate (drop very common syms) ===
    print("\n[Strategy 5] Bayesian + IDF gate (drop sym with df>30/49)", flush=True)
    DF_GATE = 30
    common = {s for s, df in sym_df.items() if df > DF_GATE}
    print(f"  Filtering {len(common)} common symptoms (>{DF_GATE} diseases)", flush=True)
    t1 = t3 = t10 = 0
    for m in matched:
        ps_filt = {x for x in m["ps"] if x not in common}
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            # Filtered total
            s_filt = {k: v for k, v in s.items() if k not in common}
            tw = sum(s_filt.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((s_filt[x]+0.1)/tw+1e-10) if x in s_filt else math.log(0.1/tw+1e-10) for x in ps_filt)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    # === Strategy 6: Combined Bayesian × IDF ===
    print("\n[Strategy 6] Bayesian + IDF weight, gate at df>20", flush=True)
    DF_GATE = 20
    t1 = t3 = t10 = 0
    for m in matched:
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            ll = 0
            for x in m["ps"]:
                df_x = sym_df.get(x, 1)
                if df_x > DF_GATE: continue  # skip very common
                w = math.log(N_diseases / df_x)
                if x in s:
                    ll += w * math.log((s[x]+0.1)/tw+1e-10)
                else:
                    ll += w * math.log(0.1/tw+1e-10)
            sc[dc] = ll
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    # === Strategy 7: BM25-like ===
    print("\n[Strategy 7] BM25-like scoring", flush=True)
    k1, b = 1.5, 0.75
    avgdl = sum(disease_total.values()) / len(disease_total)
    t1 = t3 = t10 = 0
    for m in matched:
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            dl = disease_total[dc]
            ll = 0
            for x in m["ps"]:
                if x not in s: continue
                f = s[x]
                idf_x = idf.get(x, 0)
                num = f * (k1 + 1)
                den = f + k1 * (1 - b + b * dl / avgdl)
                ll += idf_x * num / den
            sc[dc] = ll
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        if ranked[0][0] == m["true_dc"]: t1 += 1
        if m["true_dc"] in [r[0] for r in ranked[:3]]: t3 += 1
        if m["true_dc"] in [r[0] for r in ranked[:10]]: t10 += 1
    print(f"  @1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @10={100*t10/n:.1f}%", flush=True)

    print("\n" + "=" * 80, flush=True)


if __name__ == "__main__":
    main()
