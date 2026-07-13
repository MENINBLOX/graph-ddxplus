#!/usr/bin/env python3
"""KG 크기 정규화/min-count 필터/엔트로피 보정 베이지안.

발견: Localized edema는 636 syms (junk attractor)
가설: KG 크기를 정규화하면 baseline @1 올라갈 것
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


def main():
    print("=" * 80, flush=True)
    print("KG 정규화 / min-count / 엔트로피 보정 비교", flush=True)
    print("=" * 80, flush=True)

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

    # IDF / DF
    sym_df = Counter()
    for d, syms in ds.items():
        for s in syms: sym_df[s] += 1
    N_diseases = len(dcs)
    idf = {s: math.log(N_diseases / df) for s, df in sym_df.items()}

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

    # 10K 테스트
    print("\n[1] 10K 테스트 로드...", flush=True)
    matched = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 10000: break
            tdc = fr2cui.get(row["PATHOLOGY"])
            if not tdc: continue
            ps = text_match(ast.literal_eval(row["EVIDENCES"]))
            matched.append({"true_dc": tdc, "ps": ps,
                            "age": row.get("AGE", "30"), "sex": row.get("SEX", "M")})
    print(f"  {len(matched):,}명", flush=True)

    def eval_(scorer):
        t1 = t3 = t5 = t10 = 0
        for m in matched:
            sc = scorer(m)
            ranked = sorted(sc.items(), key=lambda x: -x[1])
            top10 = [r[0] for r in ranked[:10]]
            if top10[0] == m["true_dc"]: t1 += 1
            if m["true_dc"] in top10[:3]: t3 += 1
            if m["true_dc"] in top10[:5]: t5 += 1
            if m["true_dc"] in top10: t10 += 1
        n = len(matched)
        return f"@1={100*t1/n:.1f}% @3={100*t3/n:.1f}% @5={100*t5/n:.1f}% @10={100*t10/n:.1f}%"

    # === Strategy 0: Baseline ===
    def s0(m):
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in m["ps"])
        return sc
    print(f"\n[S0] Baseline                          : {eval_(s0)}", flush=True)

    # === Min-count filter (ds 정리: count>=N만 유지) ===
    for MIN_C in [2, 3, 5]:
        ds_filt = {dc: {s: c for s, c in syms.items() if c >= MIN_C} for dc, syms in ds.items()}
        def make_s(ds_f):
            def s(m):
                sc = {}
                for dc in dcs:
                    sy = ds_f.get(dc, {})
                    if not sy: sc[dc] = -1e6; continue
                    tw = sum(sy.values()) + len(all_s) * 0.1
                    sc[dc] = sum(math.log((sy[x]+0.1)/tw+1e-10) if x in sy else math.log(0.1/tw+1e-10) for x in m["ps"])
                return sc
            return s
        print(f"[S1.{MIN_C}] min count={MIN_C} (filter rare in d-KG)  : {eval_(make_s(ds_filt))}", flush=True)

    # === DF gate: 너무 흔한 증상 제거 ===
    for DF_GATE in [40, 35, 30, 25, 20, 15]:
        common = {s for s, df in sym_df.items() if df > DF_GATE}
        ds_filt = {dc: {s: c for s, c in syms.items() if s not in common} for dc, syms in ds.items()}
        n_common = len(common)
        def make_s(ds_f):
            def s(m):
                ps = {x for x in m["ps"] if x not in common}
                sc = {}
                for dc in dcs:
                    sy = ds_f.get(dc, {})
                    if not sy: sc[dc] = -1e6; continue
                    tw = sum(sy.values()) + len(all_s) * 0.1
                    sc[dc] = sum(math.log((sy[x]+0.1)/tw+1e-10) if x in sy else math.log(0.1/tw+1e-10) for x in ps)
                return sc
            return s
        print(f"[S2.{DF_GATE}] DF≤{DF_GATE} ({n_common} common dropped) : {eval_(make_s(ds_filt))}", flush=True)

    # === Combined: min count + DF gate ===
    print("\n[Combined] min_count=2, DF gate=30", flush=True)
    common = {s for s, df in sym_df.items() if df > 30}
    ds_clean = {dc: {s: c for s, c in syms.items() if c >= 2 and s not in common} for dc, syms in ds.items()}
    def s_clean(m):
        ps = {x for x in m["ps"] if x not in common}
        sc = {}
        for dc in dcs:
            sy = ds_clean.get(dc, {})
            if not sy: sc[dc] = -1e6; continue
            tw = sum(sy.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((sy[x]+0.1)/tw+1e-10) if x in sy else math.log(0.1/tw+1e-10) for x in ps)
        return sc
    print(f"[S3] cleaned KG (min=2, DF≤30)            : {eval_(s_clean)}", flush=True)

    # === KG size penalty (length normalization) ===
    print("\n[Length norm]", flush=True)
    for LAMBDA in [0.0, 0.1, 0.3, 0.5, 1.0]:
        d_size = {dc: len(ds.get(dc, {})) for dc in dcs}
        def make_s(L):
            def s(m):
                sc = {}
                for dc in dcs:
                    sy = ds.get(dc, {})
                    if not sy: sc[dc] = -1e6; continue
                    tw = sum(sy.values()) + len(all_s) * 0.1
                    base = sum(math.log((sy[x]+0.1)/tw+1e-10) if x in sy else math.log(0.1/tw+1e-10) for x in m["ps"])
                    sc[dc] = base - L * math.log(d_size[dc] or 1)
                return sc
            return s
        print(f"[S4.λ={LAMBDA:>3.1f}] log-likelihood - λ·log|KG|        : {eval_(make_s(LAMBDA))}", flush=True)

    # === Disease-symptom probability normalized (P(s|d) only, no smoothing for missing) ===
    print("\n[Pure positive matching]", flush=True)
    def s_pos(m):
        sc = {}
        for dc in dcs:
            sy = ds.get(dc, {})
            if not sy: sc[dc] = -1e6; continue
            tw = sum(sy.values()) or 1
            ll = 0
            for x in m["ps"]:
                if x in sy: ll += math.log(sy[x] / tw + 1e-10)
            sc[dc] = ll
        return sc
    print(f"[S5] only positive matches                : {eval_(s_pos)}", flush=True)

    # === Hybrid: positive + IDF + KG size penalty ===
    print("\n[Hybrid IDF + length norm]", flush=True)
    for L in [0.5, 1.0, 1.5, 2.0]:
        d_size = {dc: len(ds.get(dc, {})) for dc in dcs}
        def make_s(LX):
            def s(m):
                sc = {}
                for dc in dcs:
                    sy = ds.get(dc, {})
                    if not sy: sc[dc] = -1e6; continue
                    tw = sum(sy.values()) or 1
                    ll = 0
                    for x in m["ps"]:
                        if x in sy:
                            ll += idf.get(x, 1.0) * math.log(sy[x] / tw + 1e-10)
                    sc[dc] = ll - LX * math.log(d_size[dc] or 1)
                return sc
            return s
        print(f"[S6.{L:>3.1f}] IDF·posLog - λ·log|KG|             : {eval_(make_s(L))}", flush=True)

    # === Ranked positive count (just count matched syms, weighted by IDF) ===
    print("\n[Match count IDF]", flush=True)
    def s_count(m):
        sc = {}
        for dc in dcs:
            sy = ds.get(dc, {})
            ll = sum(idf.get(x, 1.0) for x in m["ps"] if x in sy)
            sc[dc] = ll
        return sc
    print(f"[S7] sum IDF(s) for s in patient ∩ disease: {eval_(s_count)}", flush=True)

    print("\n" + "=" * 80, flush=True)


if __name__ == "__main__":
    main()
