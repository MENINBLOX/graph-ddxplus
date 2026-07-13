#!/usr/bin/env python3
"""실패 패턴 분석: top10에 있지만 @1 실패하는 케이스의 confusion matrix.

목표: 어느 질환에서 가장 많이 실패하고, 어떤 다른 질환과 헷갈리는지 파악.
"""
from __future__ import annotations
import ast, csv, json, math, re
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
    print("실패 패턴 분석: Bayesian @1, top-K 확률, 혼동 행렬", flush=True)
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

    # 테스트 데이터 일부 (10K로 빠르게)
    print("\n[1] 테스트 (10K샘플)...", flush=True)
    patients = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 10000: break
            patients.append({"evidences": ast.literal_eval(row["EVIDENCES"]),
                             "pathology": row["PATHOLOGY"], "age": row.get("AGE", "30"),
                             "sex": row.get("SEX", "M")})

    print("\n[2] Bayesian per-rank 분포...", flush=True)
    rank_count = Counter()  # rank → count
    confusion = defaultdict(Counter)  # true_disease → predicted_top1
    fail_in_top10 = defaultdict(int)  # disease → fail count where in top10 but not @1
    in_top10_total = defaultdict(int)
    total_per_disease = Counter()

    for p in patients:
        tdc = fr2cui.get(p["pathology"])
        if not tdc: continue
        ps = text_match(p["evidences"])
        sc = {}
        for dc in dcs:
            s = ds.get(dc, {})
            if not s: sc[dc] = -1e6; continue
            tw = sum(s.values()) + len(all_s) * 0.1
            sc[dc] = sum(math.log((s[x]+0.1)/tw+1e-10) if x in s else math.log(0.1/tw+1e-10) for x in ps)
        ranked = sorted(sc.items(), key=lambda x: -x[1])
        ranks = {dc: i+1 for i, (dc, _) in enumerate(ranked)}
        true_rank = ranks.get(tdc, 99)
        rank_count[true_rank] += 1
        total_per_disease[tdc] += 1
        if ranked[0][0] != tdc:
            confusion[tdc][ranked[0][0]] += 1
            if true_rank <= 10:
                fail_in_top10[tdc] += 1
        if true_rank <= 10:
            in_top10_total[tdc] += 1

    n = sum(rank_count.values())
    print(f"\n  Rank 분포 ({n}명):", flush=True)
    cum = 0
    for r in sorted(rank_count.keys())[:15]:
        cum += rank_count[r]
        print(f"    rank {r:2d}: {rank_count[r]:>5} ({100*rank_count[r]/n:>5.1f}%) cum={100*cum/n:>5.1f}%", flush=True)

    print(f"\n  Top10 외 (실패): {n - cum} ({100*(n-cum)/n:.1f}%)", flush=True)

    print("\n[3] 가장 많이 실패하는 질환 (top10 안인데 @1 못하는):", flush=True)
    fail_rate = []
    for dc, fc in fail_in_top10.items():
        if total_per_disease[dc] >= 30:
            rate = fc / total_per_disease[dc]
            fail_rate.append((dc, fc, total_per_disease[dc], rate))
    fail_rate.sort(key=lambda x: -x[1])
    for dc, fc, tot, rate in fail_rate[:20]:
        dn = cui2name.get(dc, dc)[:35]
        # 가장 자주 헷갈리는 잘못된 답
        wrong = confusion[dc].most_common(3)
        wrong_str = " | ".join(f"{cui2name.get(d, d)[:25]}({c})" for d, c in wrong)
        print(f"    {dn:<35} fail={fc:>4}/{tot:>4} ({100*rate:>4.0f}%) → {wrong_str}", flush=True)

    print("\n[4] Confusion matrix top pairs (가장 빈번한 잘못 분류):", flush=True)
    pairs = []
    for tdc, wrongs in confusion.items():
        for wdc, cnt in wrongs.items():
            pairs.append((tdc, wdc, cnt))
    pairs.sort(key=lambda x: -x[2])
    for tdc, wdc, cnt in pairs[:25]:
        tn = cui2name.get(tdc, tdc)[:30]
        wn = cui2name.get(wdc, wdc)[:30]
        print(f"    {tn:<30} → {wn:<30} ({cnt})", flush=True)

    # 5. 각 질환의 KG 크기 / 증상 양 분석
    print("\n[5] KG-disease 증상 분포:", flush=True)
    for dc, _, tot, rate in fail_rate[:10]:
        dn = cui2name.get(dc, dc)[:30]
        sym_count = len(ds.get(dc, {}))
        top_syms = sorted(ds.get(dc, {}).items(), key=lambda x: -x[1])[:5]
        ts = ", ".join(f"{cp.get(s, s)[:20]}({c})" for s, c in top_syms)
        print(f"    {dn:<30} ({sym_count} syms): {ts}", flush=True)


if __name__ == "__main__":
    main()
