"""Use DDXPlus's per-symptom ATTRIBUTES (the thesis's core signal) for diagnosis.

DDXPlus encodes Bates attributes on each symptom: carac=character, endroitducorps/irrad=
location, intens=severity(NRS 0-10), soudain=onset. We were collapsing these to the base
CUI (all pain -> "Pain") and throwing the discriminative signal away. Here we parse them,
normalize (location->CUI, severity->{mild,mod,severe}, onset->{sudden,gradual}), and add an
attribute-match bonus when a patient finding's attribute equals the KG disease->phenotype
edge's qualified attribute -- exactly the attribute-based diagnosis this thesis proposes.
Compares base overlap vs +attributes (leave-in), KG-only, §3 safe.
"""
import sys, json, csv, ast, re, math
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U
import onlykg_eval_unified2 as U2
from umls_link import link as umls_link

LOC_TUI = {"T023", "T029", "T030", "T024", "T025", "T018", "T031", "T022", "T017"}
LAMBDA = 1.0
SUBQ = {"carac": "character", "endroitducorps": "location", "irrad": "location",
        "intens": "severity", "soudain": "onset"}


def norm(s): return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    kg, prof, idf, dz_names = U.load_kg("pilot/cache/prod/kg.json")
    children = json.load(open("pilot/cache/umls_maps/disease_children.json"))
    resolver = U.Resolver(prof, dz_names, children)
    # KG edge attributes per (disease, phenotype)
    edge_attr = defaultdict(dict)
    for e in json.load(open("pilot/cache/prod/kg.json"))["edges"]:
        edge_attr[e["d"]][e["p"]] = e.get("attrs", {})

    umap = json.load(open("data/ddxplus/umls_mapping.json"))["mapping"]
    dmap = json.load(open("data/ddxplus/disease_umls_mapping.json"))["mapping"]
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    ev_def = json.load(open("data/ddxplus/release_evidences_en.json"))
    fr2en = {}
    for info in ev_def.values():
        for mm in info.get("value_meaning", {}).values():
            if mm.get("fr") and mm.get("en") and mm["en"] != "NA":
                fr2en[norm(mm["fr"])] = mm["en"]
    fr2en_dis = {info.get("cond-name-fr") or info.get("condition_name"): en for en, info in cond.items()}

    STOPLOC = {"the", "of", "a", "region", "area", "part", "left", "right"}
    def loc_toks(text):
        return {w for w in re.findall(r"[a-z]+", (text or "").lower())
                if w not in STOPLOC and len(w) > 2}

    loc_cache = {}
    def loc_en(val):
        if val not in loc_cache:
            loc_cache[val] = fr2en.get(norm(val.replace("_", " ")), "")
        return loc_cache[val]

    def parse_evidences(evs):
        """-> {base_cui: {'location':set(cui), 'severity':str, 'onset':str}}"""
        find = defaultdict(lambda: {"location": set(), "severity": None, "onset": None})
        base_present = set()
        for e in evs:
            left = e.split("_@_")[0]
            val = e.split("_@_")[1] if "_@_" in e else None
            base, subq = left, None
            if left not in umap:
                parts = left.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in SUBQ:
                    base, subq = parts[0], SUBQ[parts[1]]
            cui = umap.get(base, {}).get("cui")
            if not cui:
                continue
            base_present.add(cui)
            if subq == "location" and val:
                find[cui]["location"] |= loc_toks(loc_en(val))
            elif subq == "severity" and val is not None:
                try:
                    v = int(val); find[cui]["severity"] = (
                        "mild" if v <= 3 else "moderate" if v <= 6 else "severe")
                except ValueError:
                    pass
            elif subq == "onset" and val is not None:
                try:
                    find[cui]["onset"] = "sudden" if int(val) >= 3 else "gradual"
                except ValueError:
                    pass
        return base_present, find

    cand = {en: (dmap[en].get("name_en", en), dmap[en]["umls_cui"])
            for en in dmap if dmap[en].get("umls_cui")}
    # cluster profiles + per-phenotype merged edge attrs
    prof_of = {}; attr_of = {}
    for k, (nm, cu) in cand.items():
        cl = resolver.cluster(nm, cu)
        prof_of[k] = U.merged_profile(cl, prof)
        pa = {}
        for c in cl:
            for p, a in edge_attr.get(c, {}).items():
                if a:
                    pa.setdefault(p, {}).update(a)
        attr_of[k] = pa

    keys = list(cand)
    def evaluate(use_attr):
        n = c1 = c3 = c5 = c10 = 0
        with open("data/ddxplus/release_test_patients.csv") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if n >= 3000:
                    break
                en = fr2en_dis.get(row["PATHOLOGY"])
                if en not in prof_of:
                    continue
                base_cuis, finds = parse_evidences(ast.literal_eval(row["EVIDENCES"]))
                if not base_cuis:
                    continue
                sc = {}
                for k in keys:
                    pf = prof_of[k]; pa = attr_of[k]; s = 0.0
                    for c in base_cuis:
                        if c in pf:
                            w = idf.get(c, 0.0); s += w
                            if use_attr and c in pa:
                                fa = finds.get(c) or {}; ea = pa[c]
                                if fa.get("location") and (fa["location"] & loc_toks(ea.get("location"))):
                                    s += LAMBDA * w
                                if fa.get("severity") and ea.get("severity") == fa["severity"]:
                                    s += LAMBDA * w
                                if fa.get("onset") and ea.get("onset") == fa["onset"]:
                                    s += LAMBDA * w
                    sc[k] = s / max(len(pf) ** 0.5, 1)
                ranked = sorted(keys, key=lambda k: -sc[k])
                r = ranked.index(en) + 1
                n += 1; c1 += r == 1; c3 += r <= 3; c5 += r <= 5; c10 += r <= 10
        return n, 100*c1/n, 100*c3/n, 100*c5/n, 100*c10/n

    for ua in (False, True):
        n, a1, a3, a5, a10 = evaluate(ua)
        tag = "+attributes" if ua else "base       "
        print(f"  {tag}  n={n}  @1={a1:.1f} @3={a3:.1f} @5={a5:.1f} @10={a10:.1f}")


if __name__ == "__main__":
    main()
