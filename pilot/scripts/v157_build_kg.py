"""S4 KG BUILD — assemble the attribute knowledge graph from normalized IE records.

Input  = kg_records.json (v155 output): {disease_cui: {disease, phenotypes:[rec...]}}, where each
         rec has name_cui/_snomed/_tui and optional qualified attributes (location_cui, severity,
         onset, character, timing, aggravating, relieving).
Output = a bipartite attribute-KG (CLAUDE.md §11): Disease node --HAS_PHENOTYPE--> Phenotype CUI,
         single direction; a Disease CUI may also be another disease's phenotype target (history).
         Attributes ride on the EDGE as a qualified statement (GA4GH-Phenopackets style), NOT as
         separate nodes.

Principled processing (no magic thresholds):
  - §12 noise filter: drop phenotype CUIs whose UMLS semantic type is demographic/temporal/
    qualifier (STOP_TUI) — deterministic type membership, not a frequency cut.
  - #8 discriminativeness: per-phenotype IDF = log(N_disease / df) (Spärck Jones). Over-represented
    hubs (Pain/Fever) keep a near-zero weight rather than being removed; genuine noise is gone via
    STOP_TUI. See docs/evidence_weighting_dedup.md.
  - provenance retained per edge (thesis traceability axis).

Also emits the §12 topology report (degree distributions, hubs, singleton rate) — mandatory after
every KG build.

Usage: python v157_build_kg.py [--records pilot/cache/prod/kg_records.json] [--out ...kg.json]"""
from __future__ import annotations
import json, os, math, argparse
from collections import defaultdict, Counter

# demographic / temporal / qualifier / conceptual semantic types = structural noise (§12).
STOP_TUI = {"T098", "T099", "T100", "T096", "T097",  # population/age/family/group
            "T079",                                    # temporal concept
            "T078", "T077", "T170", "T171",            # idea/conceptual/intellectual product/language
            "T102", "T169",                            # group attribute / functional concept
            "T080", "T081",                            # qualitative/quantitative (keep OFF for findings)
            # Anatomical STRUCTURES are not findings. They reached the graph because the IE names a
            # body part when the source discusses it ("epiglottis" as a phenotype of epiglottitis)
            # and nothing filtered the type -- 681 phenotypes (1.7%) were structures. A site belongs
            # in the location attribute, which is where the anatomy axis already lives.
            "T017", "T018", "T021", "T022", "T023", "T024", "T025", "T026", "T029", "T030"}
ATTR_FIELDS = ["location_cui", "location_snomed", "location", "severity", "onset",
               "character", "timing", "aggravating", "relieving"]
# open-schema qualifiers are a LIST, not a string, so they are merged separately from the
# string-valued slots below
QUAL_FIELD = "qualifiers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="pilot/cache/prod/kg_records.json")
    ap.add_argument("--out", default="pilot/cache/prod/kg.json")
    ap.add_argument("--preds", default="",
                    help="v153 IE predictions, used only to record how much SOURCE TEXT each "
                         "disease was built from (documentation depth); see diseases[].n_src_chars")
    ap.add_argument("--keep-qual", action="store_true",
                    help="keep T080/T081-typed phenotypes (character concepts) instead of dropping")
    a = ap.parse_args()
    recs = json.load(open(a.records, encoding="utf-8"))
    src_chars = {}
    if a.preds:
        src_chars = {c: len(v.get("source", "") or "")
                     for c, v in json.load(open(a.preds, encoding="utf-8")).items()}
    stop = STOP_TUI - ({"T080", "T081"} if a.keep_qual else set())

    diseases, phenotypes = {}, {}
    edges = []
    df = Counter()                         # phenotype document frequency (distinct diseases)
    dropped_noise = Counter()
    seen_pair = set()
    edge_of = {}          # (d,p) -> its edge dict, so a repeat pair is merged in O(1)

    for dcui, obj in recs.items():
        dname = obj.get("disease", dcui)
        # DOCUMENTATION DEPTH, recorded as a first-class disease property. A missing edge in this
        # KG mostly means "no source we read mentioned it", not "the disease lacks that finding"
        # (median degree is 7). Without this field the scorer cannot tell a genuinely narrow
        # disease from an under-documented one, so profile size leaks into the ranking: enriching
        # 2.8% of diseases moved @3 for the *unenriched* patients too, via df/IDF. Downstream
        # scoring uses it to discount ABSENCE for thinly-documented diseases.
        diseases[dcui] = {"name": dname, "n_extracted": len(obj.get("phenotypes", []))}
        if dcui in src_chars:
            diseases[dcui]["n_src_chars"] = src_chars[dcui]
        for rec in obj.get("phenotypes", []):
            pcui = rec.get("name_cui")
            tui = rec.get("name_tui", "")
            if not pcui:
                continue                    # unlinked finding -> kept only as free-text elsewhere
            if tui in stop:
                dropped_noise[tui] += 1
                continue
            key = (dcui, pcui)
            attrs = {k: rec[k] for k in ATTR_FIELDS if rec.get(k)}
            if rec.get(QUAL_FIELD):
                attrs[QUAL_FIELD] = list(rec[QUAL_FIELD])
            if key in seen_pair:
                # Merge attributes into the existing edge (same disease-phenotype pair, §(a) dedup).
                # An attribute legitimately has SEVERAL values -- angina's chest pain is described
                # as "pressure-like, heaviness, tightness" in one source and "burning" in another --
                # and `update()` kept only the last. 9.5% of pairs are built from more than one
                # record, and for those a distinct value was being discarded (8,500 locations,
                # 1,252 severities, 1,087 characters). Values are unioned instead; matching reads
                # the text for controlled classes, so a joined value simply contributes all of them.
                e = edge_of[key]
                for k, v in attrs.items():
                    old = e["attrs"].get(k)
                    if k == QUAL_FIELD:
                        seen_q = {x["quote"] for x in (old or [])}
                        e["attrs"][k] = (old or []) + [x for x in v if x["quote"] not in seen_q]
                    elif not old:
                        e["attrs"][k] = v
                    elif str(v).lower() not in str(old).lower():
                        e["attrs"][k] = f"{old}; {v}"
                continue
            seen_pair.add(key)
            df[pcui] += 1
            phenotypes.setdefault(pcui, {"name": rec.get("name", ""), "tui": tui,
                                         "snomed": rec.get("name_snomed")})
            e = {"d": dcui, "p": pcui, "attrs": attrs}
            edges.append(e)
            edge_of[key] = e

    # #8 IDF weight (Spärck Jones); N = number of diseases
    N = max(1, len(diseases))
    for pcui, cnt in df.items():
        phenotypes[pcui]["df"] = cnt
        phenotypes[pcui]["idf"] = round(math.log(N / cnt), 4)

    kg = {"meta": {"n_diseases": len(diseases), "n_phenotypes": len(phenotypes),
                   "n_edges": len(edges), "n_dropped_noise": sum(dropped_noise.values())},
          "diseases": diseases, "phenotypes": phenotypes, "edges": edges}
    json.dump(kg, open(a.out, "w"))
    print(f"KG -> {a.out}")
    print(f"  diseases={len(diseases)}  phenotypes={len(phenotypes)}  edges={len(edges)}")
    print(f"  dropped noise phenotypes (STOP_TUI): {sum(dropped_noise.values())} {dict(dropped_noise)}")

    # === §12 topology report ===
    ddeg = Counter()
    for e in edges:
        ddeg[e["d"]] += 1
    pdeg = df
    degs = sorted(ddeg.values())
    med = degs[len(degs) // 2] if degs else 0
    low = sum(1 for v in ddeg.values() if v <= 5)
    print("\n=== TOPOLOGY (§12) ===")
    print(f"  disease degree: median={med}  min={degs[0] if degs else 0}  max={degs[-1] if degs else 0}"
          f"  (<=5 evidence: {low}/{len(ddeg)} = {low/max(1,len(ddeg)):.1%})")
    singletons = sum(1 for v in pdeg.values() if v == 1)
    print(f"  phenotype degree: singletons(df=1) {singletons}/{len(pdeg)} = {singletons/max(1,len(pdeg)):.1%}")
    hubs = [(c, n) for c, n in pdeg.items() if n > 1000]
    print(f"  hub phenotypes (df>1000): {len(hubs)}")
    for c, n in sorted(hubs, key=lambda x: -x[1])[:15]:
        print(f"    {c} {phenotypes[c]['name'][:32]:32} df={n} tui={phenotypes[c]['tui']} idf={phenotypes[c]['idf']}")
    # attribute coverage on edges
    acov = Counter()
    for e in edges:
        for k in ("location_cui", "severity", "onset", "character", "timing", "aggravating", "relieving"):
            if e["attrs"].get(k):
                acov[k] += 1
    print("  edge attribute coverage:", {k: f"{acov[k]}/{len(edges)} ({acov[k]/max(1,len(edges)):.0%})"
                                         for k in ("location_cui", "severity", "onset", "character")})


if __name__ == "__main__":
    main()
