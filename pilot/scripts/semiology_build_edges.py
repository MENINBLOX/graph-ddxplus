"""Merge the semiology attribute IE (cause -> symptom -> attributes) into the KG as enriched
edge attributes. Disease-oriented sources (StatPearls/PMC) gave disease->phenotype edges with
mostly-EMPTY attrs; the symptom-organized Clinical Methods IE supplies the disease-SPECIFIC
attribute contrasts (angina chest pain = pressure/exertion/rest vs reflux = burning/recumbency/
antacid). This links both sides to CUIs and fills the empty attribute cells on MATCHING existing
edges (enrichment only -- does NOT add disease-symptom edges, so the base graph is unchanged;
the attribute layer is what improves).

Writes kg_merged3c_semio.json (KG copy with enriched edge attrs) + prints enrichment stats.
§3-safe: textbook IE, benchmark-blind, no disease-symptom mapping table.
"""
from __future__ import annotations
import json, sys, argparse
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
from umls_link import link as umls_link
from disease_link import link_disease_or_generalize
import onlykg_eval_unified as U

DISO = {"T047", "T048", "T191", "T019", "T046", "T184", "T033", "T037", "T020", "T190"}
SYMP = {"T184", "T033", "T047", "T046", "T048", "T190", "T191"}
ATTR_KEYS = ["character", "location", "radiation", "timing", "onset", "aggravating", "relieving"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg_merged3c.json")
    ap.add_argument("--ie", default="pilot/cache/prod/semiology_ie.json")
    ap.add_argument("--out", default="pilot/cache/prod/kg_merged3c_semio.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    a = ap.parse_args()

    kg = json.load(open(a.kg))
    ie = json.load(open(a.ie))
    kg_ph = set(kg["phenotypes"])

    # 1) link semiology causes+symptoms to CUIs -> overlay {(dcui, scui): merged attrs}
    _cache = {}
    def link(term, tuis):
        key = (term.lower(), tuple(sorted(tuis)))
        if key not in _cache:
            _cache[key] = umls_link(term, tuis)[0]
        return _cache[key]

    # the CAUSE is a disease name, so it must not go through the phenotype linker's head-noun
    # fallback: that sent 20% of causes into "Disease"/"Syndrome", attaching this textbook's
    # aggravating/relieving attributes to junk nodes. The SYMPTOM is a phrase and keeps umls_link.
    _dcache = {}
    def link_cause(term):
        if term not in _dcache:
            _dcache[term] = link_disease_or_generalize(term, DISO)[0]
        return _dcache[term]

    overlay = defaultdict(lambda: {k: "" for k in ATTR_KEYS})
    n_triples = 0
    for nbk, rec in ie.items():
        scui = link(rec["symptom"].split(" / ")[0], SYMP)
        if not scui:
            continue
        for c in rec["causes"]:
            dcui = link_cause(c["cause"])
            if not dcui:
                continue
            cell = overlay[(dcui, scui)]
            for k in ATTR_KEYS:
                v = str(c.get(k, "")).strip()
                if v and not cell[k]:
                    cell[k] = v
            n_triples += 1
    print(f"semiology triples linked: {n_triples} -> {len(overlay)} (dcui,scui) attr cells", flush=True)

    # 2) enrich matching KG edges. Resolve each disease's lineage so a cause CUI matches the KG
    #    disease node even at a neighbouring granularity.
    kg, prof, idf, dz_names = U.load_kg(a.kg)
    resolver = U.Resolver(prof, dz_names, json.load(open(a.children)))
    # disease-cluster membership: map every KG disease to the set of CUIs it resolves to
    dcui_to_kgd = defaultdict(set)
    for d in kg["diseases"]:
        for c in resolver.cluster(dz_names.get(d, ""), d) | {d}:
            dcui_to_kgd[c].add(d)

    kgj = json.load(open(a.kg))
    edge_index = defaultdict(list)   # (d,p) -> list of edge dicts
    for e in kgj["edges"]:
        edge_index[(e["d"], e["p"])].append(e)

    enriched = 0
    added_attr = defaultdict(int)
    for (dcui, scui), attrs in overlay.items():
        if scui not in kg_ph:
            continue
        for kgd in dcui_to_kgd.get(dcui, {dcui}):
            for e in edge_index.get((kgd, scui), []):
                ea = e.setdefault("attrs", {})
                for k in ATTR_KEYS:
                    if attrs[k] and not ea.get(k):
                        ea[k] = attrs[k]
                        added_attr[k] += 1
                enriched += 1

    json.dump(kgj, open(a.out, "w"))
    print(f"\nedges enriched: {enriched}")
    print("attribute cells added to existing edges:")
    for k in ATTR_KEYS:
        print(f"  {k:12} +{added_attr[k]}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
