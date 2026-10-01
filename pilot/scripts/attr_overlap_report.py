"""Report the quantity that actually gates the attribute channel: BOTH-SIDES overlap.

Per-side attribute fill was misleading us for a long time. The KG held 21,361 Clinical-course
values and the benchmark's patients held 940, yet the two never once described the SAME
(disease, finding) pair -- so the channel scored exactly zero no matter how much either side was
enriched. Fill rates cannot show that; only the intersection can.

For each HPO modifier family (plus the UMLS anatomy axis this ontology does not cover) it reports:
  patient has            -- patient findings stating a value in that family
  ... and edge exists    -- of those, how many the gold disease has an edge for at all
  ... and KG also states -- of those, how many the edge also qualifies  <- the real ceiling
  agree                  -- of those, how many actually match

Run it before and after an extraction change: "patient has" is fixed by the benchmark, so the only
number an IE change can move is the third one.

Usage: python attr_overlap_report.py --kg KG.json [--queries nlice_queries_v2.json]
"""
from __future__ import annotations
import argparse, collections, json, sys
sys.path.insert(0, "pilot/scripts")
import hpo_modifier
from hpo_family_ablation import hpo_terms, famname


def terms_of(obj, voc):
    t = hpo_terms(obj, voc)
    loc = obj.get("location_cui")
    return t | ({"LOC:" + loc} if loc else set())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--queries", default="pilot/cache/prod/nlice_queries_v2.json")
    a = ap.parse_args()
    voc = hpo_modifier.load()

    kg = json.load(open(a.kg, encoding="utf-8"))
    prof = collections.defaultdict(dict)
    for e in kg["edges"]:
        prof[e["d"]][e["p"]] = terms_of(e.get("attrs", {}) or {}, voc)

    Q = json.load(open(a.queries, encoding="utf-8"))
    pat = collections.Counter(); edge = collections.Counter()
    both = collections.Counter(); agree = collections.Counter()
    n_f = n_edge = 0
    for pt in Q["patients"]:
        pf = prof.get(pt["true_cui"])
        for f in pt["findings"]:
            n_f += 1
            t = terms_of(f, voc)
            fams = {famname(h, voc) for h in t}
            for fam in fams:
                pat[fam] += 1
            if pf is None or f["cui"] not in pf:
                continue
            n_edge += 1
            cell = pf[f["cui"]]
            kf = {famname(h, voc) for h in cell}
            for fam in fams:
                edge[fam] += 1
                if fam in kf:
                    both[fam] += 1
                    if {h for h in t if famname(h, voc) == fam} & \
                       {h for h in cell if famname(h, voc) == fam}:
                        agree[fam] += 1

    print(f"patient findings {n_f}; with an edge on the gold disease {n_edge} "
          f"({100*n_edge/max(1,n_f):.0f}%)\n")
    print(f"{'family':26}{'patient has':>12}{'+edge':>8}{'+KG states':>12}{'agree':>8}")
    for fam, n in pat.most_common():
        print(f"  {fam[:24]:26}{n:12}{edge[fam]:8}{both[fam]:12}{agree[fam]:8}")
    tot_both = sum(both.values())
    print(f"\n  total matchable attribute opportunities: {tot_both}"
          f"   (this is the number an IE change has to move)")


if __name__ == "__main__":
    main()
