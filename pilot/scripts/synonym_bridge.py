"""Collapse KG phenotypes that UMLS itself declares synonymous but keeps on separate CUIs.

Case-level forensics on the DDXPlus regression: our KG links "sore throat" to C0031350 and
"throat pain" to C0242429, and the HPO bridge cannot join them because HPO assigns each a different
term (HP:0025439 vs HP:0033050). A patient reporting one never matches a disease documented with the
other, and when re-normalization shifted the balance toward "sore throat", the throat-pain edge
collapsed from 173 diseases to 32 -- which is exactly where the lost rank-1 cases were (URTI, viral
pharyngitis, epiglottitis).

UMLS already states the equivalence: MRREL carries `SY` (synonymy) and `RQ` ("related and possibly
synonymous") between those two CUIs. This builds synonym groups from those relations, restricted to
clinical-finding semantic types, so the two sides of the graph can meet without inventing a mapping
of our own. Benchmark-blind and label-free (§3): the evidence is UMLS relations only.

Transitivity is applied but bounded -- a group that grows past --max-group is dropped rather than
allowed to swallow a whole semantic area, since `RQ` is looser than `SY`.

Output: {"cui2key": {cui: "SYN:<representative cui>"}, ...} -> synonym_bridge.json
"""
from __future__ import annotations
import argparse, collections, json

# clinical finding / symptom / disease types. A synonym link between two ANATOMY concepts, say,
# is not useful here and only risks merging unrelated phenotypes.
KEEP_TUI = {"T184", "T033", "T046", "T047", "T048", "T049", "T050", "T019", "T020", "T037",
            "T190", "T191", "T034"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrrel", default="data/umls_extracted/MRREL.RRF")
    ap.add_argument("--sty", default="pilot/cache/umls_maps/cui_sty.tsv")
    ap.add_argument("--kg", default="", help="restrict to phenotypes present in this KG")
    ap.add_argument("--rel", default="SY,RQ", help="MRREL REL values treated as synonymy")
    ap.add_argument("--max-group", type=int, default=8,
                    help="drop groups larger than this (RQ is loose; a runaway group would merge "
                         "clinically distinct findings)")
    ap.add_argument("--out", default="pilot/cache/prod/synonym_bridge.json")
    a = ap.parse_args()

    sty = {}
    for line in open(a.sty, encoding="utf-8"):
        q = line.rstrip("\n").split("\t")
        if len(q) >= 2:
            sty[q[0]] = q[1]
    scope = None
    if a.kg:
        scope = set(json.load(open(a.kg, encoding="utf-8"))["phenotypes"])
    rels = set(a.rel.split(","))

    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_edge = 0
    for line in open(a.mrrel, encoding="utf-8", errors="ignore"):
        p = line.split("|")
        if len(p) < 5 or p[3] not in rels:
            continue
        c1, c2 = p[0], p[4]
        if c1 == c2:
            continue
        if sty.get(c1) not in KEEP_TUI or sty.get(c2) not in KEEP_TUI:
            continue
        if scope is not None and (c1 not in scope or c2 not in scope):
            continue
        n_edge += 1
        r1, r2 = find(c1), find(c2)
        if r1 != r2:
            parent[r2] = r1

    groups = collections.defaultdict(set)
    for c in list(parent):
        groups[find(c)].add(c)
    kept = {g: cs for g, cs in groups.items() if 1 < len(cs) <= a.max_group}
    dropped = sum(1 for cs in groups.values() if len(cs) > a.max_group)

    cui2key = {}
    for g, cs in kept.items():
        key = "SYN:" + min(cs)
        for c in cs:
            cui2key[c] = key
    json.dump({"cui2key": cui2key, "n_groups": len(kept)}, open(a.out, "w"))

    sizes = sorted((len(cs) for cs in kept.values()), reverse=True)
    print(f"synonymy relations used ({a.rel}): {n_edge}")
    print(f"groups kept: {len(kept)}  (dropped {dropped} groups over size {a.max_group})")
    print(f"  concepts collapsed: {len(cui2key)}   largest kept group: {sizes[0] if sizes else 0}")
    if scope is not None:
        print(f"  KG phenotypes covered: {sum(1 for c in cui2key if c in scope)}/{len(scope)}")
    for probe in ("C0031350", "C0242429"):
        print(f"  {probe} -> {cui2key.get(probe)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
