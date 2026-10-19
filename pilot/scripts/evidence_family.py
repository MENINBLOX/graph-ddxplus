"""Collapse the many CUIs that denote ONE piece of clinical evidence into a single key.

The association IE links each item correctly, but a source can phrase the same history item a dozen
ways and UMLS gives each phrasing its own concept:

    travel   C0489542 travel history · C2919443 recent travel · C0040802 travel ·
             C4554677 international travel · C3489667 air travel
    smoking  C0037369 smoking · C0700219 cigarette smoking · C0543414 tobacco use ·
             C0337664 smokers · C0453996 tobacco smoking · C4721824 smoking history

A benchmark picks exactly one of them -- DDXPlus asks C0489542 -- so our evidence lands beside the
question instead of on it. "Travel History" alone accounts for 2,857 missed DDXPlus evidence
instances, the single largest gap after the association IE. UMLS cannot fix it: MRREL relates
neither C0489542-C0040802 nor C0040336-C0037369.

The key is the concept's content words minus the qualifiers that only say HOW the history was taken
("history", "recent", "current", "status", "use", "exposure", ...). So "travel history", "recent
travel" and "travel" all key to `travel`, while "air travel" keys separately because `air` is
content, not a qualifier.

Benchmark-blind: built from UMLS names and our own extracted vocabulary; no benchmark list is read.

Output: {"cui2key": {cui: "EV:<key>"}} -> evidence_family.json
"""
from __future__ import annotations
import argparse, collections, json, re

# words that describe how a fact was ascertained, not what it is
_QUALIFIER = {
    "history", "histories", "recent", "recently", "current", "currently", "past", "prior",
    "previous", "status", "use", "usage", "used", "using", "exposure", "exposed", "of", "the",
    "a", "an", "to", "in", "on", "at", "with", "and", "or", "for", "by", "finding", "findings",
    "disorder", "disorders", "syndrome", "problem", "problems", "abuse", "dependence",
    "consumption", "intake", "habit", "habits", "behaviour", "behavior", "positive", "known",
    "documented", "reported", "self", "patient", "personal", "chronic", "acute", "nos",
}
# semantic types that can carry patient evidence (mirrors v154's EVIDENCE_TUI)
EVIDENCE_TUI = {"T047", "T048", "T046", "T191", "T019", "T020", "T037", "T190", "T033", "T184",
                "T061", "T060", "T121", "T109", "T195", "T200", "T054", "T055", "T056", "T097",
                "T098", "T031", "T204", "T007", "T005", "T040", "T073", "T170", "T201"}


def key_of(name: str) -> str | None:
    """Concept name -> its evidence key, or None when nothing but qualifiers remains."""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _QUALIFIER and len(w) > 2]
    if not words:
        return None
    # order-insensitive so "smoking of tobacco" and "tobacco smoking" agree
    return "EV:" + " ".join(sorted(set(words)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrconso", default="data/umls_extracted/MRCONSO.RRF")
    ap.add_argument("--sty", default="pilot/cache/umls_maps/cui_sty.tsv")
    ap.add_argument("--kg", default="", help="restrict to CUIs present in this KG (+ its evidence)")
    ap.add_argument("--min-group", type=int, default=2,
                    help="only emit keys shared by at least this many CUIs (a singleton key adds "
                         "nothing and would only bloat the map)")
    ap.add_argument("--out", default="pilot/cache/prod/evidence_family.json")
    a = ap.parse_args()

    sty = {}
    for line in open(a.sty, encoding="utf-8"):
        q = line.rstrip("\n").split("\t")
        if len(q) >= 2:
            sty[q[0]] = q[1]
    scope = None
    if a.kg:
        kg = json.load(open(a.kg, encoding="utf-8"))
        scope = set(kg["phenotypes"]) | set(kg["diseases"])

    # one key per CUI, from its shortest English name (the least qualified surface form)
    best: dict[str, str] = {}
    for line in open(a.mrconso, encoding="utf-8", errors="ignore"):
        p = line.split("|")
        if len(p) < 15 or p[1] != "ENG":
            continue
        c = p[0]
        if sty.get(c) not in EVIDENCE_TUI:
            continue
        if scope is not None and c not in scope:
            continue
        s = p[14]
        if c not in best or len(s) < len(best[c]):
            best[c] = s

    cui2key, groups = {}, collections.defaultdict(set)
    for c, nm in best.items():
        k = key_of(nm)
        if k:
            groups[k].add(c)
    for k, cs in groups.items():
        if len(cs) >= a.min_group:
            for c in cs:
                cui2key[c] = k
    json.dump({"cui2key": cui2key, "n_keys": len({v for v in cui2key.values()})}, open(a.out, "w"))

    sizes = sorted((len(cs) for cs in groups.values() if len(cs) >= a.min_group), reverse=True)
    print(f"evidence-typed CUIs read: {len(best)}")
    print(f"keys shared by >={a.min_group} CUIs: {len(sizes)}   CUIs collapsed: {len(cui2key)}")
    print(f"  largest group: {sizes[0] if sizes else 0}")
    for probe in ("C0489542", "C0040802", "C2919443", "C0040336", "C0037369", "C0453996"):
        print(f"  {probe} -> {cui2key.get(probe)}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
