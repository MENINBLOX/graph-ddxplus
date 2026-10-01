"""Bridge an HPO composite phenotype to the base concept it modifies.

HPO fuses a qualifier and a concept into one term -- `Very short stature`, `Generalized hypotonia`,
`Psychomotor delay` -- while our KG stores the base concept and attaches the qualifier to the edge.
The two never meet, and on LIRICAL that is the dominant failure: the gold profile explains a median
28.6% of a patient's phenotypes and NOTHING at all for 13.7% of patients, with the most-missed
terms being exactly these composites. The base concepts are in the KG in abundance -- `short
stature` on 1,594 diseases against `very short stature` on 94, `hypotonia` on 1,999 against
`generalized hypotonia` on 137 -- so the knowledge is present and only the surface form differs.

Emits a bridge key shared by a composite and its base, in the same {cui2key} form the scorer's
other bridges use. Built from concept NAMES in the KG plus a modifier list; no benchmark file is
read, so it stays §5 benchmark-blind.
"""
from __future__ import annotations
import argparse, collections, json, re

# Leading qualifiers HPO stacks onto a base concept. Severity, extent, laterality and course --
# the same axes our edges carry as attributes, which is why they must not block a match.
MOD = (r"very|severe|severely|profound|profoundly|marked|markedly|mild|mildly|moderate|"
       r"moderately|slight|slightly|abnormally|generalized|generalised|global|diffuse|"
       r"widespread|bilateral|unilateral|left|right|progressive|progressively|chronic|acute|"
       r"recurrent|episodic|intermittent|persistent|complete|partial|total|proximal|distal|"
       r"focal|multifocal|primary|secondary|congenital|neonatal|infantile|juvenile|adult|"
       r"early|late|onset|mixed|isolated|non|increased|decreased|reduced|elevated|excessive|"
       r"delayed|abnormal|impaired|poor|frequent|severe")
_LEAD = re.compile(rf"^(?:(?:{MOD})\s+)+", re.I)
_TRAIL = re.compile(r"\s*\((?:hpo|disease|finding|symptom)[^)]*\)\s*$", re.I)


def base_form(name: str) -> str:
    """"Very short stature" -> "short stature"; unchanged when nothing leads."""
    s = _TRAIL.sub("", (name or "").strip().lower())
    s = _LEAD.sub("", s).strip()
    return re.sub(r"\s+", " ", s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", required=True)
    ap.add_argument("--out", default="pilot/cache/prod/hpo_base_bridge.json")
    ap.add_argument("--min-group", type=int, default=2,
                    help="only emit a key shared by at least this many CUIs; a key holding one "
                         "concept bridges nothing and would only enlarge the map")
    a = ap.parse_args()
    kg = json.load(open(a.kg, encoding="utf-8"))
    groups = collections.defaultdict(set)
    for c, i in kg["phenotypes"].items():
        nm = (i.get("name") or "").strip().lower()
        if not nm or len(nm) < 4:
            continue
        b = base_form(nm)
        if b and len(b) >= 4:
            groups["HB:" + b].add(c)
    cui2key = {c: k for k, cs in groups.items() if len(cs) >= a.min_group for c in cs}
    # compound CUI -> the CUI of its BASE concept, so a loader can emit the base alongside the
    # compound. Bridging alone makes the match common and therefore uninformative (`short stature`
    # sits on 1,594 diseases); pairing the base match with the modifier carried as an ATTRIBUTE is
    # what restores discrimination.
    name_of = {c: (i.get("name") or "").strip().lower() for c, i in kg["phenotypes"].items()}
    cui2base = {}
    for k, cs in groups.items():
        base_name = k[3:]
        exact = [c for c in cs if name_of.get(c) == base_name]
        if not exact:
            continue
        b = min(exact, key=lambda c: len(name_of.get(c, "")))
        for c in cs:
            if c != b and name_of.get(c) != base_name:
                cui2base[c] = b
    json.dump({"cui2key": cui2key, "n_keys": len({v for v in cui2key.values()}),
               "cui2base": cui2base},
              open(a.out, "w"))
    print(f"compound->base pairs: {len(cui2base)}")
    sizes = sorted((len(cs) for cs in groups.values() if len(cs) >= a.min_group), reverse=True)
    print(f"phenotypes {len(kg['phenotypes'])}; bridged CUIs {len(cui2key)}; "
          f"keys {len(sizes)}; largest group {sizes[0] if sizes else 0}")
    for probe in ("very short stature", "short stature", "generalized hypotonia", "hypotonia"):
        cs = [c for c, i in kg["phenotypes"].items()
              if (i.get("name") or "").lower().strip() == probe]
        print(f"  {probe:24} -> {[cui2key.get(c) for c in cs][:2]}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
