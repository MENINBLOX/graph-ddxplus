"""HPO-native phenotype linker + is-a DAG, built from the HPO ontology (hp.json).

The redesign root-cause fix (2026-07-22): our old pipeline linked IE finding names to arbitrary
UMLS CUIs (MRXNS), so (a) near-synonyms landed on DIFFERENT CUIs than the benchmark's HPO-encoded
evidence ("poor weight gain" vs "failure to thrive"), and (b) off-topic case-report fragments
became spurious CUIs (profile contamination). Normalizing phenotypes to HPO fixes BOTH: HPO
synonyms collapse near-synonyms onto one term, the is-a DAG bridges the rest by shared ancestor,
and non-phenotype text simply fails to link (a free quality filter).

`link(name)` -> HP:####### or None (label/synonym exact match after morphological normalization).
`ancestors(hp, cap)` -> set of is-a ancestors (for Phenomizer-style hierarchy matching).
Loads once from data/medkg/hp.json; caches the compiled index to hpo_index.json.
"""
from __future__ import annotations
import json, os, re, functools

HP_JSON = "data/medkg/hp.json"
CACHE = "data/medkg/hpo_index.json"
_PLURAL = re.compile(r"(ies|es|s)$")


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    toks = [t for t in s.split() if t not in ("the", "a", "an", "of", "with", "and", "in")]
    # light morphology: singularize each token (abnormalities->abnormalit.., seizures->seizure)
    toks = [re.sub(r"ies$", "y", t) if t.endswith("ies") else
            re.sub(r"(ses|xes)$", lambda m: m.group(0)[:-2], t) if re.search(r"(ses|xes)$", t) else
            t[:-1] if t.endswith("s") and len(t) > 4 else t for t in toks]
    return " ".join(toks)


def _build():
    d = json.load(open(HP_JSON))
    g = d["graphs"][0]
    idx, label, parents = {}, {}, {}
    for n in g["nodes"]:
        nid = n.get("id", "")
        if "HP_" not in nid:
            continue
        hp = "HP:" + nid.split("HP_")[1]
        lbl = n.get("lbl", "")
        if not lbl:
            continue
        label[hp] = lbl
        idx.setdefault(_norm(lbl), hp)
        for s in n.get("meta", {}).get("synonyms", []):
            # only exact/broad/narrow lexical synonyms (skip related that drift)
            if s.get("pred") in ("hasExactSynonym", "hasNarrowSynonym", "hasBroadSynonym", None):
                idx.setdefault(_norm(s["val"]), hp)
    for e in g["edges"]:
        if e.get("pred") == "is_a" and "HP_" in e.get("sub", "") and "HP_" in e.get("obj", ""):
            c = "HP:" + e["sub"].split("HP_")[1]
            p = "HP:" + e["obj"].split("HP_")[1]
            parents.setdefault(c, []).append(p)
    return {"idx": idx, "label": label, "parents": parents}


@functools.lru_cache(maxsize=1)
def _load():
    if os.path.exists(CACHE) and os.path.getmtime(CACHE) >= os.path.getmtime(HP_JSON):
        return json.load(open(CACHE))
    data = _build()
    json.dump(data, open(CACHE, "w"))
    return data


def link(name: str):
    return _load()["idx"].get(_norm(name))


def label(hp: str) -> str:
    return _load()["label"].get(hp, hp)


def ancestors(hp: str, cap: int = 5) -> set:
    par = _load()["parents"]
    seen, fr, d = set(), [hp], 0
    while fr and d < cap:
        nx = []
        for x in fr:
            for p in par.get(x, ()):
                if p not in seen:
                    seen.add(p); nx.append(p)
        fr, d = nx, d + 1
    return seen


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        h = link(a)
        print(f"{a!r} -> {h} ({label(h) if h else 'NO LINK'})")
