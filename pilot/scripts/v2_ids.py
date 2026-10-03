"""Disease identifiers for corpus v2 (research design D13).

Disease nodes are the universe keys: a UMLS CUI when the disease has one, otherwise a MONDO ID.
Benchmarks name gold diseases by UMLS CUI, OMIM or Orphanet; `gold_keys` resolves any of them to
universe keys through UMLS (MRCONSO) and MONDO cross-references, so a gold disease that UMLS lacks
can still be matched through MONDO.
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache

V2 = "pilot/cache/v2"


@lru_cache(maxsize=1)
def universe() -> dict:
    return json.load(open(f"{V2}/disease_universe.json"))


@lru_cache(maxsize=1)
def _maps():
    uni = universe()
    st = json.load(open(f"{V2}/mondo_struct.json"))
    mondo2key: dict[str, str] = {}
    ext2key: dict[str, set[str]] = defaultdict(set)
    for mid, xs in st["xref"].items():
        cuis = [x[5:] for x in xs if x.startswith("UMLS:") and x[5:] in uni]
        key = cuis[0] if cuis else (mid if mid in uni else None)
        if key is None:
            continue
        mondo2key[mid] = key
        for x in xs:
            if x.startswith(("OMIM:", "ORPHA:")):
                ext2key[x].add(key)
    for mid in uni:
        if mid.startswith("MONDO:"):
            mondo2key.setdefault(mid, mid)
    om = json.load(open("pilot/cache/umls_maps/omim_orpha_to_cui.json"))
    for k, cuis in list(om["omim"].items()) + list(om["orpha"].items()):
        for c in cuis:
            if c in uni:
                ext2key[k].add(c)
    return mondo2key, ext2key, st["is_a"]


def gold_keys(ids: set[str]) -> set[str]:
    """Benchmark gold identifiers (UMLS CUI / OMIM:n / ORPHA:n / MONDO:n) -> universe keys."""
    uni = universe()
    mondo2key, ext2key, _ = _maps()
    out = set()
    for i in ids:
        if i in uni:
            out.add(i)
        elif i.startswith("MONDO:") and i in mondo2key:
            out.add(mondo2key[i])
        else:
            out |= ext2key.get(i, set())
    return out


def disease_isa() -> list[tuple[str, str]]:
    """(child_key, parent_key) from MONDO is_a and UMLS disease hierarchy, both ends in the universe."""
    uni = universe()
    mondo2key, _, isa = _maps()
    pairs = set()
    for c, p in isa:
        kc, kp = mondo2key.get(c), mondo2key.get(p)
        if kc and kp and kc != kp:
            pairs.add((kc, kp))
    children = json.load(open("pilot/cache/umls_maps/disease_children.json"))
    for p, cs in children.items():
        if p in uni:
            for c in cs:
                if c in uni and c != p:
                    pairs.add((c, p))
    return sorted(pairs)
