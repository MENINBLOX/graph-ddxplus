#!/usr/bin/env python3
"""HDF1 (Hierarchical Diagnostic F1) metric, ICD-10 ancestral expansion.

Re-implementation of the metric described in arXiv:2510.03700 (no public code).

Idea
----
A predicted ICD-10 code that is wrong at the 5-char level may still be
correct at the chapter or 3-char level. Standard exact-match F1 throws all
that credit away. HDF1 computes precision and recall over the *ancestral
chain* of each code with depth-decaying weights.

Implementation
--------------
ICD-10 ancestry levels we use (most specific first):
   level 0 : exact code (up to 7 chars)
   level 1 : 5-char block          e.g. J44.10 → J44.10
   level 2 : 4-char block          e.g. J44.10 → J44.1
   level 3 : 3-char category       e.g. J44.10 → J44
   level 4 : chapter letter        e.g. J44.10 → J

The set of ancestors for a code c is ANC(c) = {c, c[:5], c[:4], c[:3], c[:1]}.
Each ancestor a contributes weight w(a) = decay ** depth(a).

For a predicted code p (top-1) and true code t:
   match_weight(p, t)  = sum over shared ancestors of (w_p * w_t)
   self_weight(c)      = sum over c's ancestors of (w * w)
   precision           = match_weight / self_weight(p)
   recall              = match_weight / self_weight(t)
   F1                  = 2 PR / (P + R)

For a top-k list this is computed as the *max F1* over the k predictions
(takes the best alignment). Supports CUI input by mapping CUI -> ICD-10 via
data/ddxplus/disease_icd10_cui_mapping.json.

Code is case-insensitive; dots are stripped for parsing then put back when
slicing.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

DEFAULT_DECAY = 0.5
DEFAULT_MAPPING_PATH = Path("/home/max/Graph-DDXPlus/data/ddxplus/"
                            "disease_icd10_cui_mapping.json")


def _canonicalize(code: str) -> str:
    """Upper-case and drop dots: 'j44.10' → 'J4410'."""
    return code.upper().replace(".", "").strip()


def icd10_ancestors(code: str) -> list[str]:
    """Return ancestral chain (most specific first), no dots, no duplicates."""
    c = _canonicalize(code)
    if not c:
        return []
    seen = []
    for n in (len(c), 5, 4, 3, 1):
        if n <= len(c):
            a = c[:n]
            if a and a not in seen:
                seen.append(a)
    return seen


def _ancestor_weights(code: str, decay: float) -> dict[str, float]:
    """Each ancestor's weight = decay ** depth_from_full_code."""
    chain = icd10_ancestors(code)
    return {a: decay ** depth for depth, a in enumerate(chain)}


def hdf1_pair(pred: str, true: str, decay: float = DEFAULT_DECAY) -> dict:
    """HDF1 for a single (pred, true) ICD-10 pair."""
    if not pred or not true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    wp = _ancestor_weights(pred, decay)
    wt = _ancestor_weights(true, decay)
    overlap = sum(wp[a] * wt[a] for a in wp if a in wt)
    self_p = sum(v * v for v in wp.values()) or 1e-9
    self_t = sum(v * v for v in wt.values()) or 1e-9
    p = overlap / self_p
    r = overlap / self_t
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def hdf1_topk(preds: Iterable[str], true: str,
              decay: float = DEFAULT_DECAY) -> dict:
    """Max-F1 over top-k predictions (best alignment).

    The argmax is the prediction whose chain best overlaps the true chain;
    we return its precision / recall / f1.
    """
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "best_pred": None}
    for p in preds:
        if p is None:
            continue
        r = hdf1_pair(p, true, decay)
        if r["f1"] > best["f1"]:
            best = {**r, "best_pred": p}
    return best


# ---- CUI <-> ICD-10 mapping helper ---------------------------------------

@lru_cache(maxsize=1)
def _load_cui_icd_map(path: str = str(DEFAULT_MAPPING_PATH)) -> dict[str, str]:
    """CUI -> ICD-10 string (case-preserving)."""
    if not Path(path).exists():
        return {}
    raw = json.load(open(path))
    out = {}
    for _name, rec in raw.items():
        cui = rec.get("cui"); icd = rec.get("icd10")
        if cui and icd:
            out[cui] = icd
    return out


def cui_to_icd10(cui: str, mapping_path: str = str(DEFAULT_MAPPING_PATH)
                 ) -> str | None:
    return _load_cui_icd_map(mapping_path).get(cui)


def hdf1_topk_cui(preds_cui: Iterable[str], true_cui: str,
                  decay: float = DEFAULT_DECAY,
                  mapping_path: str = str(DEFAULT_MAPPING_PATH)) -> dict:
    """HDF1 over top-k where inputs are CUIs (mapped to ICD-10 internally)."""
    true_icd = cui_to_icd10(true_cui, mapping_path)
    if not true_icd:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "best_pred": None,
                "skipped": True}
    pred_icds = [cui_to_icd10(c, mapping_path) for c in preds_cui]
    pred_icds = [p for p in pred_icds if p]
    return hdf1_topk(pred_icds, true_icd, decay)


# ---- CLI sanity test -----------------------------------------------------

def _selftest() -> None:
    print("--- HDF1 self-test ---")
    # Exact match → F1 = 1.0
    print("exact J44.10 / J44.10 :", hdf1_pair("J44.10", "J44.10"))
    # Same 3-char (J44.1 vs J44.0)
    print("J44.1 / J44.0         :", hdf1_pair("J44.1", "J44.0"))
    # Same chapter only (J44.1 vs J81.0)
    print("J44.1 / J81.0         :", hdf1_pair("J44.1", "J81.0"))
    # Different chapter (J44.1 vs A00.0)
    print("J44.1 / A00.0         :", hdf1_pair("J44.1", "A00.0"))
    # top-k
    print("topk [A00.0, J44.0] / J44.10:",
          hdf1_topk(["A00.0", "J44.0"], "J44.10"))


if __name__ == "__main__":
    _selftest()
