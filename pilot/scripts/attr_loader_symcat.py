"""Attribute/frequency-aware SymCat patient loader (standalone).

Why this exists
---------------
The production loader ``onlykg_eval_unified.load_symcat`` uses the SymCat
per-(disease, symptom) *inclusion frequency* only as a Bernoulli sampling
probability and then throws it away: the patient's evidence is a plain
``set`` of CUIs, so the scorer treats a 90%-prevalent symptom and a 2%-rare
symptom identically. That discards the single strongest query-side signal
SymCat actually has -- ``freq`` is a direct estimate of P(symptom | disease).

Investigation summary (see module docstring / task report):
  * SymCat raw data (``symcat-474-symptoms.csv``, ``symcat_parsed_full.json``,
    ``parse.py``) has NO structured per-symptom clinical attributes
    (nature / location / intensity / character). The only structured extras
    are demographic risk factors (age / sex / race) and disease common-causes.
    Clinical attributes are *baked into the symptom name* ("Sharp abdominal
    pain", "Upper abdominal pain", "Burning chest pain") -- not separable
    fields.  => the query-side attribute story for the "SymCat setting" must
    come from NLICE (arXiv:2401.13756), which augments SymCat diseases with a
    ``nature:location:intensity:duration:onset:excitation`` schema.
  * The one signal SymCat *does* carry and we currently discard is ``freq``.

This loader preserves ``freq`` per finding so a downstream scorer can weight
present evidence by prevalence instead of treating presence as binary.

Public API
----------
``load(ppd=50, sample=True, seed=42)`` -> ``(cand, patients)`` where
  * ``cand``    : ``{disease_name: (disease_name, disease_cui)}`` (unchanged
                  from ``load_symcat`` so it is a drop-in for the same scorer
                  candidate set).
  * ``patients``: list of ``(target_keys, findings, "SymCat")`` where
                  ``findings`` is a ``list[dict]``, each
                  ``{"cui": str, "freq": float, "name": str}``. ``freq`` is the
                  0-100 SymCat inclusion percentage (== P(symptom|disease)).

Backwards compatibility
-----------------------
A scorer that only wants presence can do ``{f["cui"] for f in findings}`` to
recover exactly the old ``set``-based evidence.

Notes
-----
  * ``sample=True`` (default) reproduces the current protocol: Bernoulli-sample
    each symptom by its freq, ``ppd`` patients per disease, ``random.seed`` for
    determinism. Included findings still carry their freq for weighting.
  * ``sample=False`` yields ONE "soft" patient per disease carrying *every*
    mapped symptom with its freq as a weight -- the pure freq-weighted-evidence
    setting (no Bernoulli noise), useful for the ablation in the task report.
  * Uses ``data/symcat/symcat_parsed.json`` -- the SAME source file as the
    production ``load_symcat`` (note: that file holds only 50 diseases; the
    full 801-disease table lives in ``symcat_parsed_full.json``. Pass
    ``full=True`` to use the complete set).
"""

from __future__ import annotations

import json
import random
from typing import Optional

DISEASE_MAP = "data/symcat/disease_umls_mapping.json"
SYMPTOM_MAP = "data/symcat/symptom_umls_mapping.json"
PARSED_50 = "data/symcat/symcat_parsed.json"
PARSED_FULL = "data/symcat/symcat_parsed_full.json"


def _load_maps() -> tuple[dict, dict]:
    """Return (candidate disease map, symptom-name -> CUI map)."""
    sym_umls = json.load(open(SYMPTOM_MAP))["mapping"]
    dis_umls = json.load(open(DISEASE_MAP))["mapping"]
    sym2cui = {s: i["umls_cui"] for s, i in sym_umls.items() if i.get("umls_cui")}
    cand = {
        dn: (dn, info["umls_cui"])
        for dn, info in dis_umls.items()
        if info.get("umls_cui")
    }
    return cand, sym2cui


def load(
    ppd: int = 50,
    sample: bool = True,
    seed: Optional[int] = 42,
    full: bool = False,
):
    """Load SymCat candidates + frequency-aware simulated patients.

    Parameters
    ----------
    ppd : int
        Patients per disease when ``sample=True`` (ignored when ``sample=False``).
    sample : bool
        True  -> Bernoulli-sample symptoms by freq, ``ppd`` patients/disease
                 (mirrors ``load_symcat`` sampling; findings keep their freq).
        False -> one soft patient/disease carrying every mapped symptom w/ freq.
    seed : int | None
        RNG seed for reproducible sampling (default 42, matching production).
    full : bool
        Use the complete 801-disease table instead of the 50-disease dev subset
        that ``load_symcat`` currently reads.

    Returns
    -------
    (cand, patients)
        ``patients`` items are ``(target_keys, findings, "SymCat")`` with
        ``findings`` a list of ``{"cui", "freq", "name"}`` dicts.
    """
    if seed is not None:
        random.seed(seed)

    cand, sym2cui = _load_maps()
    parsed = json.load(open(PARSED_FULL if full else PARSED_50))

    patients: list[tuple[list[str], list[dict], str]] = []
    for dn, sym_list in parsed["disease_symptom_pairs"].items():
        if dn not in cand or not sym_list:
            continue
        # (cui, freq, original symptom name) for every mapped symptom
        mapped = [
            (sym2cui[s], float(f), s)
            for s, f in sym_list
            if sym2cui.get(s)
        ]
        if not mapped:
            continue

        if not sample:
            findings = [{"cui": c, "freq": fr, "name": nm} for c, fr, nm in mapped]
            patients.append(([dn], findings, "SymCat"))
            continue

        for _ in range(ppd):
            findings = [
                {"cui": c, "freq": fr, "name": nm}
                for c, fr, nm in mapped
                if random.random() * 100 < fr
            ]
            if findings:
                patients.append(([dn], findings, "SymCat"))
    return cand, patients


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ppd", type=int, default=50)
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    cand, pats = load(ppd=a.ppd, sample=not a.no_sample, full=a.full)
    n_find = sum(len(f) for _, f, _ in pats)
    print(f"candidates: {len(cand)} diseases")
    print(f"patients:   {len(pats)}")
    print(f"findings:   {n_find} total, {n_find / max(len(pats), 1):.1f}/patient avg")
    if pats:
        tk, fnd, ds = pats[0]
        print(f"sample patient target={tk} source={ds}")
        for f in fnd[:5]:
            print(f"  {f}")
