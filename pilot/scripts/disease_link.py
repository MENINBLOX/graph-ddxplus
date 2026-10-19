"""Exact-match linking of DISEASE TITLES to UMLS CUIs (no generic head-noun fallback).

`umls_link.link` is built for phenotype PHRASES, where backing off to the head noun ("severe
substernal chest pain" -> "chest pain") is the right generalization. Applied to a disease title
that fallback is destructive: "KCNC1-Related Disorders" -> "Disease" (C0012634), "Loeys-Dietz
Syndrome" -> "Syndrome" (C0039082). Measured on the GeneReviews corpus, 31% of chapters (272/888)
collapsed this way -- 75 distinct rare diseases all became C0012634 -- so those diseases had no KG
node of their own (unrankable) while the generic sinks became noise hubs (CLAUDE.md §12).

This module instead matches the title against the UMLS string index (MRCONSO ENG, restricted to
disease-typed CUIs), which avoids the SPECIALIST-norm spelling variants that made MRXNS miss even
canonical titles ("maple syrup urine disease" is normalized to "... sirup ..." in MRXNS_ENG).
Still threshold-free exact matching -- only the fallback behaviour differs.

link_disease(title) -> (cui, mode) | (None, None). Modes: exact | nosuffix | nogene.
"""
from __future__ import annotations
import json, os, re

STR2CUI = "pilot/cache/umls_maps/disease_str2cui.json"
CUI2NAME = "pilot/cache/umls_maps/disease_cui2name.json"
STY_TSV = "pilot/cache/umls_maps/cui_sty.tsv"

# editorial suffixes GeneReviews/StatPearls add to a chapter title that are not part of the concept
_SUFFIX = re.compile(r"\s*[-–—]?\s*(overview|summary|chapter|review)\s*$", re.I)
# gene-symbol prefix used as a disease qualifier ("SCN8A-Related Epilepsy", "GJB2-Related ...")
_GENE_PREFIX = re.compile(r"^[A-Z][A-Z0-9\-]{1,9}\d?\s*[-–—]?\s*(related|associated)\s+", re.I)
_PAREN = re.compile(r"\(.*?\)")
# words that carry no disease identity on their own: a form reduced to these only would link to a
# generic concept ("KCNC1-Related Disorders" minus the gene is just "Disorders" -> C0012634)
_GENERIC = {"disease", "diseases", "disorder", "disorders", "syndrome", "syndromes", "deficiency",
            "condition", "conditions", "defect", "defects", "anomaly", "anomalies", "abnormality",
            "abnormalities", "dysfunction", "impairment", "susceptibility", "predisposition",
            "malformation", "failure", "type", "related", "associated", "and", "or", "of", "the"}

# semantic-type priority for deterministic disambiguation (specific disease before finding)
_TUI_RANK = {t: i for i, t in enumerate(
    ["T047", "T019", "T020", "T190", "T191", "T037", "T046", "T048", "T049", "T050"])}

_IDX: dict | None = None
_NAME: dict | None = None
_TOKIDX: dict | None = None
_STY: dict | None = None


def _load() -> tuple[dict, dict]:
    global _IDX, _TOKIDX, _STY
    if _IDX is None:
        _IDX = json.load(open(STR2CUI, encoding="utf-8"))
        # order-insensitive view: UMLS writes the same concept as "Ehlers-Danlos syndrome, classic
        # type" where a chapter titles it "Classic Ehlers-Danlos Syndrome". Same token multiset ->
        # still an exact match, just not an exact ORDER match.
        _TOKIDX = {}
        for s, cs in _IDX.items():
            _TOKIDX.setdefault(" ".join(sorted(s.split())), set()).update(cs)
        _STY = {}
        if os.path.exists(STY_TSV):
            for line in open(STY_TSV, encoding="utf-8"):
                q = line.rstrip("\n").split("\t")
                if len(q) >= 2:
                    _STY[q[0]] = q[1]
    return _IDX, _STY


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def _pick(cuis: list[str]) -> str:
    """Deterministic: highest semantic-type priority, tie-break by smallest CUI."""
    return min(cuis, key=lambda c: (_TUI_RANK.get((_STY or {}).get(c, ""), 99), c))


def link_disease(title: str) -> tuple[str | None, str | None]:
    """Disease title -> (cui, mode). Returns (None, None) rather than a generic concept."""
    idx, _ = _load()
    base = _PAREN.sub("", title).strip()
    forms = [("exact", base), ("nosuffix", _SUFFIX.sub("", base).strip())]
    stripped = _GENE_PREFIX.sub("", _SUFFIX.sub("", base)).strip()
    # Only drop the gene qualifier if what remains is still a SPECIFIC disease name. One content
    # word after the generics ("Disorders", "Neurodevelopmental Disorders") is a category, and
    # linking there is the sink bug; two or more ("Autosomal Recessive Nonsyndromic Hearing Loss")
    # names a real narrower entity.
    if stripped and stripped != base and len(set(_norm(stripped).split()) - _GENERIC) >= 2:
        forms.append(("nogene", stripped))
    for mode, form in forms:
        key = _norm(form)
        if not key:
            continue
        if key in idx:
            return _pick(idx[key]), mode
        tok = " ".join(sorted(key.split()))
        if _TOKIDX and tok in _TOKIDX:
            return _pick(sorted(_TOKIDX[tok])), mode + "-tok"
    return None, None


def link_disease_or_generalize(title: str, valid_tuis=None) -> tuple[str | None, str | None]:
    """Exact link first; if that fails, allow `umls_link`'s broader cascade but REFUSE a result
    whose concept name is purely generic.

    `umls_link` is right for phenotype phrases and wrong for disease names: its head-noun fallback
    sends 20% of the semiology textbook's causes into "Disease"/"Syndrome", so those diseases'
    aggravating/relieving attributes are attached to a junk node. Strict linking alone is too lossy
    here (it drops half the causes), so the fallback is kept and only its generic outputs rejected."""
    cui, mode = link_disease(title)
    if cui:
        return cui, mode
    from umls_link import link as _umls_link
    cui, mode = _umls_link(title, valid_tuis)
    if not cui:
        return None, None
    # judge by the concept's PREFERRED name only. Any synonym would do ("medical condition" has a
    # non-generic token), which is how C0012634 "Disease" kept slipping through.
    global _NAME
    if _NAME is None:
        _NAME = json.load(open(CUI2NAME, encoding="utf-8")) if os.path.exists(CUI2NAME) else {}
    pref = _NAME.get(cui, "")
    if pref and set(_norm(pref).split()) - _GENERIC:
        return cui, (mode or "") + "+broad"
    return None, None


if __name__ == "__main__":  # smoke test on the titles that exposed the sink bug
    for t in ["KCNC1-Related Disorders", "Hyaline Fibromatosis Syndrome", "Loeys-Dietz Syndrome",
              "Maple Syrup Urine Disease", "Hereditary Hyperekplexia Overview",
              "SCN8A-Related Epilepsy and/or Neurodevelopmental Disorder",
              "Classic Ehlers-Danlos Syndrome", "Long QT Syndrome Overview"]:
        print(f"{t[:55]:57} -> {link_disease(t)}")
