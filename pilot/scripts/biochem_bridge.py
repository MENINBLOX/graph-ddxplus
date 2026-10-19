"""Reconcile the two ways a biochemical abnormality is written, so patient evidence joins the KG.

The rare-disease benchmarks state metabolic findings in HPO's COMPOSITIONAL style -- direction +
analyte + compartment ("Elevated plasma citrulline", "Decreased plasma free carnitine",
"Abnormal circulating alanine concentration") -- while the KG, built by IE over textbook prose,
carries the CONTRACTED clinical form ("citrullinemia", "carnitine deficiency", "hypertyrosinemia").
They are the same finding on different CUIs, so they never join: only 68% of RAMEDIS evidence CUIs
exist in the KG phenotype vocabulary at all, and the unjoinable head is exactly these -- i.e. the
pathognomonic signal of the metabolic diseases RAMEDIS is made of.

This is the rare-benchmark analogue of the DDXPlus chest-pain composition (pain+location vs
chest pain): a deterministic, benchmark-blind normalization of CONCEPT NAMES into a shared key
(direction, analyte), emitted as a bridge the scorer adds as an extra match channel. It reads
only UMLS concept names -- no disease-symptom mapping, no benchmark labels (CLAUDE.md §3).

Output: {"cui2key": {cui: "B:<dir>:<analyte>"}, "n_keys": int} -> pilot/cache/prod/biochem_bridge.json
"""
from __future__ import annotations
import argparse, json, re, sys, collections

# compartment / measurement words that carry no analyte identity
_COMPARTMENT = {"plasma", "serum", "blood", "urine", "urinary", "circulating", "csf", "cerebrospinal",
                "fluid", "concentration", "concentrations", "level", "levels", "content", "excretion",
                "in", "of", "the", "free", "total", "fasting", "postprandial"}
# words that mark the finding as biochemical rather than an anatomical/clinical sign
_ANALYTE_HINT = re.compile(
    r"(acid|amino|carnitine|ammonia|lactate|lactic|pyruvate|glucose|galactose|fructose|glycine|"
    r"alanine|citrulline|tyrosine|phenylalanine|methionine|leucine|valine|lysine|ornithine|"
    r"arginine|homocystein|cystine|porphyrin|bilirubin|creatine|creatinine|urate|uric|cholesterol|"
    r"triglyceride|ketone|glycogen|copper|iron|ferritin|zinc|calcium|magnesium|phosphate|"
    r"transaminase|kinase|dehydrogenase|enzyme|hormone|cortisol|thyroxine|insulin|"
    r"mucopolysaccharide|oligosaccharide|sulfatide|sphingomyelin|cerebroside|sterol)", re.I)

_UP_W = {"elevated", "elevation", "elevations", "increased", "increase", "high", "higher", "excess",
         "excessive", "raised", "accumulation", "hyper"}
_DOWN_W = {"decreased", "decrease", "reduced", "low", "lower", "deficient", "deficiency",
           "depletion", "depleted", "loss", "hypo"}
_ABN_W = {"abnormal", "abnormality", "abnormalities", "altered", "aberrant"}
_DIR_W = _UP_W | _DOWN_W | _ABN_W

# -emia / -uria contractions. "hyper<x>emia" = raised <x> in blood; a bare "<x>emia" for a metabolite
# is also a raised level ("tyrosinemia", "citrullinemia"), but many -emia words are NOT metabolite
# levels at all, so those are excluded explicitly rather than guessed.
_NOT_ANALYTE_EMIA = {"anemia", "ischemia", "leukemia", "septicemia", "bacteremia", "viremia",
                     "uremia", "toxemia", "hypoxemia", "glycemia", "lipemia", "eclampsia",
                     "hyperemia", "fungemia", "parasitemia", "hypovolemia", "hypervolemia"}


def _words(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def _stem(w: str) -> str:
    """Canonical analyte token. The -emia contraction eats a trailing 'e' ("tyrosine" ->
    "tyrosinemia" -> "tyrosin"), so both sides drop it: tyrosine/tyrosin, valine/valin, lactate/lactat."""
    w = w.rstrip("s") if len(w) > 4 and w.endswith("s") else w
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def _analyte(text: str) -> str:
    """Reduce a phrase to its analyte tokens (drop compartment/measurement/direction words), sorted
    so word order does not matter ("plasma free carnitine" == "free carnitine in plasma")."""
    toks = [_stem(w) for w in _words(text)
            if w not in _COMPARTMENT and w not in _DIR_W and len(w) > 2]
    return " ".join(sorted(set(toks)))


def _from_contraction(words: list[str]) -> tuple[str, str] | None:
    """"hypertyrosinemia" / "methylmalonic aciduria" / "hypocarnitinemia" -> (dir, analyte)."""
    if not words:
        return None
    last = words[-1]
    for suffix in ("emia", "uria"):
        if not last.endswith(suffix) or len(last) <= len(suffix) + 2 or last in _NOT_ANALYTE_EMIA:
            continue
        stem, direction = last[: -len(suffix)], "up"
        if stem.startswith("hyper"):
            stem = stem[5:]
        elif stem.startswith("hypo"):
            stem, direction = stem[4:], "down"
        if len(stem) < 3:
            return None
        # modifier words stay part of the analyte: "methylmalonic aciduria" -> "methylmalonic acid"
        analyte = _analyte(" ".join(words[:-1] + [stem]))
        return (direction, analyte) if analyte else None
    return None


def key_for(name: str) -> str | None:
    """Concept name -> shared biochemical key, or None if it is not a biochemical level finding."""
    if not name or not _ANALYTE_HINT.search(name):
        return None
    words = _words(name)
    # contraction first: "hyper-tyrosinemia" must not be read as the direction word "hyper"
    c = _from_contraction(words)
    if c:
        return f"B:{c[0]}:{c[1]}"
    dirs = {w for w in words} & _DIR_W
    if dirs & _UP_W:
        direction = "up"
    elif dirs & _DOWN_W:
        direction = "down"
    elif dirs & _ABN_W:
        direction = "abn"
    else:
        return None
    analyte = _analyte(name)
    return f"B:{direction}:{analyte}" if analyte else None


def keys_for(name: str) -> list[str]:
    """All keys a concept contributes: its own, plus the direction-agnostic parent so HPO's
    "Abnormal circulating alanine concentration" still meets a KG "elevated alanine" (an is-a
    generalization, the same move as the HPO-ancestor bridge)."""
    k = key_for(name)
    if not k:
        return []
    _, direction, analyte = k.split(":", 2)
    return [k] if direction == "abn" else [k, f"B:abn:{analyte}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrconso", default="data/umls_extracted/MRCONSO.RRF")
    ap.add_argument("--out", default="pilot/cache/prod/biochem_bridge.json")
    a = ap.parse_args()

    # one preferred English name per CUI, then key it
    seen: dict[str, str] = {}
    for line in open(a.mrconso, encoding="utf-8", errors="ignore"):
        p = line.split("|")
        if len(p) > 14 and p[1] == "ENG" and p[2] == "P" and p[4] == "PF" and p[0] not in seen:
            seen[p[0]] = p[14]
    cui2key = {}
    for cui, name in seen.items():
        ks = keys_for(name)
        if ks:
            cui2key[cui] = ks
    groups = collections.Counter(k for ks in cui2key.values() for k in ks)
    json.dump({"cui2key": cui2key, "n_keys": len(groups)}, open(a.out, "w"))
    print(f"concepts read: {len(seen)}   biochemical concepts keyed: {len(cui2key)}   "
          f"distinct keys: {len(groups)}")
    print("largest key groups (these are the CUIs that now join each other):")
    for k, n in groups.most_common(12):
        if n > 1:
            print(f"  {k:44} {n}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
