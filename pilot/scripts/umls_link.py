"""Threshold-FREE UMLS concept linking via the UMLS normalized string index (MRXNS_ENG).

Academic basis: standard dictionary-based UMLS concept mapping (cf. MetaMap, QuickUMLS,
cTAKES) — morphological normalization + EXACT lexical match against the UMLS Metathesaurus.
No similarity threshold (unlike approximate candidate generation). The normalization follows
the UMLS SPECIALIST Lexicon 'norm' used to build MRXNS_ENG (lowercase, punctuation->space,
stop-word removal, inflectional reduction, word-order sort); query terms are normalized the
same way plus deterministic reductions (quantifier stripping, verb un-inflection, head-noun).

Disambiguation of a normalized string mapping to multiple CUIs is deterministic: prefer a
clinically-valid semantic type (cui_sty), else the lexicographically smallest CUI.

link(term, valid_tuis) -> (cui, mode) or (None, None). Loads MRXNS once (~6s, ~6M strings)."""
import re, os

MRXNS = "data/umls_subset/MRXNS_ENG.RRF"
STY_TSV = "pilot/cache/umls_maps/cui_sty.tsv"
STOP = set("a an and as at be by for from has have he in into is it its of on or that the to was were will "
           "with without due".split())
SEV = set("mild mildly moderate moderately severe severely marked markedly slight slightly minimal minimally "
          "profound extensive serious significant acute chronic recurrent frequent excessive increased "
          "decreased reduced worsened worsening abnormal excruciating florid massive sudden gradual".split())
QUANT = re.compile(r'^(amount|quantity|colou?r|appearance|presence|degree|sense|episodes?|lack|loss)\s+of\s+', re.I)

_IDX = None; _STY = None

def _singular(w):
    if len(w) <= 3: return w
    if w.endswith(('sis', 'us', 'is', 'ss', 'ous', 'ius', 'ates', 'ei', 'ysis')): return w
    if w.endswith('ies'): return w[:-3] + 'y'
    if w.endswith(('ses', 'xes', 'zes', 'ches', 'shes')): return w[:-2]
    if w.endswith('s'): return w[:-1]
    return w

_IRREG = {"teeth": "tooth", "feet": "foot", "men": "man", "women": "woman", "children": "child",
          "people": "person", "mice": "mouse", "geese": "goose", "lice": "louse"}

def _deverb(w):
    """candidate base forms for an inflected word (exact-match validates each, so we can be generous:
    -ing/-ed deverbing, Latin/Greek plurals (rhonchi->rhonchus, vertebrae->vertebra), irregulars)."""
    outs = {w}
    if w in _IRREG: outs.add(_IRREG[w])
    if len(w) > 3 and w.endswith('i'): outs.add(w[:-1] + 'us')      # rhonchi->rhonchus, alveoli->alveolus
    if len(w) > 3 and w.endswith('ae'): outs.add(w[:-1])            # vertebrae->vertebra
    if len(w) > 4 and w.endswith('a'): outs.add(w[:-1] + 'um')      # crateria? (rare) safe via exact-match
    for suf in ('ing', 'ed'):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            b = w[:-len(suf)]
            outs |= {b, b + 'e'}                                    # grunt / bulge
            if len(b) > 1 and b[-1] == b[-2]: outs.add(b[:-1])      # running->run
    return sorted(outs)

def norm(s, strip_sev=False, deverb=False):
    s = QUANT.sub('', s)
    words = [x for x in re.findall(r'[a-z0-9]+', s.lower()) if x not in STOP]
    if strip_sev: words = [x for x in words if x not in SEV]
    if not words: return [""]
    # produce one or more normalized candidates (verb base variants expand the set)
    per_word = []
    for x in words:
        base = _singular(x)
        variants = _deverb(base) if deverb else [base]
        per_word.append(sorted(set(variants)))
    # cartesian is overkill; only expand if a single inflected head — keep it simple:
    cands = [" ".join(sorted(w[0] for w in per_word))]
    if deverb:
        # also try replacing each word by each of its variants (small terms only)
        import itertools
        if sum(len(v) for v in per_word) <= 8:
            for combo in itertools.product(*per_word):
                cands.append(" ".join(sorted(combo)))
    return list(dict.fromkeys(cands))

def _load():
    global _IDX, _STY
    if _IDX is None:
        _IDX = {}
        for line in open(MRXNS, encoding="utf-8"):
            p = line.split("|")
            if len(p) >= 3: _IDX.setdefault(p[1], set()).add(p[2])
        _STY = {}
        if os.path.exists(STY_TSV):
            for line in open(STY_TSV, encoding="utf-8"):
                q = line.rstrip("\n").split("\t")
                if len(q) >= 2: _STY[q[0]] = q[1]
    return _IDX, _STY

# deterministic semantic-type priority for disambiguation (symptom/finding beats neoplasm/anatomy)
_TUI_RANK = {t: i for i, t in enumerate(
    ["T184", "T033", "T034", "T046", "T047", "T048", "T037", "T190", "T019", "T020", "T049", "T050",
     "T191",  # neoplastic last among "valid" — avoid "lymph nodes"->neoplasm
     "T023", "T029", "T030", "T024", "T025", "T031"])}

def _pick(cuis, valid_tuis):
    """deterministic disambiguation: among the valid-typed CUIs, prefer by semantic-type priority
    (Sign/Symptom > Finding > ... > Neoplasm), tie-break by smallest CUI; else smallest CUI overall."""
    cs = sorted(cuis)
    if valid_tuis:
        val = [c for c in cs if _STY.get(c, "") in valid_tuis]
        if val:
            return min(val, key=lambda c: (_TUI_RANK.get(_STY.get(c, ""), 99), c))
    return min(cs, key=lambda c: (_TUI_RANK.get(_STY.get(c, ""), 99), c))

def link(term, valid_tuis=None):
    """Exact-match link cascade (no threshold). Returns (cui, mode) or (None, None)."""
    idx, sty = _load()
    for mode, forms in [("exact", norm(term, True) + norm(term, False)),
                        ("deverb", norm(term, True, deverb=True))]:
        for f in forms:
            if f and f in idx:
                return _pick(idx[f], valid_tuis), mode
    # head noun-phrase / head word (still exact-match)
    words = [w for w in re.findall(r'[a-z]+', term.lower()) if w not in STOP and w not in SEV]
    for k in (2, 1):
        if len(words) >= k:
            for f in norm(" ".join(words[-k:]), True, deverb=True):
                if f and f in idx:
                    return _pick(idx[f], valid_tuis), f"head{k}"
    return None, None
