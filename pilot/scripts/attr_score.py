"""Discriminative attribute scoring for KG diagnosis.

The earlier attribute-BONUS (add on match) did not discriminate: a shared symptom's attribute
matched many candidate diseases equally. The clinically-correct signal is a MISMATCH PENALTY:
among diseases that share a finding, the one whose qualified-edge attribute CONTRADICTS the
patient's attribute (e.g. patient reports crushing/severe/sudden pain but the disease's typical
pain is dull/mild/gradual) should be down-weighted. Controlled-vocabulary attributes
(severity ordinal, onset binary) give clean agree/disagree; free-text ones (character) only
reward agreement (surface-form mismatch is not a true contradiction). Location is noisy
(granularity), so a smaller penalty.

Consumes the per-benchmark attribute loaders' finding format:
  finding = {"cui", "location":set(cui), "location_txt":set(tok), "character":set(tok),
             "severity": mild|moderate|severe|None, "onset": sudden|gradual|None}
and the KG edge's merged attrs: {"location_cui","location"(txt),"character","severity","onset"}.
"""
from __future__ import annotations
import os, collections
import re

_SEV_ORD = {"mild": 0, "moderate": 1, "severe": 2}

# Controlled pain-quality vocabulary (SNOMED-style qualifier classes). Free-text character on
# BOTH sides (patient report + KG-extracted edge) is collapsed to these classes so that
# surface-form variants ("crushing"/"pressure"/"tight", "colicky"/"cramp"/"spasm") agree. This
# is the fix for character's 0% realized agreement: the discriminative signal is in the class,
# not the exact word (patient says "a cramp", the source says "cramping").
_CHAR_CLASS = {
    "burn": "burning", "gnaw": "burning",
    "sharp": "sharp",
    "stab": "stabbing", "knife": "stabbing",
    "cramp": "cramping", "colic": "cramping", "spasm": "cramping",
    "pressure": "pressure", "heav": "pressure", "tight": "pressure",
    "squeez": "pressure", "constrict": "pressure", "crush": "pressure",
    "throb": "throbbing", "puls": "throbbing", "pound": "throbbing",
    "tear": "tearing", "rip": "tearing",
    "dull": "dull", "ach": "dull",
    "shoot": "shooting", "lancinat": "shooting",
}


# Timing and the aggravating/relieving pair had NO scoring channel, so three of the six candidate
# attributes could neither contribute to a diagnosis nor be leave-one-out ablated -- the KG stored
# them and the scorer ignored them. They are free text on both sides, so they are handled like
# character: collapse each side to a controlled class, reward agreement only. A surface mismatch
# between two free-text phrases is not a contradiction, so no penalty is applied.
_TIMING_CLASS = {
    "nocturn": "nocturnal", "night": "nocturnal", "sleep": "nocturnal",
    "morning": "morning", "wake": "morning",
    "postprandial": "postprandial", "after eat": "postprandial", "after meal": "postprandial",
    "meal": "postprandial", "food": "postprandial",
    "exertion": "exertional", "exercis": "exertional", "activity": "exertional", "effort": "exertional",
    "episod": "episodic", "intermittent": "episodic", "paroxysm": "episodic", "attack": "episodic",
    "paroxysmal": "episodic", "paroxysms": "episodic", "recurren": "episodic", "paroxystic": "episodic",
    "periodic": "episodic", "cyclic": "episodic", "fluctuat": "episodic",
    "persistent": "continuous", "continuous": "continuous", "constant": "continuous",
    "progress": "progressive", "worsen over": "progressive", "chronic": "continuous",
    # NLICE's `frequency` vocabulary
    "nightly": "nocturnal", "periodic": "episodic", "often": "frequent", "frequent": "frequent",
    "rarely": "rare", "occasional": "rare", "sometimes": "rare", "seldom": "rare",
}

# Aggravating and relieving share one class space: the same circumstance (exertion, recumbency,
# a meal) appears on either side depending on the disease, and the classes are the standard
# history-taking categories.
# A stem maps to a SET of classes, because one circumstance genuinely belongs to several: walking
# is both exertion and movement, and a single-label table made the two sides miss each other --
# measured on NLICE, 597 findings had an aggravating factor on BOTH sides and agreed 0 times,
# because the patient said "Walking" (exertion) where the KG said "movement" (movement).
_TRIGGER_CLASS = {
    "exert": {"exertion", "movement"}, "exercis": {"exertion", "movement"},
    "activity": {"exertion", "movement"}, "effort": {"exertion"},
    "walk": {"exertion", "movement"}, "climb": {"exertion", "movement"},
    "run": {"exertion", "movement"}, "ambulat": {"exertion", "movement"},
    "hold": {"exertion"}, "lift": {"exertion", "movement"}, "carry": {"exertion"},
    "weight-bearing": {"exertion", "movement"}, "weight bearing": {"exertion", "movement"},
    "intercourse": {"exertion"}, "coitus": {"exertion"}, "sexual": {"exertion"},
    "lying": {"recumbency"}, "recumben": {"recumbency"}, "supine": {"recumbency"},
    "lie down": {"recumbency"}, "flat": {"recumbency"},
    "meal": {"ingestion"}, "eat": {"ingestion"}, "food": {"ingestion"}, "swallow": {"ingestion"},
    "drink": {"ingestion"}, "alcohol": {"ingestion"}, "fast": {"ingestion"}, "fatty": {"ingestion"},
    "breath": {"respiration"}, "inspir": {"respiration"}, "cough": {"respiration", "movement"},
    "sneez": {"respiration", "movement"}, "strain": {"movement", "exertion"},
    # musculoskeletal provocation: the KG writes these as manoeuvres ("30 seconds of lumbar
    # extension"), which had no class at all and so could never match anything
    "movement": {"movement"}, "motion": {"movement"}, "moving": {"movement"},
    "extension": {"movement"}, "flexion": {"movement"}, "rotation": {"movement"},
    "bend": {"movement", "posture"}, "stoop": {"movement", "posture"}, "twist": {"movement"},
    "palpat": {"palpation"}, "press": {"palpation"}, "touch": {"palpation"},
    "pressure on": {"palpation"}, "bump": {"movement"}, "car_travel": {"movement"},
    "car travel": {"movement"}, "jolt": {"movement"}, "vibration": {"movement"},
    "sitting": {"posture"}, "sit up": {"posture"}, "stand": {"posture"}, "lean": {"posture"},
    "upright": {"posture"}, "position": {"posture"}, "squat": {"posture", "movement"},
    "rest": {"rest"}, "sleep": {"rest"}, "lying still": {"rest"}, "immobil": {"rest"},
    "cold": {"temperature"}, "heat": {"temperature"}, "warm": {"temperature"},
    "sun": {"temperature"},
    "stress": {"stress"}, "anxiet": {"stress"}, "emotion": {"stress"},
    "menstru": {"hormonal"}, "menses": {"hormonal"}, "pregnan": {"hormonal"},
    "puberty": {"hormonal"},
    "antacid": {"antacid"}, "nitroglycerin": {"nitrate"}, "nitrate": {"nitrate"},
    "analgesi": {"analgesic"}, "nsaid": {"analgesic"}, "aspirin": {"analgesic"},
    "paracetamol": {"analgesic"}, "acetaminophen": {"analgesic"}, "antipyretic": {"analgesic"},
    "infection": {"infection"}, "illness": {"infection"}, "fever": {"infection"},
    "urinat": {"voiding"}, "micturit": {"voiding"}, "defec": {"voiding"}, "stool": {"voiding"},
}


def _toks(text):
    return {w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2}


def _classes(value, table):
    """Collapse a free-text attribute value (or token set) to its controlled classes.

    A table entry may map a stem to one class or to several (a term can belong to more than one
    category: walking is exertion AND movement)."""
    text = value.lower() if isinstance(value, str) else " ".join(value or ()).lower()
    out = set()
    for stem, cls in table.items():
        if stem in text:
            out |= cls if isinstance(cls, (set, frozenset)) else {cls}
    return out


def timing_classes(value):
    """Free-text timing -> controlled temporal-pattern classes."""
    return _classes(value, _TIMING_CLASS)


def trigger_classes(value):
    """Free-text aggravating/relieving factor -> controlled history-taking classes."""
    return _classes(value, _TRIGGER_CLASS)


def char_classes(chars):
    """Collapse a set of character tokens (or a free-text string) to controlled pain-quality
    classes. Accepts a set of tokens or a raw string; returns a set of canonical class labels."""
    if isinstance(chars, str):
        text = chars.lower()
    else:
        text = " ".join(chars or ()).lower()
    return {cls for stem, cls in _CHAR_CLASS.items() if stem in text}


def merge_edge_attrs(cluster, edge_attr):
    """phenotype cui -> merged qualified attributes over the disease cluster.

    Values are UNIONED, not first-wins. A disease cluster spans subtypes that each describe the
    shared finding differently ("pressure-like" in one, "burning" in another); keeping only the
    first makes the match depend on iteration order. Class extraction reads the whole string, so a
    joined value simply contributes every class it mentions."""
    pa = {}
    for d in cluster:
        for p, at in edge_attr.get(d, {}).items():
            if not at:
                continue
            cell = pa.setdefault(p, {})
            for k, v in at.items():
                if not v:
                    continue
                old = cell.get(k)
                if not old:
                    cell[k] = v
                elif str(v).lower() not in str(old).lower():
                    cell[k] = f"{old}; {v}"
    return pa


def hpo_delta(finding, ea, w, lam=1.0, families=None):
    """Attribute contribution computed in HPO Clinical-modifier space instead of hand-made classes.

    Both sides' qualifier text is normalized to HPO terms (`hpo_modifier`), so a match no longer
    depends on a class table we wrote -- which is what silently zeroed the aggravating channel when
    the patient's "Walking" and the KG's "movement" landed in different homemade classes.
    `families` restricts scoring to particular HPO modifier families, which is how a leave-one-out
    ablation is run over a schema that was never hand-declared."""
    import hpo_modifier
    voc = hpo_modifier.load()

    def terms(obj):
        out = set()
        for q in obj.get("qualifiers", ()) or ():
            out |= voc.classes(q.get("quote", ""), q.get("relation", ""))
        for k in ("location", "character", "severity", "onset", "timing",
                  "aggravating", "relieving"):
            v = obj.get(k)
            if v:
                out |= voc.classes(v if isinstance(v, str) else " ".join(v), k)
        return out

    ft, et = terms(finding), terms(ea)
    if families is not None:
        ft = {h for h in ft if voc.label.get(voc.family(h), "") in families}
        et = {h for h in et if voc.label.get(voc.family(h), "") in families}
    # exact concept agreement, else agreement at the family level (a coarser but real match)
    if ft & et:
        return lam * w
    if {voc.family(h) for h in ft} & {voc.family(h) for h in et}:
        return 0.5 * lam * w
    return 0.0


# How a finding's several attribute channels combine. "sum" (the original) lets one finding
# collect a bonus per agreeing channel, so a finding whose location, character, severity and onset
# all agree outweighs four separate matched findings -- which is why adding channels measured WORSE
# than the best single channel (DDXPlus: onset alone @1 49.17, all four 46.33). "mean" keeps the
# statement at "this finding's qualifiers agree", once, however many channels carry it; "max" takes
# the single strongest channel.
ATTR_COMBINE = os.environ.get("KG_ATTR_COMBINE", "sum")

# Selectivity weight for an attribute agreement, by the same logic IDF applies to terms: an
# attribute value shared by most candidates separates nothing. Measured on NLICE, agreeing
# channels reward very different numbers of candidates out of 37 -- character 3.0 (gold among
# them 65%), location 2.0 (42%), but aggravating 6.3 (35%), because its `movement` and `exertion`
# classes cover a sixth of the candidate set. Given equal weight, that broad reward is what made
# the aggravating channel measure NEGATIVE (-1.22). Populated from the KG by build_attr_df();
# empty = every agreement counts the same (the previous behaviour).
_ATTR_DF: dict = {}
_ATTR_N = 0


def build_attr_df(edges) -> None:
    """Count, per (channel, normalized class), how many DISEASES carry it. KG-derived and
    benchmark-blind."""
    global _ATTR_N
    seen = collections.defaultdict(set)
    for e in edges:
        at = e.get("attrs", {}) or {}
        d = e["d"]
        for ch, fn in (("character", char_classes), ("timing", timing_classes),
                       ("aggravating", trigger_classes), ("relieving", trigger_classes)):
            v = at.get(ch)
            if v:
                for c in fn(v):
                    seen[(ch, c)].add(d)
        for ch in ("severity", "onset"):
            v = at.get(ch)
            if v:
                seen[(ch, v)].add(d)
        if at.get("location_cui"):
            seen[("location", at["location_cui"])].add(d)
    _ATTR_DF.clear()
    _ATTR_DF.update({k: len(v) for k, v in seen.items()})
    _ATTR_N = len({e["d"] for e in edges}) or 1


def attr_sel(channel: str, classes) -> float:
    """Selectivity multiplier in (0,1]: 1.0 for a value unique to one disease, small for a value
    most diseases share. Normalized by log(N) so it never rescales the whole channel away."""
    if not _ATTR_DF or not classes or os.environ.get("KG_ATTR_SEL", "1") == "0":
        return 1.0
    dfs = [_ATTR_DF.get((channel, c)) for c in classes]
    dfs = [d for d in dfs if d]
    if not dfs:
        return 1.0
    import math
    return max(0.05, math.log(_ATTR_N / min(dfs)) / math.log(max(_ATTR_N, 2)))


def attr_delta(finding, ea, w, lam=1.0, mu=1.0, use=("severity", "onset", "location", "character")):
    """Signed attribute contribution for one matched finding (idf-weight w). +agree, -contradict.
    `use` selects which attribute channels are active (per-benchmark: e.g. DDXPlus onset-only)."""
    s = 0.0
    _parts = []
    # severity: controlled ordinal -> distance-scaled penalty
    if "severity" in use:
        fs, es = finding.get("severity"), ea.get("severity")
        if fs and es:
            d = abs(_SEV_ORD.get(fs, 1) - _SEV_ORD.get(es, 1))
            _parts.append((lam if d == 0 else -mu * (d / 2.0)) * w)
    # onset: controlled binary
    if "onset" in use:
        fo, eo = finding.get("onset"), ea.get("onset")
        if fo and eo:
            _parts.append((lam if fo == eo else -mu) * w)
    # location: CUI or token overlap; noisy -> half penalty
    if "location" in use:
        floc = finding.get("location") or set()
        floct = finding.get("location_txt") or set()
        eloc = ea.get("location_cui")
        eloct = _toks(ea.get("location"))
        if (floc or floct) and (eloc or eloct):
            agree = (eloc is not None and eloc in floc) or bool(floct & eloct)
            _parts.append((lam if agree else -mu * 0.5) * w)
    # character: collapse both sides to controlled pain-quality classes, then reward agreement
    # only (surface mismatch != contradiction; free-text is too noisy for a penalty).
    if "character" in use:
        fc = char_classes(finding.get("character") or set())
        ec = char_classes(ea.get("character"))
        if fc and ec and (fc & ec):
            _parts.append(lam * w * attr_sel("character", fc & ec))
    # timing / aggravating / relieving: free text on both sides -> controlled classes, agreement
    # only (same rationale as character). Without these the leave-one-out ablation could not test
    # three of the six candidate attributes at all.
    if "timing" in use:
        ft = timing_classes(finding.get("timing"))
        et = timing_classes(ea.get("timing"))
        if ft and et and (ft & et):
            _parts.append(lam * w * attr_sel("timing", ft & et))
    for chan in ("aggravating", "relieving"):
        if chan in use:
            fa = trigger_classes(finding.get(chan))
            eb = trigger_classes(ea.get(chan))
            if fa and eb and (fa & eb):
                _parts.append(lam * w * attr_sel(chan, fa & eb))
    if not _parts:
        return 0.0
    if ATTR_COMBINE == "mean":
        return sum(_parts) / len(_parts)
    if ATTR_COMBINE == "max":
        return max(_parts, key=abs)
    return sum(_parts)
