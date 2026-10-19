"""Open attribute vocabulary: HPO's Clinical modifier subtree as the attribute space.

The KG's attribute schema used to be six slots chosen by hand from Bates. Two problems showed up
in measurement. (1) A reviewer can always ask why those six. (2) Matching two free-text values
required a hand-written class table, and that table was wrong in ways that silently destroyed the
signal: the patient's "Walking" (class `exertion`) never met the KG's "movement" (class
`movement`), so 597 findings with an aggravating factor on BOTH sides agreed 0 times.

HPO already publishes the space this schema was approximating: `Clinical modifier` (HP:0012823),
357 concepts under ~16 families (Triggered by, Aggravated by, Ameliorated by, Pain characteristic,
Severity, Clinical course, Position, ...). Using it means the attribute inventory is defined by a
maintained ontology rather than by us, the values are controlled rather than free text, and both
the KG side and the benchmark side can normalize into ONE vocabulary instead of meeting through a
hand-written table.

Caveat to carry into the paper: the rare benchmarks barely use these terms on the patient side --
measured, LIRICAL 0.13%, HMS 0.12%, MME 0%, and RAMEDIS's 21% is entirely mortality-outcome terms
("Death in infancy") that the evidence filter already drops. So an open attribute space widens what
the KG can hold, but the benchmark side has to be normalized into it too, and for RareBench there
is genuinely nothing to match.

  load() -> Vocab with .classes(text) -> {HP:...}, .family(hp) -> HP:... (the top-level modifier)
"""
from __future__ import annotations
import json, re, collections, functools

HP_JSON = "data/medkg/hp.json"
ROOT = "http://purl.obolibrary.org/obo/HP_0012823"      # Clinical modifier

# Terms whose match would be spurious: single very common words, or outcome/administrative
# modifiers that are not symptom qualifiers (the evidence filter drops these anyway).
_DROP_LABEL = re.compile(r"(?i)^(death|neonatal death|stillbirth|miscarriage|abortion)|death in ")
_STOP_WORDS = {"the", "of", "and", "or", "by", "in", "on", "at", "to", "a", "an", "with"}


def _hp_id(uri: str) -> str:
    return "HP:" + uri.rsplit("_", 1)[1] if "_" in uri else uri


# The relation phrases HPO uses to build a relational modifier. Everything after one of these is
# the ARGUMENT ("Triggered by exertion" -> "exertion").
_REL = re.compile(r"^(triggered by|aggravated by|ameliorated by|exacerbated by|worsened by|"
                  r"relieved by|improved by|precipitated by|provoked by)\s+", re.I)
# direction cue in ordinary clinical prose -> which HPO families may be used
_WORSE = re.compile(r"\b(worse|worsen\w*|aggravat\w*|exacerbat\w*|precipitat\w*|provok\w*|"
                    r"trigger\w*|brought on|induced|on exertion|with exertion)\b", re.I)
_BETTER = re.compile(r"\b(reliev\w*|ameliorat\w*|improv\w*|better|eased|alleviat\w*|"
                     r"responds? to|abolish\w*)\b", re.I)
# Everyday wording for an argument that HPO writes canonically. This is a BRIDGE INTO a standard
# vocabulary, not a taxonomy of our own: each entry only says which HPO argument a phrase means.
_ARG_SYNONYM = {
    "exertion": ("exercise", "exert", "activity", "walking", "walk", "climbing", "running",
                 "effort", "physical activity", "ambulation"),
    "exercise": ("exertion", "exert", "activity", "walking", "physical activity"),
    "activity": ("exertion", "exercise", "movement", "moving", "motion"),
    "cold": ("cold exposure", "low temperature", "chill"),
    "heat": ("hot", "warmth", "high temperature", "sun exposure"),
    "emotion": ("stress", "anxiety", "emotional", "upset"),
    "menstruation": ("menses", "menstrual", "period"),
    "flexion": ("bending", "bend", "stooping", "leaning forward"),
    "fasting": ("hunger", "empty stomach", "not eating", "skipping meals"),
    "food": ("meal", "meals", "eating", "ingestion", "postprandial", "after eating"),
    "carbohydrate ingestion": ("carbohydrate", "sugar", "glucose load"),
    "ethanol ingestion": ("alcohol", "ethanol", "drinking"),
    "sunlight": ("sun", "sun exposure", "uv", "ultraviolet"),
    "rest": ("resting", "lying still", "immobility", "sleep"),
    "nsaid": ("ibuprofen", "naproxen", "aspirin", "anti-inflammatory"),
    "gastric acid-reducing therapy": ("antacid", "antacids", "ppi", "proton pump inhibitor"),
    "medication": ("drug", "drugs", "medicine"),
    "palpation": ("pressing", "pressure applied", "touch", "tenderness to touch"),
    "infection": ("illness", "febrile illness", "intercurrent infection"),
}


# a caller-declared relation (the field the value came from) -> HPO families
_FIELD_FAMILIES = {
    "aggravating": {"Triggered by", "Aggravated by"},
    "relieving": {"Ameliorated by"},
    "trigger": {"Triggered by", "Aggravated by"},
}


_SYN = {" ".join(re.findall(r"[a-z0-9]+", k.lower())): v for k, v in _ARG_SYNONYM.items()}


@functools.lru_cache(maxsize=100000)
def _phrase_re(phrase: str):
    return re.compile(r"\b" + re.escape(phrase) + r"\b")


def _has_phrase(text: str, phrase: str) -> bool:
    return bool(_phrase_re(phrase).search(text))


def _direction_families(text: str) -> set[str]:
    """Which HPO relational families the text's direction cue licenses."""
    fams = set()
    if _WORSE.search(text):
        fams |= {"Triggered by", "Aggravated by"}
    if _BETTER.search(text):
        fams |= {"Ameliorated by"}
    return fams


class Vocab:
    def __init__(self, label: dict, syn: dict, parent: dict, family: dict) -> None:
        self.label = label            # HP:id -> preferred label
        self.parent = parent          # HP:id -> set of parent HP:ids
        self._family = family         # HP:id -> top-level family HP:id
        # surface form -> HP id. Longest forms are tried first so "exacerbated by exercise" wins
        # over "exercise".
        self.surface = {}
        for hp, forms in syn.items():
            for f in forms:
                key = self._norm(f)
                if len(key) >= 4 and key not in _STOP_WORDS:
                    self.surface.setdefault(key, hp)
        self._ordered = sorted(self.surface, key=len, reverse=True)
        # argument index for the relational families: "Triggered by exertion" -> "exertion"
        self.by_arg: dict[str, set[str]] = {}
        for hp, forms in syn.items():
            for f in forms:
                m = _REL.match(f.strip())
                if m:
                    arg = self._norm(f[m.end():])
                    if len(arg) >= 3:
                        self.by_arg.setdefault(arg, set()).add(hp)

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", s.lower()))

    def classes(self, text, direction: str = "") -> set[str]:
        """Free text (or token set) -> the HPO clinical modifiers it states.

        Two matching routes, because HPO writes relational modifiers as a canonical phrase that
        clinical prose never uses literally. "Triggered by exertion" is the term; the source says
        "worse on exertion" and the benchmark says "Walking".
          * direct  -- the term's own label/synonym occurs in the text ("Sharp", "Episodic",
                       "Bilateral", "Mild").
          * relational -- the text carries a DIRECTION cue (worse/relieved/...) and the term's
                       ARGUMENT ("exertion", "heat", "menstruation"); the direction selects the
                       family (Triggered/Aggravated vs Ameliorated) so that "worse with heat" and
                       "better with heat" do not collapse onto the same concept."""
        if not text:
            return set()
        t = self._norm(text if isinstance(text, str) else " ".join(text))
        if not t:
            return set()
        # word-boundary match: plain substring matching made "sitting upright" contain "right"
        # and stamped a laterality modifier onto every posture statement
        out = {self.surface[f] for f in self._ordered if _has_phrase(t, f)}
        # `direction` lets a caller supply the relation when the VALUE alone carries none: NLICE
        # stores an aggravating factor as the bare word "Walking", and DDXPlus/our KG store it in a
        # field whose name is the relation. Without this the argument can never be placed in a
        # family and the value is lost.
        # `direction` may be a known field name, or a free-form relation label the extractor
        # wrote itself ("what makes it worse"); both resolve to the same families.
        fams = (_direction_families(t) | _FIELD_FAMILIES.get(direction, set())
                | (_direction_families(self._norm(direction)) if direction else set()))
        if fams:
            for arg, hps in self.by_arg.items():
                if _has_phrase(t, arg) or any(_has_phrase(t, s) for s in _SYN.get(arg, ())):
                    out |= {h for h in hps if self.label.get(self.family(h), "") in fams}
        return out

    def family(self, hp: str) -> str:
        return self._family.get(hp, hp)

    def families(self, text, direction: str = "") -> set[str]:
        return {self.family(h) for h in self.classes(text, direction)}


@functools.lru_cache(maxsize=1)
def load(path: str = HP_JSON) -> Vocab:
    g = json.load(open(path, encoding="utf-8"))["graphs"][0]
    label, syn = {}, {}
    for n in g["nodes"]:
        uri, lbl = n.get("id", ""), n.get("lbl")
        if not lbl:
            continue
        hp = _hp_id(uri)
        label[hp] = lbl
        forms = [lbl]
        for s in (n.get("meta") or {}).get("synonyms", []) or []:
            if s.get("pred") in ("hasExactSynonym", "hasNarrowSynonym") and s.get("val"):
                forms.append(s["val"])
        syn[hp] = forms
    kids = collections.defaultdict(set)
    parent = collections.defaultdict(set)
    for e in g["edges"]:
        if e.get("pred") == "is_a":
            kids[_hp_id(e["obj"])].add(_hp_id(e["sub"]))
            parent[_hp_id(e["sub"])].add(_hp_id(e["obj"]))
    root = _hp_id(ROOT)

    # the modifier subtree, and which top-level family each member belongs to
    family, seen = {}, set()
    for top in kids.get(root, ()):
        stack = [top]
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c); family[c] = top
            stack.extend(kids.get(c, ()))
    keep = {h for h in seen if not _DROP_LABEL.search(label.get(h, ""))}
    return Vocab({h: label[h] for h in keep if h in label},
                 {h: syn[h] for h in keep if h in syn}, parent, family)


if __name__ == "__main__":
    v = load()
    print(f"clinical modifiers: {len(v.label)}  surface forms: {len(v.surface)}  "
          f"families: {len(set(v._family.values()))}")
    for t in ["worse on exertion", "Walking", "relieved by sitting forward", "burning",
              "nocturnal", "progressively worsening", "after a heavy meal", "lying supine"]:
        cs = v.classes(t)
        print(f"  {t[:30]:32} -> {[v.label[c] for c in cs][:3]}  fam="
              f"{[v.label.get(v.family(c), '') for c in cs][:2]}")
