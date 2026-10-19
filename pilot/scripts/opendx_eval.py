"""Open-set differential diagnosis evaluation (research design D3, D12, D14).

Every patient is ranked against EVERY disease node of the KG, not against the benchmark's own label
list. Only three kinds of information can move a score (design §3): literature IE (the KG), concept
vocabulary/hierarchy, and the patient's own findings. Nothing here reads benchmark-wide statistics.

Gold matching (design §8-6, provisional): a prediction is correct if it is the gold concept
("strict") or the gold concept or one of its descendants in the disease hierarchy ("lineage"; a
more specific diagnosis than the gold label implies the gold). Both are reported.

    python opendx_eval.py --kg KG.json --bench ddxplus-validate,rarebench,pumch,phenopacket,symcat
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import zipfile
from pathlib import Path

DATA = Path("data")
UM = Path("pilot/cache/umls_maps")
KS = (1, 3, 10)
RAW_GOLD = False   # set by graph-based evaluation to also return original gold identifiers
# Confirmation set E (research design E2). The development loaders below refuse its DDXPlus rows and
# its SymCat seed; load_confirm_e at the end of the file is the only reader.
CONFIRM_E_FLAG = "CONFIRM_E_UNSEAL"
CONFIRM_E_MANIFEST = DATA / "confirm_e/MANIFEST.json"
CONFIRM_E_SYMCAT_SEED = 202


# ----------------------------------------------------------------------------- benchmarks
def _rb_maps():
    dis = json.load(open(DATA / "rarebench/disease_umls_mapping.json"))["mapping"]
    hpo = json.load(open(DATA / "rarebench/hpo_umls_mapping.json"))["mapping"]
    om = json.load(open(UM / "omim_orpha_to_cui.json"))

    def dz(i: str) -> set[str]:
        out = set(om["omim"].get(i, [])) | set(om["orpha"].get(i, []))
        if dis.get(i, {}).get("umls_cui"):
            out.add(dis[i]["umls_cui"])
        if RAW_GOLD:
            out.add(i)          # keep the original OMIM/ORPHA id for MONDO-keyed graphs
        return out

    def hp(h: str) -> str | None:
        v = hpo.get(h)
        return v.get("umls_cui") if isinstance(v, dict) else v if isinstance(v, str) else None
    return dz, hp


# DDXPlus region cues (English value meanings) -> region-specific symptom concept. DDXPlus encodes
# "chest pain" as pain + location value; the literature names the specific symptom.
_REGION_SYMPTOM = [
    ({"chest", "thorax", "thoracic", "breast", "sternum", "sternal", "pectoral"}, "C0008031"),  # chest pain
    ({"abdomen", "abdominal", "epigastric", "epigastrium", "hypochondrium", "flank", "iliac",
      "umbilicus", "umbilical", "hypogastrium", "belly", "pelvis"}, "C0000737"),             # abdominal pain
    ({"back", "lumbar", "loin", "scapula"}, "C0004604"),                                      # back pain
    ({"head", "temple", "forehead", "occiput", "skull"}, "C0018681"),                          # headache
    ({"throat", "pharynx", "tonsil"}, "C0242429"),                                            # sore throat
]


def ddx_crosswalk():
    """DDXPlus evidence tokens -> UMLS concepts, value-aware (design §3: vocabulary mapping only).

    A token "question_@_value" whose value is the question's DEFAULT answer ("N", "nowhere", "NA",
    0) is not evidence: v1 mapped every token by its question code alone, so "trav1_@_N" (did NOT
    travel) became "Travel History" in 93% of patients. Pain locations are composed into the
    region-specific symptom the literature uses (pain + chest -> chest pain)."""
    meta = json.load(open(DATA / "ddxplus/release_evidences.json"))
    umap = json.load(open(DATA / "ddxplus/umls_mapping.json"))["mapping"]

    def base_cui(code):
        for c in (code, code.split("_")[0]):
            if c in umap:
                return umap[c]["cui"]
        return None

    def convert(tokens):
        out = set()
        for t in tokens:
            code, _, val = t.partition("_@_")
            m = meta.get(code, {})
            if val and str(val) == str(m.get("default_value")):
                continue
            c = base_cui(code)
            if c:
                out.add(c)
            if val and code.endswith("_endroitducorps") and code.startswith("douleur"):
                en = str((m.get("value_meaning", {}).get(val) or {}).get("en", val)).lower()
                words = set(re.findall(r"[a-z]+", en))
                for cues, cui in _REGION_SYMPTOM:
                    if words & cues:
                        out.add(cui)
        return out
    vocab = {base_cui(k) for k in meta} | {c for _, c in _REGION_SYMPTOM}
    return convert, {c for c in vocab if c}


def ddx_sample_indices(n_rows: int, n: int, seed: int) -> list[int]:
    """Row indices of the DDXPlus sample drawn by load_ddxplus, in draw order.

    random.sample chooses by position only, so sampling range(n_rows) draws exactly the rows that
    sampling the row list itself draws."""
    return random.Random(seed).sample(range(n_rows), min(n, n_rows))


def _ddx_patients(rows) -> list:
    cond = json.load(open(DATA / "ddxplus/release_conditions_en.json"))
    dmap = json.load(open(DATA / "ddxplus/disease_umls_mapping_v2.json"))["mapping"]
    fr2en = {info.get("cond-name-fr") or info.get("condition_name"): en for en, info in cond.items()}
    convert, _ = ddx_crosswalk()
    pats = []
    for r in rows:
        en = fr2en.get(r["PATHOLOGY"])
        gold = dmap.get(en, {}).get("umls_cui")
        ev = convert(ast.literal_eval(r["EVIDENCES"]))
        if gold and ev:
            pats.append(({gold}, ev))
    return pats


def load_ddxplus(split: str, n: int, seed: int = 0):
    rows = list(csv.DictReader(open(DATA / f"ddxplus/release_{split}_patients.csv")))
    idx = ddx_sample_indices(len(rows), n, seed)
    # A missing manifest is an error, not a pass: deleting it must not open the sealed rows.
    if split == "test":
        sealed = set(json.load(open(CONFIRM_E_MANIFEST))["ddxplus"]["row_indices"])
        if not sealed.isdisjoint(idx):
            raise RuntimeError(f"DDXPlus test sample (n={n}, seed={seed}) contains rows of the sealed "
                               f"confirmation set E ({CONFIRM_E_MANIFEST})")
    return {f"DDXPlus-{split}": _ddx_patients(rows[i] for i in idx)}


def load_rarebench():
    dz, hp = _rb_maps()
    out = {}
    for ds in ("RAMEDIS", "LIRICAL", "HMS", "MME"):
        pats = []
        for line in open(DATA / f"rarebench/data/{ds}.jsonl"):
            r = json.loads(line)
            gold = set().union(*(dz(t) for t in r["RareDisease"])) if r["RareDisease"] else set()
            ev = {c for c in (hp(h) for h in r["Phenotype"]) if c}
            pats.append((gold, ev))
        out[f"RareBench-{ds}"] = pats
    return out


def load_pumch():
    dz, hp = _rb_maps()
    out = {}
    for ds in ("PUMCH_L", "PUMCH-ADM"):
        d = json.load(open(DATA / f"external_benchmarks/phenobrain/Test cases/{ds}.json"))
        out[ds] = [(set().union(*(dz(t) for t in g)) if g else set(),
                    {c for c in (hp(h) for h in ph) if c}) for ph, g in d]
    return out


def load_phenopacket():
    dz, hp = _rb_maps()
    z = zipfile.ZipFile(DATA / "external_benchmarks/phenopacket_store/all_phenopackets.zip")
    pats = []
    for name in sorted(z.namelist()):
        if not name.endswith(".json"):
            continue
        p = json.loads(z.read(name))
        gold = set().union(*(dz(d["term"]["id"]) for d in p.get("diseases", []))) if p.get("diseases") else set()
        ev = {c for c in (hp(f["type"]["id"]) for f in p.get("phenotypicFeatures", [])
                          if not f.get("excluded")) if c}
        pats.append((gold, ev))
    return {"Phenopacket-store": pats}


def load_symcat(ppd: int, seed: int = 0):
    """SymCat's own protocol: a synthetic patient reports each symptom with the disease's listed
    probability. All 801 diseases, seeded."""
    if seed == CONFIRM_E_SYMCAT_SEED and os.environ.get(CONFIRM_E_FLAG) != "1":
        raise RuntimeError(f"SymCat seed {seed} is the sealed confirmation set E; use load_confirm_e")
    full = json.load(open(DATA / "symcat/symcat_parsed_full.json"))
    sym = json.load(open(DATA / "symcat/symptom_umls_mapping.json"))["mapping"]
    dmap = {v.get("symcat_slug"): v for v in json.load(open(DATA / "symcat/disease_umls_mapping.json"))["mapping"].values()}
    s2c = {i["symcat_slug"]: i["umls_cui"] for i in sym.values() if i.get("umls_cui")}
    rng = random.Random(seed)
    pats = []
    for slug, info in sorted(full["diseases"].items()):
        gold = dmap.get(slug, {}).get("umls_cui")
        syms = [(s2c.get(s), float(v["probability"])) for s, v in (info.get("symptoms") or {}).items()]
        syms = [(c, p) for c, p in syms if c]
        if not gold or not syms:
            continue
        for _ in range(ppd):
            ev = {c for c, p in syms if rng.random() * 100 < p}
            if ev:
                pats.append(({gold}, ev))
    return {"SymCat": pats}


# ----------------------------------------------------------------------------- scorer
class OpenScorer:
    def __init__(self, kg_path: str, alpha: float, hier: int, decay: float, unified: bool,
                 cov_mu: float = 0.0):
        kg = json.load(open(kg_path))
        self.prof: dict[str, set[str]] = collections.defaultdict(set)
        for e in kg["edges"]:
            self.prof[e["d"]].add(e["p"])
        self.dz = set(kg["diseases"])
        self.inv: dict[str, list[str]] = collections.defaultdict(list)
        for d, ps in self.prof.items():
            for p in ps:
                self.inv[p].append(d)
        n = len(self.prof)
        self.idf = {p: math.log(n / len(ds)) for p, ds in self.inv.items()}
        # Coverage smoothing (design §3: KG-derived only). A disease we read little about has a
        # short profile because it is under-documented, not because it is specific; without this the
        # open set is won by 1-3-edge diseases. Effective length n' = n + mu*(1-c)*Lbar with
        # documentation coverage c = s/(s+s_med) from the source text size recorded in the KG.
        src = {d: (info or {}).get("n_src_chars") or 0 for d, info in kg["diseases"].items()}
        med = sorted(v for v in src.values() if v)
        s_med = med[len(med) // 2] if med else 1
        lbar = sum(len(ps) for ps in self.prof.values()) / max(1, len(self.prof))
        self.norm = {}
        for d, ps in self.prof.items():
            c = src.get(d, 0) / (src.get(d, 0) + s_med)
            self.norm[d] = max(1.0, len(ps) + cov_mu * (1.0 - c) * lbar) ** alpha
        self.parents = json.load(open(UM / "phenotype_parents.json")) if hier else {}
        self.hier, self.decay, self.unified = hier, decay, unified
        ch = json.load(open(UM / "disease_children.json"))
        self.children = {k: set(v) for k, v in ch.items()}

    def lineage(self, cui: str, depth: int = 3) -> set[str]:
        out, frontier = {cui}, {cui}
        for _ in range(depth):
            frontier = {c for f in frontier for c in self.children.get(f, ())} - out
            out |= frontier
        return out

    def expand(self, ev: set[str]) -> dict[str, float]:
        w = {c: 1.0 for c in ev}
        frontier = dict(w)
        for _ in range(self.hier):
            nxt = {}
            for c, v in frontier.items():
                for p in self.parents.get(c, []):
                    if p not in w:
                        nxt[p] = max(nxt.get(p, 0.0), v * self.decay)
            w.update(nxt)
            frontier = nxt
        return w

    def rank(self, ev: set[str]) -> list[str]:
        q = self.expand(ev)
        score: dict[str, float] = collections.defaultdict(float)
        for c, w in q.items():
            idf = self.idf.get(c)
            if idf is None:
                continue
            for d in self.inv[c]:
                score[d] += w * idf
        out = {d: s / self.norm[d] for d, s in score.items()}
        if self.unified:   # a patient finding that IS a disease concept supports that lineage
            for c in ev:
                if c in self.dz:
                    for d in self.lineage(c, 1):
                        out[d] = out.get(d, 0.0) + self.idf.get(c, 1.0)
        return sorted(out, key=lambda d: (-out[d], d))


def evaluate(scorer: OpenScorer, name: str, pats, boot: int = 0):
    n_all = len(pats)
    pats = [(g, e) for g, e in pats if g and e]
    in_kg = [(g, e) for g, e in pats if g & scorer.dz]
    hits = {r: {k: [] for k in KS} for r in ("strict", "lineage")}
    for gold, ev in in_kg:
        ranked = scorer.rank(ev)[:max(KS)]
        lin = set().union(*(scorer.lineage(g) for g in gold))
        for rule, ok in (("strict", gold), ("lineage", lin)):
            pos = next((i + 1 for i, d in enumerate(ranked) if d in ok), None)
            for k in KS:
                hits[rule][k].append(pos is not None and pos <= k)
    m = len(in_kg)
    res = {"bench": name, "patients": n_all, "usable": len(pats), "gold_in_kg": m}
    for rule in hits:
        for k in KS:
            res[f"{rule}@{k}"] = round(100 * sum(hits[rule][k]) / max(1, m), 2)
    print(f"{name:20} n={m:6} (of {n_all}) | strict " +
          " ".join(f"@{k} {res[f'strict@{k}']:5.1f}" for k in KS) + " | lineage " +
          " ".join(f"@{k} {res[f'lineage@{k}']:5.1f}" for k in KS), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg_merged29.json")
    ap.add_argument("--bench", default="ddxplus-validate,rarebench,pumch,phenopacket,symcat")
    ap.add_argument("--ddx_n", type=int, default=3000)
    ap.add_argument("--symcat_ppd", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="length-normalization exponent")
    ap.add_argument("--hier", type=int, default=1)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--unified", type=int, default=1)
    ap.add_argument("--cov_mu", type=float, default=0.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sc = OpenScorer(a.kg, a.alpha, a.hier, a.decay, bool(a.unified), a.cov_mu)
    print(f"open set: {len(sc.prof)} diseases with edges, {len(sc.inv)} phenotype concepts", flush=True)
    sets = {}
    for b in a.bench.split(","):
        if b.startswith("ddxplus-"):
            sets.update(load_ddxplus(b.split("-", 1)[1], a.ddx_n))
        elif b == "rarebench":
            sets.update(load_rarebench())
        elif b == "pumch":
            sets.update(load_pumch())
        elif b == "phenopacket":
            sets.update(load_phenopacket())
        elif b == "symcat":
            sets.update(load_symcat(a.symcat_ppd))
    res = [evaluate(sc, k, v) for k, v in sets.items()]
    if a.out:
        json.dump({"args": vars(a), "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- confirmation sets
CONFIRM_FLAG = "CONFIRM_UNSEAL"


def load_confirm():
    """Sealed confirmation sets (data/confirm/MANIFEST.json, research design D).

    Evaluated once, after the design and the system are frozen. Refuses to load unless the
    environment variable CONFIRM_UNSEAL=1 is set, so development runs cannot touch them.
    """
    import os
    if os.environ.get(CONFIRM_FLAG) != "1":
        raise RuntimeError("confirmation sets are sealed; set CONFIRM_UNSEAL=1 only for the final run")
    man = json.load(open(DATA / "confirm/MANIFEST.json"))
    out = {}
    import ast
    cond = json.load(open(DATA / "ddxplus/release_conditions_en.json"))
    dmap = json.load(open(DATA / "ddxplus/disease_umls_mapping_v2.json"))["mapping"]
    fr2en = {info.get("cond-name-fr") or info.get("condition_name"): en for en, info in cond.items()}
    convert, _ = ddx_crosswalk()
    rows = list(csv.DictReader(open(DATA / "ddxplus/release_test_patients.csv")))
    pats = []
    for i in man["ddxplus"]["confirm_row_indices"]:
        r = rows[i]
        gold = dmap.get(fr2en.get(r["PATHOLOGY"]), {}).get("umls_cui")
        ev = convert(ast.literal_eval(r["EVIDENCES"]))
        if gold and ev:
            pats.append(({gold}, ev))
    out["Confirm-DDXPlus"] = pats
    dz, hp = _rb_maps()
    skip = set(man["phenopacket_new"]["excluded_source_pmid_in_corpus"])
    pats = []
    for name in sorted(man["phenopacket_new"]["sha256"]):
        if name in skip:
            continue
        p = json.load(open(DATA / "confirm/phenopacket_new" / name))
        gold = set().union(*(dz(d["term"]["id"]) for d in p.get("diseases", []))) if p.get("diseases") else set()
        ev = {c for c in (hp(f["type"]["id"]) for f in p.get("phenotypicFeatures", [])
                          if not f.get("excluded")) if c}
        pats.append((gold, ev))
    out["Confirm-Phenopacket-new"] = pats
    out["Confirm-SymCat"] = load_symcat(man["symcat"]["patients_per_disease"], seed=man["symcat"]["seed"])["SymCat"]
    return out


def _check_sha256(path: Path, want: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != want:
        raise RuntimeError(f"{path} differs from the file sealed in {CONFIRM_E_MANIFEST}")


def load_confirm_e():
    """Sealed confirmation set E (data/confirm_e/MANIFEST.json, research design E2).

    New patients of the development diseases: DDXPlus test rows outside the seed-0 development
    sample and the D confirmation rows, and SymCat seed 202. Patients are built by the development
    loaders' code. Evaluated once, after the systems are frozen; refuses to load unless the
    environment variable CONFIRM_E_UNSEAL=1 is set.
    """
    if os.environ.get(CONFIRM_E_FLAG) != "1":
        raise RuntimeError("confirmation set E is sealed; set CONFIRM_E_UNSEAL=1 only for the final run")
    man = json.load(open(CONFIRM_E_MANIFEST))
    ddx, sym = man["ddxplus"], man["symcat"]
    csv_path = DATA / "ddxplus/release_test_patients.csv"
    _check_sha256(csv_path, ddx["csv_sha256"])
    for rel, want in sym["sha256"].items():
        _check_sha256(DATA / rel, want)
    if sym["seed"] != CONFIRM_E_SYMCAT_SEED:
        raise RuntimeError(f"manifest SymCat seed {sym['seed']} != {CONFIRM_E_SYMCAT_SEED}")
    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) != ddx["n_rows"]:
        raise RuntimeError(f"{csv_path}: {len(rows)} rows, manifest says {ddx['n_rows']}")
    return {"ConfirmE-DDXPlus": _ddx_patients(rows[i] for i in ddx["row_indices"]),
            "ConfirmE-SymCat": load_symcat(sym["patients_per_disease"], seed=sym["seed"])["SymCat"]}
