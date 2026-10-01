"""NLICE attribute-aware loader for onlykg_eval_unified2 (findings-with-attributes format).

NLICE (arXiv:2401.13756) is SymCat-grounded synthetic patients WITH per-symptom Bates
attributes -- the controlled setting where the attribute contribution can be measured
(the other public benchmarks lack separable query-side attributes; see
docs/attribute_benchmark_analysis.md). NLICE SYMPTOMS token = ':'-delimited 10 fields:
  0 name | 1 nature | 2 location | 3 sublocation | 4 intensity | 5 duration | 6 onset |
  7 excitation | 8 frequency | 9 prob%
Mapped to our normalized schema:
  character <- nature (tokens)      location <- location+sublocation (CUI + tokens)
  severity  <- intensity (NRS/word) onset    <- onset (sudden/gradual)

Exposes `load(cap=150, nlice_dir=..., kg=...) -> (cand, patients)` with
  patient = (target_keys, findings, "NLICE"),
  finding = {"cui", "location":set[cui], "location_txt":set[tok], "character":set[tok],
             "severity": mild|moderate|severe|None, "onset": sudden|gradual|None}
matching the format consumed by attr_score.attr_delta.
"""
from __future__ import annotations
import json, os, re, glob
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from umls_link import link as umls_link
from attr_normalize import normalize_onset, normalize_severity

FINDING_TUI = {"T184", "T047", "T048", "T046", "T191", "T019", "T190", "T037",
               "T020", "T049", "T050", "T033", "T034"}
LOC_TUI = {"T023", "T029", "T030", "T024", "T025", "T018", "T031", "T022", "T017"}
NLICE_DIR = ("/tmp/claude-1000/-home-max-thesis/"
             "edda7dda-7b76-45bb-a79a-ce62d7e31a1b/scratchpad/NLICE/sample_data")


def _clean(s: str) -> str:
    s = s.replace("_", " ").strip()
    return re.sub(r"\(.*?\)", "", s).strip()


def _toks(s: str) -> set:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2}


def load(cap: int = 150, nlice_dir: str = NLICE_DIR, kg: str = "pilot/cache/prod/kg.json"):
    kgj = json.load(open(kg))
    kg_dz = set(kgj["diseases"])

    _dz, _sym, _loc = {}, {}, {}
    def link_dz(n):
        if n not in _dz:
            _dz[n] = umls_link(_clean(n), FINDING_TUI)[0]
        return _dz[n]
    def link_sym(n):
        if n not in _sym:
            _sym[n] = umls_link(_clean(n), FINDING_TUI)[0]
        return _sym[n]
    def link_loc(n):
        if n not in _loc:
            _loc[n] = umls_link(_clean(n), LOC_TUI)[0]
        return _loc[n]

    cand, patients = {}, []
    for fp in sorted(glob.glob(os.path.join(nlice_dir, "*.csv"))):
        dname = os.path.basename(fp)[:-4]
        tcui = link_dz(dname)
        if not tcui or tcui not in kg_dz:
            continue
        cand[dname] = (dname, tcui)
        kept = 0
        with open(fp, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                if kept >= cap:
                    break
                cell = line.rstrip("\n").split(",")[-1]
                if not cell:
                    continue
                findings = []
                for tok in cell.split(";"):
                    p = tok.split(":")
                    if not p or not p[0].strip():
                        continue
                    p += [""] * (10 - len(p))
                    name, nature, loc1, loc2, intens, dur, onset, exc, fq, _pr = p[:10]
                    scui = link_sym(name)
                    if not scui:
                        continue
                    # excitation and frequency were discarded (`_exc`, `_fq`). NLICE is the only
                    # benchmark stating an AGGRAVATING factor on the patient side (excitation filled
                    # 12% of findings) and a temporal pattern (frequency, 7%), so dropping them made
                    # three of the six candidate attributes untestable. duration is kept as free
                    # text only and is NOT scored (CLAUDE.md: timing sub-field, excluded).
                    f = {"cui": scui, "location": set(), "location_txt": set(),
                         "character": set(), "severity": None, "onset": None,
                         "timing": "", "aggravating": "", "relieving": "", "duration": ""}
                    if nature.strip():
                        f["character"] = _toks(_clean(nature))
                    for lc in (loc1, loc2):
                        if lc.strip():
                            f["location_txt"] |= _toks(_clean(lc))
                            c = link_loc(lc)
                            if c:
                                f["location"].add(c)
                    if intens.strip():
                        f["severity"] = normalize_severity(_clean(intens)) or None
                    if onset.strip():
                        f["onset"] = normalize_onset(_clean(onset)) or None
                    if exc.strip():
                        f["aggravating"] = _clean(exc)
                    if fq.strip():
                        f["timing"] = _clean(fq)
                    if dur.strip():
                        f["duration"] = _clean(dur)
                    findings.append(f)
                if findings:
                    patients.append(([dname], findings, "NLICE"))
                    kept += 1
    return cand, patients


if __name__ == "__main__":
    import argparse
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    a = ap.parse_args()
    cand, pats = load(a.cap, kg=a.kg)
    nf = sum(len(p[1]) for p in pats)
    d = Counter()
    for _, fs, _ in pats:
        for f in fs:
            for k in ("location", "location_txt", "character", "severity", "onset",
                      "timing", "aggravating", "relieving", "duration"):
                if f[k]:
                    d[k] += 1
    print(f"diseases={len(cand)} patients={len(pats)} findings={nf} ({nf/max(1,len(pats)):.1f}/pt)")
    for k in ("character", "location", "location_txt", "severity", "onset",
              "timing", "aggravating", "relieving", "duration"):
        print(f"  {k:12} {d[k]:6} ({100*d[k]/max(1,nf):.1f}%)")
