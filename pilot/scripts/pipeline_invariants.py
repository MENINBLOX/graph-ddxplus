"""Seam checks for the source -> IE -> normalize -> KG pipeline.

Every defect found in this pipeline so far degraded quality WITHOUT raising an error, so it
survived until someone opened the raw artifacts by hand:

  1. sources     - a 12,000-char cap discarded 51% of GeneReviews phenotype prose (100% of chapters)
  2. sources     - 31% of chapters collapsed onto a generic CUI ("Disease"), erasing 272 diseases
  3. IE          - an output over max_new produced unparseable JSON, discarding the whole window
  4. IE cleaner  - timing/aggravating/relieving were extracted and then dropped by the cleaner

Each is a violation of an obvious conservation property between two stages. This module states
those properties and reports violations, so the next one is caught by running a check rather than
by noticing a score that failed to improve. Reporting only -- it never edits artifacts.

Usage: python pipeline_invariants.py --sources DIR [--preds F] [--records F] [--kg F]
Exit code 1 if any check fails, so it can gate a rebuild.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from collections import Counter

# a "generic" concept is one that many distinct source documents collapse onto; treating it as a
# disease node both loses those diseases and creates a hub. See disease_link.py.
SINK_SHARE_MAX = 0.05        # >5% of sources sharing one CUI means the linker is over-generalizing
CAP_SHARE_MAX = 0.10         # >10% of blobs at exactly the same length means a hard cap is active
ZERO_FINDING_MAX = 0.05      # >5% of documented diseases yielding nothing means the IE is failing


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str) -> None:
        self.rows.append((ok, name, detail))

    def failed(self) -> int:
        return sum(1 for ok, _, _ in self.rows if not ok)

    def show(self) -> None:
        for ok, name, detail in self.rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:34} {detail}")
        print(f"\n{len(self.rows) - self.failed()}/{len(self.rows)} checks passed")


def check_sources(d: str, r: Report) -> None:
    blobs = [json.load(open(p, encoding="utf-8"))
             for p in glob.glob(os.path.join(d, "*.json")) if not p.endswith("manifest.json")]
    if not blobs:
        r.check(False, "sources/present", f"no source blobs in {d}")
        return
    lens = [len(b.get("text", "")) for b in blobs]
    # (1) truncation: a hard cap shows up as many blobs sharing one exact length
    top_len, top_n = Counter(lens).most_common(1)[0]
    share = top_n / len(lens)
    r.check(share <= CAP_SHARE_MAX, "sources/no-hard-cap",
            f"{top_n}/{len(lens)} ({share:.0%}) blobs are exactly {top_len} chars"
            + (" -> a length cap is truncating the corpus" if share > CAP_SHARE_MAX else ""))
    # (2) identity collapse: sources sharing one CUI
    ids = Counter(b.get("cui") for b in blobs)
    (top_cui, n_top), = ids.most_common(1)
    r.check(n_top / len(blobs) <= SINK_SHARE_MAX, "sources/no-cui-sink",
            f"largest CUI group {top_cui} holds {n_top}/{len(blobs)} sources")
    r.check(True, "sources/size", f"{len(blobs)} blobs, median {sorted(lens)[len(lens)//2]} chars")


def check_preds(f: str, src_dir: str | None, r: Report) -> None:
    preds = json.load(open(f, encoding="utf-8"))
    nf = [len(v.get("findings", [])) for v in preds.values()]
    zero = sum(1 for x in nf if x == 0)
    r.check(zero / max(1, len(nf)) <= ZERO_FINDING_MAX, "ie/no-empty-extractions",
            f"{zero}/{len(nf)} ({zero/max(1,len(nf)):.0%}) diseases yielded 0 findings")
    # (4) attribute schema: the prompt asks for 7 attributes; a cleaner that keeps fewer is a
    # silent schema truncation, not an empty corpus
    keys: set[str] = set()
    for v in list(preds.values())[:500]:
        for x in v.get("findings", []):
            keys |= set(x)
    want = {"location", "character", "severity", "onset", "timing", "aggravating", "relieving"}
    missing = sorted(want - keys)
    r.check(not missing, "ie/attribute-schema-intact",
            "all 7 attribute fields survive the cleaner" if not missing
            else f"fields dropped before storage: {missing}")
    # (3) every documented disease should have been read
    if src_dir:
        srcs = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(src_dir, "*.json"))
                if not p.endswith("manifest.json")}
        lost = srcs - set(preds)
        r.check(not lost, "ie/every-source-consumed",
                f"{len(lost)} sources have no IE record" if lost else f"{len(srcs)} sources all consumed")


def check_records(f: str, preds_f: str | None, r: Report) -> None:
    recs = json.load(open(f, encoding="utf-8"))
    if preds_f:
        preds = json.load(open(preds_f, encoding="utf-8"))
        lost = set(preds) - set(recs)
        r.check(not lost, "normalize/no-disease-lost",
                f"{len(lost)} diseases dropped in normalization" if lost
                else f"{len(recs)} diseases carried through")
    n = sum(len(v.get("phenotypes", [])) for v in recs.values())
    linked = sum(1 for v in recs.values() for p in v.get("phenotypes", []) if p.get("name_cui"))
    r.check(linked / max(1, n) >= 0.8, "normalize/cui-link-rate",
            f"{linked}/{n} ({linked/max(1,n):.0%}) findings linked to a CUI")


def check_kg(f: str, r: Report) -> None:
    kg = json.load(open(f, encoding="utf-8"))
    prof: dict[str, set] = {}
    for e in kg["edges"]:
        prof.setdefault(e["d"], set()).add(e["p"])
    degs = sorted(len(v) for v in prof.values())
    thin = sum(1 for x in degs if x <= 5) / max(1, len(degs))
    # CLAUDE.md §12: >10% of diseases with <=5 evidence means the IE is incomplete. This is a
    # KNOWN-FAILING check kept visible on purpose -- it is the current research bottleneck.
    r.check(thin <= 0.10, "kg/disease-degree(§12)",
            f"{thin:.0%} of diseases have <=5 phenotypes (median degree {degs[len(degs)//2]})")
    has_prov = sum(1 for i in kg["diseases"].values() if i.get("n_src_chars"))
    r.check(has_prov == len(kg["diseases"]), "kg/documentation-provenance",
            f"{has_prov}/{len(kg['diseases'])} diseases record how much source they were built from")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="")
    ap.add_argument("--preds", default="")
    ap.add_argument("--records", default="")
    ap.add_argument("--kg", default="")
    a = ap.parse_args()
    r = Report()
    if a.sources:
        check_sources(a.sources, r)
    if a.preds:
        check_preds(a.preds, a.sources or None, r)
    if a.records:
        check_records(a.records, a.preds or None, r)
    if a.kg:
        check_kg(a.kg, r)
    r.show()
    sys.exit(1 if r.failed() else 0)


if __name__ == "__main__":
    main()
