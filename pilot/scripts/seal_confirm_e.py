"""Seal confirmation set E (research design E2): writes data/confirm_e/MANIFEST.json.

Selects DDXPlus test rows by index and hash only; no patient is converted or printed. Deterministic
except the Phenopacket-store check, which asks GitHub whether main has a commit after the base
dated on or before the snapshot date and stops if it does. If the manifest already exists it is
compared with a fresh computation and never overwritten.

    python pilot/scripts/seal_confirm_e.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402

CSV = O.DATA / "ddxplus/release_test_patients.csv"
CONFIRM_D = O.DATA / "confirm/MANIFEST.json"
DEV_N, DEV_SEED = 3000, 0          # load_ddxplus("test", 3000): the development sample
CONFIRM_N = 3000
SYMCAT_PPD = 10
SYMCAT_FILES = ("symcat/symcat_parsed_full.json", "symcat/symptom_umls_mapping.json",
                "symcat/disease_umls_mapping.json")
PP_REPO = "monarch-initiative/phenopacket-store"
PP_BASE = "85ffd78f3c17442a7dbb143df916ed6f8680066e"
PP_SNAPSHOT = "2026-10-19"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gh(path: str, jq: str) -> str:
    return subprocess.run(["gh", "api", path, "--jq", jq], check=True, capture_output=True,
                          text=True).stdout.strip()


def ddxplus() -> dict:
    lines = CSV.read_bytes().split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    header, raw = lines[0], lines[1:]
    with open(CSV, newline="") as f:
        records = list(csv.reader(f))[1:]
    # Row index i must name data row i of csv.DictReader (what the loaders index) and physical line
    # i + 1, so the per-row hash identifies the same patient; a quoted newline would break this.
    if len(records) != len(raw) or any(not r for r in records) or any(b"\r" in ln for ln in raw):
        raise SystemExit(f"{CSV}: records do not map one-to-one onto lines")
    n_rows = len(raw)
    dev = set(O.ddx_sample_indices(n_rows, DEV_N, DEV_SEED))
    confirm_d = set(json.load(open(CONFIRM_D))["ddxplus"]["confirm_row_indices"])
    if dev & confirm_d:
        raise SystemExit("development sample and D confirmation rows overlap")
    remaining = set(range(n_rows)) - dev - confirm_d
    rows = sorted(random.Random(202).sample(sorted(remaining), CONFIRM_N))
    for i in rows:
        if next(csv.reader(io.StringIO(raw[i].decode("utf-8")))) != records[i]:
            raise SystemExit(f"row {i}: raw line does not parse to the csv record")
    return {
        "split": "test",
        "csv": str(CSV),
        "csv_sha256": sha256_file(CSV),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "n_rows": n_rows,
        "excluded": {
            "dev_sample": f"load_ddxplus('test', {DEV_N}) = opendx_eval.ddx_sample_indices(n_rows, {DEV_N}, "
                          f"seed={DEV_SEED})",
            "dev_rows": len(dev),
            "confirm_d": f"{CONFIRM_D} ddxplus.confirm_row_indices",
            "confirm_d_rows": len(confirm_d),
        },
        "pool_rows": len(remaining),
        "seed": 202,
        "draw": "sorted(random.Random(202).sample(sorted(pool_row_indices), 3000))",
        "row_index": "0-based data row of csv.DictReader = physical line index + 1 (line 0 is the header)",
        "row_hash": "sha256 of the raw CSV line, UTF-8 bytes without the trailing newline",
        "row_indices": rows,
        "row_sha256": [hashlib.sha256(raw[i]).hexdigest() for i in rows],
        "overlap_with_dev": len(dev.intersection(rows)),
        "overlap_with_confirm_d": len(confirm_d.intersection(rows)),
    }


def phenopacket() -> dict:
    head = gh(f"repos/{PP_REPO}/commits/main", ".sha")
    status = gh(f"repos/{PP_REPO}/compare/{PP_BASE}...{head}", ".status")
    base_date = gh(f"repos/{PP_REPO}/commits/{PP_BASE}", ".commit.committer.date")
    # UTC end of day is later than the end of the snapshot day in KST, so this window is a superset.
    later = gh(f"repos/{PP_REPO}/commits?sha=main&since={base_date}&until={PP_SNAPSHOT}T23:59:59Z&per_page=100",
               "[.[].sha] | join(\" \")").split()
    later = [s for s in later if s != PP_BASE]
    if status not in ("identical", "ahead") or later:
        raise SystemExit(f"phenopacket-store main has commits after {PP_BASE[:7]} by {PP_SNAPSHOT}: "
                         f"compare={status}, {len(later)} commits; the design assumes none")
    return {
        "snapshot_date": PP_SNAPSHOT,
        "base_commit": PP_BASE,
        "new_cases": 0,
        "note": "no commits after the base on the snapshot date; no rare-disease confirmation set",
        "main_head_seen": head,
    }


def main() -> None:
    man = {
        "created": PP_SNAPSHOT,
        "purpose": "confirmation set E for research design E (evaluate once, after the systems are frozen); "
                   "new patients of the development diseases",
        "design_e_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                                          text=True).stdout.strip(),
        "unseal": f"opendx_eval.load_confirm_e() with {O.CONFIRM_E_FLAG}=1",
        "ddxplus": ddxplus(),
        "symcat": {
            "patients_per_disease": SYMCAT_PPD,
            "seed": O.CONFIRM_E_SYMCAT_SEED,
            "note": "development used seed 0, D confirmation seed 101",
            "sha256": {rel: sha256_file(O.DATA / rel) for rel in SYMCAT_FILES},
        },
        "phenopacket": phenopacket(),
    }
    d = man["ddxplus"]
    print(f"DDXPlus test rows {d['n_rows']}: dev {d['excluded']['dev_rows']}, D confirm "
          f"{d['excluded']['confirm_d_rows']}, pool {d['pool_rows']}, sealed {len(d['row_indices'])} "
          f"(overlap dev {d['overlap_with_dev']}, D {d['overlap_with_confirm_d']})")
    print(f"SymCat seed {man['symcat']['seed']}, {SYMCAT_PPD} per disease; "
          f"Phenopacket new cases {man['phenopacket']['new_cases']}")
    out = O.CONFIRM_E_MANIFEST
    if out.exists():
        old = json.load(open(out))
        # the design commit and the head seen at sealing move on later; the sealed content must not
        def sealed(m):
            pp = {k: v for k, v in m["phenopacket"].items() if k != "main_head_seen"}
            return {**{k: v for k, v in m.items() if k not in ("design_e_commit", "phenopacket")}, "phenopacket": pp}
        same = sealed(old) == sealed(man)
        print(f"{out} exists; recomputed content {'matches' if same else 'DIFFERS'} (not overwritten)")
        sys.exit(0 if same else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
