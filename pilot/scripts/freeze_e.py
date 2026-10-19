"""Freeze the design E systems (docs/research_design.md E3, E5) into pilot/cache/v2/devD/frozen_config_E.json.

Records, for every graph system, the variant, its selected offtopic_w and the build summary fingerprint the
confirmation run must reproduce; for S-bm25 the selected configuration and the sha256 of every index file;
for S-comb the selected fusion weight; and the sha256 of the extraction inputs of S-mix. Refuses to
overwrite an existing freeze.

    python freeze_e.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

V = Path("pilot/cache/v2")
VAR = V / "devD/variants"
R = Path("pilot/cache/v3/retrieval")
OUT = V / "devD/frozen_config_E.json"
GRAPHS = {"S-mix": "v3llmd_ns", "S-D": "base_nm", "S-llm": "v3llm_ns", "S-dict": "v3umls_ns", "S-union": "v3unionu_ns"}
INPUTS = ["pilot/cache/v3/llmd_v2.jsonl.gz", "pilot/cache/v3/llmd_v3.jsonl.gz", "pilot/cache/v3/phrase2cui_v3.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} exists; a freeze is never overwritten")
    analysis = load(VAR / "analysis.json")["dev"]
    scoring = load(V / "devD/frozen_config_D.json")["gemma_1hop"]
    systems = {}
    for sid, variant in GRAPHS.items():
        summary = load(VAR / f"graph_{variant}.json")
        if "fingerprint" not in summary:
            sys.exit(f"{variant}: build summary has no fingerprint; rebuild it first")
        ot = analysis[variant]["selected_offtopic_w"]
        systems[sid] = {"variant": variant, "offtopic_w": float(ot), "fingerprint": summary["fingerprint"],
                        "edges": summary["edges"], "diseases": summary["diseases"],
                        "dev_macro": [analysis[variant]["macro@10"], analysis[variant]["macro@1"]]}
    bm25 = load(R / "tune_bm25.json")
    index = {str(p.relative_to(R)): sha256(p) for p in sorted(R.rglob("*"))
             if p.is_file() and (p.name in ("chunks.jsonl", "docs.json", "chunk_doc.npy") or p.parent.name == "bm25")}
    comb = load(R / "fuse_tune_smix.json")
    frozen = {"design": "E", "scoring": scoring, "graph_systems": systems,
              "S-bm25": {"selected": bm25["selected"], "dev_macro": bm25["selected_score"], "index_sha256": index},
              "S-comb": {"w": comb["selected"]["w"], "dev_macro": comb["selected_score"],
                         "weights_tried": [t["w"] for t in comb["trail"]]},
              "inputs_sha256": {p: sha256(Path(p)) for p in INPUTS}}
    OUT.write_text(json.dumps(frozen, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
