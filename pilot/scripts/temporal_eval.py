"""Time-split diagnosis on phenopacket-store cases (research design T4).

Ranks every dated phenopacket case against the graph currently loaded at --uri with a frozen 1-hop
configuration and writes one row per case: source PMID, publication date, age relative to the
knowledge snapshot, lineage position and whether the gold disease has any finding edge in the graph.

    python temporal_eval.py --uri bolt://localhost:7688 --snapshot 2024-12-31 \
        --cfg gemma_1hop --tag lit_2024 --out pilot/cache/v2/temporal/lit_2024.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opendx_eval as O  # noqa: E402
import v2_ids  # noqa: E402
from graph_dx import GraphDx  # noqa: E402

FROZEN = "pilot/cache/v2/devD/frozen_config_D.json"
NEW = "data/confirm/phenopacket_new"


def pub_date(rec: dict) -> date | None:
    s = rec.get("sortpubdate", "")[:10]
    return date(*map(int, s.split("/"))) if s else None


def load_cases() -> list[dict]:
    """phenopacket-store 0.1.27 plus the cases added on main afterwards, each with its source PMID date."""
    O.RAW_GOLD = True
    dz, hp = O._rb_maps()
    dates = json.load(open(O.DATA / "external_benchmarks/phenopacket_store/pmid_dates.json"))
    z = zipfile.ZipFile(O.DATA / "external_benchmarks/phenopacket_store/all_phenopackets.zip")
    srcs = [(n, "0.1.27", lambda n=n: z.read(n)) for n in sorted(z.namelist()) if n.endswith(".json")]
    srcs += [(n, "new", lambda n=n: open(f"{NEW}/{n}", "rb").read()) for n in sorted(os.listdir(NEW))]
    out = []
    for name, rel, read in srcs:
        m = re.search(r"PMID_(\d+)", name)
        d = pub_date(dates.get(m.group(1), {})) if m else None
        p = json.loads(read())
        omim = sorted({x["term"]["id"] for x in p.get("diseases", [])})
        gold = set().union(*(dz(o) for o in omim)) if omim else set()
        ev = {c for c in (hp(f["type"]["id"]) for f in p.get("phenotypicFeatures", []) if not f.get("excluded")) if c}
        if d and ev and v2_ids.gold_keys(gold):
            out.append({"file": name.rsplit("/", 1)[-1], "release": rel, "pmid": m.group(1), "date": d,
                        "omim": omim, "gold": gold, "ev": ev})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="bolt://localhost:7688")
    ap.add_argument("--auth", default=os.environ.get("NEO4J_AUTH", "neo4j/thesis-dev-only"))
    ap.add_argument("--snapshot", required=True, help="knowledge snapshot date, YYYY-MM-DD")
    ap.add_argument("--cfg", required=True, help="gemma_1hop or hpoa / primekg (frozen design D configs)")
    ap.add_argument("--override", default="{}", help="JSON of config keys to replace (sensitivity runs only)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fz = json.load(open(FROZEN))
    cfg = fz[a.cfg] if a.cfg in fz else fz["comparators"][a.cfg]["onehop"]
    cfg = {**cfg, **json.loads(a.override)}
    snap = date.fromisoformat(a.snapshot)
    cases = load_cases()
    dx = GraphDx(a.uri, a.auth, **cfg)
    with dx.drv.session() as s:
        known = {r["id"] for r in s.run("MATCH (d:Disease) WHERE d.n_findings > 0 RETURN d.id AS id")}
    rows = []
    for c in cases:
        gold = v2_ids.gold_keys(c["gold"])
        ranked = dx.rank(c["ev"], max(O.KS))
        lin = dx.lineage(gold)
        pos = next((i + 1 for i, d in enumerate(ranked) if d in lin), None)
        rows.append({"file": c["file"], "release": c["release"], "pmid": c["pmid"], "omim": c["omim"],
                     "date": c["date"].isoformat(), "age_days": (c["date"] - snap).days, "rank": pos,
                     "gold_known": bool(gold & known)})
    dx.drv.close()
    json.dump({"tag": a.tag, "snapshot": a.snapshot, "cfg": cfg, "rows": rows}, open(a.out, "w"))
    n = len(rows)
    print(f"{a.tag}: n={n} @1 {100 * sum(r['rank'] == 1 for r in rows) / n:.1f} "
          f"@10 {100 * sum(r['rank'] is not None and r['rank'] <= 10 for r in rows) / n:.1f} "
          f"known {100 * sum(r['gold_known'] for r in rows) / n:.1f}", flush=True)


if __name__ == "__main__":
    main()
