"""Dataset properties of the public lists against the train and validation records.

    uv run python reference/dataset_properties.py [--out FILE]

From the train split: the patients with a positive evidence off their disease's list, and for each of the list
entries (disease, evidence) the share of that disease's train patients in whom the evidence is positive. From the
validation split: how the reference differential relates to the gold disease and to the lists, and whether the
records hold the categorical and multi-choice evidences of the gold disease's list, and for the diseases whose
list holds another disease's list, how many of their patients the lists cannot tell from that disease. A positive
evidence is one recorded with a value other than its default (`simulator.positive_codes`). No sealed set is read.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ddxkb.data import REPO_ROOT, Patient, iter_train, load_kb, load_validation, unseal_requested
from ddxkb.kb import KB
from ddxkb.simulator import positive_codes

RARE_BELOW = 0.05
NEAR_CERTAIN_FROM = 0.99


def list_entries(root: Path) -> dict[str, object]:
    kb = load_kb(root)
    patients = np.zeros(kb.n_diseases, dtype=np.int64)
    positive = np.zeros((kb.n_diseases, kb.n_evidences), dtype=np.int64)
    column = {e.code: j for j, e in enumerate(kb.evidences)}
    outside = n = 0
    for p in iter_train(root):
        d = kb.disease_index[p.pathology]
        codes = positive_codes(kb, p)
        cols = [column[c] for c in codes]
        patients[d] += 1
        positive[d, cols] += 1
        outside += bool((~kb.membership[d, cols]).any())
        n += 1
    rates = (positive / patients[:, None])[kb.membership]
    return {
        "n": n,
        "outside_support": outside,
        "list_entries": int(kb.membership.sum()),
        "never_positive": int((rates == 0).sum()),
        "share_below_rare": float((rates < RARE_BELOW).mean()),
        "share_near_certain": float((rates >= NEAR_CERTAIN_FROM).mean()),
        "deciles": [float(q) for q in np.quantile(rates, np.arange(0.1, 1.0, 0.1))],
        "definitions": {
            "rate": "of a disease's train patients, the share in whom a listed evidence is positive",
            "share_below_rare": f"list entries with a rate below {RARE_BELOW}",
            "share_near_certain": f"list entries with a rate of at least {NEAR_CERTAIN_FROM}",
        },
    }


def differential(root: Path) -> dict[str, object]:
    kb = load_kb(root)
    top = strictly_top = 0
    excluded_mass = []
    patients = load_validation(root)
    for p in patients:
        probs = dict(p.differential)
        best = max(probs.values())
        gold = probs.get(p.pathology, 0.0)
        top += gold == best
        strictly_top += gold == best and sum(v == best for v in probs.values()) == 1
        consistent = kb.diseases_containing(positive_codes(kb, p))
        total = sum(probs.values())
        off = sum(v for name, v in probs.items() if not consistent[kb.disease_index[name]])
        excluded_mass.append(off / total)
    mass = np.asarray(excluded_mass)
    return {
        "n": len(patients),
        "valued_records": valued_records(kb, patients),
        "nested_lists": nested_lists(kb, patients),
        "gold_is_top": {"count": int(top), "share": top / len(patients)},
        "gold_is_only_top": {"count": int(strictly_top), "share": strictly_top / len(patients)},
        "mass_on_excluded": {"mean": float(mass.mean()), "median": float(np.median(mass))},
        "definitions": {
            "gold_is_top": "the gold disease has the largest probability of the reference differential (ties count)",
            "gold_is_only_top": "the same without a tie",
            "mass_on_excluded": (
                "the share of the differential's probability on diseases whose list does not hold every positive"
                " evidence of the patient"
            ),
        },
    }


def valued_records(kb: KB, patients: Sequence[Patient]) -> dict[str, object]:
    """How the records hold the categorical and multi-choice evidences against the gold disease's list."""
    valued = {e.code for e in kb.evidences if not e.is_binary}
    listed_missing = recorded_unlisted = listed = 0
    for p in patients:
        on_list = kb.support(p.pathology) & valued
        recorded = {code for code, _ in p.evidences if code in valued}
        listed += len(on_list)
        listed_missing += len(on_list - recorded)
        recorded_unlisted += len(recorded - on_list)
    return {
        "listed": listed,
        "listed_not_recorded": listed_missing,
        "recorded_not_listed": recorded_unlisted,
        "definitions": {
            "listed": "pairs of a patient and a categorical or multi-choice evidence on the gold disease's list",
            "listed_not_recorded": "of these, the pairs whose evidence the record does not hold with any value",
            "recorded_not_listed": "categorical or multi-choice evidences a record holds off the gold disease's list",
        },
    }


def nested_lists(kb: KB, patients: Sequence[Patient]) -> list[dict[str, object]]:
    """For each disease whose list holds another disease's whole list, the share of its patients whose positive
    evidences all lie on the smaller list, so that the lists cannot tell the two diseases apart."""
    out: list[dict[str, object]] = []
    for big in kb.diseases:
        for small in kb.diseases:
            if big == small or not kb.support(small) < kb.support(big):
                continue
            own = [p for p in patients if p.pathology == big]
            inside = sum(positive_codes(kb, p) <= kb.support(small) for p in own)
            out.append(
                {
                    "disease": big,
                    "contained_list": small,
                    "patients": len(own),
                    "inside": inside,
                    "share": inside / len(own),
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    args = ap.parse_args()
    if unseal_requested():
        raise SystemExit("refused: DDXKB_UNSEAL is set")
    out = {"train": list_entries(args.root), "validation": differential(args.root)}
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    if args.out is None:
        print(text)
    else:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
