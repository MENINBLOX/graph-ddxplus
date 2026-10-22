"""Synthetic KB and CSV builders in the DDXPlus release format, and the real-data skip marker."""

from __future__ import annotations

import hashlib
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ddxkb.data import KB_ZIP, REPO_ROOT, VALIDATION_CSV, Patient
from ddxkb.freeze import SELECTED
from ddxkb.frequencies import N_AGE_BINS, FrequencyCounts, age_bin, sex_index
from ddxkb.kb import KB
from ddxkb.protocol import P2_VARIANTS, P1Config, P2Config
from ddxkb.provenance import environment


def _binary(code: str, antecedent: bool = False) -> dict[str, Any]:
    return {
        "name": code,
        "code_question": code,
        "question_en": f"{code}?",
        "is_antecedent": antecedent,
        "default_value": 0,
        "value_meaning": {},
        "possible-values": [],
        "data_type": "B",
    }


def _valued(
    code: str, kind: str, values: list[Any], default: Any, parent: str, antecedent: bool = False
) -> dict[str, Any]:
    return {
        "name": code,
        "code_question": parent,
        "question_en": f"{code}?",
        "is_antecedent": antecedent,
        "default_value": default,
        "value_meaning": {},
        "possible-values": values,
        "data_type": kind,
    }


TOY_EVIDENCES: dict[str, dict[str, Any]] = {
    "fievre": _binary("fievre"),
    "toux": _binary("toux"),
    "douleur": _binary("douleur"),
    "douleur_intens": _valued("douleur_intens", "C", [0, 1, 2, 3], 0, "douleur"),
    "douleur_lieu": _valued("douleur_lieu", "M", ["nulle_part", "tete", "dos", "ventre"], "nulle_part", "douleur"),
    "trav1": _valued("trav1", "C", ["N", "AfriqN", "Asie"], "N", "trav1", antecedent=True),
    "atcd": _binary("atcd", antecedent=True),
}


def _condition(name: str, symptoms: list[str], antecedents: list[str]) -> dict[str, Any]:
    return {
        "condition_name": name,
        "cond-name-fr": name,
        "cond-name-eng": f"{name} (en)",
        "icd10-id": "X00",
        "symptoms": {s: {} for s in symptoms},
        "antecedents": {a: {} for a in antecedents},
        "severity": 3,
    }


TOY_CONDITIONS: dict[str, dict[str, Any]] = {
    "Alpha": _condition("Alpha", ["fievre", "toux"], ["trav1"]),
    "Beta": _condition("Beta", ["douleur", "douleur_intens", "douleur_lieu"], ["atcd", "trav1"]),
    "Gamma": _condition("Gamma", ["fievre", "douleur", "douleur_intens", "douleur_lieu"], ["trav1"]),
}


def binary_kb(diseases: dict[str, list[str]], evidences: list[str]) -> KB:
    """A KB of binary evidences only; diseases and evidences in the given file order."""
    catalogue = {e: _binary(e) for e in evidences}
    conditions = {d: _condition(d, s, []) for d, s in diseases.items()}
    return KB.from_json(conditions, catalogue)


def toy_counts() -> FrequencyCounts:
    """Counts for the toy KB: Alpha 10 rows, Beta 8, Gamma 2; positive columns fievre ... atcd, value columns
    douleur_intens 1-3, douleur_lieu tete/dos/ventre, trav1 AfriqN/Asie."""
    rows = np.array([10, 8, 2])
    by_age_sex = np.zeros((3, N_AGE_BINS, 2), dtype=np.int64)
    by_age_sex[0, age_bin(40), sex_index("F")] = 10
    by_age_sex[1, age_bin(40), sex_index("F")] = 6
    by_age_sex[1, age_bin(1), sex_index("M")] = 2
    by_age_sex[2, age_bin(40), sex_index("M")] = 2
    positive = np.array([[9, 7, 0, 0, 0, 1, 0], [0, 0, 8, 8, 6, 0, 3], [2, 0, 2, 2, 2, 0, 0]])
    value = np.array([[0, 0, 0, 0, 0, 0, 0, 1], [1, 3, 4, 4, 3, 1, 0, 0], [0, 2, 0, 2, 0, 0, 0, 0]])
    return FrequencyCounts(rows=rows, by_age_sex=by_age_sex, positive=positive, value=value)


def toy_patient(evidences: list[str], initial: str, pathology: str = "Beta", row: int = 0) -> Patient:
    pairs = []
    for e in evidences:
        code, sep, value = e.partition("_@_")
        pairs.append((code, value if sep else None))
    return Patient(
        split="validate",
        row=row,
        age=40,
        sex="F",
        pathology=pathology,
        initial=initial,
        evidences=tuple(pairs),
        differential=((pathology, 1.0),),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Rows in the release CSV format; EVIDENCES and DIFFERENTIAL_DIAGNOSIS as Python list literals."""
    lines = ["AGE,DIFFERENTIAL_DIAGNOSIS,SEX,PATHOLOGY,EVIDENCES,INITIAL_EVIDENCE"]
    for r in rows:
        diff = repr([[d, p] for d, p in r["differential"]])
        lines.append(f'{r["age"]},"{diff}",{r["sex"]},{r["pathology"]},"{r["evidences"]!r}",{r["initial"]}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_manifest(root: Path, name: str, csv_rel: str, n_rows: int, indices: list[int]) -> None:
    d = root / "paper1/manifests"
    d.mkdir(parents=True, exist_ok=True)
    m = {"set": name, "csv": csv_rel, "csv_data_rows": n_rows, "n": len(indices), "row_indices": indices}
    (d / f"{name}.json").write_text(json.dumps(m))


def real_data_present() -> bool:
    return (REPO_ROOT / KB_ZIP).exists() and (REPO_ROOT / VALIDATION_CSV).exists()


requires_data = pytest.mark.skipif(not real_data_present(), reason="DDXPlus release files not present")


SYNTH_DISEASES = tuple(f"Maladie{k}" for k in range(5))
SYNTH_SYMPTOMS = tuple(f"e{j:02d}" for j in range(34))
# e00 and e01 share a CUI; e30-e33 and the pain subattributes have none, as 19 real codes have none.
SYNTH_CODE_CUI = {
    **{code: f"C{j:03d}" for j, code in enumerate(SYNTH_SYMPTOMS[:30])},
    "e01": "C000",
    "douleur": "C100",
    "trav1": "C200",
}


def _synth_lists() -> dict[str, list[str]]:
    lists = {}
    for k, name in enumerate(SYNTH_DISEASES):
        codes = [*SYNTH_SYMPTOMS[6 * k : 6 * k + 9], "trav1"]
        if k in (1, 3):
            codes += ["douleur", "douleur_intens", "douleur_lieu"]
        if k == 4:
            codes.append("e33")
        lists[name] = codes
    return lists


def synth_kb_json() -> tuple[dict[str, Any], dict[str, Any]]:
    """A KB of 38 evidences (enough for |T| = 30), 5 diseases, a categorical antecedent and pain subattributes."""
    evidences: dict[str, Any] = {code: _binary(code, antecedent=code >= "e30") for code in SYNTH_SYMPTOMS}
    evidences["douleur"] = _binary("douleur")
    evidences["douleur_intens"] = _valued("douleur_intens", "C", [0, 1, 2, 3], 0, "douleur")
    evidences["douleur_lieu"] = _valued(
        "douleur_lieu", "M", ["nulle_part", "tete", "dos", "ventre"], "nulle_part", "douleur"
    )
    evidences["trav1"] = _valued("trav1", "C", ["N", "AfriqN", "Asie"], "N", "trav1", antecedent=True)
    conditions = {}
    for name, codes in _synth_lists().items():
        antecedents = [c for c in codes if evidences[c]["is_antecedent"]]
        conditions[name] = _condition(name, [c for c in codes if c not in antecedents], antecedents)
    return conditions, evidences


def synth_rows(n: int, seed: int) -> list[dict[str, Any]]:
    """Patients drawn from the synthetic lists: a mapped binary initial evidence, other positives on the list."""
    rng = random.Random(seed)
    lists = _synth_lists()
    rows = []
    for _ in range(n):
        k = rng.randrange(len(SYNTH_DISEASES))
        name = SYNTH_DISEASES[k]
        binary = [c for c in lists[name] if c.startswith("e") or c == "douleur"]
        chosen = rng.sample(binary, rng.randint(2, 6))
        initial = next((c for c in chosen if c in SYNTH_CODE_CUI), binary[0])
        evidences = list(dict.fromkeys([initial, *chosen]))
        if "douleur" in evidences:
            evidences += [f"douleur_intens_@_{rng.randint(0, 3)}", f"douleur_lieu_@_{rng.choice(['tete', 'dos'])}"]
        evidences.append("trav1_@_" + rng.choice(["N", "N", "Asie"]))
        other = SYNTH_DISEASES[(k + 1) % len(SYNTH_DISEASES)]
        rows.append(
            {
                "age": rng.randint(1, 90),
                "sex": rng.choice(["F", "M"]),
                "pathology": name,
                "evidences": evidences,
                "initial": initial,
                "differential": [(name, 0.7), (other, 0.3)],
            }
        )
    return rows


def make_repo(root: Path, n_validation: int = 45, n_train: int = 150, n_test: int = 60) -> None:
    """A repository root with the synthetic KB zip, the three CSVs, sealed H1/H2 manifests and v1's CUI files."""
    conditions, evidences = synth_kb_json()
    data = root / "data/ddxplus"
    data.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / KB_ZIP, "w") as z:
        z.writestr("release_conditions.json", json.dumps(conditions))
        z.writestr("release_evidences.json", json.dumps(evidences))
    write_csv(root / VALIDATION_CSV, synth_rows(n_validation, seed=1))
    sets = {
        "H2": ("data/ddxplus/release_train_patients.csv", synth_rows(n_train, seed=2), list(range(0, n_train, 6))),
        "H1": ("data/ddxplus/release_test_patients.csv", synth_rows(n_test, seed=3), list(range(3, n_test))),
    }
    manifests = root / "paper1/manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for name, (csv_rel, rows, sealed) in sets.items():
        write_csv(root / csv_rel, rows)
        manifest = {
            "set": name,
            "csv": csv_rel,
            "csv_sha256": hashlib.sha256((root / csv_rel).read_bytes()).hexdigest(),
            "csv_data_rows": len(rows),
            "n": len(sealed),
            "row_indices": sealed,
        }
        (manifests / f"{name}.json").write_text(json.dumps(manifest))
    (data / "umls_mapping.json").write_text(json.dumps({"mapping": {c: {"cui": u} for c, u in SYNTH_CODE_CUI.items()}}))
    diseases = {name: {"name_fr": name, "umls_cui": f"D{k:03d}"} for k, name in enumerate(SYNTH_DISEASES)}
    (data / "disease_umls_mapping.json").write_text(json.dumps({"mapping": diseases}))
    edges = sorted(
        {
            (SYNTH_CODE_CUI[c], f"D{k:03d}")
            for k, codes in enumerate(_synth_lists().values())
            for c in codes
            if c in SYNTH_CODE_CUI
        }
    )
    (data / "kg_edges.json").write_text(json.dumps({"edges": edges}))


SELECTED_P1 = P1Config(0.5, 1e-3, 0.99)


def write_selected(
    root: Path,
    p1: P1Config = SELECTED_P1,
    p2: dict[str, dict[str, Any]] | None = None,
    limit: int | None = None,
) -> None:
    """A development selection in the format of run_dev's selected.json, made in this environment."""
    if p2 is None:
        p2 = {v.name: P2Config(v, 1.0, 0.99).as_dict() for v in P2_VARIANTS}
    selected = {
        "objective": "test",
        "n": 45,
        "limit": limit,
        "P1": {"config": p1.as_dict()},
        "P2": {name: {"config": config} for name, config in p2.items()},
        "provenance": {"environment": environment()},
    }
    path = root / SELECTED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected))
