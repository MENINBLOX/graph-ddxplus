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

from ddxkb.data import KB_ZIP, REPO_ROOT, TEST_CSV, TRAIN_CSV, VALIDATION_CSV, Patient, iter_train, load_kb
from ddxkb.freeze import SELECTED
from ddxkb.frequencies import FrequencyCounts, sex_index
from ddxkb.kb import KB
from ddxkb.policies.bed import BED_STATS, BedKnowledge, count_knowledge, text_default_codes
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
    """Counts for the toy KB: Alpha 10 rows, Beta 8, Gamma 2, aged 40 but two of Beta's aged 1; positive columns
    fievre ... atcd, value columns douleur_intens 1-3, douleur_lieu tete/dos/ventre, trav1 AfriqN/Asie."""
    rows = np.array([10, 8, 2])
    by_age_sex = np.zeros((3, 41, 2), dtype=np.int64)
    by_age_sex[0, 40, sex_index("F")] = 10
    by_age_sex[1, 40, sex_index("F")] = 6
    by_age_sex[1, 1, sex_index("M")] = 2
    by_age_sex[2, 40, sex_index("M")] = 2
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
    """A manifest in the seal's format; the CSV, when it exists, gives its sha256."""
    d = root / "paper1/manifests"
    d.mkdir(parents=True, exist_ok=True)
    m: dict[str, Any] = {"set": name, "csv": csv_rel, "csv_data_rows": n_rows, "n": len(indices)}
    m["selection"] = {"rule": f"synthetic {name}"}
    m["row_indices"] = indices
    if (root / csv_rel).exists():
        m["csv_sha256"] = hashlib.sha256((root / csv_rel).read_bytes()).hexdigest()
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
    """A KB of 38 evidences (enough for IL <= 31), 5 diseases, a categorical antecedent and pain subattributes."""
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


def bed_stats_json(kb: KB, k: BedKnowledge) -> dict[str, Any]:
    """`k` in the format of BED's stats file, diseases in name order as pandas groups them."""
    out: dict[str, Any] = {}
    for d in sorted(kb.diseases):
        i = kb.disease_index[d]
        findings: dict[str, Any] = {}
        for j in map(int, np.flatnonzero(k.known[i])):
            e = kb.evidences[j]
            if e.is_binary:
                findings[e.code] = float(k.binary[i, j])
            else:
                findings[e.code] = {o: float(p) for o, p in zip(k.options[j], k.option_probs[j][i], strict=True)}
        out[d] = {"prior": float(k.prior[i]), "findings": findings}
    return out


def bed_stats_sha256(root: Path) -> str:
    """The digest of the synthetic BED stats file `make_repo` writes, which tests put in place of BED's."""
    return hashlib.sha256((root / BED_STATS).read_bytes()).hexdigest()


def make_repo(root: Path, n_validation: int = 45, n_train: int = 150, n_test: int = 60) -> None:
    """A repository root with the synthetic KB zip, the three CSVs, v1's CUI files, a BED stats file counted from
    the synthetic train rows, and the manifests: H2 (every sixth train row), T (every test row), and its reporting
    subsets H1 (rows 3 on), T-dedup (the even rows) and T-streambench (rows 10-29)."""
    conditions, evidences = synth_kb_json()
    data = root / "data/ddxplus"
    data.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / KB_ZIP, "w") as z:
        z.writestr("release_conditions.json", json.dumps(conditions))
        z.writestr("release_evidences.json", json.dumps(evidences))
    write_csv(root / VALIDATION_CSV, synth_rows(n_validation, seed=1))
    write_csv(root / TRAIN_CSV, synth_rows(n_train, seed=2))
    write_csv(root / TEST_CSV, synth_rows(n_test, seed=3))
    sealed = {
        "H2": (TRAIN_CSV, n_train, list(range(0, n_train, 6))),
        "T": (TEST_CSV, n_test, list(range(n_test))),
        "H1": (TEST_CSV, n_test, list(range(3, n_test))),
        "T-dedup": (TEST_CSV, n_test, list(range(0, n_test, 2))),
        "T-streambench": (TEST_CSV, n_test, list(range(10, 30))),
    }
    for name, (csv_rel, n_rows, rows) in sealed.items():
        write_manifest(root, name, csv_rel, n_rows, rows)
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
    kb = load_kb(root)
    knowledge = count_knowledge(kb, iter_train(root), text_default_codes(evidences))
    (root / BED_STATS).parent.mkdir(parents=True, exist_ok=True)
    (root / BED_STATS).write_text(json.dumps(bed_stats_json(kb, knowledge)))


SELECTED_P1 = P1Config(0.5, 1e-3, 0.99)


SELECTION_COMMIT = "5e1ec7" + "0" * 34


def write_selected(
    root: Path,
    p1: P1Config = SELECTED_P1,
    p2: dict[str, dict[str, Any]] | None = None,
    limit: int | None = None,
    paper1_changes: list[str] | None = None,
) -> None:
    """A development selection in the format of run_dev's selected.json, made in this environment from the
    KB and train files under `root`, by the code of SELECTION_COMMIT with `paper1_changes` uncommitted (default:
    the development outputs only)."""
    if p2 is None:
        p2 = {v.name: P2Config(v, 1.0, 0.99).as_dict() for v in P2_VARIANTS}
    if paper1_changes is None:
        paper1_changes = ["?? paper1/outputs/v2/dev/configs_tried.jsonl"]
    inputs = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in (KB_ZIP, TRAIN_CSV)}
    selected = {
        "objective": "test",
        "n": 45,
        "limit": limit,
        "P1": {"config": p1.as_dict()},
        "P2": {name: {"config": config} for name, config in p2.items()},
        "provenance": {
            "environment": environment(),
            "inputs": inputs,
            "commit": SELECTION_COMMIT,
            "paper1_changes": paper1_changes,
        },
    }
    path = root / SELECTED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected))
