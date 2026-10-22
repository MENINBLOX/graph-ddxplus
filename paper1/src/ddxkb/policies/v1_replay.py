"""Deterministic port of the earlier study's (v1) inquiry rule, for the appendix of design §3.

The rule is archive/v1_ddxplus/scripts/experiment_final_75.py as the review (§2.2-2.5) reconstructs it, with
v1's final settings: deny threshold 6, no antecedent priority, the v15 score, stop on a stable top 3 and at
most 223 questions. Questions are yes/no about the symptom nodes of a disease-symptom graph:

- stage A, while no node besides the initial one is confirmed: the unasked node on the most lists of the
  diseases whose list holds the initial node;
- stage B: over the diseases whose list holds a confirmed node and fewer than `deny` denied nodes (no
  denied-node filter when deny <= 0), the unasked node maximizing coverage (how many of those lists hold it)
  times co-occurrence (how many confirmed nodes share one of those lists with it).

A disease whose list holds a confirmed node scores c / (c + d + 1) * c, c and d counting the confirmed and
denied nodes on its list; other diseases are unranked. The episode stops once the ordered top 3 has been the
same after each of the last `stable_window` questions, when stage A or B finds no question, or after
`max_questions` questions.

v1 sorted without secondary keys, so its ties followed the database's return order. Here every tie follows
one fixed order of node ids: ascending, or shuffled by `random.Random(tie_seed)`.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ddxkb.data import REPO_ROOT, Patient
from ddxkb.kb import KB, BoolArray, FloatArray
from ddxkb.policies.bayes import IndexArray
from ddxkb.policies.episode import Episode
from ddxkb.simulator import positive_codes

V1_DENY = 6
V1_MAX_QUESTIONS = 223
V1_STABLE_WINDOW = 5
V1_DATA_DIR = "data/ddxplus"

Edge = tuple[str, str]


@dataclass(frozen=True, eq=False)
class V1Graph:
    """The v1 question space: symptom nodes, and for each KB disease the nodes on its list.

    `membership[i, s]` says node s is on KB disease i's list; `disease_nodes[i]` is disease i's node id (its
    CUI, or its name at code level). `code_node` maps an evidence code to its node's column; a code without a
    node is never asked and its answer is never seen.
    """

    kb: KB
    nodes: tuple[str, ...]
    membership: BoolArray
    disease_nodes: tuple[str, ...]
    code_node: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.membership.shape != (self.kb.n_diseases, len(self.nodes)):
            raise ValueError(f"membership shape {self.membership.shape} does not match the nodes")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("node ids must be distinct")
        if len(self.disease_nodes) != self.kb.n_diseases or len(set(self.disease_nodes)) != self.kb.n_diseases:
            raise ValueError("every KB disease needs its own node id")
        if any(not 0 <= s < len(self.nodes) for s in self.code_node.values()):
            raise ValueError("code_node points outside the nodes")
        self.membership.flags.writeable = False


def subattributes(kb: KB) -> frozenset[str]:
    """Categorical and multi-choice evidences refining another evidence's answer (pain, skin lesion and oedema
    details; 14 in DDXPlus). v1's code-to-CUI mapping gave none of them a CUI."""
    return frozenset(e.code for e in kb.evidences if not e.is_binary and e.parent != e.code)


def code_graph(kb: KB, drop_subattributes: bool = False) -> V1Graph:
    """One node per evidence code; disease node ids are the KB disease names."""
    dropped = subattributes(kb) if drop_subattributes else frozenset()
    keep = [j for j, e in enumerate(kb.evidences) if e.code not in dropped]
    return V1Graph(
        kb=kb,
        nodes=tuple(kb.evidences[j].code for j in keep),
        membership=np.array(kb.membership[:, keep]),
        disease_nodes=kb.diseases,
        code_node={kb.evidences[j].code: s for s, j in enumerate(keep)},
    )


def cui_graph(kb: KB, edges: Iterable[Edge], code_cui: Mapping[str, str], disease_cui: Mapping[str, str]) -> V1Graph:
    """Nodes are the symptom CUIs of `edges`, (symptom CUI, disease CUI) pairs; codes sharing a CUI share a node.

    `code_cui` maps evidence codes to CUIs (unknown codes are ignored), `disease_cui` KB disease names to CUIs.
    """
    edge_set = set(edges)
    disease_nodes = tuple(disease_cui[d] for d in kb.diseases)
    row = {cui: i for i, cui in enumerate(disease_nodes)}
    nodes = tuple(sorted({s for s, _ in edge_set}))
    column = {s: j for j, s in enumerate(nodes)}
    membership = np.zeros((kb.n_diseases, len(nodes)), dtype=np.bool_)
    for s, d in edge_set:
        if d not in row:
            raise ValueError(f"edge ({s}, {d}) ends at a CUI that names no KB disease")
        membership[row[d], column[s]] = True
    code_node = {code: column[cui] for code, cui in code_cui.items() if cui in column and code in kb.evidence_index}
    return V1Graph(kb=kb, nodes=nodes, membership=membership, disease_nodes=disease_nodes, code_node=code_node)


def folded_cui_edges(kb: KB, code_cui: Mapping[str, str], disease_cui: Mapping[str, str]) -> frozenset[Edge]:
    """The membership KB folded to (symptom CUI, disease CUI) pairs; codes without a CUI drop out."""
    return frozenset(
        (code_cui[kb.evidences[j].code], disease_cui[d])
        for i, d in enumerate(kb.diseases)
        for j in np.flatnonzero(kb.membership[i])
        if kb.evidences[j].code in code_cui
    )


def load_cui_mappings(root: Path = REPO_ROOT) -> tuple[dict[str, str], dict[str, str]]:
    """v1's evidence code -> CUI map and French disease name -> CUI map."""
    data = root / V1_DATA_DIR
    codes = json.loads((data / "umls_mapping.json").read_text())["mapping"]
    diseases = json.loads((data / "disease_umls_mapping.json").read_text())["mapping"]
    code_cui = {str(code): str(m["cui"]) for code, m in codes.items() if m.get("cui")}
    disease_cui = {str(m["name_fr"]): str(m["umls_cui"]) for m in diseases.values()}
    return code_cui, disease_cui


def load_v1_edges(root: Path = REPO_ROOT) -> list[Edge]:
    """The (symptom CUI, disease CUI) edges of v1's graph as archived in kg_edges.json."""
    edges = json.loads((root / V1_DATA_DIR / "kg_edges.json").read_text())["edges"]
    return [(str(s), str(d)) for s, d in edges]


def v1_cui_graph(kb: KB, root: Path = REPO_ROOT) -> V1Graph:
    return cui_graph(kb, load_v1_edges(root), *load_cui_mappings(root))


@dataclass(frozen=True, eq=False, slots=True)
class V1Episode(Episode):
    """`questions` are node ids. `order` ranks the scored KB diseases best first with ties in the replay's
    tie order, so v1's own prediction is order[0]; `posterior` holds the scores normalized to sum 1."""

    order: tuple[int, ...]

    def in_top(self, k: int) -> bool:
        return self.gold in self.order[:k]


@dataclass(frozen=True, eq=False)
class V1Replay:
    """The v1 rule on `graph`.

    `trav1_bug` answers yes to every recorded code whatever its value, as v1's oracle did; otherwise default
    values are absent, as in the main protocol. In v1's CUI graph the bug reached only trav1_@_N (no travel),
    recorded for most patients, because no other categorical or multi-choice evidence had a CUI; in a code
    graph with subattributes it also answers yes to their recorded default values.
    """

    graph: V1Graph
    deny: int = V1_DENY
    trav1_bug: bool = False
    tie_seed: int | None = None
    max_questions: int = V1_MAX_QUESTIONS
    stable_window: int = V1_STABLE_WINDOW
    _node_key: IndexArray = field(init=False, repr=False)
    _disease_key: IndexArray = field(init=False, repr=False)
    _membership_f: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_questions < 0 or self.stable_window < 1:
            raise ValueError("max_questions must be >= 0 and stable_window >= 1")
        g = self.graph
        names = sorted(set(g.nodes) | set(g.disease_nodes))
        if self.tie_seed is not None:
            random.Random(self.tie_seed).shuffle(names)
        rank = {name: i for i, name in enumerate(names)}
        object.__setattr__(self, "_node_key", np.array([rank[n] for n in g.nodes], dtype=np.intp))
        object.__setattr__(self, "_disease_key", np.array([rank[n] for n in g.disease_nodes], dtype=np.intp))
        object.__setattr__(self, "_membership_f", g.membership.astype(np.float64))

    def answered_yes(self, patient: Patient) -> BoolArray:
        """The nodes the patient answers yes to."""
        g = self.graph
        codes = [code for code, _ in patient.evidences] if self.trav1_bug else positive_codes(g.kb, patient)
        yes = np.zeros(len(g.nodes), dtype=np.bool_)
        yes[[g.code_node[c] for c in codes if c in g.code_node]] = True
        return yes

    def ranking(self, confirmed: BoolArray, denied: BoolArray) -> tuple[IndexArray, FloatArray]:
        """The ranked KB diseases best first, and every KB disease's score (0 when unranked)."""
        m = self.graph.membership
        c = (m & confirmed).sum(axis=1)
        d = (m & denied).sum(axis=1)
        linked = np.flatnonzero(c > 0)
        scores = np.zeros(self.graph.kb.n_diseases)
        scores[linked] = c[linked] / (c[linked] + d[linked] + 1.0) * c[linked]
        order = linked[np.lexsort((self._disease_key[linked], -scores[linked]))]
        return order, scores

    def next_question(self, initial: int, confirmed: BoolArray, denied: BoolArray, asked: BoolArray) -> int | None:
        m = self.graph.membership
        if np.count_nonzero(confirmed) == 1:
            score = m[m[:, initial]].sum(axis=0)
        else:
            linked = (m & confirmed).any(axis=1)
            valid = linked if self.deny <= 0 else linked & ((m & denied).sum(axis=1) < self.deny)
            lists = self._membership_f[valid]
            shared_lists = lists.T @ lists[:, confirmed]
            score = lists.sum(axis=0) * np.count_nonzero(shared_lists > 0, axis=1)
        score = np.where(asked, 0, score)
        best = score.max()
        if best <= 0:
            return None
        tied = np.flatnonzero(score == best)
        return int(tied[np.argmin(self._node_key[tied])])

    def run(self, patient: Patient) -> V1Episode:
        g = self.graph
        if patient.initial not in g.code_node:
            raise ValueError(f"initial evidence {patient.initial} has no node in this graph")
        yes = self.answered_yes(patient)
        initial = g.code_node[patient.initial]
        confirmed = np.zeros(len(g.nodes), dtype=np.bool_)
        denied = np.zeros_like(confirmed)
        asked = np.zeros_like(confirmed)
        confirmed[initial] = asked[initial] = True
        questions: list[int] = []
        top3: list[tuple[int, ...]] = []
        while len(questions) < self.max_questions:
            s = self.next_question(initial, confirmed, denied, asked)
            if s is None:
                break
            asked[s] = True
            (confirmed if yes[s] else denied)[s] = True
            questions.append(s)
            top3.append(tuple(int(i) for i in self.ranking(confirmed, denied)[0][:3]))
            recent = top3[-self.stable_window :]
            if len(recent) == self.stable_window and all(t == recent[-1] for t in recent):
                break
        order, scores = self.ranking(confirmed, denied)
        total = scores.sum()
        return V1Episode(
            row=patient.row,
            gold=g.kb.disease_index[patient.pathology],
            questions=tuple(g.nodes[s] for s in questions),
            il=len(questions) + 1,
            posterior=scores / total if total > 0 else scores,
            budgets=(),
            budget_posteriors=np.empty((0, g.kb.n_diseases)),
            budget_gains=np.empty(0),
            order=tuple(int(i) for i in order),
        )
