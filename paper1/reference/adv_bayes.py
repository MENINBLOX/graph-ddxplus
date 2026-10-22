"""Adversarial check (validation only): Bayes + EIG over the published membership KB vs train-frequency Bayes.

Question unit = DDXPlus evidence code (223); answer = code present with a non-default value.
Initial evidence is given; up to 29 further questions (|T_i| <= 30 incl. initial).
Usage: python -I adv_bayes.py <n_val> <n_train> <nproc>
"""
import ast
import csv
import json
import random
import sys
from multiprocessing import Pool

import numpy as np

D = "/home/max/thesis/data/ddxplus/"
N_VAL, N_TRAIN, NPROC = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
BUDGETS = (0, 5, 10, 15, 20, 29)

fr = json.load(open(D + "release_evidences.json"))
en = json.load(open(D + "release_evidences_en.json"))
cond = json.load(open(D + "release_conditions.json"))
q2fr = {v["question_en"].strip(): k for k, v in fr.items()}
e2f = {k: q2fr[v["question_en"].strip()] for k, v in en.items()}
CODES = sorted(fr)
CI = {c: i for i, c in enumerate(CODES)}
DEF = {k: str(v.get("default_value")) for k, v in fr.items() if v["data_type"] in "MC"}
DIS = sorted(c["cond-name-fr"] for c in cond.values())
DI = {d: i for i, d in enumerate(DIS)}
M = np.zeros((len(DIS), len(CODES)), dtype=bool)
for c in cond.values():
    for x in list(c["symptoms"]) + list(c["antecedents"]):
        M[DI[c["cond-name-fr"]], CI[e2f[x]]] = True


def pos_codes(ev):
    out = set()
    for e in ev:
        if "_@_" in e:
            code, val = e.split("_@_", 1)
            if DEF.get(code) == val:
                continue
            out.add(code)
        else:
            out.add(e)
    return out


def train_freq():
    cnt = np.zeros((len(DIS), len(CODES)))
    ny = np.zeros(len(DIS))
    with open(D + "release_train_patients.csv") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= N_TRAIN:
                break
            y = DI[row["PATHOLOGY"]]
            ny[y] += 1
            for c in pos_codes(ast.literal_eval(row["EVIDENCES"])):
                cnt[y, CI[c]] += 1
    return (cnt + 0.5) / (ny[:, None] + 1.0), ny / ny.sum()


def ent(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def run(P, prior, init, pos, stop_tau, tie_key):
    """Greedy EIG. Returns (correct@1 per budget dict, final top1-correct expected, top3, IL incl initial)."""
    asked = np.zeros(len(CODES), dtype=bool)
    lp = np.log(prior).copy()
    ii = CI[init]
    lp += np.log(P[:, ii])
    asked[ii] = True
    il = 1
    snaps = {}

    def post():
        q = np.exp(lp - lp.max())
        return q / q.sum()

    def top1_exp(q):
        m = q.max()
        tied = np.isclose(q, m, rtol=1e-9, atol=0)
        return tied[TRUE] / tied.sum() if tied[TRUE] else 0.0

    nq = 0
    stopped = None
    while True:
        q = post()
        if nq in BUDGETS:
            snaps[nq] = top1_exp(q)
        if stopped is None and (q.max() >= stop_tau or nq == 29):
            stopped = (top1_exp(q), TRUE in np.argsort(-q, kind="stable")[:3], il)
        if nq == 29:
            break
        py = q @ P
        h0 = ent(q)
        best, bi = -1.0, -1
        jy = q[:, None] * P
        jn = q[:, None] * (1 - P)
        with np.errstate(divide="ignore", invalid="ignore"):
            hy = -np.nansum(np.where(jy > 0, jy * np.log(jy / py), 0), axis=0)
            hn = -np.nansum(np.where(jn > 0, jn * np.log(jn / (1 - py)), 0), axis=0)
        eig = h0 - (hy + hn)
        eig[asked] = -np.inf
        cand = np.flatnonzero(eig >= eig.max() - 1e-12)
        bi = min(cand, key=tie_key)
        if eig[bi] < 1e-6 and stopped is None:
            stopped = (top1_exp(q), TRUE in np.argsort(-q, kind="stable")[:3], il)
        asked[bi] = True
        nq += 1
        il += 1 if stopped is None else 0
        lp += np.log(P[:, bi] if bi in pos else 1 - P[:, bi])
    return snaps, stopped


TRUE = None


def work(args):
    global TRUE
    rows, cfgs = args
    out = {name: {"snap": {b: 0.0 for b in BUDGETS}, "c1": 0.0, "c3": 0, "il": 0, "full": 0.0} for name in cfgs}
    for row in rows:
        TRUE = DI[row["PATHOLOGY"]]
        pos = {CI[c] for c in pos_codes(ast.literal_eval(row["EVIDENCES"]))}
        for name, (P, prior, tau) in cfgs.items():
            snaps, (c1, c3, il) = run(P, prior, row["INITIAL_EVIDENCE"], pos, tau, lambda j: j)
            o = out[name]
            for b in BUDGETS:
                o["snap"][b] += snaps[b]
            o["c1"] += c1
            o["c3"] += int(c3)
            o["il"] += il
            x = np.zeros(len(CODES), dtype=bool)
            x[list(pos)] = True
            lpf = np.log(prior) + np.log(np.where(x, P, 1 - P)).sum(axis=1)
            m = lpf.max()
            tied = np.isclose(lpf, m, rtol=0, atol=1e-9)
            o["full"] += tied[TRUE] / tied.sum() if tied[TRUE] else 0.0
    return out


def main():
    with open(D + "release_validate_patients.csv") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(0)
    rows = rng.sample(rows, N_VAL) if N_VAL < len(rows) else rows
    Pf, pr = train_freq()
    uni = np.full(len(DIS), 1 / len(DIS))
    cfgs = {}
    for a, eps, tau in [(0.5, 1e-2, 0.99), (0.5, 1e-3, 0.9), (0.5, 1e-3, 0.99), (0.5, 1e-4, 0.99),
                        (0.3, 1e-3, 0.99), (0.7, 1e-3, 0.99)]:
        Pk = np.where(M, a, eps)
        cfgs[f"KB a={a} eps={eps} prior=uniform tau={tau}"] = (Pk, uni, tau)
    for tau in (0.9, 0.99, 0.999):
        cfgs[f"FREQ(train {N_TRAIN}) tau={tau}"] = (Pf, pr, tau)
    chunks = [rows[i::NPROC] for i in range(NPROC)]
    with Pool(NPROC) as p:
        parts = p.map(work, [(c, cfgs) for c in chunks])
    n = len(rows)
    for name in cfgs:
        snap = {b: sum(pt[name]["snap"][b] for pt in parts) / n for b in BUDGETS}
        c1 = sum(pt[name]["c1"] for pt in parts) / n
        c3 = sum(pt[name]["c3"] for pt in parts) / n
        il = sum(pt[name]["il"] for pt in parts) / n
        full = sum(pt[name]["full"] for pt in parts) / n
        print(f"{name:45s} stop: GTPA@1(tie-exp)={c1:.4f} @3={c3:.4f} IL(incl init)={il:.2f} full-profile@1={full:.4f} | budget@1 "
              + " ".join(f"{b}:{snap[b]:.3f}" for b in BUDGETS), flush=True)


if __name__ == "__main__":
    main()
