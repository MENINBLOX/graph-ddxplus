"""Confusion-pair forensic for the discrimination bottleneck.

Reuses the v103 cosine+IDF scoring (identical to v103_eval_algoideas.evaluate, mode=base).
For every DDXPlus test patient whose GT is rankable, bucket into:
  - @1 correct
  - in top-10 but NOT #1   <- the discrimination failure we care about
  - outside top-10

For the top-10-not-#1 bucket, tabulate (GT, predicted-#1) confusion pairs, then
for the most frequent pairs explain WHY: which evidence is shared (non-discriminating)
vs which CUIs each profile uniquely carries that the patient also has.

CPU-only. Does not touch the GPU / running IE.
"""
import sys, json, math, argparse, csv, ast
from collections import defaultdict, Counter
sys.path.insert(0, "pilot/scripts")
import pickle
from v103_eval_algoideas import load_patients, build_profile
from onlykg_eval_v71_selfaware import compute_idf, reweight


def cui2name():
    """cui -> English disease name (invert DDXPlus icd map)."""
    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    return {info["cui"]: dn for dn, info in icd.items() if "cui" in info}


def rank_patient(pos, profile, idf, beta, dlist, all_evs):
    pos = pos & all_evs
    if not pos:
        return None, None
    patv = {e: idf.get(e, 1.0) ** beta for e in pos}
    pn = math.sqrt(sum(v * v for v in patv.values())) or 1e-9
    scores = {}
    for d in dlist:
        prof = profile[d]
        dn = math.sqrt(sum(v * v for v in prof.values())) or 1e-9
        dot = sum(patv[e] * prof[e] for e in pos if e in prof)
        scores[d] = dot / (pn * dn)
    ranked = sorted(dlist, key=lambda d: -scores[d])
    return ranked, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="pilot/data/cache/v103deep120m_kg.pkl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--top_pairs", type=int, default=15)
    ap.add_argument("--explain", type=int, default=6)
    ap.add_argument("--out", default="docs/forensic_v103_confusion.md")
    args = ap.parse_args()

    G = pickle.load(open(args.graph, "rb"))
    dcs, pats = load_patients(args.n)
    profile, all_evs = build_profile(G, dcs)
    idf = compute_idf(profile, 0.12)
    profile = reweight(profile, idf, 1.0, args.beta)
    dlist = [d for d in profile if profile[d]]
    name = cui2name()
    nm = lambda c: name.get(c, c)

    n = c1 = ctop10 = cout = 0
    conf = Counter()                 # (gt, pred1) for in-top10-not-#1
    pair_patients = defaultdict(list)
    for p in pats:
        tc = p["true"]
        if tc not in profile or not profile[tc]:
            continue
        ranked, scores = rank_patient(p["pos"], profile, idf, args.beta, dlist, all_evs)
        if ranked is None:
            continue
        rk = ranked.index(tc) + 1
        n += 1
        if rk == 1:
            c1 += 1
        elif rk <= 10:
            ctop10 += 1
            pred1 = ranked[0]
            conf[(tc, pred1)] += 1
            if len(pair_patients[(tc, pred1)]) < 3:
                pair_patients[(tc, pred1)].append(p)
        else:
            cout += 1

    lines = []
    lines.append(f"# v103 혼동쌍 forensic — discrimination 병목\n")
    lines.append(f"KG: `{args.graph}`  /  patients scored: {n}\n")
    lines.append(f"- @1 정답: {c1} ({100*c1/n:.1f}%)")
    lines.append(f"- top-10엔 있으나 #1 실패 (혼동): {ctop10} ({100*ctop10/n:.1f}%)")
    lines.append(f"- top-10 밖: {cout} ({100*cout/n:.1f}%)\n")
    lines.append(f"**개선 여지 = 혼동 {ctop10}건 (top-10 안에 정답이 이미 있음).**\n")

    lines.append(f"## 가장 잦은 혼동쌍 (GT → 잘못 뽑힌 #1), top {args.top_pairs}\n")
    lines.append("| # | GT (정답) | 오답 #1 | 건수 |")
    lines.append("|---|---|---|---|")
    for i, ((gt, pr), cnt) in enumerate(conf.most_common(args.top_pairs), 1):
        lines.append(f"| {i} | {nm(gt)} | {nm(pr)} | {cnt} |")
    lines.append("")

    lines.append(f"## 왜 안 갈렸나 — 상위 {args.explain} 쌍 evidence 분석\n")
    for (gt, pr), cnt in conf.most_common(args.explain):
        gp, pp = set(profile[gt]), set(profile[pr])
        reps = pair_patients[(gt, pr)]
        # union of patient pos CUIs across reps, restricted to vocab
        ppos = set()
        for p in reps:
            ppos |= (p["pos"] & all_evs)
        shared = ppos & gp & pp                  # in patient + both profiles -> non-discriminating
        gt_only = (ppos & gp) - pp               # patient has, only GT profile -> SHOULD favor GT
        pr_only = (ppos & pp) - gp               # patient has, only wrong profile -> pulls to wrong
        def top(cuis, prof, k=8):
            return sorted(cuis, key=lambda c: -prof.get(c, 0))[:k]
        lines.append(f"### {nm(gt)} → {nm(pr)}  ({cnt}건)")
        lines.append(f"- 공유(비변별) evidence {len(shared)}개: " +
                     ", ".join(top(shared, profile[gt])) + ("…" if len(shared) > 8 else ""))
        lines.append(f"- GT만 가진 환자 evidence {len(gt_only)}개(정답 쪽이어야): " +
                     ", ".join(top(gt_only, profile[gt])) if gt_only else "- GT-only evidence 없음 ⚠️ (환자 증거가 GT 프로필에 안 잡힘 = IE 부족)")
        lines.append(f"- 오답만 가진 환자 evidence {len(pr_only)}개(오답으로 끌림): " +
                     ", ".join(top(pr_only, profile[pr])) if pr_only else "- 오답-only evidence 없음")
        lines.append("")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines[:14]))
    print(f"\n... full → {args.out}")


if __name__ == "__main__":
    main()
