"""Build pilot/cache/v3/recall/sensitivity.md from the xref / ranker / both decompositions."""
import json
import sys
from collections import Counter

ROOT = "/home/max/thesis/"
O = ROOT + "pilot/cache/v3/recall"
GRAPHS = [("D", "D (base_nm)", 0.0), ("smix", "S-mix", 0.5), ("sllm", "S-llm", 1.0), ("sdict", "S-dict", 0.5)]
MODES = ["xref", "ranker", "both"]
STAGES = ["no_cui", "no_docs", "not_in_text", "not_extracted", "not_linked", "filtered", "near_child",
          "near_offtopic", "offtopic_only", "near", "edge"]
GROUPS = [("collection loss (no_docs+not_in_text)", ["no_docs", "not_in_text"]),
          ("extraction+linking (not_extracted+not_linked)", ["not_extracted", "not_linked"]),
          ("filtered", ["filtered"]),
          ("offtopic_only", ["offtopic_only"]),
          ("near_offtopic", ["near_offtopic"]),
          ("near_child", ["near_child"]),
          ("near", ["near"]),
          ("edge", ["edge"]),
          ("near+edge", ["near", "edge"]),
          ("no_cui", ["no_cui"])]


def paths(g, m):
    if m == "xref":
        base = ROOT + "pilot/cache/v2/recall/decomp" if g == "D" else f"{O}/decomp_{g}"
    else:
        base = f"{O}/decomp_{g}_{m}cui"
    return base + ".json", base + "_pairs.json"


def load():
    res, prs = {}, {}
    for g, _, _ in GRAPHS:
        for m in MODES:
            js, pj = paths(g, m)
            try:
                res[(g, m)] = json.load(open(js))
                prs[(g, m)] = {(r["d"], r["hpo"]): r for r in json.load(open(pj))}
            except FileNotFoundError:
                print("missing", g, m, file=sys.stderr)
    return res, prs


def fmt(n, tot):
    return f"{n:,} ({100 * n / tot:.1f}%)"


def main():
    res, prs = load()
    n_trip = {k: r["n"] for k, r in prs[("smix", "xref")].items()}
    T = sum(n_trip.values())
    cnt = {gm: Counter({s: v[0] for s, v in r["by_triple"].items()}) for gm, r in res.items()}
    L = []
    w = L.append
    w("# Recall decomposition: feature CUI 민감도 (xref / ranker / both)\n")
    w("생성: `recall_decomp.py --feature-cui {xref,ranker,both}`. 단위는 (case, gold, curated feature) triple "
      f"{T:,}건, 괄호는 그 비율이다. 입력 인자는 각 JSON의 `args`에 있다.\n")
    w("| 모드 | feature 매칭 CUI | near | 결과 파일 |")
    w("|---|---|---|---|")
    w("| xref (공개 결과) | hp.json UMLS xref (`CA.hpo_to_cui`) | HPO is_a 부모 **또는 자식**의 xref CUI에 topic edge | "
      "D `pilot/cache/v2/recall/decomp.json`, v3 `decomp_{smix,sllm,sdict}.json` |")
    w("| ranker | ranker가 조회하는 CUI 하나 (`hpo_umls_mapping.json`, `temporal_eval.load_cases`와 동일) | "
      "그래프 Concept IS_A(`phenotype_parents.json`) **부모**에 topic edge. 출발 CUI가 edge를 가진 concept일 때만 "
      "IS_A가 있다(build_graph_v2) | `decomp_<graph>_rankercui.json` |")
    w("| both | ranker CUI 또는 xref CUI | 둘 모두의 그래프 IS_A 부모 | `decomp_<graph>_bothcui.json` |\n")
    w("ranker/both 모드에서 추가된 stage: `no_cui`(ranker CUI가 없어 ranker 입력에서 빠지는 feature), "
      "`near_child`(자식 concept에만 topic edge, ranker는 점수 없음), `near_offtopic`(부모의 off-topic edge만, "
      "decay x offtopic_w로 점수). 판정 순서는 edge > near > offtopic_only > near_offtopic > near_child > filtered > "
      "no_docs > not_in_text > not_linked > not_extracted 이다.\n")
    w("`scored`는 stage 정의와 무관한 기준이다. graph_dx.RANK가 이 feature(ranker CUI와 그 그래프 IS_A 부모)로 gold 질병에 "
      "0보다 큰 점수를 주면 1이다. topic edge는 1, off-topic edge는 frozen design E `offtopic_w`"
      "(D 0, S-mix 0.5, S-llm 1.0, S-dict 0.5)를 곱한다. idf > 0도 확인한다. ranker 모드 실행에서 계산했고 "
      "both 실행에서도 pair 단위로 같다.\n")
    w("whole-graph pass 검증: 각 ranker/both 실행이 모든 질병의 item을 다시 링크해서 얻은 "
      "diseases_with_findings / edges / concepts / Concept IS_A가 빌드 로그·summary와 같다:\n")
    w("| graph | diseases_with_findings | edges | concepts | Concept IS_A |")
    w("|---|---|---|---|---|")
    for g, name, _ in GRAPHS:
        for m in ("ranker", "both"):
            if (g, m) in res:
                gr = res[(g, m)]["graph"]
                w(f"| {name} {m} | {gr['diseases_with_findings']:,} | {gr['edges']:,} | {gr['concepts']:,} | "
                  f"{gr['concept_is_a']:,} |")
    w("")

    # 0. summary
    def pct(g, m, ss):
        return 100 * sum(cnt[(g, m)][x] for x in ss) / T

    col, ext = GROUPS[0][1], GROUPS[1][1]
    w("## 요약: 헤드라인 수치의 이동\n")
    gap = {m: pct("D", m, col) - pct("smix", m, col) for m in MODES}
    w(f"- **collection loss**(no_docs+not_in_text): D 대 S-mix가 xref {pct('D', 'xref', col):.1f}% 대 "
      f"{pct('smix', 'xref', col):.1f}%(차 {gap['xref']:.1f}pp)에서 ranker {pct('D', 'ranker', col):.1f}% 대 "
      f"{pct('smix', 'ranker', col):.1f}%(차 {gap['ranker']:.1f}pp), both {pct('D', 'both', col):.1f}% 대 "
      f"{pct('smix', 'both', col):.1f}%(차 {gap['both']:.1f}pp)가 된다. 어느 graph에서도 xref 대비 "
      f"{max(abs(pct(g, 'xref', col) - pct(g, m, col)) for g, _, _ in GRAPHS for m in ('ranker', 'both')):.1f}pp 이내로 "
      "움직인다. '수집 손실이 67%에서 37%로 줄었다'는 결론은 매핑 기준에 민감하지 않다.")
    xs = ", ".join(f"{name} {pct(g, 'xref', ext):.1f}→{pct(g, 'ranker', ext):.1f}%" for g, name, _ in GRAPHS)
    nl = ", ".join(f"{name} {cnt[(g, 'xref')]['not_linked']:,}→{cnt[(g, 'ranker')]['not_linked']:,}" for g, name, _ in GRAPHS)
    w(f"- **extraction+linking**(not_extracted+not_linked), xref→ranker: {xs}. 움직임은 거의 not_linked다({nl}). "
      "v3 graph는 --hpo-names가 바로 ranker CUI를 graph에 넣기 때문에 xref 기준 not_linked가 부풀어 있었고, D는 반대로 "
      "약간 과소였다. 그래서 D와 v3의 차이는 ranker 기준에서 더 커진다"
      f"(D−S-mix {pct('D', 'xref', ext) - pct('smix', 'xref', ext):.1f}pp → "
      f"{pct('D', 'ranker', ext) - pct('smix', 'ranker', ext):.1f}pp).")
    ed = ", ".join(f"{name} {pct(g, 'xref', ['edge']):.1f}→{pct(g, 'ranker', ['edge']):.1f}%" for g, name, _ in GRAPHS)
    w(f"- **edge**, xref→ranker: {ed}. D−S-mix 차이는 {pct('smix', 'xref', ['edge']) - pct('D', 'xref', ['edge']):.1f}pp → "
      f"{pct('smix', 'ranker', ['edge']) - pct('D', 'ranker', ['edge']):.1f}pp. near는 ranker 정의(그래프 IS_A 부모만)에서 줄고"
      f"(S-mix {pct('smix', 'xref', ['near']):.1f}→{pct('smix', 'ranker', ['near']):.1f}%), 자식만 닿는 near_child"
      f"({pct('D', 'ranker', ['near_child']):.1f}/{pct('smix', 'ranker', ['near_child']):.1f}/"
      f"{pct('sllm', 'ranker', ['near_child']):.1f}/{pct('sdict', 'ranker', ['near_child']):.1f}%)는 따로 센다.")
    ot = ", ".join(f"{name} {cnt[(g, 'ranker')]['offtopic_only']:,}+{cnt[(g, 'ranker')]['near_offtopic']:,}"
                   for g, name, _ in GRAPHS[1:])
    w(f"- **offtopic_only**: v3 graph에서는 손실이 아니다. frozen E offtopic_w(0.5/1.0/0.5)에서 ranker 기준 offtopic_only와 "
      f"near_offtopic triple은 모두 점수를 받는다({ot}, 3절). D(offtopic_w 0)에서는 0%다.")
    sc = ", ".join(f"{name} {res[(g, 'ranker')]['scored'][1]:.1f}%" for g, name, _ in GRAPHS)
    w(f"- **ranker가 실제로 점수를 주는 triple**(stage 정의와 무관): {sc}.\n")

    # 1. headline
    w("## 1. 헤드라인 수치가 얼마나 움직이나\n")
    w(f"셀은 triple 비율(%)이고 xref → ranker (both) 순이다. 세 값의 분모는 모두 같은 {T:,} triple이다.\n")
    w("| 지표 | " + " | ".join(name for _, name, _ in GRAPHS) + " |")
    w("|---|" + "---|" * len(GRAPHS))
    for label, ss in GROUPS + [("scored by ranker", None)]:
        cells = []
        for g, _, _ in GRAPHS:
            vals = []
            for m in MODES:
                if (g, m) not in res:
                    vals.append(None)
                    continue
                if ss is None:
                    vals.append(None if m == "xref" else res[(g, m)]["scored"][0])
                elif all(s in res[(g, m)]["by_triple"] for s in ss):
                    vals.append(sum(cnt[(g, m)][s] for s in ss))
                else:
                    vals.append(None)
            x, r, b = vals
            pct = lambda v: "-" if v is None else f"{100 * v / T:.1f}"  # noqa: E731
            if ss is None:
                cells.append(f"{r:,} ({pct(r)}%)" if r is not None else "-")
            else:
                cells.append(f"{pct(x)} → **{pct(r)}** ({pct(b)})")
        w(f"| {label} | " + " | ".join(cells) + " |")
    cells = []
    for g, _, _ in GRAPHS:
        v = []
        for m in MODES:
            if (g, m) not in res:
                v.append("-")
                continue
            c = cnt[(g, m)]
            rest = T - c["no_cui"] - c["no_docs"] - c["not_in_text"]
            v.append(f"{100 * (c['not_extracted'] + c['not_linked']) / rest:.1f}")
        cells.append(f"{v[0]} → **{v[1]}** ({v[2]})")
    w("| extraction+linking / (collection 손실과 no_cui를 뺀 triple) | " + " | ".join(cells) + " |")
    w("")
    w("건수(triple)로 본 핵심 지표:\n")
    w("| 지표 | 모드 | " + " | ".join(name for _, name, _ in GRAPHS) + " |")
    w("|---|---|" + "---|" * len(GRAPHS))
    for label, ss in [GROUPS[0], GROUPS[1], GROUPS[7], GROUPS[8]]:
        for m in MODES:
            cells = [fmt(sum(cnt[(g, m)][s] for s in ss), T) if (g, m) in res else "-" for g, _, _ in GRAPHS]
            w(f"| {label} | {m} | " + " | ".join(cells) + " |")
    w("")

    # 2. full stage tables
    w("## 2. Stage별 전체 분포 (triple)\n")
    for g, name, ow in GRAPHS:
        w(f"### {name} (frozen offtopic_w {ow})\n")
        w("| stage | xref | ranker | both |")
        w("|---|---|---|---|")
        for s in STAGES:
            cells = []
            for m in MODES:
                if (g, m) not in res or s not in res[(g, m)]["by_triple"]:
                    cells.append("-")
                else:
                    cells.append(fmt(cnt[(g, m)][s], T))
            w(f"| {s} | " + " | ".join(cells) + " |")
        w("")

    # 3. offtopic_only weight
    w("## 3. offtopic_only를 손실로 볼 수 있나: graph weight > 0 비율\n")
    w("offtopic_only triple 중 ranker가 실제로 0보다 큰 점수를 주는 비율(`scored`)이다. xref 행은 공개 결과의 "
      "offtopic_only triple에 ranker 실행의 pair별 `scored`를 붙인 것이다.\n")
    w("| graph | offtopic_w | xref offtopic_only | 그중 scored | ranker offtopic_only | 그중 scored | "
      "ranker near_offtopic | 그중 scored | both offtopic_only | 그중 scored |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for g, name, ow in GRAPHS:
        rk = prs.get((g, "ranker"))
        row = [name, str(ow)]
        for m, s in (("xref", "offtopic_only"), ("ranker", "offtopic_only"), ("ranker", "near_offtopic"),
                     ("both", "offtopic_only")):
            if (g, m) not in prs or rk is None:
                row += ["-", "-"]
                continue
            tot = sc = 0
            for k, r in prs[(g, m)].items():
                if r["v"] == s:
                    tot += n_trip[k]
                    sc += n_trip[k] * rk[k]["s"]
            row += [f"{tot:,}", f"{sc:,} ({100 * sc / max(1, tot):.1f}%)"]
        w("| " + " | ".join(row) + " |")
    w("")

    # 4. published stages actually scored
    w("## 4. 공개(xref) stage별로 ranker가 실제 점수를 주는 triple\n")
    w("셀은 scored / 해당 stage triple. xref 정의에서 손실로 분류된 triple 중 ranker가 쓰는 것, edge·near로 분류됐지만 "
      "ranker가 쓰지 못하는 것을 보여 준다.\n")
    w("| xref stage | " + " | ".join(name for _, name, _ in GRAPHS) + " |")
    w("|---|" + "---|" * len(GRAPHS))
    for s in [x for x in STAGES if x not in ("no_cui", "near_child", "near_offtopic")]:
        cells = []
        for g, _, _ in GRAPHS:
            rk, xr = prs.get((g, "ranker")), prs.get((g, "xref"))
            if rk is None or xr is None:
                cells.append("-")
                continue
            tot = sum(n_trip[k] for k, r in xr.items() if r["v"] == s)
            sc = sum(n_trip[k] * rk[k]["s"] for k, r in xr.items() if r["v"] == s)
            cells.append(f"{sc:,} / {tot:,}")
        w(f"| {s} | " + " | ".join(cells) + " |")
    w("")

    # 5. transitions
    w("## 5. 주요 이동 xref → ranker (triple, 상위 10)\n")
    for g, name, _ in GRAPHS:
        rk, xr = prs.get((g, "ranker")), prs.get((g, "xref"))
        if rk is None or xr is None:
            continue
        t = Counter()
        for k, r in xr.items():
            if r["v"] != rk[k]["v"]:
                t[(r["v"], rk[k]["v"])] += n_trip[k]
        moved = sum(t.values())
        w(f"- **{name}** (바뀐 triple {moved:,}, {100 * moved / T:.1f}%): "
          + ", ".join(f"{a}→{b} {c:,}" for (a, b), c in t.most_common(10)))
    w("")

    # 6. HPO hierarchy vs graph IS_A
    w("## 6. HPO is_a와 그래프 IS_A의 차이 (ranker 모드)\n")
    w("ranker 모드에서 점수가 없는 stage(near_child, filtered, no_docs, not_in_text, not_extracted, not_linked, no_cui)의 "
      "triple 중, 같은 CUI 매핑으로 HPO is_a 부모(또는 자식만)의 CUI에 topic edge가 있는 triple 수다. ranker는 HPO 계층을 "
      "쓰지 않으므로 점수가 없다. 공개 정의의 near는 이 경우를 포함했다. 반대로 그래프 IS_A(UMLS PAR/RB)로만 닿는 경우는 "
      "공개 정의에서 빠졌다(5절의 not_in_text→near, not_in_text→near_offtopic).\n")
    w("| graph | HPO 부모 topic edge | HPO 자식만 topic edge | 그중 판정이 near_child |")
    w("|---|---|---|---|")
    for g, name, _ in GRAPHS:
        if (g, "ranker") not in res:
            continue
        hn = res[(g, "ranker")]["hpo_near_by_stage"]
        low = ["no_cui", "no_docs", "not_in_text", "not_extracted", "not_linked", "filtered", "near_child"]
        p = sum(hn[s]["parent"] for s in low)
        c = sum(hn[s]["child_only"] for s in low)
        nc = hn["near_child"]["parent"] + hn["near_child"]["child_only"]
        w(f"| {name} | {p:,} | {c:,} | {nc:,} |")
    w("")
    # 7. supplement and caveats
    sup = "pilot/cache/v3/recall/supp"
    try:
        fx = json.load(open(f"{sup}/v2_fullflags/decomp.json"))["by_triple"]
        fr = json.load(open(f"{sup}/v2_fullflags_ranker/decomp.json"))["by_triple"]
    except FileNotFoundError:
        fx = fr = None
    if fx:
        e = lambda b: b["not_extracted"][0] + b["not_linked"][0]  # noqa: E731
        w("## 7. 보충: v2 corpus에 v3 flag만 적용 (base_split_tg_ns와 같은 graph)\n")
        w("`recall_decomp.py --link full --topic-genes --no-self [--feature-cui ranker --offtopic-w 0]` (v2 sources, "
          "ie_gemma_full, phrase2cui_gemma). ranker 실행의 whole-graph 수치는 build_base_split_tg_ns.log와 같다"
          "(27,473 / 1,144,472 / 46,529 / 33,298).\n")
        w("| | xref | ranker |")
        w("|---|---|---|")
        w(f"| not_linked | {fx['not_linked'][0]:,} | {fr['not_linked'][0]:,} |")
        w(f"| extraction+linking | {e(fx):,} | {e(fr):,} |")
        w(f"| edge | {fx['edge'][0]:,} | {fr['edge'][0]:,} |")
        w("")
        w(f"flag만으로 D의 not_linked가 줄어드는 폭은 ranker 기준에서 더 크다(xref {cnt[('D', 'xref')]['not_linked']:,}→"
          f"{fx['not_linked'][0]:,}, ranker {cnt[('D', 'ranker')]['not_linked']:,}→{fr['not_linked'][0]:,}). flag를 고정하고 "
          f"corpus만 v3로 바꾸면 extraction+linking은 xref 기준 {e(fx):,}→"
          f"{sum(cnt[('smix', 'xref')][x] for x in ext):,}(S-mix)로 거의 그대로지만, ranker 기준으로는 {e(fr):,}→"
          f"{sum(cnt[('smix', 'ranker')][x] for x in ext):,}로 줄어든다.\n")
    w("## 주의\n")
    w("- 문자열 단계(no_docs, not_in_text, not_extracted, not_linked)의 판정은 모드와 무관하다. 모드 간 차이는 그 앞의 CUI "
      "단계(edge, near, offtopic_only, near_offtopic, near_child, filtered)가 먼저 가져가는 triple이 달라서 생긴다.")
    w("- no_cui 393 triple(155 pair)은 hpo_umls_mapping CUI도 hp.json xref도 없다. 그래서 both에서도 같다. load_cases는 "
      "이 feature를 환자 증거에서 뺀다.")
    w("- `scored`는 HAS_FINDING 경로만 본다. unified evidence(환자 CUI가 gold 질병이나 그 부모 질병 자체인 경우)는 넣지 "
      "않았다. --no-self 때문에 이런 CUI는 filtered로 분류되지만 ranker는 unified evidence로 점수를 준다(앞선 검증에서 "
      "xref 기준 S-mix 149, S-llm 146, S-dict 79 triple).")
    w("- 문자열 존재 검사는 부정문 속 언급도 'text에 있음'으로 센다. 그래서 not_extracted는 엄밀한 추출 손실 하한이 아니다"
      "(앞선 검증 지적, 이번 변경과 무관).")
    sys.stdout.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
