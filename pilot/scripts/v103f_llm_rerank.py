"""LLM top-1 reranker over the KG top-10 candidates (VIOLATES 원칙 #4 — LLM at
inference — but user-authorized to probe the ceiling above the strong recall
base @10=81%). Records as a constraint-violating reference number.

Pipeline:
1. KG cosine+IDF (best recall KG) → top-10 candidate diseases per patient.
2. Render the patient's findings in English (DDXPlus question_en + value meaning).
3. LLM CoT: given findings + the 10 candidate disease names, pick the single
   most likely. Optionally show the KG-matched phenotypes (source grounding).
4. Measure @1 of the LLM's pick.
"""
import sys, json, csv, ast, math, argparse, re, pickle
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
from onlykg_eval_v71_selfaware import compute_idf, reweight

VALUE_CUIS = "/windows/data/medkg/kg/ddxplus_evidence_value_cuis.json"


def render_patient(evs, evmeta):
    lines = []
    for ev in evs:
        if "_@_" in ev:
            base, val = ev.split("_@_", 1)
            m = evmeta.get(base, {})
            q = m.get("question_en", base)
            vm = m.get("value_meaning", {}).get(val, {})
            ven = vm.get("en", val) if isinstance(vm, dict) else val
            lines.append(f"{q} -> {ven}")
        else:
            m = evmeta.get(ev, {})
            lines.append(f"{m.get('question_en', ev)} -> yes")
    # dedup keep order
    seen = set(); out = []
    for l in lines:
        if l not in seen: seen.add(l); out.append(l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--ground", action="store_true", help="show KG-matched phenotypes")
    ap.add_argument("--ground2", action="store_true", help="show each candidate's KG signature phenotypes")
    ap.add_argument("--sc", type=int, default=1, help="self-consistency samples (majority vote)")
    args = ap.parse_args()

    icd = json.load(open("data/ddxplus/disease_icd10_cui_mapping.json"))
    cond = json.load(open("data/ddxplus/release_conditions_en.json"))
    evmeta = json.load(open("data/ddxplus/release_evidences.json"))
    value_cuis = json.load(open(VALUE_CUIS))
    fr2cui = {info.get("cond-name-fr", ""): icd[dn]["cui"] for dn, info in cond.items() if dn in icd}
    cui2name = {info["cui"]: dn for dn, info in icd.items() if "cui" in info}
    dcs = sorted(set(fr2cui.values()))

    def L(f): return pickle.load(open(f"pilot/data/cache/{f}", "rb"))
    Gs = [L("v103ddx49_kg.pkl"), L("v103pres_ddx49_sci_kg.pkl"), L("v103sci_ddx49_kg.pkl")]
    cui2pname = {}
    for G in Gs:
        for nd, dat in G.nodes(data=True):
            if dat.get("ntype") == "phenotype" and dat.get("name"):
                cui2pname.setdefault(nd, dat["name"])
    P = {}
    for d in dcs:
        w = defaultdict(float)
        for G in Gs:
            if d in G:
                for _, p, e in G.out_edges(d, data=True):
                    if e.get("etype") == "HAS_PHENOTYPE": w[p] += e.get("n_mentions", 0.0)
        P[d] = {p: x/(x+2.0) for p, x in w.items() if x > 0}
    all_evs = set().union(*[set(p) for p in P.values()])
    idf = compute_idf(P, 0.12); beta = 0.75
    Pw = reweight(dict(P), idf, 1.0, beta)
    dl = [d for d in Pw if Pw[d]]

    tasks = []
    with open("data/ddxplus/release_test_patients.csv") as f:
        for row in csv.DictReader(f):
            if len(tasks) >= args.n: break
            tc = fr2cui.get(row["PATHOLOGY"])
            if tc not in P or not P[tc]: continue
            evs = ast.literal_eval(row["EVIDENCES"])
            pos = set()
            for ev in evs:
                base = ev.split("_@_", 1)[0] if "_@_" in ev else ev
                m = value_cuis.get(base, {})
                v = m.get("_question", []); pos.update(v if isinstance(v, list) else [])
                if "_@_" in ev:
                    vv = m.get(ev.split("_@_", 1)[1], []); pos.update(vv if isinstance(vv, list) else [])
            posm = pos & all_evs
            if not posm: continue
            patv = {e: idf.get(e, 1.0)**beta for e in posm}
            pn = math.sqrt(sum(v*v for v in patv.values())) or 1e-9
            sc = {}
            for d in dl:
                pr = Pw[d]; dn = math.sqrt(sum(v*v for v in pr.values())) or 1e-9
                sc[d] = sum(patv[e]*pr[e] for e in posm if e in pr)/(pn*dn)
            cand = sorted(dl, key=lambda d: -sc[d])[:args.topk]
            findings = render_patient(evs, evmeta)
            tasks.append((tc, cand, findings, posm))

    print(f"{len(tasks)} patients, building prompts (ground={args.ground})", flush=True)
    base_c1 = sum(1 for tc, cand, _, _ in tasks if cand[0] == tc)

    prompts = []
    for tc, cand, findings, posm in tasks:
        fl = "\n".join(f"- {x}" for x in findings)
        cl = []
        for i, d in enumerate(cand, 1):
            line = f"{i}. {cui2name.get(d, d)}"
            if args.ground:
                matched = [p for p in posm if p in P[d]]
                line += f"  (KG-matched findings: {len(matched)})"
            if args.ground2:
                top = sorted(P[d].items(), key=lambda x: -x[1])[:12]
                sig = ", ".join((cui2pname.get(c, c) + ("*" if c in posm else "")) for c, _ in top)
                line += f"\n   KG profile (* = patient has): {sig}"
            cl.append(line)
        cl = "\n".join(cl)
        prompt = (f"A patient presents with the following findings:\n{fl}\n\n"
                  f"Candidate diagnoses (already shortlisted):\n{cl}\n\n"
                  f"Think step by step about which single diagnosis best explains ALL the findings, "
                  f"especially the discriminating ones. Then end with a line exactly: ANSWER: <number>")
        prompts.append(prompt)

    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E4B-it", dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True,
              limit_mm_per_prompt={"image": 0, "audio": 0})
    temp = 0.0 if args.sc == 1 else 0.6
    sp = SamplingParams(temperature=temp, max_tokens=1024, n=args.sc)
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=True)

    from collections import Counter
    c1 = 0
    for (tc, cand, _, _), o in zip(tasks, outs):
        votes = Counter()
        for samp in o.outputs:
            m = re.findall(r"ANSWER:\s*(\d+)", samp.text)
            if m:
                idx = int(m[-1]) - 1
                if 0 <= idx < len(cand): votes[cand[idx]] += 1
        pick = votes.most_common(1)[0][0] if votes else cand[0]
        c1 += (pick == tc)
    n = len(tasks)
    print(f"\nN={n} sc={args.sc}  KG base @1={100*base_c1/n:.2f}  LLM-rerank @1={100*c1/n:.2f}", flush=True)


if __name__ == "__main__":
    main()
