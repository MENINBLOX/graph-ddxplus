"""KG + LLM re-ranker: the KG produces a traceable top-10 shortlist; a LOCAL 8B LLM picks.

This is a SEPARATE system variant from the strict-zero-shot KG-only model: it uses an
inference-time LLM, so it is compared against the LLM/agentic paradigm (DeepRare etc.), not
reported under the strict-zero-shot claim. The LLM only re-ranks the KG's candidate list (it
never sees the gold, never does free diagnosis, no disease-symptom tables) and runs on the
same local Gemma-E4B (~8B) -- so the cost story (no GPT-4o/API) is preserved.

Input: the per-benchmark dumps from onlykg_eval_unified2.py --rerank_dump (each patient =
{gold_cuis, gold_rank(KG), top10:[[key,name,score]], evidence:[names]}).
Output: LLM Recall@1/@3 vs the KG baseline (bounded above by KG Recall@10, the shortlist ceiling).

Usage: python pilot/scripts/llm_rerank.py --dump <path>.<Bench>.json [--bs 32]
"""
from __future__ import annotations
import json, re, sys, glob, argparse
sys.path.insert(0, "pilot/scripts")
from v146_decomp import build_backend

PROMPT = '''You are an expert physician making a differential diagnosis.

A patient presents with the following clinical findings:
{evidence}

Below is a shortlist of candidate diagnoses retrieved for this patient. Using clinical reasoning about which diagnosis best explains ALL of the findings, choose the THREE most likely, most likely first.

Candidates:
{cands}

Answer with ONLY the three candidate numbers, most likely first, comma-separated (for example: 4, 1, 7). Do not add any other text.'''


def parse_ranks(text, k):
    nums = [int(x) for x in re.findall(r"\d+", text)]
    seen, out = set(), []
    for x in nums:
        if 1 <= x <= k and x not in seen:
            seen.add(x); out.append(x)
        if len(out) == 3:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="glob of <path>.<Bench>.json rerank dumps")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.5, help="KG dominance margin: keep KG #1 if (s1-s2)/s1 >= margin, else defer to LLM")
    a = ap.parse_args()

    files = sorted(glob.glob(a.dump))
    chat = build_backend("vllm", a.bs, a.max_new)
    for f in files:
        bench = f.split(".")[-2]
        data = json.load(open(f))
        prompts = []
        for p in data:
            cands = "\n".join(f"{i+1}. {c[1]}" for i, c in enumerate(p["top10"]))
            ev = ", ".join(p["evidence"]) or "(no findings provided)"
            prompts.append(PROMPT.replace("{evidence}", ev).replace("{cands}", cands))
        outs = chat(prompts)

        n = len(data)
        kg1 = kg3 = llm1 = llm3 = ceil10 = bl1 = bl3 = 0
        for p, o in zip(data, outs):
            gold = set(p["gold_cuis"])
            kr = p["gold_rank"]
            kg1 += (kr == 1); kg3 += (kr <= 3); ceil10 += (kr <= 10)
            top = p["top10"]
            picks = parse_ranks(o, len(top))
            keys = [top[i - 1][0] for i in picks]                  # LLM-ranked candidate keys
            if keys and keys[0] in gold:
                llm1 += 1
            if any(k in gold for k in keys[:3]):
                llm3 += 1
            # --- KG-score-aware blend: trust a DOMINANT KG #1, else defer to the LLM ---
            # dominance = KG top-1 score margin over top-2 (normalized). If the KG is confident
            # (clear margin), keep its #1; otherwise adopt the LLM's re-ranking.
            s1 = top[0][2] if top else 0.0
            s2 = top[1][2] if len(top) > 1 else 0.0
            dominant = s1 > 0 and (s1 - s2) / s1 >= a.margin
            if dominant or not keys:
                b_order = [c[0] for c in top]                       # keep KG order
            else:
                lk = keys + [c[0] for c in top if c[0] not in keys]  # LLM first, then rest KG
                b_order = lk
            bl1 += (b_order and b_order[0] in gold)
            bl3 += any(k in gold for k in b_order[:3])
        print(f"\n=== {bench}  (n={n}) ===")
        print(f"  KG-only   : @1={100*kg1/n:5.2f}  @3={100*kg3/n:5.2f}   (shortlist ceiling @10={100*ceil10/n:.2f})")
        print(f"  KG+LLM    : @1={100*llm1/n:5.2f}  @3={100*llm3/n:5.2f}   (Δ@1 {100*(llm1-kg1)/n:+.2f})")
        print(f"  KG+LLM blend(m={a.margin}): @1={100*bl1/n:5.2f}  @3={100*bl3/n:5.2f}   (Δ@1 {100*(bl1-kg1)/n:+.2f})")


if __name__ == "__main__":
    main()
