"""Self-supervised KG embedding (KG+ML) for the disease-phenotype graph.

The LSA channel is a linear SVD factorization. This learns disease/phenotype embeddings by
SELF-SUPERVISED link prediction on the SAME KG (predict which phenotypes a disease has, vs
negatives) — a non-linear, negative-sampled representation that captures higher-order structure
(disease-disease via shared phenotypes; phenotype co-occurrence). §3-safe: trains ONLY on the
KG we built from IE (benchmark-blind), no benchmark labels, no disease-symptom tables, no
inference-time LLM. At diagnosis: score(d | patient P) = <e_d, mean_{p in P} e_p> — a drop-in
richer alternative/complement to the LSA cosine.

Trains on GPU (torch). Saves {disease_emb, phenotype_emb, di, pi} to kg_emb.pt.
Usage: python pilot/scripts/kg_embed.py --kg pilot/cache/prod/kg_merged3c.json --dim 128 --epochs 8
"""
from __future__ import annotations
import json, math, argparse, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg_merged3c.json")
    ap.add_argument("--out", default="pilot/cache/prod/kg_emb.pt")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--neg", type=int, default=5)
    ap.add_argument("--bs", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=0.01)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    kg = json.load(open(a.kg))
    dz = sorted(kg["diseases"]); ph = sorted(kg["phenotypes"])
    di = {d: i for i, d in enumerate(dz)}; pi = {p: i for i, p in enumerate(ph)}
    # IDF over phenotypes (edge weight + negative-sampling by phenotype popularity^0.75)
    from collections import Counter
    df = Counter(e["p"] for e in kg["edges"])
    edges = [(di[e["d"]], pi[e["p"]]) for e in kg["edges"] if e["d"] in di and e["p"] in pi]
    src = torch.tensor([e[0] for e in edges], device=dev)
    dst = torch.tensor([e[1] for e in edges], device=dev)
    nP = len(ph)
    # unigram^0.75 negative sampling table over phenotypes
    freq = torch.tensor([df.get(p, 1) for p in ph], dtype=torch.float, device=dev) ** 0.75
    negp = freq / freq.sum()
    print(f"KG embed: {len(dz)} dz, {len(ph)} ph, {len(edges)} edges, dim={a.dim}, dev={dev}", flush=True)

    Ed = torch.nn.Embedding(len(dz), a.dim).to(dev)
    Ep = torch.nn.Embedding(len(ph), a.dim).to(dev)
    torch.nn.init.normal_(Ed.weight, std=0.1); torch.nn.init.normal_(Ep.weight, std=0.1)
    opt = torch.optim.Adam(list(Ed.parameters()) + list(Ep.parameters()), lr=a.lr)
    M = src.shape[0]
    for ep in range(a.epochs):
        perm = torch.randperm(M, device=dev)
        tot = 0.0; t0 = time.time()
        for i in range(0, M, a.bs):
            idx = perm[i:i + a.bs]
            d = src[idx]; p = dst[idx]
            ed = Ed(d)                                   # (B, dim)
            ep_pos = Ep(p)
            pos = (ed * ep_pos).sum(-1)                  # <e_d, e_p+>
            negs = torch.multinomial(negp, idx.shape[0] * a.neg, replacement=True).view(idx.shape[0], a.neg)
            ep_neg = Ep(negs)                            # (B, neg, dim)
            neg = torch.bmm(ep_neg, ed.unsqueeze(-1)).squeeze(-1)   # (B, neg)
            # logistic link-prediction loss (word2vec-style)
            loss = -(torch.nn.functional.logsigmoid(pos).mean()
                     + torch.nn.functional.logsigmoid(-neg).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"  epoch {ep+1}/{a.epochs}  loss={tot/(M//a.bs+1):.4f}  ({time.time()-t0:.1f}s)", flush=True)

    torch.save({"disease_emb": Ed.weight.detach().cpu(), "phenotype_emb": Ep.weight.detach().cpu(),
                "di": di, "pi": pi}, a.out)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
