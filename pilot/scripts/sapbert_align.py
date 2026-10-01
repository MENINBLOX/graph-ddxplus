"""SapBERT soft CUI-alignment (ML technique, strict-zero-shot safe).

The bottleneck for common-disease benchmarks is CUI-vocabulary mismatch: a patient evidence
concept and the KG phenotype for the SAME finding often get DIFFERENT UMLS CUIs (granularity /
linker disagreement), so exact-CUI overlap misses them (DDXPlus true-recall 34%, of which +25%
is only hierarchy-reachable). SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext) is a
biomedical encoder trained on UMLS synonymy to embed concept NAMES so synonyms/near-granularity
terms are close. We use it to align each benchmark evidence CUI to its nearest KG-phenotype CUIs
by cosine similarity -- a learned vocabulary bridge that knows nothing about the benchmark
(no labels, no disease-symptom mappings) -> §3 safe.

Exposes embed_names() and build_alignment(); imported by the evaluator.
"""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModel

# panel of biomedical concept encoders (same family) for an academically-defensible comparison
MODELS = {
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "coder": "GanjinZero/UMLSBert_ENG",                       # CODER, UMLS-trained
    "biolord": "FremyCompany/BioLORD-2023-C",                 # BioLORD concept encoder
    "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",       # general-domain baseline
}

_MODEL = MODELS["sapbert"]
_pool = "cls"
_tok = _mod = None
_cur = None


def set_model(key):
    """Switch the active encoder (by MODELS key or raw HF id). Resets the cache."""
    global _MODEL, _tok, _mod, _cur, _pool
    _MODEL = MODELS.get(key, key)
    _pool = "mean" if key in ("biolord", "minilm") else "cls"
    _tok = _mod = None
    _cur = key
    return _MODEL


def _load():
    global _tok, _mod
    if _mod is None:
        _tok = AutoTokenizer.from_pretrained(_MODEL)
        _mod = AutoModel.from_pretrained(_MODEL).cuda().eval()
    return _tok, _mod


@torch.no_grad()
def embed_names(names, bs=512):
    """[CLS] embeddings, L2-normalized. names: list[str] -> (N, H) float tensor on GPU."""
    tok, mod = _load()
    out = []
    for i in range(0, len(names), bs):
        batch = names[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=32, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        hs = mod(**enc).last_hidden_state
        if _pool == "mean":
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            emb = hs[:, 0]                                  # CLS
        emb = torch.nn.functional.normalize(emb, dim=1)
        out.append(emb)
    return torch.cat(out, 0)


def build_alignment(ev_cuis, ev_names, kg_cuis, kg_names, topk=5, tau=0.80):
    """For each evidence CUI, its nearest KG-phenotype CUIs by SapBERT cosine.
    Returns {ev_cui: [(kg_cui, sim), ...]} keeping matches with sim>=tau (always the top-1)."""
    ev_emb = embed_names(ev_names)                          # (E, H)
    kg_emb = embed_names(kg_names)                          # (K, H)
    align = {}
    B = 256
    for i in range(0, len(ev_cuis), B):
        sims = ev_emb[i:i + B] @ kg_emb.T                  # (b, K)
        vals, idx = sims.topk(min(topk, len(kg_cuis)), dim=1)
        for r in range(vals.size(0)):
            e = ev_cuis[i + r]
            lst = []
            for j in range(vals.size(1)):
                s = float(vals[r, j])
                if j == 0 or s >= tau:
                    lst.append((kg_cuis[idx[r, j]], s))
            align[e] = lst
    return align
