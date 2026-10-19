"""Unified-evidence evaluator (CLAUDE.md §11) — the eval-method fix.

Forensic (PKU): the KG holds correct PKU knowledge, but a RareBench patient's pathognomonic
finding 'Hyperphenylalaninemia' is a DISEASE node in our KG (the parent concept of PKU), not
a phenotype edge -> phenotype-only overlap scores it 0. §11 says treat every CUI as unified
evidence and let disease CUIs be match targets. So we score a candidate disease D against
patient evidence E by, for each e in E, the MAX of:
  (a) phenotype match   : e (or a hierarchy neighbor) is in D's merged subtype phenotype profile
  (b) disease match     : e is a disease-hierarchy relative (ancestor/descendant) of D, weighted
                          by the information content of e  (rare parent concept => discriminative)
Everything is UMLS structure -> strict zero-shot safe. Reuses loaders/Resolver from
onlykg_eval_unified.
"""
from __future__ import annotations
import sys, json, math, argparse
from collections import defaultdict
sys.path.insert(0, "pilot/scripts")
import onlykg_eval_unified as U


def build_lsa(kg, prof, idf, k=256):
    """Truncated-SVD latent space of the disease x phenotype matrix (LSA). Returns disease-idx,
    phenotype-idx, L2-normalized disease embeddings, phenotype embeddings."""
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    dz = sorted(prof); ph = sorted(kg["phenotypes"])
    di = {d: i for i, d in enumerate(dz)}; pi = {p: i for i, p in enumerate(ph)}
    rows, cols, vals = [], [], []
    for d in dz:
        for p, w in prof[d].items():
            if p in pi:
                rows.append(di[d]); cols.append(pi[p]); vals.append(w)
    M = csr_matrix((vals, (rows, cols)), shape=(len(dz), len(ph)), dtype=np.float32)
    # ARPACK starts from a random vector, so without a fixed seed the latent prior differs slightly
    # run to run and near-tied candidates swap places: the same KG and configuration gave DDXPlus
    # Top-3 75.43 and 75.40 on two runs with an identical hash seed. Seeding makes every reported
    # number reproducible.
    U_, S, Vt = svds(M, k=k, random_state=0)
    S = S[::-1]; U_ = U_[:, ::-1]; Vt = Vt[::-1]
    d_emb = (U_ * S).astype(np.float32)
    d_emb /= (np.linalg.norm(d_emb, axis=1, keepdims=True) + 1e-9)
    return di, pi, d_emb, Vt.T.astype(np.float32)


def build_disease_hierarchy(children):
    rev = defaultdict(set)
    for p, chs in children.items():
        for c in chs:
            rev[c].add(p)
    return children, rev


def closure(cui, adj, cap):
    seen, fr, d = set(), {cui}, 0
    while fr and d < cap:
        nx = set()
        for c in fr:
            for k in adj.get(c, ()):
                if k not in seen:
                    seen.add(k); nx.add(k)
        fr, d = nx, d + 1
    return seen


def closure_dist(cui, adj, cap):
    """{cui: hop-distance} within cap hops (0 for self)."""
    dist = {cui: 0}
    fr, d = {cui}, 0
    while fr and d < cap:
        nx = set()
        for c in fr:
            for k in adj.get(c, ()):
                if k not in dist:
                    dist[k] = d + 1; nx.add(k)
        fr, d = nx, d + 1
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg.json")
    ap.add_argument("--children", default="pilot/cache/umls_maps/disease_children.json")
    ap.add_argument("--parents", default="pilot/cache/umls_maps/phenotype_parents.json")
    ap.add_argument("--benchmarks", default="rarebench,symcat")
    ap.add_argument("--ppd", type=int, default=30)
    ap.add_argument("--hier", type=int, default=2, help="phenotype ancestor depth")
    ap.add_argument("--dz_cap", type=int, default=4, help="disease hierarchy depth")
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--dz_decay", type=float, default=1.0)
    ap.add_argument("--unified", type=int, default=1,
                    help="1 = unified evidence (§11): a disease concept in the patient's evidence can "
                         "match a candidate through the disease hierarchy. 0 = phenotype channel only, "
                         "for the component ablation; --dz_cap 0 is not an ablation because the "
                         "candidate's self-match survives it.")
    ap.add_argument("--normexp", type=float, default=0.5)
    ap.add_argument("--cov_mu", type=float, default=0.0,
                    help="coverage-aware length smoothing (0 = off). A disease's profile length is "
                         "used as if it meant 'this disease is narrow', but in this KG it mostly "
                         "means 'we read little about it' — median degree 7, and enriching 2.8% of "
                         "diseases moved @3 for the un-enriched patients too. This adds "
                         "mu*(1-coverage)*avg_len to the denominator, so a THINLY DOCUMENTED "
                         "disease is no longer rewarded for its own missing edges, while a "
                         "well-documented narrow disease keeps its short profile. Dirichlet-style "
                         "smoothing; coverage = src_chars/(src_chars+median) (no hard threshold).")
    ap.add_argument("--pivot_b", type=float, default=-1.0,
                    help="pivoted length normalization (Singhal et al. 1996): divide by "
                         "(1-b + b*len/avglen) instead of len**normexp. The power form rewards a "
                         "THIN profile, but in this KG a thin profile means missing IE, not a "
                         "specific disease -- SymCat's top-1 is repeatedly a 2-phenotype disease "
                         "matching one generic symptom. -1 = disabled (use normexp).")
    ap.add_argument("--sapbert", type=int, default=0, help="1 = enable SapBERT vocab alignment")
    ap.add_argument("--emb_model", default="biolord", help="encoder key: sapbert/coder/biolord/...")
    ap.add_argument("--tau", type=float, default=0.80, help="min cosine for a soft match")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--align_all", type=int, default=0,
                    help="1 = build soft alignment for ALL evidence, not just vocabulary-gap "
                         "evidence (lets a generic evidence CUI that only exact-matches a hub "
                         "phenotype also soft-match near-synonymous specific phenotypes)")
    ap.add_argument("--lsa", type=int, default=0, help="1 = fuse LSA latent score")
    ap.add_argument("--lsa_k", type=int, default=256)
    ap.add_argument("--lsa_w", type=float, default=0.3, help="LSA fusion weight (0..1)")
    ap.add_argument("--attr", type=int, default=0, help="1 = discriminative attribute channel")
    ap.add_argument("--attr_lam", type=float, default=1.0, help="attribute agreement bonus")
    ap.add_argument("--attr_mu", type=float, default=1.0, help="attribute contradiction penalty")
    ap.add_argument("--attr_w", type=float, default=0.0,
                    help="fixed weight for the attribute delta (0=use base-symptom idf). The "
                         "attribute's discriminative value is INDEPENDENT of the base symptom's "
                         "rarity (pain is a hub, idf~2, yet its character carries ~3 bits); "
                         "weighting the delta by the base idf suppresses exactly that signal. "
                         "Set to ~median idf (~9-10) so an attribute agreement counts like a "
                         "specific-concept match.")
    ap.add_argument("--attr_use", default="severity,onset,location,character")
    ap.add_argument("--dump_ranks", default="", help="if set, write per-patient [rank, has_attr] to <path>.<bench>.json")
    ap.add_argument("--trace", default="", help="if set, write per-patient rich case trace to <path>.<bench>.json")
    ap.add_argument("--rerank_dump", default="", help="if set, write per-patient {gold, top10 candidates, evidence names} to <path>.<bench>.json for LLM re-ranking")
    ap.add_argument("--attr_combos", type=int, default=0, help="1 = sweep all 2^4 subsets of {location,character,severity,onset} and print an ablation table (requires --attr 1)")
    ap.add_argument("--stop_cuis", default="", help="json list of non-phenotype CUIs to drop from patient evidence")
    ap.add_argument("--hpo_bridge", default="", help="hpo_bridge.json (cui2hpo+ancestors) to add an HPO-key match channel")
    ap.add_argument("--hpo_anc", type=int, default=0, help="HPO ancestor hops to add (0=exact synonym collapse only)")
    ap.add_argument("--hpo_w", type=float, default=1.0, help="scale for the HPO-bridge key weight")
    ap.add_argument("--bio_bridge", default="",
                    help="biochem_bridge.json — joins the compositional HPO phrasing of a metabolic "
                         "finding ('Elevated plasma citrulline') to the KG's contracted form "
                         "('citrullinemia') via a (direction, analyte) key")
    ap.add_argument("--bio_w", type=float, default=1.0, help="scale for the biochem-bridge key weight")
    ap.add_argument("--syn_bridge", default="",
                    help="synonym_bridge.json — joins KG phenotypes that UMLS declares synonymous "
                         "(MRREL SY/RQ) but keeps on separate CUIs, e.g. sore throat C0031350 vs "
                         "throat pain C0242429, which the HPO bridge cannot merge because HPO "
                         "gives them different terms")
    ap.add_argument("--syn_w", type=float, default=1.0, help="scale for the synonym-bridge key weight")
    ap.add_argument("--ev_bridge", default="",
                    help="evidence_family.json — one piece of history evidence is spread over many "
                         "CUIs (travel history / recent travel / travel), and a benchmark asks for "
                         "exactly one of them; this keys them together")
    ap.add_argument("--ev_w", type=float, default=1.0, help="scale for the evidence-family key weight")
    ap.add_argument("--neg_w", type=float, default=0.0,
                    help="penalty for a candidate's OWN salient findings that the patient did not "
                         "report. A benchmark whose records come from a fixed question panel has "
                         "asked about them, so silence is informative: it is what separates the "
                         "answer from the rest of a shortlist (@10 is 92 while @1 is 42 -- the "
                         "answer is already retrieved, it just is not first). Only findings that "
                         "SOME patient in the benchmark reports are counted, i.e. only questions "
                         "the panel actually asks; 0 disables.")
    ap.add_argument("--ppmi_bridge", default="",
                    help="pheno_ppmi.json — phenotype-phenotype association mined from the KG's "
                         "own disease profiles (positive PMI). The graph has been strictly "
                         "bipartite and single-hop, so a patient finding the candidate lacks "
                         "contributes nothing even when the candidate carries a phenotype that "
                         "co-occurs with it far above chance. This lets such a finding be PARTLY "
                         "explained, which is the one structural axis left: no benchmark file and "
                         "no inference-time LLM are involved.")
    ap.add_argument("--ppmi_w", type=float, default=0.0,
                    help="scale for propagated (2-hop) evidence; 0 = the previous single-hop KG")
    ap.add_argument("--verb_w", type=float, default=0.0,
                    help="two-stage length normalization: fraction of a profile's length treated "
                         "as VERBOSITY (we read more) rather than SCOPE (the disease manifests "
                         "more). Robertson's two hypotheses; 0 = single-stage.")
    ap.add_argument("--local_idf", type=float, default=0.0,
                    help="blend in candidate-set-local IDF (0=corpus IDF only, 1=local only). The "
                         "stored IDF is computed over all 35k KG diseases, but a ranking happens "
                         "inside one benchmark's candidate list: a finding shared by every "
                         "candidate is globally rare yet locally useless, and one that separates "
                         "two candidates is globally common yet locally decisive. Standard IR "
                         "practice is to weight by the collection actually being searched.")
    ap.add_argument("--idf_pow", type=float, default=1.0, help="raise profile IDF weights to this power (>1 emphasises specific matches)")
    ap.add_argument("--kge_emb", default="", help="path to kg_emb.pt: use learned KG embeddings for the latent channel instead of SVD-LSA")
    ap.add_argument("--fuse", default="sum", choices=["sum", "rrf", "max", "mult", "summax", "minrank"],
                    help="channel fusion: 'sum' = weighted-sum + LSA late-fusion (default, "
                         "unchanged); 'rrf' = Reciprocal Rank Fusion of per-channel rankings")
    ap.add_argument("--rrf_k", type=float, default=60.0, help="RRF constant k (score=w/(k+rank))")
    ap.add_argument("--rrf_channels", default="phen,dz,lsa",
                    help="channels to fuse under --fuse rrf (subset of phen,dz,lsa)")
    ap.add_argument("--rrf_w", default="1,1,1",
                    help="per-channel RRF weights for phen,dz,lsa (comma list, aligns with the "
                         "3 canonical channels regardless of --rrf_channels selection)")
    a = ap.parse_args()
    # parse RRF config once
    _rrf_names = ["phen", "dz", "lsa"]
    _rw = [float(x) for x in a.rrf_w.split(",")] if a.rrf_w else [1, 1, 1]
    _rw += [1.0] * (3 - len(_rw))
    RRF_W = dict(zip(_rrf_names, _rw))
    RRF_CH = [c.strip() for c in a.rrf_channels.split(",") if c.strip() in _rrf_names]

    kg, prof, idf, dz_names = U.load_kg(a.kg)
    # selectivity weights for attribute agreements (KG-derived, benchmark-blind)
    import attr_score as _A
    _A.build_attr_df(kg["edges"])
    children = json.load(open(a.children))
    child_adj, parent_adj = build_disease_hierarchy(children)
    resolver = U.Resolver(prof, dz_names, children)
    U.PARENTS.update(json.load(open(a.parents)))
    N_DZ = len(kg["diseases"])
    KGDZ = set(kg["diseases"])
    NAMES = U.cui_names(kg) if a.sapbert else {}
    # documentation depth per disease node (written by v157_build_kg --preds). Diseases built
    # before that field existed fall back to "fully documented", i.e. no smoothing, so the flag
    # is a no-op on an older KG rather than silently mis-scoring it.
    SRC_CHARS = {d: info.get("n_src_chars") for d, info in kg["diseases"].items()}
    _sc = sorted(v for v in SRC_CHARS.values() if v)
    SRC_MED = _sc[len(_sc) // 2] if _sc else 0
    if a.cov_mu > 0:
        print(f"coverage smoothing on: {len(_sc)}/{len(SRC_CHARS)} diseases have source-size "
              f"provenance, median {SRC_MED} chars", flush=True)
    STOP = set(json.load(open(a.stop_cuis))) if a.stop_cuis else set()
    # HPO bridge: map each phenotype CUI -> its HPO term (+ is-a ancestors) so that near-synonym
    # concepts on different CUIs ("poor weight gain" vs "failure to thrive") match via a shared
    # HPO key. Fixes the vocabulary-granularity gap without the noise of embedding soft-matching.
    BRIDGE = json.load(open(a.hpo_bridge)) if a.hpo_bridge else None
    C2H = BRIDGE["cui2hpo"] if BRIDGE else {}
    HANC = BRIDGE["anc"] if BRIDGE else {}
    # Biochemical bridge: the rare benchmarks write metabolic findings compositionally ("Elevated
    # plasma citrulline") while the KG carries the contracted clinical form ("citrullinemia"), so
    # the pathognomonic evidence of a metabolic disease never joins. Keying both sides by
    # (direction, analyte) lifts RAMEDIS evidence joinability 67.9% -> 77.2%. Benchmark-blind: the
    # key is derived from the concept NAME only (CLAUDE.md §3).
    C2S = json.load(open(a.syn_bridge))["cui2key"] if a.syn_bridge else {}
    C2E = json.load(open(a.ev_bridge))["cui2key"] if a.ev_bridge else {}
    PPMI = json.load(open(a.ppmi_bridge)) if (a.ppmi_bridge and a.ppmi_w > 0) else {}
    PPMI_MAX = max((w for lst in PPMI.values() for _, w in lst), default=1.0) if PPMI else 1.0
    C2B = json.load(open(a.bio_bridge))["cui2key"] if a.bio_bridge else {}
    if C2B:
        import biochem_bridge
        for c, info in kg["phenotypes"].items():   # KG names are IE prose, not UMLS preferred names
            ks = set(C2B.get(c, ())) | set(biochem_bridge.keys_for(info.get("name", "")))
            if ks:
                C2B[c] = sorted(ks)

    def hpo_expand(d, decay=0.6):
        """augment a {cui: weight} map with HP:-prefixed keys (the CUI's HPO term + up to
        a.hpo_anc is-a ancestors). a.hpo_anc=0 = exact synonym collapse only (no generalization)."""
        if not C2H and not C2B and not C2S and not C2E:
            return d
        out = dict(d)
        for c, w in list(d.items()):
            ek = C2E.get(c)
            if ek:
                we = w * a.ev_w
                if we > out.get(ek, 0):
                    out[ek] = we
            sk = C2S.get(c)
            if sk:
                ws = w * a.syn_w
                if ws > out.get(sk, 0):
                    out[sk] = ws
            for bk in C2B.get(c, ()):
                wb = w * a.bio_w
                if wb > out.get(bk, 0):
                    out[bk] = wb
            h = C2H.get(c)
            if not h:
                continue
            wb = w * a.hpo_w                    # scale the bridge (weak bridge = rescue only)
            k = "H:" + h
            if wb > out.get(k, 0):
                out[k] = wb
            for j, anc in enumerate(HANC.get(h, ())[:a.hpo_anc]):
                ak = "H:" + anc
                aw = wb * (decay ** (j + 1))
                if aw > out.get(ak, 0):
                    out[ak] = aw
        return out

    ALIGN = {}                                            # {gap_evidence_cui: [(kg_cui, sim)]}

    # threshold-free evidence-specificity gate for soft alignment (used by --align_all).
    # Generic (low-idf / hub) evidence -> weight ~1 (soft match fully active); specific
    # (high-idf) evidence -> weight ~0 (soft match suppressed so sharp rare-disease evidence
    # is not diluted). Resolves the common-vs-rare tension of blanket soft matching without a
    # magic threshold: the weight is the evidence's own IC rescaled to the observed range.
    _idf_vals = list(idf.values()) or [1.0]
    _idf_lo, _idf_hi = min(_idf_vals), max(_idf_vals)
    def genericness(e):
        v = idf.get(e)
        if v is None or _idf_hi <= _idf_lo:
            return 1.0
        return (_idf_hi - v) / (_idf_hi - _idf_lo)

    EDGE_ATTR = defaultdict(dict)
    if a.attr:
        import attr_score
        for e in kg["edges"]:
            EDGE_ATTR[e["d"]][e["p"]] = e.get("attrs", {})

    LSA = None
    if a.lsa:
        import numpy as np
        if a.kge_emb:   # use learned KG embeddings (KG+ML) instead of the SVD/LSA factorization
            import torch
            e = torch.load(a.kge_emb)
            LSA = {"di": e["di"], "pi": e["pi"],
                   "demb": np.asarray(e["disease_emb"], dtype=np.float32),
                   "pemb": np.asarray(e["phenotype_emb"], dtype=np.float32), "np": np}
            print(f"using KGE embeddings from {a.kge_emb} (dim={LSA['demb'].shape[1]})", flush=True)
        else:
            print(f"building LSA (k={a.lsa_k}) ...", flush=True)
            _di, _pi, _demb, _pemb = build_lsa(kg, prof, idf, a.lsa_k)
            LSA = {"di": _di, "pi": _pi, "demb": _demb, "pemb": _pemb, "np": np}

    def build_align(cand, pats):
        ALIGN.clear()
        if not a.sapbert:
            return
        import sapbert_align as SB
        prof_cuis = set()
        for nm, cu in cand.values():
            prof_cuis |= set(U.merged_profile(resolver.cluster(nm, cu), prof))
        ev_cuis = set()
        for _, ev, _ in pats:
            ev_cuis |= ({f["cui"] for f in ev} if (ev and isinstance(next(iter(ev)), dict))
                        else set(ev))
        gap = sorted(ev_cuis if a.align_all else ev_cuis - prof_cuis)  # gap-only, or all evidence
        if not gap or not prof_cuis:
            return
        pl = sorted(prof_cuis)
        al = SB.build_alignment(gap, [NAMES.get(c, c) for c in gap],
                                pl, [NAMES.get(c, c) for c in pl], topk=a.topk, tau=a.tau)
        for e, lst in al.items():
            keep = [(c, s) for c, s in lst if s >= a.tau and c != e]  # drop self (double-count)
            if keep:
                ALIGN[e] = keep

    # Resnik/Phrank ancestor-IC closure: {cui} ∪ phenotype ancestors that are KG phenotypes,
    # each keyed by its information content (idf). MICA IC = max idf over shared ancestors.
    _anc_ic = {}
    def anc_ic(cui):
        if cui in _anc_ic:
            return _anc_ic[cui]
        out = {}
        if cui in idf:
            out[cui] = idf[cui]
        for anc in closure(cui, U.PARENTS, a.hier):
            if anc in idf and idf[anc] > out.get(anc, 0):
                out[anc] = idf[anc]
        _anc_ic[cui] = out
        return out

    # information content of a disease CUI: rarer (fewer descendants) => higher IC
    ic_cache = {}
    def ic_dz(cui):
        if cui not in ic_cache:
            nd = len(closure(cui, child_adj, 6) & KGDZ)
            ic_cache[cui] = math.log(N_DZ / (1 + nd))
        return ic_cache[cui]

    def candidate_signature(name, cui):
        cl = resolver.cluster(name, cui)                 # subtypes (for phenotype profile)
        raw = U.merged_profile(cl, prof)
        raw_n = len(raw)                                 # intrinsic breadth (pre-expansion)
        pf = U.expand_profile(raw, a.hier, a.decay) if raw else {}
        pf = hpo_expand(pf)   # add HPO-key channel (near-synonym bridge)
        if a.idf_pow != 1.0:  # emphasise specific (high-IDF) profile matches for top-3 discrimination
            pf = {c: (w ** a.idf_pow if w > 0 else w) for c, w in pf.items()}
        # unified disease signature: ancestors ∪ descendants ∪ self, IC-weighted and
        # proximity-decayed (a closer relative is a sharper match -> breaks Top-1 ties).
        dsig = {}
        if not a.unified:
            return pf, dsig, raw_n, set(raw)
        base = ({cui} if cui else set()) | cl
        for b in base:
            dd = closure_dist(b, parent_adj, a.dz_cap)   # ancestors w/ distance
            for r, dist in closure_dist(b, child_adj, a.dz_cap).items():  # descendants
                if dist < dd.get(r, 99):
                    dd[r] = dist
            for r, dist in dd.items():
                w = ic_dz(r) * (a.dz_decay ** dist)
                if w > dsig.get(r, 0):
                    dsig[r] = w
        return pf, dsig, raw_n, set(raw)

    def evaluate(name, cand, patients, coref=None):
        """coref: list of code-groups the BENCHMARK declares co-referent (same disease at
        different codes, e.g. OMIM+ORPHA of one RareBench disease). We merge ONLY these --
        NOT UMLS-hierarchy neighbours (that over-merges clinically distinct diseases)."""
        keys = sorted(cand)
        sig = {}
        for k in keys:
            nm, cu = cand[k]
            sig[k] = candidate_signature(nm, cu)   # (panc, dsig, n_pheno)
        parent = {k: k for k in keys}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for grp in (coref or []):
            g = [k for k in grp if k in parent]
            for k in g[1:]:
                parent[find(k)] = find(g[0])
        group = {k: find(k) for k in keys}
        rankable = [k for k in keys if sig[k][0] or sig[k][1]]
        # the question panel: every evidence concept some patient in this benchmark reports.
        # Silence about one of these is informative; silence about anything else is not.
        # Report FREQUENCY of each evidence concept in this benchmark. The absence penalty is
        # weighted by it, which makes the penalty self-gating without a per-benchmark switch: in a
        # questionnaire benchmark a common item (cough, reported by 30% of patients) is asked of
        # everyone, so a candidate whose profile predicts it loses real mass when it is absent; in
        # a curated-annotation benchmark every HPO term is rare across patients, the frequencies
        # are ~0, and the penalty vanishes -- absence was never informative there.
        FREQ = defaultdict(float)
        if a.neg_w:
            for _p in patients:
                for _e in _p[1]:
                    FREQ[_e["cui"] if isinstance(_e, dict) else _e] += 1.0 / len(patients)
        # length-normalization pivots: the AVERAGE candidate signature size in this benchmark's own
        # candidate set (a property of the KG+candidates, not of the labels), used by _lnorm below.
        avg_pf = (sum(len(sig[k][0]) for k in rankable) / len(rankable)) if rankable else 1.0
        avg_dsig = (sum(len(sig[k][1]) for k in rankable) / len(rankable)) if rankable else 1.0

        # how completely each CANDIDATE is documented: saturating in the source text read about
        # its whole disease cluster, so it is a smooth [0,1) weight with no magic threshold.
        covw = {}
        if a.cov_mu > 0 and SRC_MED:
            for k in keys:
                nm, cu = cand[k]
                chars = sum(SRC_CHARS.get(c) or 0
                            for c in (resolver.cluster(nm, cu) | ({cu} if cu else set())))
                covw[k] = chars / (chars + SRC_MED)

        # candidate-set-local IDF: how many of THESE candidates carry each phenotype
        local_idf = {}
        if a.local_idf > 0 and rankable:
            ldf = defaultdict(int)
            for k in rankable:
                for c in sig[k][0]:
                    ldf[c] += 1
            NL = len(rankable)
            local_idf = {c: math.log((NL + 1) / (n + 0.5)) for c, n in ldf.items()}

        def _lnorm(n, avg, c=1.0):
            """Candidate-length normalizer, two-stage (Robertson's verbosity/scope split).

            A profile is long for two different reasons, and IR has names for them: VERBOSITY -- we
            simply read more text about this disease -- and SCOPE -- the disease genuinely
            manifests in more ways. One exponent penalizes both identically, which is measurably
            wrong here: among patients whose gold sits at rank 4-10 the gold covers MORE of their
            evidence than the winners (62.0% vs 57.1%) and loses only for being larger (464 vs 274
            concepts). Two-stage normalization (Na, TOIS 2015) prenormalizes by verbosity, then
            applies the length penalty to what remains, so breadth is judged after reading depth is
            divided out.

            `c` is the candidate's documentation coverage; `verb_w` sets how much of the length is
            attributed to verbosity (0 = single-stage, the previous behaviour)."""
            if a.verb_w > 0:
                # expected profile size at this reading depth; >1 means we read a lot about it
                v = 1.0 + a.verb_w * (c - 0.5) * 2.0
                n = n / max(v, 0.2)
            n = n + a.cov_mu * (1.0 - c) * avg
            if a.pivot_b < 0:
                return max(n ** a.normexp, 1.0)
            return max(1.0 - a.pivot_b + a.pivot_b * (n / max(avg, 1e-9)), 1e-6)
        # LSA cluster embeddings (mean of member disease latent vectors)
        lsa_emb = {}
        if LSA:
            np = LSA["np"]
            for k in keys:
                nm, cu = cand[k]
                idxs = [LSA["di"][c] for c in (resolver.cluster(nm, cu) | ({cu} if cu else set()))
                        if c in LSA["di"]]
                if idxs:
                    v = LSA["demb"][idxs].mean(0)
                    lsa_emb[k] = v / (np.linalg.norm(v) + 1e-9)
                else:
                    lsa_emb[k] = None
        # merged KG edge attributes per candidate (for the discriminative attribute channel)
        cattr = {}
        if a.attr:
            for k in keys:
                nm, cu = cand[k]
                cattr[k] = attr_score.merge_edge_attrs(
                    resolver.cluster(nm, cu) | ({cu} if cu else set()), EDGE_ATTR)
        n = len(patients)
        c1 = c3 = c5 = c10 = rr = 0
        cov = 0
        pranks = []   # per-patient (gold rank, has_attribute) for significance dumps
        ptrace = []   # per-patient rich trace (gold, top5, findings, gold_loc_match) for case study
        rrdump = []   # per-patient {gold, top10, evidence names} for LLM re-ranking
        for tkeys, ev_raw, ds in patients:
            # extended attribute format: ev_raw may be a list of finding dicts. Always derive the
            # CUI set from it; the discriminative attribute channel only fires when --attr is on.
            is_ext = bool(ev_raw) and isinstance(next(iter(ev_raw)), dict)
            findings = ev_raw if (a.attr and is_ext) else None
            ev = {f["cui"] for f in ev_raw} if is_ext else set(ev_raw)
            if STOP:
                ev = ev - STOP     # drop non-phenotype evidence (inheritance mode / mortality outcome)
            tset = set(tkeys)
            if not any(sig.get(t, (None, None))[0] or sig.get(t, (None, None))[1] for t in tset):
                rr += 1.0 / len(keys)
                continue
            cov += 1
            q = U.expand_query(ev, a.hier, a.decay)
            q = hpo_expand(q)   # add HPO-key channel (near-synonym bridge)
            if local_idf:
                gmax = max(idf.values()) or 1.0
                lmax = max(local_idf.values()) or 1.0
                for c in list(q):
                    g = idf.get(c, 0.0) / gmax
                    l = local_idf.get(c, math.log((len(rankable) + 1) / 0.5)) / lmax
                    mix = (1 - a.local_idf) * g + a.local_idf * l
                    q[c] *= mix / max(g, 1e-6) if g > 0 else mix
            # SapBERT vocabulary-alignment: for evidence with NO exact KG-phenotype CUI,
            # its nearest KG-phenotype CUIs (by biomedical-encoder cosine) as soft matches.
            soft = []  # list of (kg_cui, sim) proposed for this patient's exact-missed evidence
            if ALIGN:
                for e in ev:
                    ge = genericness(e) if a.align_all else 1.0  # specificity gate (align_all only)
                    for kgc, sim in ALIGN.get(e, ()):
                        soft.append((kgc, sim * ge))
            scores = {}
            ch_dm = {}   # disease-hierarchy channel (per-candidate, own-cardinality normalized)
            ch_pm = {}   # phenotype (+HPO-bridge +attr) channel
            for k in rankable:
                pf, dsig, raw_n, raw = sig[k]
                # two evidence channels normalized SEPARATELY by their own cardinality:
                #   disease-hierarchy prior (§11 unified evidence) and phenotype similarity.
                dm = pm = 0.0
                for c, qw in q.items():
                    d = dsig.get(c, 0.0)
                    p = pf.get(c, 0.0)
                    dm += qw * d
                    if p > d:
                        pm += qw * p
                for kgc, sim in soft:
                    if kgc in raw:
                        pm += sim * idf.get(kgc, 0.0)
                if PPMI:
                    # 2-hop: a reported finding the candidate lacks is partly explained by a
                    # phenotype it HAS that co-occurs with it far above chance. Credited at most
                    # once per finding and scaled by normalized PMI, so propagation can supplement
                    # direct evidence but never outweigh it.
                    for c, qw in q.items():
                        if c in raw or c in dsig:
                            continue
                        best = 0.0
                        for nb, w in PPMI.get(c, ()):
                            if nb in raw and w > best:
                                best = w
                        if best:
                            pm += a.ppmi_w * qw * (best / PPMI_MAX)
                # discriminative attribute channel: agree -> bonus, contradict -> penalty
                am = 0.0
                if findings:
                    pa = cattr[k]
                    for f in findings:
                        c = f["cui"]
                        if c in raw and c in pa:
                            aw = a.attr_w if a.attr_w > 0 else idf.get(c, 0.0)
                            am += attr_score.attr_delta(f, pa[c], aw,
                                                        a.attr_lam, a.attr_mu, tuple(a.attr_use.split(",")))
                if a.neg_w and FREQ:
                    # expected reported-but-absent mass of the candidate's own findings
                    own = [c for c in raw if c in FREQ]
                    if own:
                        pm -= a.neg_w * sum(FREQ[c] * idf.get(c, 0.0)
                                            for c in own if c not in ev) / len(own)
                ck = covw.get(k, 1.0)
                dm_n = dm / _lnorm(len(dsig), avg_dsig, ck)
                pm_n = (pm + am) / _lnorm(len(pf), avg_pf, ck)
                ch_dm[k] = dm_n
                ch_pm[k] = pm_n
                scores[k] = dm_n + pm_n
            # LSA cosine channel (hoisted so both fusers can use it)
            lsc = None
            if LSA:
                np = LSA["np"]
                qv = None
                vs = [idf.get(p, 1.0) * LSA["pemb"][LSA["pi"][p]] for p in ev if p in LSA["pi"]]
                if vs:
                    qv = np.sum(vs, 0); qv = qv / (np.linalg.norm(qv) + 1e-9)
                if qv is not None:
                    lc = {k: float(lsa_emb[k] @ qv) for k in rankable if lsa_emb.get(k) is not None}
                    if lc:
                        lsc = lc
            if a.fuse == "rrf":
                # Reciprocal Rank Fusion: rank candidates within EACH channel separately, then
                # score each candidate by sum_c w_c / (k + rank_c). Rewards candidates that rank
                # decently in MULTIPLE channels (vs weighted-sum, which a single dominant channel
                # can bury). Collapse to group level (max member per channel) BEFORE ranking so
                # the fusion is consistent with the group-level final ranking.
                chan = {"phen": ch_pm, "dz": ch_dm}
                if lsc:
                    chan["lsa"] = lsc
                gscore = defaultdict(float)
                gset = {group[k] for k in rankable}
                for cname in RRF_CH:
                    cs = chan.get(cname)
                    if not cs:
                        continue
                    gm = {}
                    for k in rankable:
                        g = group[k]; v = cs.get(k, 0.0)
                        if v > gm.get(g, -1e18):
                            gm[g] = v
                    order = sorted(gm, key=lambda g: -gm[g])
                    w = RRF_W.get(cname, 1.0)
                    for ri, g in enumerate(order, 1):
                        gscore[g] += w / (a.rrf_k + ri)
                for g in gset:               # ensure every group is rankable (score 0 if unseen)
                    gscore.setdefault(g, 0.0)
                gscore = dict(gscore)
                ranked = sorted(gscore, key=lambda g: (-gscore[g], str(g)))
            else:
                # weighted-sum + LSA late-fusion: min-max normalize both channels, blend by lsa_w
                if lsc:
                    raw_scores = dict(scores)
                    smin, smax = min(scores.values()), max(scores.values())
                    lmin, lmax = min(lsc.values()), max(lsc.values())
                    sr = (smax - smin) or 1.0; lr = (lmax - lmin) or 1.0
                    for k in rankable:
                        sn = (scores[k] - smin) / sr
                        ln = ((lsc[k] - lmin) / lr) if k in lsc else 0.0
                        if a.fuse == "summax":
                            # CombMAX over the two fusion shapes. SUM (prior can rescue a weak
                            # evidence score: carries big-profile diseases like Bronchitis, but
                            # buries multisystem SLE) and MULT (bounded prior: rescues SLE, but
                            # cannot carry Bronchitis). Each shape's failure set is the other's
                            # success set, so a candidate keeps the better of the two -- the
                            # classic CombMAX rationale, applied to fusions instead of channels.
                            _sum = (1 - min(a.lsa_w, 0.99)) * sn + min(a.lsa_w, 0.99) * ln                                 if a.lsa_w < 1 else 0.6 * sn + 0.4 * ln
                            _mult = sn * (1 + a.lsa_w * ln) / (1 + a.lsa_w)
                            scores[k] = max(_sum, _mult)
                        elif a.fuse == "mult":
                            # Multiplicative prior with bounded demotion. Read as posterior ∝
                            # likelihood × prior: the evidence channel is the likelihood, LSA's
                            # typicality is the prior, and a prior may only rescale -- the worst
                            # demotion is a factor 1/(1+w), so it can reorder candidates whose
                            # evidence is close but can never bury one the evidence ranks clearly
                            # ahead. Weighted-SUM fusion had no such bound, which is how the LSA
                            # channel took SLE from rank 6 (evidence-only) to rank 24 and 0/44 @10.
                            scores[k] = sn * (1 + a.lsa_w * ln)
                        elif a.fuse == "max":
                            # CombMAX (Fox & Shaw): a candidate keeps the BEST of its evidence
                            # score and the blended score. The blend adds LSA's typicality prior
                            # where it helps, but can no longer bury a candidate the evidence
                            # channel ranks highly -- which is exactly what it did to multisystem
                            # diseases (SLE 0/44 at @10 with the blend, 32/44 without it).
                            scores[k] = max(sn, (1 - a.lsa_w) * sn + a.lsa_w * ln)
                        else:
                            scores[k] = (1 - a.lsa_w) * sn + a.lsa_w * ln
                if a.fuse in ("summax", "minrank") and lsc:
                    # CombMAX over two fusion SHAPES, each min-max normalized across candidates
                    # before the max. SUM lets the typicality prior rescue a weak evidence score
                    # (carries big-profile Bronchitis, buries multisystem SLE); MULT bounds the
                    # prior (rescues SLE, cannot carry Bronchitis). Their failure sets are
                    # disjoint, so each candidate keeps its better shape.
                    _sn = {}; _ln = {}
                    for k2 in rankable:
                        _sn[k2] = (raw_scores[k2] - smin) / sr
                        _ln[k2] = ((lsc[k2] - lmin) / lr) if k2 in lsc else 0.0
                    f1 = {k2: 0.6 * _sn[k2] + 0.4 * _ln[k2] for k2 in rankable}
                    f2 = {k2: _sn[k2] * (1 + a.lsa_w * _ln[k2]) for k2 in rankable}
                    for fd in (f1, f2):
                        lo, hi = min(fd.values()), max(fd.values())
                        rng = (hi - lo) or 1.0
                        for k2 in fd:
                            fd[k2] = (fd[k2] - lo) / rng
                    if a.fuse == "summax":
                        for k2 in rankable:
                            scores[k2] = max(f1[k2], f2[k2])
                    else:  # minrank
                        # order by the better of the two shape-ranks; a candidate either shape
                        # places at rank r lands within the top ~2r of the merged order, so
                        # neither shape's burial can push it far down.
                        r1 = {k2: i for i, k2 in enumerate(sorted(rankable, key=lambda x: (-f1[x], str(x))))}
                        r2 = {k2: i for i, k2 in enumerate(sorted(rankable, key=lambda x: (-f2[x], str(x))))}
                        for k2 in rankable:
                            best = min(r1[k2], r2[k2])
                            # keep scores POSITIVE: the group-collapse below initializes with -1,
                            # so a negative score would silently drop its group from the ranking
                            # (measured: the ranked list shrank to ~4 groups and "gold@10" read
                            # 100%, above the true two-run union of 95.4% -- an impossible number
                            # that exposed the bug)
                            # Tiebreak on the min-rank plateau uses BOTH shapes' evidence (their
                            # sum), not the max alone: integer min-ranks tie often, and 38
                            # Bronchitis misses sat at exactly rank 11 -- losing the tiebreak, not
                            # the ranking.
                            scores[k2] = (len(rankable) - best) + 0.5 * max(f1[k2], f2[k2])
                # collapse to group scores (max member), rank groups
                gscore = {}
                for k in rankable:
                    g = group[k]
                    if scores[k] > gscore.get(g, -1):
                        gscore[g] = scores[k]
                ranked = sorted(gscore, key=lambda g: (-gscore[g], str(g)))
            tgroups = {group[t] for t in tset if t in group and group[t] in gscore}
            ranks = [ranked.index(g) + 1 for g in tgroups]
            rank = min(ranks) if ranks else len(ranked) + 1
            if rank == 1: c1 += 1
            if rank <= 3: c3 += 1
            if rank <= 5: c5 += 1
            if rank <= 10: c10 += 1
            rr += 1.0 / rank
            if a.dump_ranks:
                ha = bool(findings) and any(f.get("location") or f.get("character")
                                            or f.get("severity") or f.get("onset")
                                            for f in findings)
                pranks.append((rank, ha))
            if a.trace:
                nm = lambda c: NAMES.get(c, c)
                rec = {"i": len(ptrace), "rank": rank,
                       "gold": [cand[t][0] for t in tset if t in cand],
                       "top5": [[cand[g][0] if g in cand else nm(g), round(gscore[g], 3)]
                                for g in ranked[:5]]}
                if findings:
                    rec["findings"] = [{"name": nm(f["cui"]), "loc": sorted(f.get("location_txt", [])),
                                        "char": list(f.get("character", []))[:3],
                                        "sev": f.get("severity"), "onset": f.get("onset")}
                                       for f in findings if f.get("location") or f.get("cui")][:12]
                    # for the gold disease: which findings' location matches the KG edge location
                    gk = next((t for t in tset if t in cattr), None)
                    if gk:
                        gm = []
                        for f in findings:
                            c = f["cui"]
                            pa = cattr[gk].get(c)
                            if pa and (pa.get("location_cui") or f.get("location")):
                                gm.append({"finding": nm(c),
                                           "patient_loc": [nm(x) for x in f.get("location", ())],
                                           "kg_loc": pa.get("location"),
                                           "match": bool(pa.get("location_cui") and pa["location_cui"] in f.get("location", ()))})
                        rec["gold_loc_match"] = gm
                ptrace.append(rec)
            if a.rerank_dump:
                nm = lambda c: NAMES.get(c, c)
                def support(g):   # patient findings that match candidate g's KG profile (KG evidence FOR g)
                    raw = sig[g][3] if g in sig else set()
                    return sorted({nm(c) for c in ev if c in raw})[:15]
                rrdump.append({
                    "gold_cuis": sorted(group[t] for t in tset if t in group),
                    "gold_names": sorted({cand[t][0] for t in tset if t in cand}),
                    "gold_rank": rank,
                    "top10": [[g, cand[g][0] if g in cand else nm(g), round(gscore[g], 3), support(g)]
                              for g in ranked[:10]],
                    "evidence": sorted({nm(c) for c in ev})[:60]})
        if a.dump_ranks:
            json.dump(pranks, open(f"{a.dump_ranks}.{name.replace(':','_')}.json", "w"))
        if a.trace:
            json.dump(ptrace, open(f"{a.trace}.{name.replace(':','_')}.json", "w"))
        if a.rerank_dump:
            json.dump(rrdump, open(f"{a.rerank_dump}.{name.replace(':','_')}.json", "w"))
        print(f"=== {name}  ({n} pat, {len(rankable)} rankable) cov={100*cov/n:.1f}%  "
              f"@1={100*c1/n:.2f} @3={100*c3/n:.2f} @5={100*c5/n:.2f} @10={100*c10/n:.2f} "
              f"MRR={rr/n:.4f}")
        return {"n": n, "at1": 100*c1/n, "at3": 100*c3/n, "at5": 100*c5/n,
                "at10": 100*c10/n, "mrr": rr/n, "cov": 100*cov/n}

    def coref_from_targets(patients):
        """benchmark-declared co-reference: codes co-occurring in one patient's target list."""
        par = {}
        def find(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        for tkeys, _, _ in patients:
            for t in tkeys[1:]:
                par.setdefault(tkeys[0], tkeys[0])
                par[find(t)] = find(tkeys[0])
        groups = defaultdict(list)
        for k in par:
            groups[find(k)].append(k)
        return list(groups.values())

    if a.sapbert:
        import sapbert_align as SB
        print(f"SapBERT alignment ON: model={SB.set_model(a.emb_model)} tau={a.tau} topk={a.topk}",
              flush=True)

    # --attr routes each benchmark through its attribute-aware loader (per-benchmark
    # preprocessing), which emits findings-with-attributes instead of a bare CUI set.
    if a.attr:
        import attr_loader_ddxplus, attr_loader_rarebench, attr_loader_symcat
        ld_ddx = lambda: attr_loader_ddxplus.load(3000)
        ld_sym = lambda: attr_loader_symcat.load(a.ppd)
        ld_rb = lambda dsn: attr_loader_rarebench.load([dsn])
    else:
        ld_ddx = lambda: U.load_ddxplus(3000)
        ld_sym = lambda: U.load_symcat(a.ppd)
        ld_rb = lambda dsn: U.load_rarebench([dsn])

    import itertools
    def run_combos(name, cand, pats, coref):
        """Sweep all 2^4 subsets of the usable attributes (Prof. Bang's one-factor-removed +
        combination request). Reuses the loaded KG/embeddings/alignment; only attr_use varies."""
        ATTRS = ["location", "character", "severity", "onset"]
        rows = []
        for r in range(len(ATTRS) + 1):
            for combo in itertools.combinations(ATTRS, r):
                a.attr_use = ",".join(combo)
                rows.append((combo, evaluate(name, cand, pats, coref)))
        base = rows[0][1]
        print(f"\n{'='*82}\n  ATTRIBUTE COMBINATION ABLATION — {name} (kg={a.kg})")
        print(f"  {'combination':40} {'@1':>6} {'@3':>6} {'@10':>6} {'MRR':>8}  Δ@1")
        for combo, res in sorted(rows, key=lambda x: (len(x[0]), x[0])):
            tag = "base (no attributes)" if not combo else "+".join(combo)
            print(f"  {tag:40} {res['at1']:6.2f} {res['at3']:6.2f} {res['at10']:6.2f} "
                  f"{res['mrr']:8.4f}  {res['at1']-base['at1']:+.2f}")
        # leave-one-out view: full minus each single attribute
        full = next(res for combo, res in rows if len(combo) == 4)
        print(f"\n  -- one-factor-removed (Δ vs +all, negative = that attribute was helping) --")
        for combo, res in rows:
            if len(combo) == 3:
                removed = (set(ATTRS) - set(combo)).pop()
                print(f"  all − {removed:12}  @1={res['at1']:6.2f}  (Δ vs all {res['at1']-full['at1']:+.2f})")
        return {"combos": [{"attrs": list(c), **r} for c, r in rows]}

    summary = {}
    for b in a.benchmarks.split(","):
        if b == "rarebench":
            for dsn in ["RAMEDIS", "LIRICAL", "MME", "HMS"]:
                cand, pats = ld_rb(dsn)
                if pats:
                    build_align(cand, pats)
                    summary[f"RareBench:{dsn}"] = evaluate(f"RareBench:{dsn}", cand, pats,
                                                           coref_from_targets(pats))
        elif b == "symcat":
            cand, pats = ld_sym()
            build_align(cand, pats)
            summary["SymCat"] = evaluate("SymCat", cand, pats)   # no co-reference
        elif b == "ddxplus":
            cand, pats = ld_ddx()
            build_align(cand, pats)
            summary["DDXPlus"] = (run_combos("DDXPlus", cand, pats, None) if a.attr_combos
                                  else evaluate("DDXPlus", cand, pats))   # no co-reference
        elif b == "nlice":
            import attr_loader_nlice
            cand, pats = attr_loader_nlice.load(150, kg=a.kg)
            build_align(cand, pats)
            summary["NLICE"] = (run_combos("NLICE", cand, pats, None) if a.attr_combos
                                else evaluate("NLICE", cand, pats))   # findings carry attributes

    print("\n" + "=" * 66)
    print(f"  {'benchmark':18} {'cov%':>6} {'@1':>6} {'@3':>6} {'@5':>6} {'@10':>6}")
    for k, v in summary.items():
        if "cov" not in v:   # combo-sweep result carries no single cov/at1
            continue
        print(f"  {k:18} {v['cov']:6.1f} {v['at1']:6.2f} {v['at3']:6.2f} {v['at5']:6.2f} {v['at10']:6.2f}")
    json.dump(summary, open("pilot/cache/prod/unified2_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
