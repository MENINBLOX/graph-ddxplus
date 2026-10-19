"""S5 PROVENANCE MATERIALIZATION — attach per-edge source provenance to the built KG.

Thesis traceability axis: every disease->phenotype edge should be traceable to the source
document(s) and finding surface text it was extracted from. The build (v157) drops provenance;
this post-processor re-attaches it WITHOUT touching any scoring field.

Approach (parity-safe, additive):
  * Read the EXISTING built KG (kg_merged2.json) and add ONLY a new `prov` key to each edge.
    All scoring fields (meta / diseases / phenotypes / edge d,p,attrs) are copied verbatim, so the
    evaluator sees an identical graph.
  * The finding -> phenotype-CUI mapping is already materialized in kg_records_merged2.json
    (each phenotype rec carries name + name_cui), so NO UMLS re-linking is needed.
  * Per-finding corpus attribution was lost in the merge (findings were unioned & deduped by
    normalized name). We reconstruct it by re-matching each (disease_cui, finding_name) against the
    three per-corpus IE files:
        ie_pred_full.json = PubMed base, ie_pred_gr.json = GeneReviews, ie_pred_sp.json = StatPearls.
    A finding's corpus = whichever per-corpus file(s) contain that (cui, norm(name)) pair.
  * Document-id granularity:
        - GeneReviews : exactly one NBK per disease (source doc), recovered via crawl manifest title.
        - StatPearls  : exactly one NBK per disease, recovered via source HTML filename slug.
        - PubMed base : the per-disease source blob is a concatenation of ranked passages from
          multiple documents (mostly PubMed abstracts + a few medlineplus/statpearls/wikipedia
          sections). A specific finding CANNOT be traced to a single passage, so PubMed provenance is
          honestly reported at DISEASE-DOC-SET granularity (the full set of source docs for that
          disease). GR/SP are DOCUMENT granularity.

Output: kg_merged2_prov.json  (== kg_merged2.json + edge['prov']).
Usage: python v159_provenance.py [--kg ...kg_merged2.json] [--records ...kg_records_merged2.json]
       [--out ...kg_merged2_prov.json]"""
from __future__ import annotations
import json, os, re, glob, argparse
from collections import defaultdict


def norm(nm: str) -> str:
    """Same normalization the merge used to dedup finding names."""
    return re.sub(r"\s+", " ", str(nm).lower().strip())


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def title_norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def load_corpus_finding_index(path: str) -> dict:
    """{cui: set(norm(finding_name))} for one per-corpus IE file."""
    idx = {}
    data = json.load(open(path, encoding="utf-8"))
    for cui, obj in data.items():
        idx[cui] = {norm(f.get("name", "")) for f in obj.get("findings", []) if f.get("name")}
    return idx


def build_pubmed_docs(src_dir: str) -> dict:
    """{cui: [doc_id strings]} from the base source provenance blobs."""
    docs = {}
    for p in glob.glob(f"{src_dir}/*.json"):
        cui = os.path.splitext(os.path.basename(p))[0]
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        ids = []
        for e in d.get("provenance", []):
            kind = e.get("kind", "")
            if e.get("pmid"):
                ids.append(f"PMID:{e['pmid']}")
            elif e.get("ref"):
                ids.append(f"{kind}:{e['ref']}")
        # de-dup, preserve order
        seen = set(); uniq = []
        for x in ids:
            if x not in seen:
                seen.add(x); uniq.append(x)
        docs[cui] = uniq
    return docs


def build_gr_nbk(src_dir: str, index_json: str) -> dict:
    """{cui: NBK} for GeneReviews via crawl manifest title match."""
    idx = json.load(open(index_json, encoding="utf-8"))
    t2nbk = {title_norm(e["title"]): e["nbk"] for e in idx}
    out = {}
    for p in glob.glob(f"{src_dir}/*.json"):
        cui = os.path.splitext(os.path.basename(p))[0]
        d = json.load(open(p, encoding="utf-8"))
        nbk = t2nbk.get(title_norm(d.get("disease", "")))
        if nbk:
            out[cui] = nbk
    return out


def build_sp_nbk(src_dir: str, html_dir: str) -> dict:
    """{cui: NBK} for StatPearls via source HTML filename slug match."""
    slug2nbk = {}
    for h in glob.glob(f"{html_dir}/*.html"):
        m = re.match(r"(NBK\d+)_(.+)\.html", os.path.basename(h))
        if m:
            slug2nbk[m.group(2)] = m.group(1)
    out = {}
    for p in glob.glob(f"{src_dir}/*.json"):
        cui = os.path.splitext(os.path.basename(p))[0]
        d = json.load(open(p, encoding="utf-8"))
        nbk = slug2nbk.get(slug(d.get("disease", "")))
        if nbk:
            out[cui] = nbk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default="pilot/cache/prod/kg_merged2.json")
    ap.add_argument("--records", default="pilot/cache/prod/kg_records_merged2.json")
    ap.add_argument("--ie-full", default="pilot/cache/prod/ie_pred_full.json")
    ap.add_argument("--ie-gr", default="pilot/cache/prod/ie_pred_gr.json")
    ap.add_argument("--ie-sp", default="pilot/cache/prod/ie_pred_sp.json")
    ap.add_argument("--src-pubmed", default="pilot/cache/prod/sources")
    ap.add_argument("--src-gr", default="pilot/cache/prod/sources_gr")
    ap.add_argument("--src-sp", default="pilot/cache/prod/sources_sp")
    ap.add_argument("--gr-index", default="data/medkg/genereviews/index.json")
    ap.add_argument("--sp-html", default="data/medkg/statpearls")
    ap.add_argument("--out", default="pilot/cache/prod/kg_merged2_prov.json")
    a = ap.parse_args()

    print("loading per-corpus finding indexes ...", flush=True)
    pub_idx = load_corpus_finding_index(a.ie_full)
    gr_idx = load_corpus_finding_index(a.ie_gr)
    sp_idx = load_corpus_finding_index(a.ie_sp)
    print(f"  pubmed {len(pub_idx)}  genereviews {len(gr_idx)}  statpearls {len(sp_idx)} diseases", flush=True)

    print("building doc-id maps ...", flush=True)
    pub_docs = build_pubmed_docs(a.src_pubmed)
    gr_nbk = build_gr_nbk(a.src_gr, a.gr_index)
    sp_nbk = build_sp_nbk(a.src_sp, a.sp_html)
    print(f"  pubmed-docsets {len(pub_docs)}  GR-NBK {len(gr_nbk)}  SP-NBK {len(sp_nbk)}", flush=True)

    print("loading kg_records (finding -> name_cui) ...", flush=True)
    recs = json.load(open(a.records, encoding="utf-8"))
    # (dcui, pcui) -> list of original finding surface names
    pair_findings = defaultdict(list)
    for dcui, obj in recs.items():
        for rec in obj.get("phenotypes", []):
            pcui = rec.get("name_cui")
            nm = rec.get("name", "")
            if pcui and nm:
                pair_findings[(dcui, pcui)].append(nm)

    print("loading built KG ...", flush=True)
    kg = json.load(open(a.kg, encoding="utf-8"))

    # attribution stats
    n_edges = 0; n_edge_with_prov = 0; n_prov_entries = 0
    corpus_count = defaultdict(int)
    findings_total = 0; findings_attributed = 0

    for e in kg["edges"]:
        n_edges += 1
        d, p = e["d"], e["p"]
        prov = []
        seen = set()
        for nm in pair_findings.get((d, p), []):
            findings_total += 1
            n = norm(nm)
            corpora = []
            if d in pub_idx and n in pub_idx[d]:
                corpora.append("pubmed")
            if d in gr_idx and n in gr_idx[d]:
                corpora.append("genereviews")
            if d in sp_idx and n in sp_idx[d]:
                corpora.append("statpearls")
            if corpora:
                findings_attributed += 1
            for c in corpora:
                if c == "pubmed":
                    doc_ids = pub_docs.get(d, [])
                    gran = "disease-doc-set"
                elif c == "genereviews":
                    nbk = gr_nbk.get(d)
                    doc_ids = [nbk] if nbk else []
                    gran = "document"
                else:  # statpearls
                    nbk = sp_nbk.get(d)
                    doc_ids = [nbk] if nbk else []
                    gran = "document"
                key = (c, nm)
                if key in seen:
                    continue
                seen.add(key)
                entry = {
                    "source": c,
                    "doc_id": doc_ids[0] if doc_ids else None,
                    "finding": nm,
                    "granularity": gran,
                }
                if len(doc_ids) > 1:
                    entry["doc_ids"] = doc_ids
                prov.append(entry)
                corpus_count[c] += 1
                n_prov_entries += 1
        e["prov"] = prov
        if prov:
            n_edge_with_prov += 1

    kg.setdefault("meta", {})["prov"] = {
        "materialized": True,
        "granularity": {"pubmed": "disease-doc-set", "genereviews": "document", "statpearls": "document"},
        "n_edges_with_prov": n_edge_with_prov,
        "n_prov_entries": n_prov_entries,
        "corpus_entry_counts": dict(corpus_count),
    }

    json.dump(kg, open(a.out, "w"))
    print(f"\nPROVENANCED KG -> {a.out}")
    print(f"  edges={n_edges}  with prov={n_edge_with_prov} ({n_edge_with_prov/max(1,n_edges):.1%})")
    print(f"  prov entries={n_prov_entries}  by corpus={dict(corpus_count)}")
    print(f"  finding attribution: {findings_attributed}/{findings_total} "
          f"({findings_attributed/max(1,findings_total):.1%}) matched to >=1 corpus")


if __name__ == "__main__":
    main()
