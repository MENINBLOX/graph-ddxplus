"""Build per-disease source blobs from the crawled GeneReviews corpus, for re-IE.

GeneReviews chapters are long (~18x PubMed) free-text clinical references that contain the
rare-disease phenotypes (incl. pathognomonic biochemistry) our PubMed corpus lacks. This
script: (1) reads the crawl manifest, (2) strips HTML to text, (3) keeps the phenotype-bearing
sections (Clinical Description / Diagnosis / Suggestive Findings / Signs & symptoms), (4) maps
each chapter's disease title to a seed CUI via the same UMLS linker used to build the KG, and
(5) writes {cui}.json source blobs (schema matching v156: {cui, disease, text}) for v153 IE.
Free-text prose only; benchmark-blind (all chapters) -> strict-zero-shot safe.
"""
from __future__ import annotations
import json, os, re, glob, argparse, html
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disease_link import link_disease

DISO_TUI = {"T047", "T191", "T019", "T020", "T037", "T046", "T048", "T049", "T050", "T190"}
# GeneReviews section headers that carry phenotype content (keep), vs boilerplate (drop)
KEEP_SEC = re.compile(r"(clinical\s+description|suggestive\s+findings|signs?\s+and\s+symptoms|"
                      r"diagnosis|phenotyp|clinical\s+characteristics|clinical\s+features|"
                      r"establishing\s+the\s+diagnosis|clinical\s+manifestations|"
                      r"history\s+and\s+physical|presentation)", re.I)  # StatPearls uses "History and Physical"
DROP_SEC = re.compile(r"(management|surveillance|therapies|agents\s+to\s+avoid|genetic\s+counsel|"
                      r"molecular\s+genetics|resources|references|chapter\s+notes|"
                      r"acknowledg|author\s+history|revision\s+history|gene\s+therapy)", re.I)


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|head|nav|footer).*?</\1>", " ", raw)
    raw = re.sub(r"(?is)</(p|div|li|h[1-6]|tr|br|section)>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n\s*\n+", "\n", raw)
    return raw.strip()


def phenotype_sections(text: str, budget: int = 6000) -> str:
    """Keep phenotype-bearing sections up to a word budget; skip management/genetics boilerplate."""
    lines = text.split("\n")
    out, keep, used = [], False, 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # crude header detection: short line, title-ish
        if len(s) < 80 and (KEEP_SEC.search(s) or DROP_SEC.search(s)):
            keep = bool(KEEP_SEC.search(s)) and not DROP_SEC.search(s)
            continue
        if keep:
            out.append(s); used += len(s.split())
            if used >= budget:
                break
    joined = " ".join(out)
    return joined if len(joined.split()) >= 40 else " ".join(text.split()[:budget])  # fallback


def title_from(path: str, title: str | None) -> str:
    if title:
        return title
    base = os.path.basename(path).rsplit(".", 1)[0]
    return re.sub(r"^NBK\d+_", "", base).replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gr-dir", default="data/medkg/genereviews")
    ap.add_argument("--out-dir", default="pilot/cache/prod/sources_gr")
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--cap", type=int, default=0,
                    help="hard character cap per disease blob (0 = uncapped). A cap was needed "
                         "when the IE sent the whole blob in one prompt; the IE now windows long "
                         "text (v153 --win), so capping only discards phenotype prose. GeneReviews "
                         "phenotype sections are ~24k chars, so the old 12000 cap dropped ~51%.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    idx_path = os.path.join(a.gr_dir, "index.json")
    if os.path.exists(idx_path):
        items = json.load(open(idx_path))
        entries = [(it.get("path") or os.path.join(a.gr_dir, f"{it['nbk']}.html"), it.get("title"))
                   for it in items]
    else:
        entries = [(p, None) for p in glob.glob(os.path.join(a.gr_dir, "*.htm*"))]
    print(f"chapters: {len(entries)}", flush=True)

    link_cache: dict[str, str | None] = {}
    def link_dz(name):
        if name not in link_cache:
            link_cache[name] = link_disease(name)[0]
        return link_cache[name]

    n_ok = n_nolink = 0
    written = {}
    for path, title in entries:
        if not os.path.exists(path):
            continue
        raw = open(path, encoding="utf-8", errors="ignore").read()
        name = title_from(path, title)
        cui = link_dz(name)
        if not cui:
            # No exact UMLS concept for this rare disease (e.g. "ATP1A3-Related Disorder"). Keep the
            # chapter as its OWN node under a synthetic key instead of folding it into a generic
            # concept: the evaluator resolves gold diseases to KG nodes lexically by name too, so a
            # correctly-named node stays reachable, while a generic sink is both unreachable and a
            # noise hub (CLAUDE.md §12).
            n_nolink += 1
            cui = "GR:" + os.path.basename(path).split("_")[0]
        text = phenotype_sections(html_to_text(raw), a.budget)
        # multiple chapters can map to the same CUI -> concatenate. The IE windows long text, so
        # no cap by default (--cap 12000 restores the old token-safe single-prompt behaviour).
        prev = written.get(cui)
        if prev:
            text = prev["text"] + " " + text
        written[cui] = {"cui": cui, "disease": name,
                        "text": text[:a.cap] if a.cap else text}
        n_ok += 1

    # SEAM INVARIANT: one chapter should become one disease. Chapters collapsing onto a shared CUI
    # is how 272 rare diseases silently became "Disease" (C0012634) -- and it is invisible after
    # this stage, because the output is one file PER CUI. Check it while both counts are known.
    n_chapters = sum(1 for p, _ in entries if os.path.exists(p))
    collapse = 1.0 - len(written) / max(1, n_chapters)
    print(f"chapter->disease collapse: {n_chapters} chapters -> {len(written)} diseases "
          f"({collapse:.0%} collapsed)" + ("   <-- CHECK THE LINKER" if collapse > 0.10 else ""))

    for cui, blob in written.items():
        json.dump(blob, open(os.path.join(a.out_dir, f"{cui}.json"), "w"))
    print(f"linked chapters: {n_ok}  no-CUI: {n_nolink}  distinct-disease source blobs: {len(written)}")
    print(f"-> {a.out_dir}/")


if __name__ == "__main__":
    main()
