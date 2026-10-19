"""Index NCBI Bookshelf open-access bundles (LitArch) by chapter: NBK accession -> title, kept text.

Bookshelf web pages now answer scripted requests with a CAPTCHA, so chapters are read from the
versioned LitArch bundles instead (ftp.ncbi.nlm.nih.gov/pub/litarch/: gene_NBK1116 = GeneReviews,
statpearls_NBK430685 = StatPearls). Kept sections are clinical description, presentation,
evaluation, etiology and the differential; management, treatment and references are dropped,
as for journal articles.

    python book_index_v2.py --out pilot/cache/v2/book_index.json
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
import xml.etree.ElementTree as ET

KEEP = re.compile(r"(?i)(clinical|presentation|history and physical|evaluation|etiology|cause|"
                  r"pathophysiology|differential|diagnos|suggestive|finding|sign|symptom|feature|"
                  r"manifestation|phenotype|natural history|genotype-phenotype)")
DROP = re.compile(r"(?i)(treatment|management|therap|prognos|complication|deterrence|pearls|"
                  r"enhancing|review question|reference|counsel|resources|molecular genetics|"
                  r"epidemiolog|toxicokinetic|pharmacokinetic|continuing education|author)")


def chapter(xml_bytes: bytes) -> dict | None:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    # StatPearls chapters carry only the wrapper id ("article-20545"), GeneReviews a short name
    # ("marfan") as book-part-id; either identifies the chapter inside its versioned bundle.
    acc = next((e.text for e in root.iter("book-part-id")), None) or root.get("id")
    t = root.find(".//title-group/title")
    title = " ".join("".join(t.itertext()).split()) if t is not None else ""
    parts = []
    for sec in root.iter("sec"):
        st = sec.find("title")
        stitle = " ".join("".join(st.itertext()).split()) if st is not None else ""
        if not stitle or DROP.search(stitle) or not KEEP.search(stitle):
            continue
        for p in sec.findall("p"):
            parts.append(" ".join(" ".join(p.itertext()).split()))
    text = " ".join(x for x in parts if x)
    if not acc or not text or "(Archived)" in title:
        return None
    return {"acc": acc, "title": title, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = {}
    for path, book in (("data/litarch/gene_NBK1116.tar.gz", "genereviews"),
                       ("data/litarch/statpearls_NBK430685.tar.gz", "statpearls")):
        with tarfile.open(path) as tf:
            for m in tf:
                if not m.name.endswith(".nxml"):
                    continue
                c = chapter(tf.extractfile(m).read())
                if c:
                    c["book"] = book
                    c["doc_id"] = f"{book}:{c['acc']}"
                    idx[c["doc_id"]] = c
        print(book, sum(1 for v in idx.values() if v["book"] == book), flush=True)
    json.dump(idx, open(a.out, "w"))


if __name__ == "__main__":
    main()
