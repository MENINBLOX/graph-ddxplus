import json, os, re, sys, time, zipfile, urllib.request, urllib.parse
key = next((l.split("=",1)[1].strip().strip('"') for l in open(".env") if l.startswith("NCBI_API_KEY=")), "")
pm = set()
z = zipfile.ZipFile("data/external_benchmarks/phenopacket_store/all_phenopackets.zip")
for n in z.namelist():
    m = re.search(r"PMID_(\d+)", n)
    if m: pm.add(m.group(1))
for n in os.listdir("data/confirm/phenopacket_new"):
    m = re.search(r"PMID_(\d+)", n)
    if m: pm.add(m.group(1))
pm = sorted(pm); out = {}
for i in range(0, len(pm), 200):
    q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pm[i:i+200]), "retmode": "json", "api_key": key})
    r = json.load(urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + q))["result"]
    for u in r.get("uids", []):
        out[u] = {"pubdate": r[u].get("pubdate", ""), "epubdate": r[u].get("epubdate", ""), "sortpubdate": r[u].get("sortpubdate", "")}
    time.sleep(0.15)
json.dump(out, open(sys.argv[1], "w"))
print(len(pm), "pmids;", len(out), "dated")
