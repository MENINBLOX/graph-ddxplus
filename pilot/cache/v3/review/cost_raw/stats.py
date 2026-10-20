"""Counts units processed by each extraction stage (read-only)."""
import glob, json, os, random, sys, time
sys.path.insert(0, 'pilot/scripts')
from ie_v2 import PROMPT, windows
OUT = sys.argv[1]
res = {}
rng = random.Random(0)
samples = {}
def src_stats(tag, d):
    t = time.time(); n_files = n_docs = chars = nwin = winchars = promptchars = 0; empty = 0
    res_s = []; seen = 0
    for f in sorted(glob.glob(f'{d}/*.json')):
        rec = json.load(open(f)); n_files += 1
        if not rec.get('docs'): empty += 1
        pc = len(PROMPT.format(disease=rec['name'], src=''))
        for doc in rec.get('docs', []):
            n_docs += 1; chars += len(doc['text'])
            for off, w in windows(doc['text'], 6000, 400):
                nwin += 1; winchars += len(w); promptchars += pc
                seen += 1
                if len(res_s) < 1500: res_s.append(PROMPT.format(disease=rec['name'], src=w))
                else:
                    j = rng.randrange(seen)
                    if j < 1500: res_s[j] = PROMPT.format(disease=rec['name'], src=w)
        res[tag] = dict(files=n_files, files_without_docs=empty, docs=n_docs, chars=chars, windows=nwin,
                    window_chars=winchars, prompt_template_chars=promptchars, sec=round(time.time()-t))
    print(tag, res[tag], flush=True)
def ie_stats(tag, p):
    t = time.time(); n = nw = nraw = nloc = items = with_items = 0
    for ln in open(p):
        r = json.loads(ln); n += 1
        nw += r.get('n_windows', 0); nraw += r.get('n_raw', 0); nloc += r.get('n_located', 0)
        k = len(r.get('items', [])); items += k; with_items += k > 0
    res[tag] = dict(diseases=n, diseases_with_items=with_items, windows=nw, raw_items=nraw, located_items=nloc, items=items, sec=round(time.time()-t))
    print(tag, res[tag], flush=True)
src_stats('src_v2', 'pilot/cache/v2/sources')
src_stats('src_v3new', 'pilot/cache/v3/sources_new')
ie_stats('ie_v2', 'pilot/cache/v2/ie_gemma_full.jsonl')
ie_stats('ie_v3', 'pilot/cache/v3/ie_v3.jsonl')
json.dump(res, open(OUT, 'w'), indent=1)
print('STATS_DONE')
