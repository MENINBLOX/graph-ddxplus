"""Timing of dict_ie.py --vocab umls on a fixed 1-in-20 sample of source files (estimate by extrapolation).
Same index, same extract(), same Pool/imap/chunksize as dict_ie.main(); output is counted, not written."""
import glob, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, 'pilot/scripts')
import dict_ie as D
workers = int(sys.argv[1]); out = sys.argv[2]
res = {'workers': workers}
t = time.perf_counter(); D.NAMES.update(D.umls_index()); res['index_load_sec'] = round(time.perf_counter() - t, 1)
res['index_strings'] = len(D.NAMES)
for tag, d in (('v2', 'pilot/cache/v2/sources'), ('v3new', 'pilot/cache/v3/sources_new')):
    files = sorted(glob.glob(f'{d}/*.json'))[::20]
    chars = sum(len(doc['text']) for f in files for doc in json.load(open(f)).get('docs', []))
    t = time.perf_counter(); nbytes = 0; items = 0
    with Pool(workers) as pool:
        for ln in pool.imap(D.extract, files, chunksize=8):
            nbytes += len(ln); items += ln.count('{"t": ')
    sec = time.perf_counter() - t
    res[tag] = {'files': len(files), 'chars': chars, 'match_sec': round(sec, 1), 'items': items, 'out_bytes': nbytes,
                'chars_per_sec': chars / sec}
    print(tag, res[tag], flush=True)
json.dump(res, open(out, 'w'), indent=1)
print(res)
