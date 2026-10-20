"""S-dict items without a CUI: how many get a concept only through the LLM-derived phrase table (sample estimate).
Same 1-in-20 source sample as dict_bench.py; linked_forms as build_graph_v2 calls it for v3umls_ns
(--split-coord --hpo-names, table phrase2cui_gemma.json). Document filters and self-edge removal are not applied."""
import glob, json, sys
from multiprocessing import Pool
sys.path.insert(0, 'pilot/scripts')
import dict_ie as D
from build_graph_v2 import linked_forms, hpo_name_index
D.NAMES.update(D.umls_index())
def nocui(path):
    r = json.loads(D.extract(path))
    return len(r['items']), [it['t'] for it in r['items'] if 'cui' not in it]
samples = {}
for tag, d in (('v2', 'pilot/cache/v2/sources'), ('v3new', 'pilot/cache/v3/sources_new')):
    files = sorted(glob.glob(f'{d}/*.json'))[::20]
    tot, ts = 0, []
    with Pool(8) as pool:
        for n, t in pool.imap(nocui, files, chunksize=8):
            tot += n; ts += t
    samples[tag] = (tot, ts)
D.NAMES.clear()
hpo = hpo_name_index()
table = json.load(open('pilot/cache/v2/phrase2cui_gemma.json'))
out = {}
for tag, (tot, ts) in samples.items():
    any_with = only_table = links_with = links_without = 0
    for t in ts:
        a = linked_forms(t, table, True, hpo); b = linked_forms(t, {}, True, hpo)
        links_with += len(a); links_without += len(b)
        any_with += bool(a); only_table += bool(a) and not b
    out[tag] = dict(items=tot, items_without_cui=len(ts), linked_with_table=any_with, linked_only_via_table=only_table,
                    links_with_table=links_with, links_without_table=links_without)
    print(tag, out[tag], flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
