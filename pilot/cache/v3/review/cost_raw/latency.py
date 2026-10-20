"""Per-patient ranking latency of the frozen S-mix configuration on one Neo4j instance (read-only)."""
import json, os, statistics, sys, time
sys.path.insert(0, 'pilot/scripts')
import opendx_eval as O
import v2_ids
from graph_dx import GraphDx
uri, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
O.RAW_GOLD = True
cfg = dict(alpha=0.35, mu=2.0, decay=0.3, unified_w=2.0, support_beta=0.5, offtopic_w=0.5)  # frozen_config_E scoring + S-mix offtopic_w
t = time.time()
dx = GraphDx(uri, 'neo4j/thesis-dev-only', cfg['alpha'], cfg['mu'], cfg['decay'], cfg['unified_w'], cfg['support_beta'], cfg['offtopic_w'])
init = time.time() - t
pats = [(g, ev) for g, ev in O.load_ddxplus('test', 3000)['DDXPlus-test'] if v2_ids.gold_keys(g) and ev][:n]
res = {'uri': uri, 'n': len(pats), 'cfg': cfg, 'init_sec': round(init, 3), 'passes': []}
for p in range(2):
    rank_ms, lin_ms, nev = [], [], []
    for g, ev in pats:
        t0 = time.perf_counter(); dx.rank(ev, 10); t1 = time.perf_counter(); dx.lineage(v2_ids.gold_keys(g)); t2 = time.perf_counter()
        rank_ms.append(1000 * (t1 - t0)); lin_ms.append(1000 * (t2 - t1)); nev.append(len(ev))
    q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]
    res['passes'].append({'pass': p + 1,
        'rank_ms': {'mean': statistics.mean(rank_ms), 'median': statistics.median(rank_ms), 'p90': q(rank_ms, .9), 'max': max(rank_ms), 'min': min(rank_ms)},
        'lineage_ms': {'mean': statistics.mean(lin_ms), 'median': statistics.median(lin_ms)},
        'total_ms_mean': statistics.mean([a + b for a, b in zip(rank_ms, lin_ms)]),
        'patient_concepts_mean': statistics.mean(nev)})
    print(json.dumps(res['passes'][-1]), flush=True)
json.dump(res, open(out, 'w'), indent=1)
