"""Aggregate fill-rate + character cleanliness over an IE output dir (array schema)."""
import argparse, json, glob
from collections import Counter, defaultdict
ap=argparse.ArgumentParser(); ap.add_argument("--in",dest="ind",required=True); a=ap.parse_args()
LIST_ATTR=["location","character","radiation","aggravating","relieving","associated"]
SCALAR_ATTR=["onset","duration","severity","timing","course","context","prior_episodes"]
QUALITY=set("sharp dull burning throbbing pulsating stabbing aching cramping colicky pressure pressing tight tightness squeezing pleuritic gnawing shooting tearing ripping stinging".split())
fill=Counter(); vals=defaultdict(Counter); nfind=0; ndis=0; char_clean=0; char_tot=0
for fp in glob.glob(f"pilot/data/cache/{a.ind}/*.json"):
    if fp.endswith("_stats.json") or fp.endswith("_normstats.json"): continue
    d=json.load(open(fp)); ndis+=1
    for f in d.get("findings",[]):
        nfind+=1
        for at in LIST_ATTR:
            v=f.get(at,[]);
            if isinstance(v,str): v=[v] if v else []
            if v: fill[at]+=1
            for e in v:
                vals[at][str(e).lower()]+=1
                if at=="character": char_tot+=1; char_clean+= 1 if str(e).lower() in QUALITY else 0
        for at in SCALAR_ATTR:
            if str(f.get(at,"")).strip(): fill[at]+=1; vals[at][str(f.get(at)).lower()]+=1
stats={"in":a.ind,"diseases":ndis,"findings":nfind,"findings_per_disease":round(nfind/max(1,ndis),1),
       "fill_rate":{at:round(fill[at]/max(1,nfind),3) for at in LIST_ATTR+SCALAR_ATTR},
       "character_cleanliness":round(char_clean/max(1,char_tot),3),
       "distinct_values":{at:vals[at].most_common(60) for at in LIST_ATTR+["onset","severity","timing"]}}
json.dump(stats,open(f"pilot/data/cache/{a.ind}/_stats.json","w"),indent=2,ensure_ascii=False)
print(f"{a.ind}: diseases={ndis} findings={nfind} per_disease={stats['findings_per_disease']}")
print("fill:",{k:f"{v*100:.0f}%" for k,v in stats['fill_rate'].items()})
print("char_cleanliness:",stats['character_cleanliness'])
for at in ["character","aggravating","relieving"]:
    print(f"  {at}:", ", ".join(f"{v}({n})" for v,n in vals[at].most_common(15)))
