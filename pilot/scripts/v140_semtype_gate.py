"""Semantic-type gate on the verified onset set. Diagnosis (final round, n=30): residual
errors (precision 57%) are dominated by tempo attached to a DISEASE-ENTITY head noun
("acute heart failure", "sudden febrile illness", "slowly progressive cardiomyopathy") and
to pathologic-process entities ("striatal damage", "rupture") — linguistically identical to
valid "acute onset of <symptom>" but the head is not a symptom. Principled fix (not prompt
tweaking, not model blame): link each finding to UMLS via scispaCy and KEEP only findings
whose top concept is a Sign/Symptom (T184) or Finding (T033); DROP Disease/Syndrome (T047),
Pathologic Function (T046), Neoplastic Process (T191), Anatomical Abnormality (T190), etc.
Uses the same scispaCy+UMLS infra already in the repo. Output = gated high-precision set."""
import json, argparse
import spacy
from scispacy.linking import EntityLinker

SYMPTOM_OK = {"T184", "T033"}  # Sign or Symptom, Finding

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default="pilot/data/cache/maccrobat/onset_verified_set.json")
    ap.add_argument("--out",default="pilot/data/cache/maccrobat/onset_gated_set.json")
    ap.add_argument("--thr",type=float,default=0.80); a=ap.parse_args()
    data=json.load(open(a.src))
    nlp=spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker",config={"resolve_abbreviations":True,"linker_name":"umls"})
    linker=nlp.get_pipe("scispacy_linker")

    kept=[]; drop_reasons={}
    for d in data:
        doc=nlp(d["finding"]); ok=False; best=None
        # collect linked CUIs across entities in the finding phrase
        for ent in doc.ents:
            for cui,score in ent._.kb_ents:
                if score<a.thr: continue
                types=set(linker.kb.cui_to_entity[cui].types)
                if best is None: best=(cui,score,types)
                if types & SYMPTOM_OK: ok=True; break
            if ok: break
        if ok:
            kept.append({"idx":len(kept),"pmid":d["pmid"],"finding":d["finding"],"onset":d["onset"],"abstract":d["abstract"]})
        else:
            tps = ",".join(sorted(best[2])) if best else "NO_LINK"
            drop_reasons[d["finding"]]=tps
    json.dump(kept,open(a.out,"w"),ensure_ascii=False,indent=0)
    from collections import Counter
    print(f"semtype gate: kept {len(kept)}/{len(data)} (top concept is Sign/Symptom or Finding)")
    print(f"dist={dict(Counter(x['onset'] for x in kept))}")
    print("\nsample DROPPED (finding -> top semantic types):")
    for f,t in list(drop_reasons.items())[:18]: print(f"  [{t}] {f}")

if __name__=="__main__": main()
