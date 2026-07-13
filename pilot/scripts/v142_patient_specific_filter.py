"""V7 = V5 verified set (best base, 57%) + DETERMINISTIC patient-specificity / anti-definitional
filter. Lesson from V6: prompting an 8B model to tell "Pharyngoconjunctival fever is
characterized by abrupt onset of high fever" (definitional) from "10 inpatients developed
sudden onset of high fever" (patient-specific) FAILS — both read as "sudden onset of high
fever". The principled lever is corpus/sentence structure, not model judgment (cf.
project_v103_corpus_mismatch). Keep an onset only when (a) the abstract is a patient case
report (explicit patient anchor) and (b) the tempo-bearing sentence is not a generic
disease-class definition. Deterministic, source-grounded, benchmark-blind. No GPU."""
import json, re, argparse

# (a) abstract must contain a patient anchor (a real case, not a review/definition)
PATIENT_ANCHOR = re.compile(
    r'\b(\d{1,3}[\s-]?year[\s-]?old|\d{1,3}[\s-]?(?:yo|y/o)\b'
    r'|we (?:report|describe|present)\b|case report|a case of'
    r'|(?:the|our|a|an|this)\s+patient\b|patients? (?:was|were|presented|developed|had|underwent|admitted)'
    r'|presented with|was admitted|were admitted|referred (?:to|for))', re.I)

# (b) the tempo sentence must NOT be a generic disease-class definition / epidemiologic generalization
DEFINITIONAL = re.compile(
    r'\b(is a |is an |are a |is the |is characterized by|are characterized by|is defined as'
    r'|are defined as|refers to|is typically|are typically|usually presents|typically presents'
    r'|most frequently presents|is characteri|consists of|is known as|can be defined'
    r'|is an? entity|presents? with the|is associated with the classic)\b', re.I)
TEMPO = re.compile(r'\b(sudden|suddenly|abrupt|abruptly|acute onset|acutely|gradual|gradually'
                   r'|insidious|slowly progressive|slow loss|progressively|rapidly progressive)\b', re.I)
# study outcome measures / instruments are not patient symptoms (SF-36, QoL scores, recovery trajectories)
STUDY_OUTCOME = re.compile(r'\b(score|qol|quality of life|sf-?36|summary score|questionnaire'
                           r'|outcome measure|recovery of|health summary|drusen)\b', re.I)

def sents(t): return re.split(r'(?<=[.!?])\s+', t)

def keep(item):
    ab=item["abstract"]
    if STUDY_OUTCOME.search(item["finding"]): return False, "study_outcome_measure"
    if not PATIENT_ANCHOR.search(ab): return False, "no_patient_anchor"
    # find the tempo-bearing sentence(s); reject if any is definitional and lacks its own patient anchor
    tempo_sents=[s for s in sents(ab) if TEMPO.search(s)]
    if not tempo_sents: return True, "ok_no_tempo_sentence_found"  # finding grounded elsewhere; keep
    for s in tempo_sents:
        if DEFINITIONAL.search(s) and not PATIENT_ANCHOR.search(s):
            return False, "definitional_tempo_sentence"
    return True, "ok"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default="pilot/data/cache/maccrobat/onset_verified_set.json")
    ap.add_argument("--out",default="pilot/data/cache/maccrobat/onset_verified_set_v7.json"); a=ap.parse_args()
    data=json.load(open(a.src)); kept=[]; drop=[]
    for d in data:
        ok,why=keep(d)
        if ok: kept.append({**d,"idx":len(kept)})
        else: drop.append((why,d["finding"]))
    json.dump(kept,open(a.out,"w"),ensure_ascii=False,indent=0)
    from collections import Counter
    print(f"V7 deterministic filter: kept {len(kept)}/{len(data)}  dist={dict(Counter(x['onset'] for x in kept))}")
    print(f"drop reasons: {dict(Counter(w for w,_ in drop))}")
    print("\nsample DROPPED (reason | finding):")
    for w,f in drop[:18]: print(f"  {w:28} | {f}")

if __name__=="__main__": main()
