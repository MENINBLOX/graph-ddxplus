"""Onset prompt-improvement loop (local gemma-4-E4B). Each --variant runs onset IE
on the fixed eval pool (onset_eval_pool.json) -> items file for sub-agent QA.
Goal: reduce the 'acute/chronic in disease-name' confound while keeping principles
(source-grounded, NO few-shot). Add variants as the loop diagnoses failures."""
import json, re, argparse

BASE_RULES = '''onset = exactly "sudden" or "gradual": "sudden" if begun suddenly/abruptly/acutely; "gradual" if gradually/insidiously/slowly/progressively-over-time. Otherwise "".
Only real patient clinical findings (not study terms, not substances).'''

# V2: anti disease-name-acute confound
V2_RULES = '''onset = exactly "sudden" or "gradual", ONLY when the text describes the temporal BEGINNING of a patient SYMPTOM/SIGN: "sudden" = began suddenly/abruptly (e.g. "sudden onset of chest pain", "abruptly developed diplopia"); "gradual" = developed gradually/insidiously/slowly/progressively over time (e.g. "gradually worsening", "slowly progressive weakness"). Otherwise "".
CRITICAL — the words "acute"/"chronic" are NOT onset when they are part of a DISEASE NAME or classification: e.g. "acute myocardial infarction", "acute infection", "acute pancreatitis", "chronic kidney disease", "acute febrile illness". In such cases "acute"/"chronic" name/classify the disease, NOT the temporal onset of a symptom -> leave onset "".
Only fill onset when an explicit temporal-onset phrase applies to a specific symptom/sign the patient experienced. Exclude epidemiologic/statistical uses and histologic descriptors ("slowly evolving dysplasia"). Only real patient findings (not study terms, drugs, substances).'''

V3_RULES = '''onset = "sudden" or "gradual" — fill ONLY when the abstract uses an EXPLICIT temporal-onset-quality phrase describing how a specific PATIENT symptom/sign BEGAN:
- "sudden": "sudden onset", "suddenly", "abruptly", "acute onset of <symptom>", "within hours/minutes".
- "gradual": "gradually", "insidious onset", "slowly progressive", "developed slowly over weeks/months/years".
Leave onset "" (do NOT guess) in ALL of these cases:
- AGE of onset only — "infantile-onset", "childhood-onset", "at age 25", "onset at 5 years", "ages at onset were 5.4 years". (age is NOT sudden/gradual)
- mere sequence words — "developed X", "presented with X", "subsequently X", "from birth". (sequence is NOT a tempo)
- "acute"/"chronic"/"fulminant" that is part of a DISEASE NAME/classification — "acute myocardial infarction", "chronic kidney disease", "fulminant cardiomyopathy".
- the finding is a disease/syndrome (not a symptom), a test/EEG/imaging/histology result, a mortality outcome ("sudden death"), or a study/population/animal-model term.
Only assign onset to a genuine patient symptom/sign that has an explicit tempo phrase next to it. If unsure, leave "".'''
# V4: stepwise gate — first decide finding is a SYMPTOM/SIGN, only then assign tempo.
# (V3 still attached onset to disease/syndrome/test/outcome entities; make that gate primary.)
V4_RULES = '''Decide onset in TWO steps for each finding:

STEP 1 — is the finding a SYMPTOM or SIGN the patient experiences or that is observed on exam? (e.g. pain, weakness, headache, numbness, seizure, fever, rash, ataxia, diplopia, dyspnea).
If instead the finding is ANY of the following, set onset="" and STOP (do not look at tempo words):
- a DISEASE / SYNDROME / DIAGNOSIS name — e.g. cardiomyopathy, glomerulonephritis, LGMD, muscular dystrophy, "MCA syndrome", myocardial infarction, encephalitis, any "...-opathy / ...-itis / ...-osis / ...-emia / ... syndrome / ... disease".
- a TEST / LAB / IMAGING / PATHOLOGY result — e.g. "decreasing serum complement", "thrombus on MRI", EEG/biopsy/histology findings.
- an OUTCOME or epidemiologic term — e.g. "sudden death", mortality, "early death", a study/population/animal-model term.

STEP 2 — only for a genuine SYMPTOM/SIGN from step 1, set onset = "sudden" or "gradual" ONLY if an EXPLICIT temporal-onset phrase describes how THAT symptom began:
- "sudden": "sudden onset", "suddenly", "abruptly", "acute onset of <symptom>", began "within minutes/hours".
- "gradual": "gradually", "insidious onset", "slowly progressive", "developed slowly over weeks/months/years".
Leave onset="" if: only an AGE of onset is given ("infantile-onset", "at one year of age"); only a sequence word with no tempo ("developed", "presented with", "progressed to" with no sudden/gradual quality); or the tempo word is "subacute"/"acute"/"chronic" used to NAME a disease.
If unsure, leave onset="".'''
PROMPTS={"V1":BASE_RULES,"V2":V2_RULES,"V3":V3_RULES,"V4":V4_RULES}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--variant",required=True); a=ap.parse_args()
    rules=PROMPTS[a.variant]
    pool=json.load(open("pilot/data/cache/maccrobat/onset_eval_pool.json"))
    PROMPT='''Read the biomedical abstract. Extract the patient's clinical symptoms/findings and, for each, the ONSET.

ABSTRACT:
"""{src}"""

'''+rules+'''
Output ONLY JSON: {{"findings":[{{"name":"...","onset":""}}]}}'''
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=4096,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":PROMPT.format(src=d["text"])}] for d in pool],SamplingParams(temperature=0.0,max_tokens=1024),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    items=[]
    for d,o in zip(pool,outs):
        srcl=d["text"].lower(); txt=o.outputs[0].text
        m=re.search(r'(\{.*\})',txt,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except: pass
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower(); on=str(f.get("onset","")).strip().lower()
            if on in ("sudden","gradual") and len(nm)>=2 and in_src(nm,srcl):
                items.append({"idx":len(items),"pmid":d["pmid"],"finding":nm,"onset":on,"abstract":d["text"]})
    json.dump(items,open(f"pilot/data/cache/maccrobat/onset_{a.variant}_items.json","w"),ensure_ascii=False,indent=0)
    print(f"{a.variant}: {len(items)} onset instances on {len(pool)} abstracts")

if __name__=="__main__": main()
