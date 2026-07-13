"""Unified 4-attribute IE (location/severity/onset/character) on DISEASE clinical text —
the production setting (input = Wikipedia clinical sections, v105_sources). Replaces the
mis-scoped PubMed/onset-isolated experiment. Reuses the VALIDATED severity (R2, MACCROBAT
value-F1 0.58) and location (L1, F1 0.55) rules verbatim; onset is the under-developed
attribute (production = a single line) and is the target of deep iteration here, mirroring
the severity R2 structure (symptom-bound + anti-disease-level + adjacency). character =
faithfulness-only. gemma-4-E4B, temp=0, source-grounded, benchmark-blind, NO few-shot.

Run: --onset {O0,O1,...} selects the onset rule variant; output per-disease JSON for
source-grounded faithfulness scoring (sub-agent) on the SAME disease text."""
import json, re, glob, argparse
from pathlib import Path

# ---- validated rules (reused verbatim) ----
SEV_R2='- severity: one word: "mild", "moderate", or "severe" — BUT fill ONLY when the SOURCE states the severity of THIS SPECIFIC symptom (e.g. "severe headache", "mild cough"). If the SOURCE only describes the severity of the disease/case/episode as a whole (e.g. "in severe cases", "severe disease"), leave severity EMPTY. The word must describe the symptom\'s own intensity, not the illness. Confirm the severity word stands directly next to this symptom in the SOURCE; if not directly attached, leave it EMPTY.'
LOC_L1='- location: the specific anatomical body site the SOURCE attaches DIRECTLY to THIS finding (e.g. "cheeks","groin","left lower lobe"), copied verbatim. Fill ONLY when the site directly modifies THIS finding. EXCLUDE non-anatomical locations. If no body site is attached to this finding, leave EMPTY.'
CHAR='- character: the QUALITY of the symptom (e.g. "sharp","dull","burning","throbbing","productive"), only if the SOURCE states it. Do NOT define or restate the symptom itself (for "cyanosis" do NOT write "bluish skin").'

# ---- onset variants (the deep-iteration target) ----
ONSET={
 # O0 = current production (one line)
 "O0":'- onset: exactly one word: "sudden" or "gradual".',
 # O1 = R2-analog: symptom-bound + anti-disease-level + adjacency, with sudden/gradual definitions
 "O1":'- onset: one word "sudden" or "gradual" — fill ONLY when the SOURCE states how THIS SPECIFIC symptom BEGINS or appears (e.g. "sudden chest pain", "gradual onset of weakness", "develops insidiously"). "sudden" = begins abruptly/acutely/rapidly/suddenly; "gradual" = develops slowly/insidiously/progressively over time. If the SOURCE gives no temporal-onset wording for THIS symptom, or only states the onset of the disease as a whole, or gives only an AGE of onset, leave onset EMPTY. The wording must describe this symptom\'s own beginning, not the disease course.',
 # O2 = O1 + DISTRIBUTIVE group-onset (recall fix: O1 precision 93% but recall 44% — model
 # attaches a group onset to only one of the co-listed symptoms). On disease text, distributing
 # an explicitly-stated group onset across its listed symptoms is CORRECT (source-grounded).
 "O2":'- onset: one word "sudden" or "gradual" stating how a symptom BEGINS. "sudden" = begins abruptly/acutely/rapidly/suddenly; "gradual" = develops slowly/insidiously/progressively over time.\n'
      '  * Fill when the SOURCE states the onset of THIS symptom (e.g. "sudden chest pain", "gradual onset of weakness").\n'
      '  * DISTRIBUTE group onset: when the SOURCE states ONE onset for a LIST of symptoms together — e.g. "the onset of symptoms is sudden, including A, B, C", or "A, B, and C develop rapidly", or "symptoms come on suddenly, such as A, B, C" — assign that SAME onset to EVERY symptom explicitly named in that list (not only the first).\n'
      '  * Leave EMPTY if: no temporal-onset wording applies to this symptom or its group; only the disease/episode onset is described with no symptom list; or only an AGE of onset is given. Do NOT extend a group onset to symptoms that are NOT in the stated list.',
 # O3 = O2 + post-positioned group pattern ("A, B, C develop rapidly/suddenly") and "rapidly"->sudden
 # mapping (recall: Boerhaave "...shock develop rapidly thereafter" missed); tighten distribution to
 # symptoms named IN the onset clause only (precision: Influenza "coughing/fatigue" were over-extended).
 "O3":'- onset: one word "sudden" or "gradual" stating how a symptom BEGINS. "sudden" = begins suddenly/abruptly/acutely/rapidly/within minutes-hours; "gradual" = develops slowly/insidiously/progressively/over days-weeks-years.\n'
      '  * Fill when the SOURCE states the onset of THIS symptom directly (e.g. "sudden onset of chest pain", "gradual onset of weakness").\n'
      '  * DISTRIBUTE a group onset to EVERY symptom NAMED in the same onset clause, whether the onset word comes BEFORE the list ("the onset of symptoms is sudden, including A, B, C"; "sudden onset of A, B, and C"; "symptoms come on suddenly: A, B, C") OR AFTER the list ("A, B, and C develop rapidly/appear suddenly/come on gradually"). Map "rapidly"/"acutely"/"abruptly"->sudden, "slowly"/"insidiously"/"progressively"->gradual.\n'
      '  * PRECISION: assign the group onset ONLY to symptoms explicitly listed inside that onset clause. Do NOT give it to other symptoms mentioned elsewhere in the SOURCE (e.g. a symptom named in a different sentence with no onset wording stays EMPTY).\n'
      '  * Leave EMPTY if: no onset wording applies to this symptom or its clause; only the disease/episode onset is described with no symptom list; or only an AGE of onset is given.',
}

HEAD='''You are a clinical information extractor. Read the SOURCE about "{disease}" and extract, USING ONLY facts explicitly stated in the SOURCE, the symptoms/signs a patient experiences and their clinically discriminative attributes.

SOURCE:
"""{src}"""

Work in two steps.
STEP 1 — Reasoning: list the concrete patient symptoms/signs the SOURCE mentions, and for each, note ONLY the attributes the SOURCE explicitly states.
STEP 2 — Output one JSON object.

RULES — what counts as a "finding":
- INCLUDE only a specific symptom or sign a patient feels or shows (e.g. fever, productive cough, chest pain, malar rash, groin bulge).
- EXCLUDE headings/categories/disease-names ("general symptoms", "<disease> presentation", "signs and symptoms").

RULES — attribute values (fill ONLY if the SOURCE states it AND it adds NEW information; otherwise ""):
{onset}
{severity}
{location}
{character}
- Never put the disease name or a category into any attribute.

Output exactly:
JSON: {{"findings":[{{"name":"...","onset":"","severity":"","location":"","character":""}}]}}'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--onset",default="O1"); ap.add_argument("--src_dir",default="pilot/data/cache/v105_sources")
    ap.add_argument("--out",default=None); a=ap.parse_args()
    out=a.out or f"pilot/data/cache/maccrobat/unified4_{a.onset}.json"
    prompt_tmpl=HEAD.replace("{onset}",ONSET[a.onset]).replace("{severity}",SEV_R2).replace("{location}",LOC_L1).replace("{character}",CHAR)
    icd=json.load(open("data/ddxplus/disease_icd10_cui_mapping.json")); cui2name={info["cui"]:dn for dn,info in icd.items() if "cui" in info}
    items=[]
    for fp in sorted(glob.glob(f"{a.src_dir}/*.txt")):
        c=fp.split("/")[-1][:-4]
        if c in cui2name: items.append((c,cui2name[c],open(fp).read()))
    from vllm import LLM, SamplingParams
    llm=LLM(model="google/gemma-4-E4B-it",dtype="bfloat16",max_model_len=8192,gpu_memory_utilization=0.85,enforce_eager=True,limit_mm_per_prompt={"image":0,"audio":0})
    outs=llm.chat([[{"role":"user","content":prompt_tmpl.format(disease=dn,src=src[:2200])}] for c,dn,src in items],
                  SamplingParams(temperature=0.0,max_tokens=4096),use_tqdm=True)
    STOP=set("the a an of to in on with and or for is are be may can at as by from".split())
    def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
    def in_src(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)
    res={}
    for (c,dn,src),o in zip(items,outs):
        srcl=src.lower(); m=re.search(r'(\{.*\})',o.outputs[0].text,re.DOTALL); finds=[]
        if m:
            try: finds=json.loads(m.group(1)).get("findings",[])
            except: pass
        clean=[]
        for f in finds:
            if not isinstance(f,dict): continue
            nm=str(f.get("name","")).strip().lower()
            if len(nm)<2 or not in_src(nm,srcl): continue
            g={"name":nm}
            for at in ("onset","severity","location","character"):
                v=str(f.get(at,"")).strip().lower()
                if at=="onset": v=v if v in ("sudden","gradual") else ""
                elif v and not in_src(v,srcl): v=""   # source-grounding for free-text attrs
                g[at]=v
            clean.append(g)
        res[c]={"disease":dn,"source":src,"findings":clean}
    json.dump(res,open(out,"w"),ensure_ascii=False,indent=0)
    from collections import Counter
    fr=Counter(); n=0
    for r in res.values():
        for f in r["findings"]:
            n+=1
            for at in ("onset","severity","location","character"):
                if f[at]: fr[at]+=1
    print(f"unified4 {a.onset}: {len(res)} diseases, {n} findings. fill: " + ", ".join(f"{k}={fr[k]}({100*fr[k]//max(1,n)}%)" for k in ("onset","severity","location","character")))

if __name__=="__main__": main()
