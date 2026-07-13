"""Claude 1차 검토 of gemini 4-attr labels (before professor). Conservative:
remove only CLEAR errors so professor doesn't waste time; flag ambiguous for them.
Rules:
 - value must be source-grounded (token-majority in text) else DROP (hallucinated).
 - character that merely restates the finding name -> DROP.
 - severity not a grading word AND not source-adjacent -> FLAG (keep, note).
 - location not in source -> DROP.
Outputs reviewed file + a human-readable review report."""
import json, re
SRC=json.load(open("pilot/data/cache/maccrobat/v131_gemini_4attr_pred.json"))
GRADE={"mild","moderate","severe","marked","slight","extensive","significant","minimal","minor","small","large","high","low","massive","profuse","advanced","serious","mild-to-moderate","moderate-to-severe"}
STOP=set("the a an of to in on with and or for is are be may can at as by from of".split())
def kt(s): return [w for w in re.findall(r'[a-z]+',s.lower()) if len(w)>3 and w not in STOP]
def ground(v,srcl): ws=kt(v); return bool(ws) and sum(1 for w in ws if w in srcl)>=max(1,len(ws)//2)

report={"dropped_ungrounded":[], "dropped_char_restate":[], "flag_severity_nonstd":[], "flag_location_long":[]}
out={}
for pmid,rec in SRC.items():
    srcl=rec["text"].lower(); finds=[]
    for f in rec["findings"]:
        nm=f["name"]; g={"name":nm,"onset":"","character":"","severity":"","location":""}
        for a in ["onset","character","severity","location"]:
            v=f.get(a,"").strip()
            if not v: continue
            if a=="onset": g[a]=v if v in ("sudden","gradual") else ""; continue
            if not ground(v,srcl):
                report["dropped_ungrounded"].append((pmid,a,nm,v)); continue
            if a=="character":
                # restating finding? char tokens subset of finding tokens
                if set(kt(v)) and set(kt(v))<=set(kt(nm)):
                    report["dropped_char_restate"].append((pmid,nm,v)); continue
            if a=="severity" and v not in GRADE:
                report["flag_severity_nonstd"].append((pmid,nm,v))  # keep but flag
            if a=="location" and len(v.split())>4:
                report["flag_location_long"].append((pmid,nm,v))  # keep but flag (maybe phrase not site)
            g[a]=v
        finds.append(g)
    out[pmid]={"text":rec["text"],"findings":finds}
json.dump(out,open("pilot/data/cache/maccrobat/v132_reviewed_pred.json","w"))

def cnt(a): return sum(1 for v in out.values() for x in v["findings"] if x[a])
print("=== Claude 1차 검토 결과 ===")
print(f"속성 잔존: onset={cnt('onset')} character={cnt('character')} severity={cnt('severity')} location={cnt('location')}")
print(f"제거: ungrounded={len(report['dropped_ungrounded'])}  character-restating-finding={len(report['dropped_char_restate'])}")
print(f"플래그(교수님 주의): severity 비표준어={len(report['flag_severity_nonstd'])}  location 긴구={len(report['flag_location_long'])}")
print("\n[제거 예: character가 소견 재진술]");
for x in report["dropped_char_restate"][:6]: print("  ",x[1],"| char=",x[2])
print("\n[플래그 예: severity 비표준어(교수님 판단 필요)]")
for x in report["flag_severity_nonstd"][:8]: print("  ",x[1],"| sev=",x[2])
json.dump(report,open("pilot/data/cache/maccrobat/v132_review_report.json","w"),ensure_ascii=False,indent=1)
