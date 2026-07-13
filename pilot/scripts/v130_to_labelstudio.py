"""Converter: v129 4-attribute IE on MACCROBAT -> Label Studio tasks with predictions
(1차 라벨). Findings = labeled spans; onset/severity = per-region choices;
character/location = per-region textarea. Creates project + imports via API (session).

Densifies for onset/character (sparse, no gold): prioritizes docs that have >=1
onset OR character candidate, then fills to N with random docs (for recall).
"""
import requests, re, json, argparse, random

B="http://localhost:8080"
CONFIG='''<View>
<Header value="원문 케이스리포트 — 노란 밑줄 = 후보 소견(우리 IE). 각 소견을 클릭해 속성(onset/severity/character/location)을 검토·수정하세요. 빠진 소견은 드래그로 추가."/>
<View style="font-size:13px; color:#888">원본 논문: <Text name="link" value="$link"/></View>
<Labels name="finding" toName="text"><Label value="finding" background="#FFD54F"/></Labels>
<Text name="text" value="$text"/>
<Choices name="onset" toName="text" perRegion="true" choice="single" showInline="true">
  <Choice value="sudden"/><Choice value="gradual"/></Choices>
<Choices name="severity" toName="text" perRegion="true" choice="single" showInline="true">
  <Choice value="mild"/><Choice value="moderate"/><Choice value="severe"/><Choice value="marked"/><Choice value="slight"/><Choice value="extensive"/><Choice value="massive"/></Choices>
<TextArea name="character" toName="text" perRegion="true" rows="1" placeholder="character (e.g. sharp, dull, productive)"/>
<TextArea name="location" toName="text" perRegion="true" rows="1" placeholder="body site (e.g. left lung)"/>
</View>'''

def login():
    s=requests.Session(); s.get(B+"/user/login/")
    s.post(B+"/user/login/",data={"csrfmiddlewaretoken":s.cookies.get("csrftoken"),
        "email":"admin@meninblox.com","password":"adminpass123"},headers={"Referer":B+"/user/login/"})
    return s, {"X-CSRFToken":s.cookies.get("csrftoken"),"Content-Type":"application/json"}

def build_task(pmid, rec):
    text=rec["text"]; results=[]
    for k,x in enumerate(rec["findings"]):
        pos=text.lower().find(x["name"].lower())
        if pos<0: continue
        rid=f"r{k}"; sp={"start":pos,"end":pos+len(x["name"])}
        results.append({"id":rid,"from_name":"finding","to_name":"text","type":"labels",
                        "value":{**sp,"labels":["finding"],"text":text[pos:pos+len(x['name'])]}})
        if x.get("onset"):
            results.append({"id":rid,"from_name":"onset","to_name":"text","type":"choices","value":{**sp,"choices":[x["onset"]]}})
        if x.get("severity"):
            sv=x["severity"] if x["severity"] in ("mild","moderate","severe","marked","slight","extensive","massive") else "severe"
            results.append({"id":rid,"from_name":"severity","to_name":"text","type":"choices","value":{**sp,"choices":[sv]}})
        if x.get("character"):
            results.append({"id":rid,"from_name":"character","to_name":"text","type":"textarea","value":{**sp,"text":[x["character"]]}})
        if x.get("location"):
            results.append({"id":rid,"from_name":"location","to_name":"text","type":"textarea","value":{**sp,"text":[x["location"]]}})
    return {"data":{"text":text,"pmid":pmid,"link":f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"},
            "predictions":[{"model_version":"gemma-4-E4B-IE","result":results}]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=40); ap.add_argument("--title",default="IE 4속성 검증 (1차)")
    a=ap.parse_args()
    pred=json.load(open("pilot/data/cache/maccrobat/v129_4attr_pred.json"))
    # densify: docs with onset/character first, then random fill
    has_oc=[p for p,v in pred.items() if any(x["onset"] or x["character"] for x in v["findings"])]
    rest=[p for p in pred if p not in has_oc]
    random.seed(42); random.shuffle(rest)
    chosen=has_oc[:a.n] + rest[:max(0,a.n-len(has_oc))]
    chosen=chosen[:a.n]
    s,H=login()
    r=s.post(B+"/api/projects",headers=H,data=json.dumps({"title":a.title,"label_config":CONFIG}))
    if r.status_code not in (200,201): print("project create FAIL",r.status_code,r.text[:300]); return
    pid=r.json()["id"]; print("project",pid,"created")
    tasks=[build_task(p,pred[p]) for p in chosen]
    ok=0
    for i in range(0,len(tasks),10):
        batch=tasks[i:i+10]
        r=s.post(B+f"/api/projects/{pid}/import",headers=H,data=json.dumps(batch))
        if r.status_code in (200,201): ok+=r.json().get("task_count",len(batch))
        else: print(f"  batch {i} FAIL {r.status_code}: {r.text[:200]}")
    print(f"imported {ok}/{len(tasks)} tasks")
    print(f"\n열기: {B}/projects/{pid}/data  (docs={len(tasks)}, onset/char-dense={len([p for p in chosen if p in has_oc])})")

if __name__=="__main__": main()
