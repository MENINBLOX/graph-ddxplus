import re, sys, json
from datetime import datetime
def parse(path):
    chunks=[]; cur=None
    ts_re=re.compile(r'(?:INFO|WARNING|ERROR) (\d\d-\d\d \d\d:\d\d:\d\d)')
    for line in open(path, errors='replace'):
        if line.startswith('ie_v2:'):
            if cur: chunks.append(cur)
            m=re.search(r'diseases=(\d+) \(done (\d+)\)', line)
            cur={'diseases':int(m.group(1)),'done_before':int(m.group(2)),'start':None,'last_ts':None,'end':None,'win':0,'win_total':0,'el':0,'rate':0}
            continue
        if cur is None: continue
        m=ts_re.search(line)
        if m:
            t=datetime.strptime('2026-'+m.group(1),'%Y-%m-%d %H:%M:%S')
            if cur['start'] is None: cur['start']=t
            cur['last_ts']=t
        m=re.match(r'\s+windows (\d+)/(\d+)\s+(\d+)s\s+out-tok/s (\d+)', line)
        if m:
            cur['win'],cur['win_total'],cur['el'],cur['rate']=map(int,m.groups())
        m=re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]', line)
        if m:
            cur['end']=datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S')
    if cur: chunks.append(cur)
    return chunks
for path in sys.argv[1:]:
    ch=parse(path)
    tot_el=sum(c['el'] for c in ch); tot_tok=sum(c['el']*c['rate'] for c in ch)
    tot_win=sum(c['win'] for c in ch)
    wall=0; interrupted=[]
    for c in ch:
        end=c['end'] or c['last_ts']
        if c['start'] and end: wall+=(end-c['start']).total_seconds()
        if c['win']<c['win_total'] or c['end'] is None: interrupted.append((c['done_before'],c['win'],c['win_total'],c['el'], str(c['start']), str(end)))
    first=min(c['start'] for c in ch if c['start']); last=max((c['end'] or c['last_ts']) for c in ch)
    print(path)
    print(f" chunks {len(ch)}  windows processed {tot_win:,}  gen seconds {tot_el:,} = {tot_el/3600:.2f} h  out tokens ~{tot_tok:,.0f}  mean rate {tot_tok/tot_el:.0f}/s")
    print(f" sum chunk wall (process start->end incl. model load) {wall/3600:.2f} h ; span {first} -> {last} = {(last-first).total_seconds()/3600:.2f} h")
    print(" incomplete chunks:", interrupted)
    for c in ch:
        print('   ', c['done_before'], c['diseases'], c['win'], c['win_total'], c['el'], c['rate'], c['start'], c['end'])
