"""Input tokens per IE window: gemma chat template over a uniform random sample of windows (estimate)."""
import glob, json, os, random, sys
sys.path.insert(0, 'pilot/scripts')
from ie_v2 import PROMPT, windows
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-4-E4B-it')
out = {}
for tag, d in (('src_v2', 'pilot/cache/v2/sources'), ('src_v3new', 'pilot/cache/v3/sources_new')):
    rng = random.Random(0); res = []; seen = 0
    for f in sorted(glob.glob(f'{d}/*.json')):
        rec = json.load(open(f))
        for doc in rec.get('docs', []):
            for off, w in windows(doc['text'], 6000, 400):
                seen += 1
                if len(res) < 2000: res.append((rec['name'], w))
                else:
                    j = rng.randrange(seen)
                    if j < 2000: res[j] = (rec['name'], w)
    lens, chs, wch = [], [], []
    for nm, w in res:
        p = PROMPT.format(disease=nm, src=w)
        s = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        lens.append(len(tok(s, add_special_tokens=False)['input_ids'])); chs.append(len(p)); wch.append(len(w))
    out[tag] = dict(sample=len(lens), windows_total=seen, tok_per_window_mean=sum(lens)/len(lens),
                    chars_per_token=sum(chs)/sum(lens), window_chars_mean=sum(wch)/len(wch))
    print(tag, out[tag], flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
