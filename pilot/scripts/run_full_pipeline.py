#!/usr/bin/env python3
"""전체 파이프라인 실행: 471건 문서 × gemma4 + 노이즈 필터.

Step 1: 이미 수집된 데이터 재사용 (step1_documents.json)
Step 1b: T033/T034 제외 + 블랙리스트 5개 적용
Step 2: gemma4 ternary classification (전체 471건)
Step 3: Jensen Lab + G² + FDR
Step 4: KG 구축 + 평가
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
import scipy.stats as stats
from Bio import Entrez

Entrez.email = "max@meninblox.com"
Entrez.api_key = os.environ.get("PUBMED_API_KEY", "")

UMLS_DIR = Path("data/umls_extracted")
DATA_DIR = Path("pilot/data")
RESULTS_DIR = Path("pilot/results")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:e4b-it-bf16"
MAX_PAIRS_PER_DOC = 15

# 노이즈 필터: T033, T034 제외
ALLOWED_STYS = {"T047", "T184", "T191", "T046", "T048", "T037", "T019", "T020", "T190", "T049"}
BLACKLIST_CUIS = {"C1457887", "C3257980", "C0012634", "C0699748", "C3839861"}

PROMPT = """You are a biomedical relation extractor. Analyze this PubMed abstract and classify concept pair relationships.

RULES:
1. "present" = text EXPLICITLY states a positive relationship (symptom of, causes, associated with, co-occurs)
2. "absent" = text EXPLICITLY states a NEGATIVE relationship (not seen, absence of, rules out, excluded)
3. "not_related" = both concepts appear but NO explicit relationship between them
4. Do NOT infer relationships not explicitly stated

Text:
{text}

Pairs:
{pairs}

Respond ONLY with JSON array:
[{{"cui_a": "...", "cui_b": "...", "classification": "present|absent|not_related"}}]"""


def load_cui_stys() -> dict[str, set[str]]:
    r = defaultdict(set)
    with open(UMLS_DIR / "MRSTY.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            r[p[0]].add(p[1])
    return dict(r)


def load_cui_names() -> dict[str, str]:
    names = {}
    with open(UMLS_DIR / "MRCONSO.RRF") as f:
        for line in f:
            p = line.strip().split("|")
            if p[1] == "ENG" and (p[0] not in names or p[2] == "P"):
                names[p[0]] = p[14]
    return names


def call_ollama(prompt: str) -> tuple[str, float]:
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 4096},
    }, timeout=300)
    return resp.json().get("response", ""), time.time() - t0


def parse_json(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def dunning_g2(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return 0.0
    def g(o, e):
        return o * math.log(o / e) if o > 0 and e > 0 else 0
    ea = (a+b)*(a+c)/n; eb = (a+b)*(b+d)/n
    ec = (c+d)*(a+c)/n; ed = (c+d)*(b+d)/n
    return 2 * (g(a,ea) + g(b,eb) + g(c,ec) + g(d,ed))


def bh_fdr(pvals):
    n = len(pvals)
    if not n: return []
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    q = [0.0] * n
    prev = 1.0
    for i in range(n-1, -1, -1):
        idx, p = indexed[i]
        rank = i + 1
        q[idx] = min(prev, p * n / rank)
        prev = q[idx]
    return q


def main():
    print("=" * 80)
    print("전체 파이프라인: 471건 × gemma4 + 노이즈 필터")
    print("=" * 80)

    # ============================================================
    # Step 1: 데이터 로드 + 노이즈 필터
    # ============================================================
    print("\n[Step 1] 데이터 로드 + 노이즈 필터...")
    cui_stys = load_cui_stys()
    cui_names = load_cui_names()

    with open(DATA_DIR / "step1_documents.json") as f:
        docs_data = json.load(f)
    with open(DATA_DIR / "step1_cui_pairs.json") as f:
        all_pairs_raw = json.load(f)

    documents = docs_data["documents"]

    # 허용 CUI 필터: ALLOWED_STYS에 해당하고 블랙리스트 아닌 것
    allowed_cuis = set()
    for cui, stys in cui_stys.items():
        if (stys & ALLOWED_STYS) and cui not in BLACKLIST_CUIS:
            allowed_cuis.add(cui)

    # 문서별 필터링된 CUI 쌍
    doc_pairs = defaultdict(list)
    for p in all_pairs_raw:
        if p["cui_a"] in allowed_cuis and p["cui_b"] in allowed_cuis:
            doc_pairs[p["pmid"]].append(p)

    total_filtered_pairs = sum(len(v) for v in doc_pairs.values())
    docs_with_pairs = sum(1 for v in doc_pairs.values() if v)
    print(f"  문서: {len(documents)}, 쌍 있는 문서: {docs_with_pairs}")
    print(f"  필터링 후 CUI 쌍: {total_filtered_pairs:,}")
    print(f"  (필터 전: {len(all_pairs_raw):,}, 감소율: {1 - total_filtered_pairs/len(all_pairs_raw):.1%})")

    # ============================================================
    # Step 2: 초록 수집 + LLM 분류
    # ============================================================
    print("\n[Step 2] 초록 수집...")
    pmids = [d["pmid"] for d in documents]
    pmid_text = {}
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        try:
            h = Entrez.efetch(db="pubmed", id=batch, rettype="xml")
            records = Entrez.read(h); h.close()
            for article in records["PubmedArticle"]:
                pm = str(article["MedlineCitation"]["PMID"])
                ab = article["MedlineCitation"]["Article"].get("Abstract", {})
                pmid_text[pm] = " ".join(str(t) for t in ab.get("AbstractText", []))
            time.sleep(0.3)
        except Exception as e:
            print(f"  배치 {i} 오류: {e}")
    print(f"  초록 수집: {len(pmid_text)}건")

    # 체크포인트
    checkpoint_file = DATA_DIR / "full_checkpoint.json"
    all_cls = []
    processed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
            all_cls = ckpt.get("classifications", [])
            processed = set(c["pmid"] for c in all_cls)
            print(f"  체크포인트: {len(processed)}건 이미 처리됨")

    print(f"\n  LLM 분류 시작 ({len(documents) - len(processed)}건 남음)...")
    start = time.time()
    actual = 0
    errors = 0

    for idx, doc in enumerate(documents):
        pmid = doc["pmid"]
        if pmid in processed:
            continue
        text = pmid_text.get(pmid, "")
        pairs = doc_pairs.get(pmid, [])
        if not text or not pairs:
            continue

        pairs = pairs[:MAX_PAIRS_PER_DOC]
        pairs_text = "\n".join(
            f"- ({cui_names.get(p['cui_a'], p['cui_a'])[:40]}, "
            f"{cui_names.get(p['cui_b'], p['cui_b'])[:40]}) "
            f"[CUI: {p['cui_a']}, {p['cui_b']}]"
            for p in pairs
        )
        prompt = PROMPT.format(text=text[:2500], pairs=pairs_text)

        try:
            response, elapsed = call_ollama(prompt)
            parsed = parse_json(response)
            for item in parsed:
                cls = item.get("classification", "").lower().strip().replace(" ", "_")
                if cls in ("present", "absent", "not_related"):
                    all_cls.append({
                        "pmid": pmid, "cui_a": item.get("cui_a", ""),
                        "cui_b": item.get("cui_b", ""), "classification": cls,
                        "seed_disease": doc.get("seed_disease", ""),
                    })
            actual += 1
        except Exception as e:
            errors += 1

        if actual % 10 == 0 and actual > 0:
            elapsed_total = time.time() - start
            rate = actual / elapsed_total
            remaining = (docs_with_pairs - len(processed) - actual) / rate if rate > 0 else 0
            print(f"  [{actual + len(processed):4d}/{docs_with_pairs}] "
                  f"분류={len(all_cls):,} err={errors} "
                  f"{rate:.2f}건/s 잔여={remaining/60:.0f}분")
            # 체크포인트 저장
            with open(checkpoint_file, "w") as f:
                json.dump({"classifications": all_cls}, f)

    # 최종 저장
    with open(DATA_DIR / "full_classifications.json", "w") as f:
        json.dump(all_cls, f, indent=2, ensure_ascii=False)

    elapsed_total = time.time() - start
    dist = Counter(c["classification"] for c in all_cls)
    print(f"\n  Step 2 완료: {elapsed_total/60:.1f}분")
    print(f"  present={dist['present']}, absent={dist['absent']}, not_related={dist['not_related']}, 총={len(all_cls)}")

    # ============================================================
    # Step 3: Jensen Lab + G²
    # ============================================================
    print(f"\n[Step 3] 통계 집계...")
    pair_stats = defaultdict(lambda: {"n_present": 0, "n_absent": 0, "n_nr": 0, "pmids": set()})
    cui_doc_count = defaultdict(int)
    for doc in documents:
        for cui in set(doc["diso_cuis"]):
            if cui in allowed_cuis:
                cui_doc_count[cui] += 1
    total_docs = len(documents)

    for c in all_cls:
        key = tuple(sorted([c["cui_a"], c["cui_b"]]))
        pair_stats[key][f"n_{c['classification']}"] = pair_stats[key].get(f"n_{c['classification']}", 0) + 1
        pair_stats[key]["pmids"].add(c["pmid"])

    # present >= 1 인 쌍만
    candidates = {k: v for k, v in pair_stats.items() if v["n_present"] >= 1}
    print(f"  고유 쌍: {len(pair_stats):,}, present>=1: {len(candidates):,}")

    edges = []
    for (ca, cb), ps in candidates.items():
        np_ = ps["n_present"]
        na_ = ps["n_absent"]
        a = np_
        b = max(cui_doc_count.get(ca, 1) - np_, 0)
        c = max(cui_doc_count.get(cb, 1) - np_, 0)
        d = max(total_docs - a - b - c, 0)
        g2 = dunning_g2(a, b, c, d)
        pv = 1 - stats.chi2.cdf(g2, df=1) if g2 > 0 else 1.0

        c_ab = float(np_)
        c_a = float(max(cui_doc_count.get(ca, 1), 1))
        c_b = float(max(cui_doc_count.get(cb, 1), 1))
        alpha = 0.6
        oe = (c_ab * total_docs) / (c_a * c_b) if c_a * c_b > 0 else 0
        jensen = (c_ab ** alpha) * (oe ** (1 - alpha)) if c_ab > 0 and oe > 0 else 0

        polarity = "absent" if na_ > np_ else "present"
        edges.append({
            "cui_a": ca, "cui_b": cb, "polarity": polarity,
            "n_present": np_, "n_absent": na_,
            "jensen_score": jensen, "g2": g2, "p_value": pv,
            "pmids": sorted(ps["pmids"]),
        })

    # FDR
    pvals = [e["p_value"] for e in edges]
    qvals = bh_fdr(pvals)
    for e, q in zip(edges, qvals):
        e["q_value"] = q

    # 결과
    for fdr in [0.01, 0.05, 0.10]:
        sig = [e for e in edges if e["q_value"] < fdr]
        cuis = set()
        for e in sig:
            cuis.add(e["cui_a"]); cuis.add(e["cui_b"])
        print(f"  FDR<{fdr}: {len(sig)} 엣지, {len(cuis)} CUI")

    sig005 = [e for e in edges if e["q_value"] < 0.05]
    sig005.sort(key=lambda e: -e["jensen_score"])

    # ============================================================
    # Step 4: KG 통계 + DDXPlus 관련
    # ============================================================
    print(f"\n[Step 4] KG 구축 + 평가")
    print("=" * 80)

    unique_cuis = set()
    for e in sig005:
        unique_cuis.add(e["cui_a"]); unique_cuis.add(e["cui_b"])
    print(f"  KG: {len(unique_cuis)} 노드, {len(sig005)} 엣지 (FDR<0.05)")

    # Semantic type 분포
    sty_dist = Counter()
    for cui in unique_cuis:
        for s in cui_stys.get(cui, set()) & ALLOWED_STYS:
            sty_dist[s] += 1
    print(f"  Semantic type 분포:")
    for s, cnt in sty_dist.most_common():
        print(f"    {s}: {cnt}")

    # DDXPlus 5개 질환
    ddx = {"Pneumonia": "C0032285", "PE": "C0034065", "GERD": "C0017168",
           "Panic": "C0086769", "Bronchitis": "C0006277"}
    print(f"\n  DDXPlus 5개 질환 관련 엣지 (FDR<0.05):")
    for name, dcui in ddx.items():
        related = [e for e in sig005 if dcui in (e["cui_a"], e["cui_b"])]
        symptom_related = []
        for e in related:
            other = e["cui_b"] if e["cui_a"] == dcui else e["cui_a"]
            o_stys = cui_stys.get(other, set()) & ALLOWED_STYS
            o_name = cui_names.get(other, other)[:40]
            symptom_related.append((o_name, e["n_present"], e["jensen_score"], sorted(o_stys)))
        symptom_related.sort(key=lambda x: -x[1])
        print(f"\n  {name} ({dcui}): {len(related)}개")
        for o_name, np_, js, stys in symptom_related[:10]:
            print(f"    {o_name:<40s} n={np_:>2d} J={js:.2f} {stys}")

    # 상위 엣지
    print(f"\n  전체 상위 엣지 (FDR<0.05, 상위 20):")
    for e in sig005[:20]:
        a = cui_names.get(e["cui_a"], e["cui_a"])[:30]
        b = cui_names.get(e["cui_b"], e["cui_b"])[:30]
        print(f"    {a:30s} - {b:30s} n={e['n_present']:>3d} J={e['jensen_score']:.2f} q={e['q_value']:.4f}")

    # 저장
    output = {
        "config": {"model": MODEL, "allowed_stys": sorted(ALLOWED_STYS),
                   "blacklist": sorted(BLACKLIST_CUIS), "alpha": 0.6, "fdr": 0.05},
        "stats": {"n_docs": len(documents), "n_classifications": len(all_cls),
                  "n_edges_total": len(edges), "n_edges_sig": len(sig005),
                  "n_nodes": len(unique_cuis)},
        "edges": sig005,
        "all_edges": edges,
    }
    with open(DATA_DIR / "full_kg.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {DATA_DIR / 'full_kg.json'}")
    print("완료!")


if __name__ == "__main__":
    main()
