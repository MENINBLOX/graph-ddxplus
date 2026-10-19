"""Deterministic normalization of CATEGORICAL attributes to the controlled vocabulary.
Applied AFTER IE extracts the raw source expression (kept faithful), turning diverse
source wordings -- adjectives, NRS 0-10 numbers, grades -- into the schema vocab.

severity -> {mild, moderate, severe}   (NRS 1-3 / 4-6 / 7-10 per Bates/HPO ordinal mapping)
onset    -> {sudden, gradual}

normalize_* returns the mapped value or "" if the raw expression carries no gradable signal.
No model/LLM; pure lexical + numeric rules. Importable by the normalization pipeline."""
import re

SEV_LEX = {
    "mild": "mild", "mildly": "mild", "slight": "mild", "slightly": "mild", "minimal": "mild",
    "minimally": "mild", "faint": "mild", "low-grade": "mild", "trivial": "mild", "subtle": "mild",
    "moderate": "moderate", "moderately": "moderate", "intermediate": "moderate",
    "severe": "severe", "severely": "severe", "marked": "severe", "markedly": "severe",
    "florid": "severe", "massive": "severe", "profound": "severe", "extensive": "severe",
    "serious": "severe", "significant": "severe", "high-grade": "severe", "intense": "severe",
    "excruciating": "severe", "debilitating": "severe", "life-threatening": "severe", "acute": "",  # acute=timing not severity
}
# numeric grade words -> ordinal (grade 1/2/3, mild/mod/severe)
GRADE = {"1": "mild", "i": "mild", "2": "moderate", "ii": "moderate", "3": "severe", "iii": "severe",
         "4": "severe", "iv": "severe"}

def _nrs_to_grade(n):
    if 1 <= n <= 3: return "mild"
    if 4 <= n <= 6: return "moderate"
    if 7 <= n <= 10: return "severe"
    return ""

def normalize_severity(raw):
    if not raw: return ""
    s = str(raw).strip().lower()
    # NRS "x/10" or "x out of 10"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10', s)
    if m: return _nrs_to_grade(round(float(m.group(1))))
    # bare 0-10 number
    m = re.fullmatch(r'(\d+(?:\.\d+)?)', s)
    if m:
        n = float(m.group(1))
        if n <= 10: return _nrs_to_grade(round(n))
    # "grade N" / "stage N"
    m = re.search(r'(?:grade|stage|class)\s*([1-4]|i{1,3}v?)', s)
    if m and m.group(1) in GRADE: return GRADE[m.group(1)]
    # lexical: scan words, first gradable hit
    for w in re.findall(r'[a-z-]+', s):
        if w in SEV_LEX and SEV_LEX[w]: return SEV_LEX[w]
    return ""

ONSET_LEX = {
    "sudden": ["sudden", "suddenly", "acute", "acutely", "abrupt", "abruptly", "rapid", "rapidly",
               "immediate", "immediately", "explosive", "thunderclap", "instant", "instantaneous",
               "within minutes", "within hours", "overnight", "paroxysmal"],
    "gradual": ["gradual", "gradually", "insidious", "insidiously", "slow", "slowly", "progressive",
                "progressively", "chronic", "chronically", "over days", "over weeks", "over months",
                "over years", "long-standing", "longstanding", "creeping", "stepwise"],
}

def normalize_onset(raw):
    if not raw: return ""
    s = str(raw).strip().lower()
    if s in ("sudden", "gradual"): return s
    for grade, syns in ONSET_LEX.items():
        if any(sy in s for sy in syns): return grade
    return ""


if __name__ == "__main__":
    tests_sev = ["severe", "7/10", "3", "grade III", "8 out of 10", "moderate", "mild", "2", "marked",
                 "significant", "5", "0", "excruciating", "low-grade", "acute", ""]
    tests_ons = ["sudden", "acute onset", "insidious", "developed over weeks", "rapidly progressive",
                 "chronic", "gradual", "abrupt", ""]
    print("SEVERITY:")
    for t in tests_sev: print(f"  {t!r:18} -> {normalize_severity(t)!r}")
    print("ONSET:")
    for t in tests_ons: print(f"  {t!r:22} -> {normalize_onset(t)!r}")
