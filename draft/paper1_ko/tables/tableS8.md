**Supplementary Table S8. Effect of the answer rule on the validation set (132,448 patients)**

| Method | Answers | Accuracy, % (95% CI) | Evidence collected, mean (95% CI) |
|---|---|---|---|
| P2, answer values | Official rules | 99.26 (99.22–99.31) | 5.10 (5.07–5.12) |
| P2, answer values with record status | Official rules, and whether the record holds a categorical or multi-choice evidence | 99.36 (99.32–99.40) | 4.79 (4.77–4.82) |
| BED re-implementation | BED simulator | 99.44 (99.40–99.48) | 5.48 (5.46–5.50) |

| Comparison | Factor changed | Point | Accuracy difference, percentage points (95% CI) | Difference in evidence collected (95% CI) | Patients correct in only the first / only the second method |
|---|---|---|---|---|---|
| P2, answer values with record status − P2, answer values | Record status added to the answers | At the stop | +0.10 (+0.06 to +0.14) | −0.30 (−0.31 to −0.29) | 397 / 267 |
|  |  | After 5 questions | +1.84 (+1.74 to +1.94) | – | 3,792 / 1,354 |
|  |  | After 10 questions | +0.33 (+0.29 to +0.36) | – | 559 / 128 |
|  |  | After 20 questions | +0.12 (+0.10 to +0.14) | – | 165 / 7 |
|  |  | After 30 questions | +0.07 (+0.06 to +0.08) | – | 93 / 1 |
|  |  | Stop threshold 0.9 | +0.15 (+0.10 to +0.21) | −0.23 (−0.23 to −0.22) | 885 / 682 |
|  |  | Stop threshold 0.95 | +0.28 (+0.23 to +0.33) | −0.22 (−0.22 to −0.21) | 777 / 406 |
|  |  | Stop threshold 0.999 | +0.09 (+0.06 to +0.11) | −0.51 (−0.52 to −0.50) | 193 / 78 |
| BED re-implementation − P2, answer values with record status | BED and the rest of its simulator | At the stop | +0.08 (+0.05 to +0.11) | +0.68 (+0.66 to +0.70) | 250 / 143 |
|  |  | After 5 questions | −8.05 (−8.23 to −7.86) | – | 2,965 / 13,627 |
|  |  | After 10 questions | −1.07 (−1.13 to −1.01) | – | 144 / 1,559 |
|  |  | After 20 questions | −0.06 (−0.08 to −0.04) | – | 29 / 107 |
|  |  | After 30 questions | −0.02 (−0.03 to −0.01) | – | 32 / 57 |
| BED re-implementation − P2, answer values | BED and its simulator | At the stop | +0.18 (+0.14 to +0.22) | +0.38 (+0.36 to +0.40) | 479 / 242 |
|  |  | After 5 questions | −6.21 (−6.41 to −6.02) | – | 4,612 / 12,836 |
|  |  | After 10 questions | −0.74 (−0.81 to −0.68) | – | 503 / 1,487 |
|  |  | After 20 questions | +0.06 (+0.04 to +0.08) | – | 188 / 108 |
|  |  | After 30 questions | +0.05 (+0.03 to +0.07) | – | 123 / 56 |

CI, confidence interval; P2, frequency-based method; BED, Bayesian experimental design.

Record status: whether the patient record holds the evidence with any value, the default value included. The lower part gives the first method minus the second on the same patients; the stop uses the threshold 0.99 unless stated. Differences were computed before rounding. The last column counts patients with ties broken in the listed order of the public data.
