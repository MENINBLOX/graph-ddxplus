**Supplementary Table S2. Additional metrics on the test set (%)**

| Method | Accuracy | Accuracy, official code | GTPA@3 | GTPA@5 | DDR | DDP | DDF1 | DSHM | PER |
|---|---|---|---|---|---|---|---|---|---|
| List-based method (P1) | 98.44 | 98.45 | 99.41 | 99.55 | 22.36 | 98.01 | 30.88 | 11.12 | 22.35 |
| P2, yes-or-no answers | 98.60 | 98.60 | 99.25 | 99.28 | 21.88 | 98.90 | 31.04 | 10.89 | 20.29 |
| P2, yes-or-no answers, age and sex | 98.66 | 98.66 | 99.28 | 99.30 | 21.87 | 98.93 | 31.05 | 10.87 | 20.04 |
| P2, answer values | 99.30 | 99.30 | 99.58 | 99.58 | 21.58 | 99.45 | 30.92 | 10.75 | 22.77 |
| P2, answer values, age and sex | 99.29 | 99.29 | 99.56 | 99.57 | 21.58 | 99.45 | 30.91 | 10.74 | 22.56 |
| BED re-implementation | 99.47 | 99.47 | 99.75 | 99.76 | 21.65 | 99.44 | 31.01 | 10.78 | 18.62 |
| BED, reported in the CASANDE paper [@casande] | NR | 99.47 | NR | NR | NR | NR | 31.01 | 10.78 | 18.62 |

P1, list-based method; P2, frequency-based method; BED, Bayesian experimental design. NR, not reported.

Accuracy, official code: the accuracy as the official code computes it (Supplementary Methods 5); the accuracy reported for BED was computed by the BED code with the same rule. GTPA@k, the true disease is among the k most probable diseases after removing diseases with probability 0.01 or less, with ties shared as for accuracy. DDR and DDP, mean over patients of the recall and precision against the reference differential diagnosis; DDF1, mean over patients of their harmonic mean. DSHM, mean over patients of the harmonic mean of the rule-in and rule-out rates of severe diseases [@casande]. PER, share of the positive evidences of the patient that were collected, the initial evidence included.
