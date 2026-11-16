**Supplementary Table S1. Additional metrics on the test set (%)**

| Method | Accuracy | Accuracy, official code | GTPA@3 | GTPA@5 | DDR | DDP | DDF1 | DSHM | PER |
|---|---|---|---|---|---|---|---|---|---|
| List-based method (P1) | 98.44 | 98.45 | 99.41 | 99.55 | 22.36 | 98.01 | 30.88 | 11.12 | 22.35 |
| Frequency-based method (P2) | 98.60 | 98.60 | 99.25 | 99.28 | 21.88 | 98.90 | 31.04 | 10.89 | 20.29 |
| P2 with age and sex | 98.66 | 98.66 | 99.28 | 99.30 | 21.87 | 98.93 | 31.05 | 10.87 | 20.04 |
| P2 with answer values | 99.30 | 99.30 | 99.58 | 99.58 | 21.58 | 99.45 | 30.92 | 10.75 | 22.77 |
| P2 with answer values, age and sex | 99.29 | 99.29 | 99.56 | 99.57 | 21.58 | 99.45 | 30.91 | 10.74 | 22.56 |
| BED re-implementation | 99.47 | 99.47 | 99.75 | 99.76 | 21.65 | 99.44 | 31.01 | 10.78 | 18.62 |

GTPA@k, the true disease is among the k most probable diseases; DDR, DDP and DDF1, recall, precision and their harmonic mean against the reference differential diagnosis; DSHM, harmonic mean of recall and precision for severe diseases; PER, share of the positive evidences of the patient that were collected.
Accuracy, official code: the accuracy as the official code computes it (Supplementary Methods 3).
