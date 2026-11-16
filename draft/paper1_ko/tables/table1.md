**Table 1. Accuracy for the true disease and number of evidences collected on the test set (134,529 patients)**

| Method | Knowledge | Input at start | Answers | Accuracy, % (95% CI) | Evidence collected, mean (95% CI) | Evidence collected, median | Stopped at the question limit, % |
|---|---|---|---|---|---|---|---|
| List-based method (P1) | Public lists | Initial evidence | Yes or no | 98.44 (98.38–98.51) | 10.35 (10.31–10.39) | 9 | 3.27 |
| P2, yes-or-no answers | Training frequencies | Initial evidence | Yes or no | 98.60 (98.54–98.66) | 7.27 (7.24–7.31) | 6 | 1.75 |
| P2, yes-or-no answers, age and sex | Training frequencies | Age, sex, initial evidence | Yes or no | 98.66 (98.60–98.72) | 7.12 (7.09–7.16) | 6 | 1.69 |
| P2, answer values | Training frequencies | Initial evidence | Answer values | 99.30 (99.26–99.35) | 5.10 (5.07–5.12) | 4 | 0.41 |
| P2, answer values, age and sex | Training frequencies | Age, sex, initial evidence | Answer values | 99.29 (99.24–99.33) | 4.97 (4.95–4.99) | 4 | 0.40 |
| BED re-implementation | Public lists and training frequencies^a)^ | Initial evidence | Answer values, BED simulator | 99.47 (99.43–99.51) | 5.48 (5.45–5.50) | 5 | 0.01 |

CI, confidence interval; P1, list-based method; P2, frequency-based method; BED, Bayesian experimental design.

Evidence collected includes the initial evidence. P1 and P2 stop when the largest posterior probability reaches 0.99, when no unasked evidence has an expected information gain of at least 10^−6^ nats, or after 30 questions. BED stops when no question has a utility of at least 0.01, or after 30 questions.

^a)^The statistics file of the training patients distributed with the BED code [@casande_code].
