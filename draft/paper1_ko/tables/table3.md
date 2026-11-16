**Table 3. Results reported on DDXPlus and their evaluation conditions**

| Study | Knowledge used | Patients evaluated | Input at start | Answers | Question limit | Accuracy, % | Evidence collected, mean^a)^ |
|---|---|---|---|---|---|---|---|
| *Whole test set, up to 30 questions after the initial evidence* |  |  |  |  |  |  |  |
| AARLC [@aarlc]^b)^ | Training patients | 134,529 | Age, sex, initial evidence | Patient record with values | 30 | 99.21 | 6.73 |
| BASD [@ddxplus; @basd]^b)^ | Training patients | 134,529 | Age, sex, initial evidence | Patient record with values | 30 | 97.15 | 17.99 |
| Diaformer [@diaformer]^c)^ | Training patients | 134,529 | Initial evidence | Patient record with values | 30 | 91.81 | 18.45 |
| CASANDE [@casande]^c)^ | Training patients | 134,529 | Age, sex, initial evidence | Patient record with values, sub-question answers filled in | 30 | 98.8 | 19.53 |
| BED [@bed]^c)^ | Public lists and training frequencies | 134,529 | Initial evidence | BED simulator | 30 | 99.47 | 5.47 |
| This study, list-based method (P1) | Public lists | 134,529 | Initial evidence | Patient record, yes or no | 30 | 98.44 | 10.35 |
| This study, P2, answer values, age and sex | Training frequencies | 134,529 | Age, sex, initial evidence | Patient record with values | 30 | 99.29 | 4.97 |
| This study, BED re-implementation | Public lists and training frequencies | 134,529 | Initial evidence | BED simulator | 30 | 99.47 | 5.48 |
| *Other conditions* |  |  |  |  |  |  |  |
| COTCAgent [@cotcagent]^d)^ | Public lists | 1,000 | Initial evidence | Patient record, yes or no | 10 | 39.6–60.4 | 6.65–7.80 |
| CaRDI [@cardi]^e)^ | Training frequencies and an LLM prior | 490 | Age, sex, initial evidence | Patient record with values | 10 | 96.12–96.69 | 4.54–4.64 |
| MEDDxAgent [@meddxagent]^f)^ | LLM with retrieval and example patients | 100 | Age, sex, chief complaint | LLM patient simulator | 15 | 86 | NR |
| StreamBench [@streambench]^g)^ | LLM | 1,764 | Full patient record | None | 0 | 70.64–92.01 | NA |

AARLC, adaptive alignment of reinforcement learning and classification; BASD, baseline automatic symptom detection; BED, Bayesian experimental design; P1, list-based method; P2, frequency-based method; LLM, large language model; NR, not reported; NA, not applicable.

^a)^Including the initial evidence.

^b)^Model trained to predict the true disease; values from Table 3 of the DDXPlus paper [@ddxplus].

^c)^Model trained to predict the true disease, and BED; values from Table 1 of the CASANDE paper [@casande].

^d)^Two baselines re-implemented in that paper that score diseases from the public lists without learned weights (its Table 3 and Table A4); how the 1,000 patients were drawn is not stated; the reported 5.65–6.80 questions after the initial evidence are shown with the initial evidence added.

^e)^Patients from the test set; range over three LLMs; the reported 3.54–3.64 questions after the initial evidence are shown with the initial evidence added.

^f)^GPT-4o on the interactive task with up to 15 questions; the paper does not name the split.

^g)^Patients from the test set, 36 per disease; GPT-4o, from zero-shot prompting to the best reported method, which uses as examples its own earlier answers on test patients that feedback marked as correct.
