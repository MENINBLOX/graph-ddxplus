**Table 3. Results reported on DDXPlus and their evaluation conditions**

| Study | Knowledge used | Patients evaluated | Input at start | Answers | Question limit | Accuracy, % | Evidence collected, mean^a)^ |
|---|---|---|---|---|---|---|---|
| *Official interactive rules, whole test set* |  |  |  |  |  |  |  |
| AARLC [@aarlc]^b)^ | Training patients | 134,529 | Age, sex, initial evidence | Patient record | 30 | 99.21 | 6.73 |
| BASD [@basd]^b)^ | Training patients | 134,529 | Age, sex, initial evidence | Patient record | 30 | 97.15 | 17.99 |
| Diaformer [@diaformer]^c)^ | Training patients | 134,529 | Age, sex, initial evidence | Patient record | 30 | 91.81 | 18.45 |
| CASANDE [@casande]^c)^ | Training patients | 134,529 | Age, sex, initial evidence | Patient record, sub-question answers filled in | 30 | 98.8 | 19.53 |
| BED [@bed]^c)^ | Public lists and training frequencies | 134,529 | Initial evidence | BED simulator | 30 | 99.47 | 5.47 |
| This study, list-based method (P1) | Public lists | 134,529 | Initial evidence | Patient record, yes or no | 30 | 98.44 | 10.35 |
| This study, frequency-based method (P2) with answer values, age and sex | Training frequencies | 134,529 | Age, sex, initial evidence | Patient record with values | 30 | 99.29 | 4.97 |
| This study, BED re-implementation | Public lists and training frequencies | 134,529 | Initial evidence | BED simulator | 30 | 99.47 | 5.48 |
| *Other conditions* |  |  |  |  |  |  |  |
| CaRDI [@cardi]^d)^ | Training frequencies and an LLM prior | 490 from the test set | Age, sex, initial evidence | Patient record with values | 10 | 96.12–96.69 | 4.54–4.64 |
| MEDDxAgent [@meddxagent]^e)^ | LLM | 100 from the StreamBench patients | Age, sex, chief complaint | LLM patient simulator | 15 | 86 | NR |
| StreamBench [@streambench]^f)^ | LLM | 1,764 from the test set | Full patient record | None | 0 | 70.64–92.01 | NA |

LLM, large language model; NR, not reported; NA, not applicable.
^a)^Including the initial evidence.
^b)^Model trained to predict the true disease; values from Table 3 of the DDXPlus paper [@ddxplus].
^c)^Model trained to predict the true disease, and BED; values from Table 1 of the CASANDE paper [@casande].
^d)^Range over three LLMs; the reported 3.54–3.64 questions after the initial evidence are shown with the initial evidence added.
^e)^GPT-4o on the interactive task after 15 questions.
^f)^GPT-4o, from zero-shot prompting to the best reported method.
