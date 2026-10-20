## 3. 방법

### 3.1 설계 경계

진단 점수에 영향을 줄 수 있는 정보의 범위는 v2 구축 전에 정했고, v3 수집에서 원인 유전자 기호를 더했다(표 1). 점수에 영향을 줄 수 있는 정보는 다섯 가지다. 첫째는 공개 문헌에서 추출한 결과, 둘째는 UMLS·HPO·MONDO의 개념–개념 관계로 이루어진 어휘·계층·동의어, 셋째는 벤치마크 근거 코드를 표준 개념으로 옮기는 어휘 사상(크로스워크), 넷째는 평가 대상 환자 본인의 입력이다. 다섯째는 멘델 질환의 원인 유전자 기호이며, 검색어와 주제 문서 판정에만 쓴다. 추출 결과는 출처 문서 ID와 함께 보존한다.

HPO 주석(phenotype.hpoa), Orphanet, OMIM clinical synopsis 같은 질환–표현형 주석은 질환과 소견의 연결을 직접 담는다. 그래서 주 시스템의 점수에는 쓰지 않고, 별도 Neo4j 인스턴스에 적재한 비교 지식원(3.7절)과 참조로만 쓴다. 벤치마크가 배포한 질환–증상 정보(DDXPlus 조건 파일의 증상 목록, SymCat 빈도), 벤치마크 학습 라벨, 벤치마크 환자 집단 통계도 점수에 쓰지 않는다. 벤치마크 정답 질환 목록은 후보로 쓰지 않고 정답 대조에만 쓴다. 수집 대상 질환과 수집 상한은 온톨로지와 유전자–질환 자원에서 같은 규칙으로 정해 모든 질환에 적용한다. 원인 유전자가 있는 질환은 유전자 검색 문서를 더 받는다. 설정은 개발 세트에서만 고르며, 확증 세트는 동결 후 한 번만 평가한다(3.6절). 진단 추론 시점에는 LLM을 호출하지 않는다.

벤치마크 증례의 원출처 논문은 수집에서 제외했다. v2 수집은 Phenopacket-store 0.1.27이 기록한 원출처 PMID 1,733편을 제외했다. v3 신규 수집은 0.1.27 이후 신규 증례의 원 논문까지 더한 1,834편을 모든 검색 경로에서 제외했다. v3 그래프는 v2 문서도 함께 쓴다. v2 문서는 1,733편 기준으로만 걸렀으므로 0.1.27 이후 증례의 원 논문 9편이 v2 문서에 남았고, 이 논문의 증례 32건은 시간 분할 평가에서 뺐다(3.9절). 원출처를 제외한 뒤에도 같은 증례가 재보고나 리뷰를 통해 남는지는 데이터 기반으로 따로 점검한다(3.9절).

원인 유전자 기호는 v3 수집의 검색어와 주제 문서 판정에 쓴다(3.2절). 유전자 목록은 HPO가 배포하는 genes_to_disease.txt(2026-10-14 내려받음)에서 연관 유형이 MENDELIAN인 질환–유전자 쌍으로 만들었다. 이 자원은 표현형을 담지 않지만 큐레이션 자원이므로 표 1에 따로 표시했다.

수집 확장 파일럿(4.5절)은 사전테스트 질환 300개 가운데 Phenopacket-store 정답 질환 22개와 무작위 질환 28개, 모두 50개 질환에만 문헌을 더했다. 정답 질환이 섞인 일부 질환만 깊게 수집했으므로 이 파일럿은 균일 수집 원칙을 벗어난다. 따라서 파일럿 그래프는 어떤 주 결과에도 쓰지 않았고, 모든 질환에 같은 상한을 둔 v3 재수집(3.2절)으로 대체했다. 파일럿의 이득이 전 질환 규모에서 재현되지 않은 결과는 4.5절에 보고한다.

Table 1. Boundary of information that may influence scores of the proposed system

| Allowed | Forbidden |
|---|---|
| Extraction results from public literature (source document IDs kept) | Disease–phenotype annotations (HPO annotation, Orphanet, OMIM clinical synopsis)^a^ |
| Concept vocabularies, hierarchies and synonyms (concept–concept relations in UMLS, HPO, MONDO) | Disease–symptom information distributed with benchmarks (DDXPlus condition file, SymCat frequencies) |
| Vocabulary mapping (crosswalk) from benchmark evidence codes to standard concepts | Benchmark training labels |
| Input of the patient under evaluation | Benchmark population statistics |
| Causal-gene symbols of Mendelian diseases (HPO genes_to_disease), only as search terms and on-topic title marks^b^ | Using benchmark gold-diagnosis lists as candidates |
| | Collecting literature in depth only for gold diagnoses^c^ |
| | Using confirmation data for configuration selection |
| | Source papers of benchmark cases (v2 collection: 1,733 PMIDs; v3 collection: 1,834 PMIDs)^d^ |
| | Calling an LLM at inference time |

^a^ HPO annotations and PrimeKG are loaded only into separate comparator Neo4j instances queried by the same ranker (Section 3.7); they never enter the proposed system's scores. ^b^ Gene–disease links, not disease–phenotype links. Time-split builds use them only for diseases annotated in HPOA release 2022-12-15 (Section 3.9). ^c^ The 50-disease collection pilot, which added literature to 22 gold diseases, is not used in any main result; its results are reported only as a non-replicated pilot (Section 4.5.1). ^d^ v2 documents, which the v3 graphs also use, were filtered with the 1,733 PMIDs only; 9 source papers of 32 later cases remain there, and those cases are excluded from the time-split evaluation (Section 3.9).

### 3.2 질환 목록과 문헌 수집

이 장에서 문서는 출처를 기록하는 단위(논문 본문, 초록, 한 논문의 표 묶음, 도서 장)이고, 논문은 원 논문 하나다. 문헌은 수집한 문서 전체를 가리킨다. v2 문헌은 v2 수집의 문서이고, v3 문헌은 v3 재수집에서 새로 얻은 문서다. v3 그래프는 두 문헌을 합친 v2+v3 문헌으로 만든다.

**질환 목록.** 후보 질환 집합은 UMLS 출처 질환에, 그 집합에 대응하는 CUI가 없는 MONDO 질환을 더해 구성했다(두 체계의 합집합). Phenopacket-store 0.1.27의 환자–질환 10,377쌍 가운데 9,522쌍은 두 체계 모두에, 792쌍은 MONDO에만, 25쌍은 UMLS에만 대응 개념이 있었고, 38쌍(0.4%)은 어느 쪽에도 없었다. 한 체계만 쓰면 정답이 더 많이 빠지므로 두 체계를 합쳤다. 수집 대상은 UMLS 출처 질환 35,590개와, 그 집합에 대응하는 CUI가 없는 MONDO 질환 16,304개, 모두 51,894개다. UMLS 출처 질환은 UMLS 질환 개념 35,281개와, 이전 지식 그래프에서 넘어온 GeneReviews 장 309개로 이루어진다. MONDO는 2026-09-01판(질환 클래스 32,111개, OMIM 교차참조 9,904개, UMLS 교차참조 21,661개)을, UMLS는 2025AB판을, HPO는 2026-06-23판을 썼다. MONDO의 '비인간 동물 질환'(MONDO:0005583) 하위 8,319개는 사람의 진단 대상이 아니므로 후보에서 뺐고, 후보 질환은 43,575개(UMLS 출처 35,590개, MONDO 7,985개)다. 이 제외 규칙은 구 시스템(Qwen3-4B IE, 그래프 v2.0)의 간선 표본 200개를 LLM 판정자 1인이 판정한 결과에서 정했다. 이 판정에서 비인간 질환 노드의 간선 39개 중 33개가 근거 문장에 지지되지 않았다. 독립 판정은 아니다. 노드 키는 UMLS 질환 개념의 CUI이고, GeneReviews 장 309개는 장 ID를, 그 집합에 대응하는 CUI가 없는 MONDO 질환은 MONDO ID를 쓴다. OMIM·Orphanet·UMLS 교차참조는 정답 사상을 위해 보존한다. 질환 계층은 MONDO is_a와 UMLS 질환 계층을 합친 것이며, 두 끝이 모두 후보 질환인 쌍은 45,118개다.

**v2 문헌 수집.** 질의 형태는 질환명, 이름 변형, 온톨로지 동의어(MONDO 동의어, 그리고 UMLS 표준 용어 동의어 최대 5개, unspecified·NOS 제외)다. PMC 오픈액세스에서 리뷰와 증례보고를 검색했고, 모든 형태로 제목 검색을 먼저 한 뒤 결과가 없을 때만 전체 필드를 검색했다. 각 단계에서는 앞 형태부터 질의해 결과가 처음 나온 형태를 썼다. UMLS 동의어는 첫 수집에서 문헌 0건이던 UMLS 질환 8,875개를 다시 수집하려고 더했고, 그중 2,489개가 문헌을 얻었다. 질환당 상한은 리뷰 15편과 증례보고 10편이다. 처음에는 리뷰 6편·증례 4편으로 수집했고, 문헌 깊이 점검 뒤 모든 질환의 상한을 함께 올려 추가분만 수집했다. 이 확장 수집은 모든 질환을 같은 질의 목록과 순서로 다시 질의했다. 논문마다 임상 관련 절의 본문을 12,000자까지 저장했다. 임상 관련 절은 제목이 없거나, 제목에 clinical, sign, symptom, presentation, manifestation, feature, diagnosis, history, physical, examination, finding, course, epidemiology, risk 중 하나가 들어가고 method, statistic, acknowledgment, funding, conflict, supplement, reference 등이 들어가지 않은 절이다. StatPearls와 GeneReviews는 NCBI LitArch 번들(2026-09/10판)에서 장 제목이 질환의 이름 형태와 일치하는 장을 연결했고, 장마다 24,000자까지 저장했다. NCBI Bookshelf 웹은 스크립트 접근에 CAPTCHA를 요구해 쓰지 않았다. 문서마다 PMID, PMCID(도서는 장 ID), 제목, 출판연도, 라이선스를 저장했다.

요청 실패(빈 응답, 속도 제한 응답)는 '문헌 없음'과 구분해 최대 6회 재시도했고, 끝내 실패한 질환은 저장하지 않아 다음 실행에서 다시 수집했다. 이름 변형 함수가 6자 미만의 이름을 버려 질의가 한 번도 나가지 않던 결함을 고친 뒤, 해당 25개 질환(Gout, Croup, Mumps 등)을 다시 수집했다. v2 코퍼스의 규모는 4.1절 표 2에 있다.

**문서 단위 규칙.** 다음 두 규칙은 벤치마크 정보 없이 코퍼스만으로 정의했고 모든 질환에 같게 적용한다. 첫째, 질환별 수집 목록에 모두 20회 이상 나타난 논문(한 질환 목록에 두 번 나오면 2회로 센다)은 특정 질환의 근거로 보기 어려우므로 모든 질환에서 뺀다(일반 문헌 규칙). 기준 20은 코퍼스만 보고 미리 정했다. 논문의 표 문서는 원 논문과 같은 논문으로 센다. 둘째, 문서 제목이 그 질환의 이름 형태(질의 이름, 이름 변형, 동의어) 중 하나를 포함하면 그 문서를 그 질환의 주제 문서로 표시한다. 전체 필드 검색은 다른 질환을 다루면서 질환명을 지나가듯 언급한 문서도 가져오기 때문이다. 이 규칙은 구 시스템의 출처 감사에서 관찰한 실패 유형(다른 질환을 다룬 문서)에서 도출했다. v3 그래프에서는 원인 유전자 기호가 제목에 있는 문서도 주제 문서로 인정한다. 단일 유전자 아형은 아형 이름 대신 원인 유전자 이름으로 보고되는 경우가 많기 때문이다. 주제 표시는 간선을 지우지 않고 채점의 가중에만 쓰인다(3.5절).

**v3 재수집.** 수집을 다시 한 근거는 두 가지 개발 측정이다. 설계 D 동결 그래프(v2 문헌)의 재현율 손실 분해에서 Phenopacket 증례의 큐레이션 소견 중 최대 67.0%(55,465건, 문자열 일치 기준 상한)가 수집 단계에서 빠졌다. 재수집 뒤 같은 분해를 v2+v3 문헌의 그래프(S-mix, S-llm, S-dict)에서 다시 했다(4.3절). HPOA가 주석한 OMIM 질환 무작위 300개(시드 7)의 사전테스트에서는 큐레이션 소견 이름이 수집 텍스트에 나오는 비율이 v2 문헌의 18.4%(대조 2.9%)에서, PMC 문헌 확대(본문 전체, 절단 없음), 초록, 원인 유전자 검색을 더하면 58.1%(대조 17.8%)로 올랐다. 대조는 다른 무작위 질환의 큐레이션 소견이 나오는 비율로, 텍스트 양만으로 생기는 출현을 잰다. 58.1%는 본문 전체를 절단 없이 쓴 값이며, 본문을 임상 절로 자른 v3 주 그래프의 텍스트에 대한 값은 아니다. 사전테스트(무작위 40개 질환)에서 PMC 논문의 12,000자 임상 절 텍스트에 없던 큐레이션 소견 181개 중 58.6%는 표에 있었다. v2 수집은 단락(`<p>`)만 저장해 표를 버렸다. 같은 PMC 리뷰·증례 30+30편에서 재현율 대 대조의 비는 임상 절과 표를 쓰면 3.8(40.9% 대 10.8%), 본문 전체와 표를 쓰면 3.1(51.1% 대 16.5%)이었다. 그래서 v3는 본문 전체 대신 임상 절과 표를 추출 대상으로 삼았다.

v3 상한은 모든 질환에 같다. PMC 오픈액세스 리뷰 30편과 증례보고 30편을 받아 임상 절(12,000자), 본문 전체, 표를 저장했다. PubMed에서는 제목·초록 검색으로 초록 50편을 받았다. 원인 유전자는 질환당 최대 2개까지 검색했고, 유전자마다 초록 50편과 PMC 오픈액세스 논문 20편을 받았다. 유전자 검색어에는 patient*, case, famil* 중 하나를 함께 요구했다. 질의어는 질환명, 이름 변형, 동의어 가운데 최대 8개 형태이며, 앞 형태부터 질의해 결과가 처음 나온 형태를 쓴다. 증례 원 논문 1,834편은 모든 검색 경로에서 제외했다. 수집에는 중단을 포함한 경과 시간 약 25시간, 실제 작업 시간 약 16시간(약 45질환/분)이 걸렸다.

v3에서 새로 추출한 것은 v2에 없는 것뿐이다. 대상은 v2에 없던 PMC 논문의 임상 절, v3 검색이 받은 모든 PMC 논문의 표, v2에 없던 초록이다. 표는 v2에 이미 있던 논문도 v3 검색이 다시 받은 경우에만 있다. 무작위 1,000개 질환 표본에서 v2 PMC 논문의 약 97%가 다시 검색되었다. 같은 논문의 PMC 본문이 이미 문서로 있으면 그 초록은 뺐다. 출처를 문서 단위로 기록하므로 초록 하나와 논문 하나의 표 묶음은 각각 별도 문서다. 표는 캡션과 행을 이어 쓴 텍스트로 저장했다. 유전자 검색으로 찾은 문서에는 그 경로를 표시했다. v3 문서에는 PMID, 문서 ID(PMCID 또는 PMID), 종류(본문, 표, 초록), 제목, 출판연도를 저장했으며, 별도의 PMCID 필드와 라이선스 필드는 없다. 주 그래프는 PMC 본문을 12,000자 임상 절로 자른 판을 쓴다. 본문 전체를 읽는 판(접미어 full)은 문헌이 다르므로 설계 E의 비교 시스템(3.6절)에서 뺐다. 이 판의 개발 결과는 4.3절에 있다. v3에서 새 문서를 얻은 질환은 43,575개 중 35,006개이고, 새 텍스트는 고유 문서 기준 21.4억 자(질환–문서 연결마다 더하면 42.6억 자)다(표 2). 2022 절단 빌드(접미어 _y2022)는 2022년까지 출판된 문서만 쓰고, 유전자 검색 문서와 유전자 제목의 주제 인정은 HPOA 2022-12-15판이 주석한 질환에만 쓴다(3.9절).

### 3.3 추출: LLM·사전·혼합

같은 문헌에 네 가지 추출 방식을 적용했다. LLM 추출, LLM 추출을 쓰지 않는 사전 추출, 두 결과의 합집합, LLM이 고른 구절 안에서 사전이 이름을 붙이는 혼합 추출이다. 네 방식의 출력은 같은 형식(질환, 문서, 구절, 근거 문장)이다. v2+v3 문헌의 네 시스템(S-llm, S-dict, S-union, S-mix)은 같은 빌더와 같은 빌더 옵션으로 그래프가 된다. 그래프 사이의 차이는 추출과 개념 연결에서 생긴다. 다만 채점에서는 주제 밖 간선 가중치만 그래프마다 개발 세트에서 다시 골랐고, 구절 표는 S-dict가 v2 구절 표를, 나머지 셋이 v3 구절 표를 쓴다. 시스템별 설정은 표 5에 있다.

**LLM 추출.** IE 모델은 gemma-4-E4B-it다. 주 IE 모델은 처음에 Qwen3-4B-Instruct-2507로 사전 지정했다. 라벨 없는 시범(40개 질환, 317창)에서 Qwen3가 질환당 고유 개념을 더 많이 냈기 때문이다(43.7개 대 31.3개). 같은 시범에서 gemma는 창당 15.3개 항목을 냈고 그중 96.3%가 원문에서 확인되었다. Qwen3는 창당 29.8개, 76.3%였다. v1 단계에서는 시험 세트 결과로 모델을 바꾸지 않았다. 사후 교체는 시험 기반 선택이 되기 때문이다. 설계 D에서 그 시험 세트 9개를 개발 세트로 바꾼 뒤 gemma 그래프를 주 시스템으로 정했다. 이어서 같은 v2 문헌·같은 연결기로 만든 Qwen3 그래프의 개발 선택을 마쳤고, 개발 9개 세트 매크로 lineage Top-10/Top-1이 gemma보다 낮아(Qwen3 29.59/11.02% 대 gemma 30.56/12.64%, 각 그래프에서 따로 고른 채점 설정, 대응 검정 없음) gemma를 유지했다. 이 9개 세트는 v1에서 시험 세트로 열람되었고, 그때 이미 gemma 그래프가 9개 중 8개에서 Top-10이 높았다. 추론은 RTX 4090(24GB) 1장에서 vLLM 배치 추론(v3 실행은 vLLM 0.25.0, bfloat16, 문맥 8,192토큰, 출력 최대 2,048토큰)으로 했다. 디코딩은 temperature 0의 탐욕 디코딩이다. 모델은 문서 창마다 한 번만 통과하며, 프롬프트에는 예시가 없고 추출 개수 제한도 없다.

프롬프트에는 질환명과 문서 창 하나만 들어간다. 프롬프트는 텍스트가 그 질환과 연관 짓는 항목을 모두 나열하도록 지시한다. 항목은 환자의 임상 소견(증상, 신체 징후, 검사실·영상·기타 검사 소견, 합병증)과 관련 요인(위험인자, 노출, 과거·동반 질환, 가족력, 약물, 연령·성별 경향)이다. 프롬프트는 각 항목을 원문에서 그대로 복사한 6단어 이하 구절 하나로 내게 하고, 문장·절 출력과 의역·번역·축약·병합을 금지한다. 텍스트가 없다고 한 항목, 치료·시술·용량, 다른 질환에 귀속된 항목은 빼도록 하되, 치료 문장 안의 소견은 남기도록 한다. 출력 형식은 줄마다 JSON 객체 {"t": 구절, "k": 종류}이고, 종류는 symptom, sign, test, complication, risk, history, exposure, medication, demographic 중 하나다. 프롬프트 전문은 공개 코드에 있다.

문서는 6,000자 창으로 나누고 인접 창을 400자 겹친다. 모델이 낸 구절은 그 창에서 대소문자를 무시하고 찾았을 때만 남긴다. 구절이 들어 있는 문장을 근거 문장으로 함께 저장하며, 문장은 앞에서부터 600자까지 저장한다. 표의 행처럼 문장 경계가 없어 그 600자 안에 구절이 없으면, 빌더가 원문에서 구절을 중심으로 600자를 다시 자른다. 이 원문 위치 확인으로 모든 항목이 원문 문장에 결정론적으로 연결된다.

v2 문헌에서는 51,894개 질환의 모든 창을 처리했다(문헌이 없는 12,886개 질환 제외). 원문에서 위치가 확인된 항목은 533만 개였다(모델 출력 560만 개). v3 문헌에서는 같은 모델과 같은 프롬프트로 v3 추출 대상만 처리했다. 43,575개 질환을 처리했다. 새 문서가 있는 35,006개 질환(창 1,866,063개) 가운데 34,654개에서 원문 위치가 확인된 항목이 나왔고, 그 항목은 1,232만 개였다(모델 출력 1,323만 개). v2와 v3를 합친 LLM 항목은 17,654,170개다. v3 실행은 CUDA 그래프와 배치 512를 썼다. 사전 측정에서 CUDA 그래프와 배치 556은 초당 9창으로 eager 모드·배치 64의 3.2배였고, 항목 수는 ±1% 안에서 같았다.

**구절 표 연결.** LLM 구절은 고유 구절마다 한 번씩 개념에 연결해 구절 표로 저장하고, 빌더가 이 표를 재사용한다. 규칙은 다음 순서로 적용한다. (1) 8단어를 넘는 구절은 버린다. (2) NegEx식 부정 표지(no, not, without, absent, negative for, ruled out, denied, free of, lack of, unremarkable, normal 등)가 있는 구절은 버린다. (3) 연결기가 후보 개념 하나를 고른다. 연결기는 구절에서 정도·발병 수식어를 지운 뒤, scispaCy(0.6.2, en_core_sci_lg 0.5.4) UMLS 후보 생성기의 후보 가운데 문자 n-gram 유사도가 0.70 이상인 것을 구절 전체, 끝 두 단어, 끝 단어 순서로 찾는다. 유사도가 0.88 미만인 후보는 개념 이름과 내용어를 하나 이상 공유해야 하며, 허용 의미유형 후보를 우선한다. 조건을 만족하는 후보가 없으면 버린다. (4) 고른 개념의 의미유형을 UMLS MRSTY 원본에서 읽어, 허용 의미유형 28개에 들지 않으면 버린다. 허용 유형은 소견·질환·검사 결과 13개와 위험인자·노출을 위한 약물·화학물질·행동·직업·인구집단·체내 물질·생물체 등의 유형이다. (5) 개념의 이름이나 별칭이 구절 안에 연속으로 나타나고 구절 내용어의 50% 이상을 덮을 때만 채택한다(근거 규칙). 연결기가 끝 단어로 고른 일반어 개념이 구절 전체를 대표하지 않게 하기 위해서다. 연결된 개념 중 검사·시술·진단 절차·의료 활동(T058–T061)은 소견이 아니므로 빌더가 뺀다. v3 LLM 그래프는 v2 구절 표를 같은 연결기로 v3 구절까지 넓힌 표를 쓴다. 이 표에서 v3 구절에 개념이 붙는 비율은 고유 구절 기준 17.0%(748,154/4,408,241), 항목 기준 37.7%였다. v2 항목은 같은 표에서 고유 구절 기준 16.8%, 항목 기준 35.1%였다.

**빌더의 연결 규칙.** 빌더는 구절 표가 붙인 개념 외에 두 가지 연결을 더할 수 있다. 등위 구절 분해는 한 항목을 쉼표, 세미콜론, 빗금, and, or, with에서 나누고, 앞의 정도어(severe, mild, progressive 등)를 뗀 부분들을 구절 표로 연결한다. LLM이 "arachnodactyly, kyphosis or scoliosis"처럼 여러 소견을 한 항목으로 내는 경우가 있기 때문이다. HPO 이름 연결은 항목이나 그 부분이 HPO 레이블 또는 exact 동의어와 일치하면, 환자 쪽 평가에 쓰는 HPO–UMLS 대응의 CUI로 연결한다. UMLS 연결기가 같은 이름에 다른 CUI를 고르는 경우(failure to thrive: C0015544와 C0231246)를 메우기 위해서다. v2+v3 문헌의 모든 시스템은 등위 구절 분해, HPO 이름 연결, 원인 유전자 제목 주제 인정을 함께 쓴다. 설계 D의 동결 시스템(S-D)은 셋 다 쓰지 않은 v2 그래프다. HPO 한정어 규칙은 변형으로만 시험했고 주 시스템에는 쓰지 않는다(4.4절). 항목은 그 질환이 보유한 문서를 인용할 때만 쓰며, 일반 문헌 규칙으로 빠진 논문의 문서에서 나온 항목은 뺀다.

**LLM 경로의 연결 손실.** v2와 v3를 합친 LLM 항목 17,654,170개 중 구절 표로 개념이 붙은 것은 36.9%(6,518,369개)다. 연결되지 않은 사유를 전체 항목 대비로 보면 근거 부족 43.0%, 의미유형 12.2%, 후보 개념 없음 4.0%, 부정 2.1%, 8단어 초과 1.7%, 표에 없음 0.1%다(부록 표 18). 근거 부족은 연결된 개념의 이름·별칭이 구절 안에 연속으로 나타나지 않거나 구절 내용어의 절반 미만만 덮는 경우(규칙 5)이며, "right ptosis"와 "serum creatine kinase was elevated"가 그 예다. 같은 연결 파이프라인으로 v2 문헌을 비교하면, 문헌 필터를 통과한 항목 중 개념에 연결되는 비율은 LLM 33.9%, HPO 사전 99.96%, UMLS 사전 98.4%였다. 그래프 간선까지 오는 항목의 비율은 LLM 22.5%(등위 분해·HPO 이름 연결을 쓰면 27.7%), HPO 사전 68.1%, UMLS 사전 67.2%였다.

**사전 추출.** LLM 추출을 쓰지 않는 기준으로, 사전 매칭을 LLM과 같은 문서의 모든 문장에 적용했다. 최장 일치를 먼저 찾으며 일치는 최대 8토큰이다. 일치 앞 5토큰 안에 부정 단서(no, not, without, absent, absence, negative, denied, denies, normal, lack)가 있으면 그 일치를 버린다. HPO 사전은 HPO 레이블과 exact 동의어로 이루어진다. HPO 사전 항목은 CUI 없이 저장되며, 빌더가 HPO 이름 연결과 구절 표로 개념을 붙인다. UMLS 사전은 소견 의미유형 13개(T184, T047, T048, T046, T191, T019, T190, T037, T020, T049, T050, T033, T034)에 속한 개념의 모든 영어 문자열(4자 이상, 8단어 이하)로 이루어진다. 매칭은 정규화 문자열의 정확 일치이며, QuickUMLS와 달리 유사도 임계값을 쓰지 않는다. 한 문자열이 여러 개념에 걸리면 그 문자열이 선호명인 개념 하나를 고른다. 그런 개념이 하나로 정해지지 않으면 CUI 없이 저장하고, 빌더가 다른 구절처럼 HPO 이름과 구절 표로 연결한다. 사전 그래프의 구절 표는 v2 구절 표다. CUI 없는 항목을 연결하는 v2 구절 표는 LLM 구절로 만들었으므로, 사전 그래프도 이 경로에서는 LLM 산출물에 기댄다(4.4.7절). S-dict는 같은 문헌의 UMLS 사전 그래프이며, HPO 사전 그래프는 변형으로만 보고한다. v2 문헌에서 항목 수는 LLM 533만 개, HPO 사전 788만 개, UMLS 사전 2,130만 개였다. 그래프 간선에 연결된 언급 수는 각각 120만 개(등위 분해·HPO 이름 연결을 쓰면 176만 개), 621만 개, 1,493만 개였다.

**합집합.** 합집합 그래프는 LLM 항목과 사전 항목을 함께 같은 빌더에 넣어 만든다. S-union은 LLM 항목과 모든 문장의 UMLS 사전 항목을 합친 그래프다.

**혼합 추출(S-mix).** 혼합 추출은 LLM 경로의 연결 손실을 사전으로 메운다. 구절 표가 개념을 붙인 LLM 항목은 그대로 둔다. 연결되지 않은 항목 가운데 부정, 8단어 초과, 표에 없음 사유를 뺀 10,455,243개는 그 LLM 구절 안에서 찾은 UMLS 소견 문자열로 교체한다. 색인과 부정 규칙은 UMLS 사전 추출과 같고, 부정 단서는 구절 안에서만 본다. 어떤 텍스트가 그 질환의 소견인지는 LLM이 정하고, 사전은 이름만 붙인다. 교체 대상 중 5,460,459개(52%)에서 사전 문자열이 일치해 새 항목 7,475,303개가 생겼고, 일치가 없는 4,994,784개는 빠졌다. 교체 방식이므로 빌더가 원래 구절에 등위 분해·HPO 이름으로 붙이던 연결도 함께 지워진다. 교체 대상 중 130만 항목에 이런 연결이 198만 개 있었다. 그중 131만 개는 같은 항목의 사전 일치가 같은 CUI로 다시 붙이므로, 순손실은 67만 개다. 사전 일치가 없는 24만 항목이 연결 27만 개를 모두 잃고, 사전 일치가 있는 항목 가운데 35만 개가 연결 40만 개를 잃는다. "increased"나 "high" 같은 일반어도 붙지만 별도 필터는 두지 않았고, 그런 개념은 IDF가 가중을 낮춘다. 재연결 항목에는 원래 LLM 구절을 src_t로 남기고, 빌더는 이를 Mention의 span 속성으로 저장한다. 근거 언급에서 'LLM이 고른 구절'과 '사전이 붙인 이름'을 따로 확인할 수 있게 하기 위해서다. 개발 결과와 설계 E의 확증 결과는 모두 Mention 없는 빌드에서 나왔고, Mention을 넣은 S-mix 빌드의 검사는 3.4절에 적는다. 근거 문장을 다시 자를 때도 사전 문자열이 아니라 LLM 구절을 중심으로 자른다. 대조 변형(보존판, 변형 이름 v3llmda_ns)은 교체 대상 항목을 지우지 않고 사전 항목만 덧붙인다(간선 5,433,244개). LLM이 구절을 고르고 사전이 개념을 붙이는 구성 자체는 흔한 설계다(2장). 이 연구는 같은 문헌·같은 순위 함수에서 구성 요소별 그래프를 비교한다. 이 비교에는 사전을 쓰는 위치 외에 재연결 대상 항목의 삭제와 그래프별 주제 밖 간선 가중치가 섞여 있으며, 삭제의 효과는 보존판으로 따로 본다(4.4절).

### 3.4 그래프 모델

추출 결과는 네 종류의 노드와 여섯 종류의 관계(IS_A는 개념 계층과 질환 계층에 함께 쓴다)로 이루어진 Neo4j 그래프에 적재한다(표 3). Disease 노드는 후보 질환 43,575개 전부이며, 소견 개념 수(n_findings)와 수집 원문 문자 수(src_chars)를 가진다. Concept 노드는 의미유형과 IDF를 가진다. IDF는 소견이 하나라도 있는 질환 수 N과 그 개념을 가진 질환 수 df로 계산한 ln(N/df)이다. Doc 노드는 원 문서이며, 논문 본문, 초록, 한 논문의 표 묶음, 도서 장이 각각 하나의 Doc이다. Mention 노드는 원문에서 위치가 확인된 항목이 개념 하나에 연결된 단위이며, 연결된 구절, 근거 문장, 주제 문서 여부(topic)를 담는다. 등위 분해·HPO 이름 연결로 한 항목이 여러 Mention이 될 수 있고, 이때 구절은 연결된 부분 구절이다. 혼합 추출로 재연결된 항목의 Mention은 원 LLM 구절(span)도 담는다. HAS_FINDING 간선은 지지 언급 수(n_mentions), 지지 문서 수(n_docs), 주제 문서 수(n_topic_docs)를 가진다. 순위화는 Disease·Concept 속성, HAS_FINDING, IS_A, IS_DISEASE만 읽고(3.5절), Mention과 Doc은 근거 경로 조회에 쓰인다.

적재가 끝나면 두 불변식을 검사한다. (가) 모든 HAS_FINDING 간선에는 그 질환(OF)과 그 개념(MENTIONS)을 함께 잇는 Mention이 하나 이상 있다. (나) Disease 노드 수는 후보 질환 수(43,575)와 같다. 어긋나면 빌드를 오류로 끝내고 빌드 요약(지문 포함)을 기록하지 않는다. 기존 그래프는 관계를 먼저, 노드를 나중에 2,000개 단위로 지운다. 실패한 배치가 하나라도 있거나 지운 뒤 노드가 남으면 중단한다. 이 검사들은 일괄 삭제의 부분 실패가 서버 경고로만 남아, 제외 대상 질환 노드 2,682개와 근거 없는 이전 간선 62,497개가 그래프에 남았던 사건 뒤에 넣었다(4.1절).

**자기 간선 제거.** 자기 간선 제거(변형 이름의 접미어 _ns)는 질환에서 그 질환 자신의 개념과 직접 상위 질환의 개념으로 가는 간선을 뺀다. 환자 소견 개념이 질환 노드이기도 하면 통합 근거(3.5절)가 이미 그 질환과 직접 하위 질환을 지지하므로, 같은 정보가 소견 점수로 한 번 더 들어가지 않게 하기 위해서다. 다만 IS_DISEASE 연결은 그래프에 Concept 노드로 남은 질환 개념에만 만들어진다. 그래서 자기 개념이 다른 질환의 소견이나 소견 개념의 상위 개념으로 나오지 않는 질환은 자기 간선과 함께 IS_DISEASE 연결(통합 근거)도 잃는다. 이런 질환의 개념은 환자에게 있어도 소견 점수와 통합 근거 어디에도 쓰이지 않는다. IS_DISEASE 연결은 v3 LLM 그래프에서 19,609개에서 16,468개로, v3 UMLS 사전 그래프에서 25,754개에서 24,769개로 줄었다. v2+v3 문헌의 네 시스템(S-mix, S-llm, S-dict, S-union)은 모두 제거판이다. 자기 간선 유무를 모두 빌드한 v3 HPO 사전·UMLS 사전·LLM 그래프에서 개발 선택 규칙은 모두 제거판을 골랐다(4.4절). 합집합과 혼합 그래프는 제거판으로만 만들었다. S-D는 설계 D에서 동결한 그대로 자기 간선을 둔다.

**비교용 빌드와 지문.** 개발 변형은 Mention과 Doc 노드 없이 별도 비교용 Neo4j 인스턴스에 빌드했다. 이 빌드에서는 불변식 (가)를 검사하지 않는다. 빌더는 순위화가 읽는 모든 행의 지문(SHA-1)을 기록한다. 지문은 질환별 n_findings와 src_chars, 개념 IDF, 간선별 지지 문서 수와 주제 문서 수, 개념·질환 계층, IS_DISEASE 연결을 덮는다. 지문이 있는 빌드는 다시 빌드했을 때 지문과 채점 설정이 모두 같을 때만 이전 평가 결과를 재사용한다. 지문을 넣기 전에 만든 빌드(S-D를 포함한 v2 변형 일부)는 요약 수치 6개(질환, 소견 있는 질환, 개념, 간선, 언급, 문서 수)가 같을 때 재사용했다. 비교 그래프들의 규모는 표 5와 4.1절에 있다.

**설계 E의 동결 검증.** 설계 E는 모든 그래프 시스템을 확증 실행 직전에 같은 비교 인스턴스에 다시 빌드하고, 빌드 요약의 지문과 적재된 HAS_FINDING 수가 동결 커밋에 기록한 값과 같을 때만 평가하도록 정했다. Disease 노드 수가 후보 질환 수(43,575)와 다르면 빌더가 멈춘다(불변식 (나)). BM25 색인은 파일의 sha256을 동결 커밋에 기록하고 확증 때 대조한다. 요약에 지문이 없는 S-D는 동결 전에 다시 빌드해 지문을 기록한다. 또 동결 전에 Mention을 넣어 빌드한 S-mix가 불변식 (가)·(나)를 통과하는지, 그리고 개발 DDXPlus 3,000명에서 정답 질환의 순위가 Mention 없는 빌드와 같은지 확인하도록 정했다. src_t 필드는 Mention 속성과 근거 문장 자르기에만 쓰이며 지문에 들어가는 행을 바꾸지 않는다. 동결 전에 필드를 넣어 다시 빌드한 S-mix의 지문은 이전 빌드와 같았고(15ba677…), 비교 인스턴스에 적재된 HAS_FINDING 간선은 빌드 요약과 같은 5,221,285개였다. 다시 빌드한 S-D의 요약 수치 6개는 이전 요약과 같았고, 이 빌드의 지문을 기록했다. Mention을 넣어 빌드한 S-mix는 불변식 (가)·(나)를 통과했다(근거 언급 없는 HAS_FINDING 0개, Disease 43,575개). 이 빌드의 근거 언급은 11,582,169개이고, 지문은 Mention 없는 빌드와 같았다(15ba677…). 개발 DDXPlus 3,000명 모두에서 정답 질환의 Top-10 내 위치(lineage, Top-10 밖이면 없음)가 Mention 없는 빌드와 같았다. 이 검사를 마친 뒤 시스템별 지문·간선 수, BM25 색인의 sha256, 채점 설정을 frozen_config_E.json에 기록해 커밋 49901bc(2026-10-19)로 동결했다. 확증은 이 커밋에서 한 번 실행했다(2026-10-19 23:13 시작, 2026-10-20 01:34 종료, 시도 1회). 다시 빌드한 다섯 그래프 시스템은 모두 지문과 적재된 HAS_FINDING 수가 frozen_config_E.json의 값과 같았고, BM25 색인 파일의 sha256도 모두 같았다(4.6절).

Table 3. Graph schema

| Element | Type | Properties | Role |
|---|---|---|---|
| Disease | Node | id (UMLS CUI, GeneReviews chapter ID, or MONDO ID), name, vocab, n_findings, src_chars | Candidate disease (43,575) |
| Concept | Node | cui, semantic type, IDF | Normalized finding concept |
| Doc | Node | id, PMID, PMCID^a^, kind, title, year, license^a^ | Source document: article body, abstract, table set of one article, or book chapter |
| Mention | Node | phrase, sentence, topic, span^b^ | Unit of provenance: one item–concept link located in the source text |
| (Disease)-[:HAS_FINDING]->(Concept) | Relationship | n_mentions, n_docs, n_topic_docs | Disease–finding association used for scoring |
| (Mention)-[:OF]->(Disease) | Relationship | – | Links a mention to its disease |
| (Mention)-[:MENTIONS]->(Concept) | Relationship | – | Links a mention to its finding concept |
| (Mention)-[:FROM]->(Doc) | Relationship | – | Links a mention to its source document |
| (Concept)-[:IS_A]->(Concept) | Relationship | – | UMLS concept hierarchy |
| (Disease)-[:IS_A]->(Disease) | Relationship | – | MONDO is_a and UMLS disease hierarchy (45,118 pairs) |
| (Concept)-[:IS_DISEASE]->(Disease) | Relationship | – | A finding concept that is itself a candidate disease (unified evidence) |

Six relationship types; IS_A serves both the concept and the disease hierarchy. Load-time invariants: (i) every HAS_FINDING edge is backed by at least one Mention linked to both of its ends; (ii) the number of Disease nodes equals the number of candidate diseases (43,575). A failed invariant ends the build with an error, and no build summary is written. Development variants are built without Doc and Mention nodes, so invariant (i) is checked only on builds with mentions. ^a^ Empty for v3 documents, which carry PMID, document ID, kind, title and year. ^b^ Only on items re-linked by mixed extraction: the LLM phrase in which the dictionary string was found.
