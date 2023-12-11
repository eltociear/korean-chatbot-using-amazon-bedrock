# Google Search API

LangChain의 [GoogleSearchAPIWrapper](https://api.python.langchain.com/en/latest/utilities/langchain.utilities.google_search.GoogleSearchAPIWrapper.html#)를 활용하는 방법에 대해 설명합니다.

## API 준비
### API Key 발급

[api_key](https://developers.google.com/custom-search/docs/paid_element?hl=ko#api_key)에서 키 가져오기를 선택합니다.

### 검색엔진 ID 만들기

[새 검색엔진 만들기](https://programmablesearchengine.google.com/controlpanel/create?hl=ko)에서 검색엔진을 설정합니다.


### 가격
- Programmable Search Element API는 광고 없는 검색 요소 쿼리 1,000개당 $5를 청구

endpoint: endpoint is https://customsearch.googleapis.com/customsearch/v1

```python
from googleapiclient.discovery import build

api_key = 'YOUR_API_KEY'
cse_id = 'YOUR_SEARCH_ENGINE_ID'

service = build("customsearch", "v1", developerKey=api_key)

print(res['searchInformation']['totalResults'])
```

## 활용

### 직업 호출하기

```python
import googleapiclient.discovery

api_key = 'your_api_key' 

service = googleapiclient.discovery.build(
  'customsearch', 'v1', developerKey=api_key)

res = service.cse().list(
  q='query string', cx='search engine id').execute()
```

### 동작확인

[Custom Search JSON API](https://developers.google.com/custom-search/v1/introduction?hl=ko)

```text
curl \
  'https://customsearch.googleapis.com/customsearch/v1?key=[YOUR_API_KEY]' \
  --header 'Accept: application/json' \
  --compressed
```

### WebResearchRetriever

[WebResearchRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/web_research)의 예제입니다.

```python
# Initialize
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities.google_search import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
)

docs = web_research_retriever.get_relevant_documents(user_input)
```

## 결과 구하기

검색한 결과를 이용하여 Q&A 결과 얻는 방법입니다.

```python
from langchain.chains import RetrievalQAWithSourcesChain
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, retriever=web_research_retriever
)
msg = qa_chain({"question": revised_question})
```

## Reference

[Public APIs Developers Can Use in Their Projects](https://ijaycent.hashnode.dev/public-apis-developers-can-use-in-their-projects)
