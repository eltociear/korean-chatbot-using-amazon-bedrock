# Google Search API

LangChain의 [GoogleSearchAPIWrapper](https://api.python.langchain.com/en/latest/utilities/langchain.utilities.google_search.GoogleSearchAPIWrapper.html#)를 활용하는 방법에 대해 설명합니다.

## API 발급

[Enable the Custom Search API](https://console.cloud.google.com/apis/library/customsearch.googleapis.com?project=red-grid-306501)

[api_key](https://developers.google.com/custom-search/docs/paid_element?hl=ko#api_key)에서 키 가져오기를 선택합니다.

- Programmable Search Element API는 광고 없는 검색 요소 쿼리 1,000개당 $5를 청구
- 

endpoint: endpoint is https://customsearch.googleapis.com/customsearch/v1

```python
from googleapiclient.discovery import build

api_key = 'YOUR_API_KEY'
cse_id = 'YOUR_SEARCH_ENGINE_ID'

service = build("customsearch", "v1", developerKey=api_key)

print(res['searchInformation']['totalResults'])
```

## Reference

[Public APIs Developers Can Use in Their Projects](https://ijaycent.hashnode.dev/public-apis-developers-can-use-in-their-projects)
