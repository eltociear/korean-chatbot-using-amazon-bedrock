# Parent Document Retrieval

문서를 large chunk와 small chunk로 나누어서 small chunk를 찾은후에 LLM의 context에는 large chunk를 사용하면 검색의 정확도는 높이고 충분한 문서를 context로 활용할 수 있습니다. 

## LangChain의 Parent Document Retriever

[(Parent Document Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/)에서는 In-memory store를 이용하여 parent-child chunking을 구현하는 방법을 설명하고 있습니다. 

AWS Lambda와 같은 경우는 event가 일정시간 없으면 초기화되므로 In-memory store를 사용할 수 없습니다. 따라서 persistant store인 opensearch를 사용합니다. 

## Parent child splitter 활용

[MongoDB Parent Document Retrieval over your data with Amazon Bedrock](https://medium.com/@dminhk/mongodb-parent-document-retrieval-over-your-data-with-amazon-bedrock-0ecf1db9d999)와 같이 parent-child spliteer를 이용합니다.


