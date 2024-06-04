# Parent Document Retrieval

문서를 large chunk와 small chunk로 나누어서 small chunk를 찾은후에 LLM의 context에는 large chunk를 사용하면 검색의 정확도는 높이고 충분한 문서를 context로 활용할 수 있습니다. 

## Parent/Child Chunking

문서 또는 이미지 파일에서 텍스트를 추출시 Parent와 Child로 Chunking을 수행합니다. 상세한 코드는 [lambda-document](./lambda-document-manager/lambda_function.py)을 참조합니다.

먼저 parent/child splitter를 지정합니다. parent의 경우에 전체 문서를 나눠야하므로 separator로 개행이나 마침표와 같은 단어를 기준으로 합니다. 여기서는, chunk_size는 2000으로 하였으나 목적에 맞게 조정할 수 있습니다. chunk_size가 크면 LLM이 충분한 정보를 가질수 있으나, 전체적으로 token 소모량이 증가하고, 문서의 수(top_k)를 많이 넣으면 LLM에 따라서는 context winodow를 초과할 수 있어 주의가 필요합니다. child splitter의 경우는 여기서는 400을 기준으로 50의 overlap을 설정하였습니다. child의 경우에 관련된 문서를 찾는 기준이 되나 실제 사용은 parent를 사용하므로 검색이 잘되도록 하는것이 중요합니다. child의 경우는 하나의 문장의 일부가 될 수 있어야 하고 제목이 하나의 chunk를 가져가면 안되므로, 아래와 같이 개행문자등을 separator로 등록하지 않았습니다. 

```python
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    # separators=["\n\n", "\n", ".", " ", ""],
    length_function = len,
)
```

분리된 parent/child chink들을 OpenSearch에 등록합니다. OpenSearch에 등록할때 사용하는 문서의 id는 OpenSearch의 성능에 영향을 줄 수 있으며, child가 가지고 있는 parent의 id를 기반으로 검색할 수 있어야 하므로, OpenSearch가 생성하는 id를 이용합니다. 아래와 같이 먼저 parent가 되는 chink의 metadata에 doc_level을 parent로 설정한 후에 OpenSearch에 add_documents()로 등록합니다. 이때 child는 생성된 id를 parent의 id로 활용하여 아래와 같이 metadata에 parent_doc_id를 등록합니다. 

```python
parent_docs = parent_splitter.split_documents(docs)
    if len(parent_docs):
        for i, doc in enumerate(parent_docs):
            doc.metadata["doc_level"] = "parent"
            print(f"parent_docs[{i}]: {doc}")
                    
        parent_doc_ids = vectorstore.add_documents(parent_docs, bulk_size = 10000)
        print('parent_doc_ids: ', parent_doc_ids)
                
        child_docs = []
        for i, doc in enumerate(parent_docs):
            _id = parent_doc_ids[i]
            sub_docs = child_splitter.split_documents([doc])
            for _doc in sub_docs:
                _doc.metadata["parent_doc_id"] = _id
                _doc.metadata["doc_level"] = "child"
            child_docs.extend(sub_docs)
        # print('child_docs: ', child_docs)
                
        child_doc_ids = vectorstore.add_documents(child_docs, bulk_size = 10000)
        print('child_doc_ids: ', child_doc_ids)
                    
        ids = parent_doc_ids+child_doc_ids
```

사용자가 문서 삭제나 업데이트를 할 경우에 기생성된 parent_doc_ids와 child_doc_ids를 이용하여 아래와 같이 삭제를 수행합니다. 이때 파일에 대한 OpenSearch의 id 리스트는 파일 생성시 만든 json파일에서 로드하여 이용합니다. 

```python
def delete_document_if_exist(metadata_key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        print('objs: ', objs)
        
        if(len(objs)>0):
            doc = s3r.Object(s3_bucket, metadata_key)
            meta = doc.get()['Body'].read().decode('utf-8')
            print('meta: ', meta)
            
            ids = json.loads(meta)['ids']
            print('ids: ', ids)
            
            result = vectorstore.delete(ids)
            print('result: ', result)        
        else:
            print('no meta file: ', metadata_key)
```



## References 

### LangChain의 Parent Document Retriever

[(Parent Document Retriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/)에서는 In-memory store를 이용하여 parent-child chunking을 구현하는 방법을 설명하고 있습니다. AWS Lambda와 같은 경우는 event가 일정시간 없으면 초기화되므로 In-memory store를 사용할 수 없습니다. 

### Parent child splitter 활용

[MongoDB Parent Document Retrieval over your data with Amazon Bedrock](https://medium.com/@dminhk/mongodb-parent-document-retrieval-over-your-data-with-amazon-bedrock-0ecf1db9d999)와 같이 parent-child spliteer를 이용합니다.


