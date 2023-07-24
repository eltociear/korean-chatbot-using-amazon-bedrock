
## Faiss

[Faiss](https://github.com/facebookresearch/faiss)는 Facebook에서 오픈소스로 제공하는 In-memory vector store로서 embedding과 document들을 저장할 수 있으며, LangChain을 지원합니다. 비슷한 역할을 하는 persistent store로는 Amazon OpenSearch, RDS Postgres with pgVector, ChromaDB, Pinecone과 Weaviate가 있습니다. 

faiss.write_index(), faiss.read_index()을 이용해서 local에서 index를 저장하고 읽어올수 있습니다. 그러나 S3에서 직접 로드는 현재 제공하고 있지 않습니다. EFS에서 저장후 S3에 업로드 하는 방식은 레퍼런스가 있습니다.

[Faiss-LangChain](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/faiss)와 같이 save_local(), load_local()을 사용할 수 있고, merge_from()으로 2개의 vector store를 저장할 수 있습니다.
