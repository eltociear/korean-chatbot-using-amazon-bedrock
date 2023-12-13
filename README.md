# Multi-RAG와 Claude LLM으로 한국어 Chatbot 만들기 

[Amazon Bedrock](https://aws.amazon.com/ko/bedrock/)의 Anthropic Claude LLM(Large Language Models) 모델을 이용하여 질문/답변(Question/Answering)을 수행하는 Chatbot을 [Knowledge Database](https://aws.amazon.com/ko/about-aws/whats-new/2023/09/knowledge-base-amazon-bedrock-models-data-sources/)를 이용하여 구현합니다. 


## 아키텍처 개요

전체적인 아키텍처는 아래와 같습니다. 사용자는 Amazon CloudFront를 통해 [Amazon S3](https://aws.amazon.com/ko/s3/)로 부터 웹페이지에 필요한 리소스를 읽어옵니다. 사용자가 Chatbot 웹페이지에 로그인을 하면, 사용자 아이디를 이용하여 Amazon DynamoDB에 저장된 채팅 이력을 로드합니다. 이후 사용자가 메시지를 입력하면 WebSocket을 이용하여 LLM에 질의를 하게 되는데, DynamoDB로 부터 읽어온 채팅 이력과 RAG를 제공하는 Vector Database로부터 읽어온 관련문서(Relevant docs)를 이용하여 적절한 응답을 얻습니다. 이러한 RAG 구현을 위하여 [LangChain을 활용](https://python.langchain.com/docs/get_started/introduction.html)하여 Application을 개발하였고, Chatbot을 제공하는 인프라는 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 통해 배포합니다. 

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/d4b20e2d-f392-438a-864c-b87d0bcd8822" width="800">

상세하게 단계별로 설명하면 아래와 같습니다.

단계1: 브라우저를 이용하여 사용자가 CloudFront 주소로 접속하면, Amazon S3에서 HTML, CSS, JS등의 파일을 전달합니다. 이때 로그인을 수행하고 채팅 화면으로 진입합니다.

단계2: Client는 사용자 아이디를 이용하여 '/history' API로 채팅이력을 요청합니다. 이 요청은 API Gateway를 거쳐서 lambda-history에 전달됩니다. 이후 DynamoDB에서 채팅 이력을 조회한 후에 다시 API Gateway와 lambda-history를 통해 사용자에게 전달합니다.

단계3: Client가 API Gateway로 WebSocket 연결을 시도하면, API Gateway를 거쳐서 lambda-chat-ws로 WebSocket connection event가 전달됩니다. 이후 사용자가 메시지를 입력하면, API Gateway를 거쳐서 lambda-chat-ws로 메시지가 전달됩니다.

단계4: lambda-chat-ws은 사용자 아이디를 이용하여 DynamoDB의 기존 채팅이력을 읽어와서, 채팅 메모리에 저장합니다.

단계5: lambda-chat-ws은 RAG에 관련된 문서(relevant docs)를 검색합니다.

단계6: lambda-chat-ws은 사용자의 질문(question), 채팅 이력(chat history), 관련 문서(relevant docs)를 Amazon Bedrock의 Enpoint로 전달합니다.

단계7: Amazon Bedrock의 사용자의 질문과 채팅이력이 전달되면, Anthropic의 Claude LLM을 이용하여 적절한 답변(answer)을 사용자에게 전달합니다. 이때, stream을 사용하여 답변이 완성되기 전에 답변(answer)를 사용자에게 보여줄 수 있습니다.

이때의 Sequence diagram은 아래와 같습니다.

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/283b858c-f94d-426e-8a06-aa7fe411c05f" width="800">

## 주요 구성

### Bedrock을 LangChain으로 연결

[Bedrock](https://python.langchain.com/docs/integrations/providers/bedrock)을 import하여 LangChain로 application을 개발할 수 있습니다. 아래와 같이 bedrock client를 정의합니다. 서비스이름은 "bedrock-runtime"입니다.

```python
import boto3
from langchain.llms.bedrock import Bedrock

boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

llm = Bedrock(
    model_id=modelId, 
    client=boto3_bedrock, 
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model_kwargs=parameters)
```

여기서 파라메터는 아래와 같습니다.

```python
def get_parameter(modelId):
    if modelId == 'anthropic.claude-v1' or modelId == 'anthropic.claude-v2':
        return {
            "max_tokens_to_sample":8191, # 8k
            "temperature":0.1,
            "top_k":250,
            "top_p": 0.9,
            "stop_sequences": [HUMAN_PROMPT]            
        }
parameters = get_parameter(modelId)
```

Bedrock의 지원모델은 [service name이 "bedrock"](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock.html)의 list_foundation_models()을 이용하여 조회합니다. 

```python
bedrock_client = boto3.client(
    service_name='bedrock',
    region_name=bedrock_region,
)
modelInfo = bedrock_client.list_foundation_models()    
print('models: ', modelInfo)
```

### Embedding

[BedrockEmbeddings](https://python.langchain.com/docs/integrations/text_embedding/bedrock)을 이용하여 Embedding을 합니다. 'amazon.titan-embed-text-v1'은 Titan Embeddings Generation 1 (G1)을 의미하며 8k token을 지원합니다.

```python
bedrock_embeddings = BedrockEmbeddings(
    client=boto3_bedrock,
    region_name = bedrock_region,
    model_id = 'amazon.titan-embed-text-v1' 
)
```

## Knowledge Store 

여기서는 Knowledge Store로 OpenSearch, Faiss, Kendra을 이용합니다.

## 메모리에 대화 저장

### RAG를 사용하지 않는 경우

lambda-chat-ws는 인입된 메시지의 userId를 이용하여 map_chat에 기저장된 대화 이력(memory_chat)가 있는지 확인합니다. 채팅 이력이 없다면 아래와 같이 [ConversationBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html?highlight=conversationbuffermemory#langchain.memory.buffer.ConversationBufferMemory)로 memory_chat을 설정합니다. 여기서, Anhropic Claude는 human과 ai의 이름으로 "Human"과 "Assistant"로 설정합니다. LLM에 응답을 요청할때에는 ConversationChain을 이용합니다. [ConversationBufferWindowMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer_window.ConversationBufferWindowMemory.html)을 이용하여 간단하게 k개로 conversation의 숫자를 제한할 수 있습니다.


```python
map_chat = dict()

if userId in map_chat:  
    memory_chat = map_chat[userId]
else:
    memory_chat = ConversationBufferWindowMemory(human_prefix='Human', ai_prefix='Assistant', k=20)
    map_chat[userId] = memory_chat
conversation = ConversationChain(llm=llm, verbose=False, memory=memory_chat)

msg = get_answer_from_conversation(text, conversation, convType, connectionId, requestId)      
def get_answer_from_conversation(text, conversation, convType, connectionId, requestId):
    conversation.prompt = get_prompt_template(text, convType)
    stream = conversation.predict(input=text)                        
    msg = readStreamMsg(connectionId, requestId, stream)

    return msg
```

### RAG를 사용하는 경우

RAG를 이용할때는 [ConversationBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html?highlight=conversationbuffermemory#langchain.memory.buffer.ConversationBufferMemory)을 이용해 아래와 같이 채팅 메모리를 지정합니다. 대화가 끝난후에는 add_user_message()와 add_ai_message()를 이용하여 새로운 chat diaglog를 업데이트 합니다.

```python
map_chain = dict()

if userId in map_chain:  
    memory_chain = map_chain[userId]
else: 
    memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
    map_chain[userId] = memory_chain

memory_chain.chat_memory.add_user_message(text)  # append new diaglog
memory_chain.chat_memory.add_ai_message(msg)

msg = get_answer_using_RAG(llm, text, conv_type, connectionId, requestId, bedrock_embeddings, rag_type)

revised_question = get_revised_question(llm, connectionId, requestId, text)
PROMPT = get_prompt_template(revised_question, conv_type, rag_type)

relevant_docs = []
capabilities = ["kendra", "opensearch", "faiss"];
for reg in capabilities:
    if reg == 'kendra':
        rel_docs = retrieve_from_kendra(query = revised_question, top_k = top_k)
    else:
        rel_docs = retrieve_from_vectorstore(query = revised_question, top_k = top_k, rag_type = reg)

    for doc in rel_docs:
        relevant_docs.append(doc)

selected_relevant_docs = priority_search(revised_question, relevant_docs, bedrock_embeddings)

for document in selected_relevant_docs:
    relevant_context = relevant_context + document['metadata']['excerpt'] + "\n\n"

stream = llm(PROMPT.format(context=relevant_context, question=revised_question))
msg = readStreamMsg(connectionId, requestId, stream)
```

### Stream 처리

여기서 stream은 아래와 같은 방식으로 WebSocket을 사용하는 client에 메시지를 전달할 수 있습니다.

```python
def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg
            }
            sendMessage(connectionId, result)
    print('msg: ', msg)
    return msg
```

여기서 client로 메시지를 보내는 sendMessage()는 아래와 같습니다. 여기서는 boto3의 [post_to_connection](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/apigatewaymanagementapi/client/post_to_connection.html)를 이용하여 메시지를 WebSocket의 endpoint인 API Gateway로 전송합니다.

```python
def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except: 
        raise Exception ("Not able to send a message")
```

### Google API를 이용한 검색기능

Multi-RAG로 검색하여 Relevant Document가 없는 경우에 Google API를 이용해 검색한 결과를 RAG에서 사용합니다. 상세한 내용은 [Google Search API](./GoogleSearchAPI.md)에서 확인합니다.

```python
from googleapiclient.discovery import build

google_api_key = os.environ.get('google_api_key')
google_cse_id = os.environ.get('google_cse_id')

api_key = google_api_key
cse_id = google_cse_id

relevant_docs = []
try:
    service = build("customsearch", "v1", developerKey = api_key)
    result = service.cse().list(q = revised_question, cx = cse_id).execute()
    print('google search result: ', result)

    if "items" in result:
        for item in result['items']:
            api_type = "google api"
            excerpt = item['snippet']
            uri = item['link']
            title = item['title']
            confidence = ""
            assessed_score = ""

            doc_info = {
                "rag_type": 'search',
                "api_type": api_type,
                "confidence": confidence,
                "metadata": {
                    "source": uri,
                    "title": title,
                    "excerpt": excerpt,                                
                },
                "assessed_score": assessed_score,
            }
        relevant_docs.append(doc_info)
```

### AWS CDK로 인프라 구현하기

[CDK 구현 코드](./cdk-qa-with-rag/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)


### CDK를 이용한 인프라 설치
[인프라 설치](https://github.com/kyopark2014/question-answering-chatbot-using-RAG-based-on-LLM/blob/main/deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


### 실행결과

[fsi_faq_ko.csv](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/fsi_faq_ko.csv)을 다운로드한 후에 파일 아이콘을 선택하여 업로드하면 Knowledge Database에 저장됩니다. 이후 아래와 같이 파일 내용을 확인할 수 있도록 요약하여 보여줍니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a0b3b5b8-6e1e-4240-9ee4-e539680fa28d)

채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. 이때의 결과는 ＂아니오”입니다. 이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c7aeca05-0209-49c3-9df9-7e04026900f2)

채팅창에 "이체를 할수 없다고 나옵니다. 어떻게 해야 하나요?” 라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/56ad9192-6b7c-49c7-9289-b6a3685cb7d4)

채팅창에 "공동인증서 창구발급 서비스는 무엇인가요?” 라고 입력하면 아래와 같은 결과를 얻을 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/95a78e6a-5a78-4879-98a1-a30aa6f7e3d5)



아래와 같이 채팅창에 "이체를 할수 없다고 나옵니다. 어떻게 해야 하나요?” 라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/849169c3-c9ee-40dd-8e44-0ec308688ce6)


채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. "영문뱅킹에서는 간편조회서비스 이용불가"하므로 좀더 자세한 설명을 얻었습니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/3a896488-af0c-42b2-811b-d2c0debf5462)

채팅창에 "공동인증서 창구발급 서비스는 무엇인가요?"라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/2e2b2ae1-7c50-4c14-968a-6c58332d99af)




#### Translation

"아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다.”로 입력하고 번역 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/818662e1-983f-44c2-bfcf-e2605ba7a1e6)

#### Extracted Topic and sentiment

“식사 가성비 좋습니다. 위치가 좋고 스카이라운지 바베큐 / 야경 최곱니다. 아쉬웠던 점 · 지하주차장이 비좁습니다.. 호텔앞 교통이 너무 복잡해서 주변시설을 이용하기 어렵습니다. / 한강나가는 길 / 주변시설에 나가는 방법등.. 필요합니다.”를 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/8c38a58b-08df-4e9e-a162-1cd8f542fb46)

#### Information extraction

“John Park. Solutions Architect | WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-5555“로 입력후에 이메일이 추출되는지 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/f613e86e-b08d-45e8-ac0e-71334427f450)

#### PII(personally identifiable information) 삭제하기

PII(Personal Identification Information)의 삭제의 예는 아래와 같습니다. "John Park, Ph.D. Solutions Architect | WWCS Amazon Web Services Email: john@amazon.com Mobile: +82-10-1234-4567"와 같이 입력하여 name, phone number, address를 삭제한 텍스트를 얻습니다. 프롬프트는 [PII](https://docs.anthropic.com/claude/docs/constructing-a-prompt)를 참조합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/a77d034c-32fc-4c84-8054-f4e1230292d6)

#### 문장 오류 고치기

"To have a smoth conversation with a chatbot, it is better for usabilities to show responsesess in a stream-like, conversational maner rather than waiting until the complete answer."로 오류가 있는 문장을 입력합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/55774e11-58e3-4eb4-b91c-5b09572456bd)

"Chatbot과 원할한 데화를 위해서는 사용자의 질문엥 대한 답변을 완전히 얻을 때까지 기다리기 보다는 Stream 형태로 보여주는 것이 좋습니다."로 입력후에 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/7b098a29-9bf5-43bf-a32f-82c94ccd04eb)

#### 복잡한 질문 (step-by-step)

"I have two pet cats. One of them is missing a leg. The other one has a normal number of legs for a cat to have. In total, how many legs do my cats have?"를 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c1bf6749-1ce8-44ba-81f1-1fb52e04a2e8)


"내 고양이 두 마리가 있다. 그중 한 마리는 다리가 하나 없다. 다른 한 마리는 고양이가 정상적으로 가져야 할 다리 수를 가지고 있다. 전체적으로 보았을 때, 내 고양이들은 다리가 몇 개나 있을까?"로 질문을 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/992c8385-f897-4411-b6cf-b185465e8690)

#### 날짜/시간 추출하기

메뉴에서 "Timestamp Extraction"을 선택하고, "지금은 2023년 12월 5일 18시 26분이야"라고 입력하면 prompt를 이용해 아래처럼 시간을 추출합니다.

![noname](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/7dd7e659-498c-4898-801c-b72830bf254b)


실제 결과 메시지는 아래와 같습니다. 

```xml
<result>
<year>2023</year>
<month>12</month>
<day>05</day>
<hour>18</hour>
<minute>26</minute>
</result>
```

#### 어린이와 대화 (Few shot example)

대화의 상대에 맞추어서 질문에 답변을하여야 합니다. 이를테면 [General Conversation]에서 "산타가 크리스마스에 선물을 가져다 줄까?"로 질문을 하면 아래와 같이 답변합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/4624727b-addc-4f5d-8f3d-94f358572326)

[9. Child Conversation (few shot)]으로 전환합니다. 동일한 질문을 합니다. 상대에 맞추어서 적절한 답변을 할 수 있었습니다. 

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/cbbece5c-5476-4f3b-89f7-c7fcf90ca796)


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://ap-northeast-2.console.aws.amazon.com/apigateway/main/apis?region=ap-northeast-2)로 접속하여 "rest-api-for-stream-chatbot", "ws-api-for-stream-chatbot"을 삭제합니다.

2) [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.


```text
cdk destroy --all
```




## Reference 

[Claude - Constructing a prompt](https://docs.anthropic.com/claude/docs/constructing-a-prompt)
