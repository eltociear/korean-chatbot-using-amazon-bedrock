# RAG 성능을 향상시켜 한국어 Chatbot 만들기

여기에서는 RAG의 성능을 향상시키는 여러가지 방법들에 대해 설명하고 이를 이용하여 기업 또는 개인의 데이터를 쉽게 활용할 수 있는 Chatbot을 만들고자 합니다. RAG는 기업 또는 개인의 중요한 데이터를 LLM에서 활용할 수 있는 기술입니다. LLM 활용에 빠지지 않는 중요한 기술이지만 제대로 활용하기 위해서는 아래와 같은 것을을 검토할 수 있습니다. 

- Multi-RAG: RAG에 반드시 필요한 [지식 저장소(Knowledge Store)](https://aws.amazon.com/ko/about-aws/whats-new/2023/09/knowledge-base-amazon-bedrock-models-data-sources/)는 별도로 구축할수도 있고, 기존의 RDB를 활용할 수 있습니다. 다양한 지식저장소를 활용하여 RAG의 활용도를 높입니다.
- Multi-Region LLM: 여러 리전에 있는 LLM을 동시에 활용함으로써 질문후 답변까지의 동작시간을 단축하고, On-Demand 방식의 동시 실행 수의 제한을 완화할 수 있습니다.
- 한영 동시 검색: RAG에 한국어와 영어 문서들이 혼재할 경우에 한국어로 영어 문서를 검색할 수 없습니다. 한국어로 한국어, 영어 문서를 모두 검색하여 RAG의 성능을 향상 시킬 수 있습니다.
- 인터넷 검색: RAG의 지식저장소에 관련된 문서가 없는 경우에 인터넷 검색을 통해 활용도를 높입니다.
- 관련도 기준으로 검색된 문서 활용: RAG는 LLM에 Context로 관련된 문서를 제공합니다. Context에 들어가는 문서의 순서에 따라 RAG의 성능이 달라집니다.

여기서는 [Amazon Bedrock](https://aws.amazon.com/ko/bedrock/)의 Anthropic Claude LLM(Large Language Models) 모델을 이용하여 질문/답변(Question/Answering)을 수행하는 Chatbot을 구성하지만, [LangChain](https://aws.amazon.com/ko/what-is/langchain/)을 기반으로 구성하므로 Llama2와 같은 다른 LLM을 활용할때에도 쉽게 응용할 수 있습니다. 




## 아키텍처 개요

전체적인 아키텍처는 아래와 같습니다. 사용자의 질문은 WebSocket을 이용하여 AWS Lambda에서 RAG와 LLM을 이용하여 답변합니다. 대화 이력(chat history)를 이용하여 사용자의 질문(Question)을 새로운 질문(Revised question)으로 생성합니다. 새로운 질문으로 지식 저장소(Knowledge Store)인 Kendra와 OpenSearch에 활용합니다. 두개의 지식저장소에는 용도에 맞는 데이터가 입력되어 있는데, 만약 같은 데이터가 가지고 있더라도, 두개의 지식저장소의 문서를 검색하는 방법의 차이로 인해, 서로 보완적인 역할을 합니다. 지식저장소에 한국어/한국어로 된 문서들이 있다면, 한국어 질문은 영어로 된 문서를 검색할 수 없습니다. 따라서 질문이 한국어라면 한국어로 한국어 문서를 먼저 검색한 후에, 영어로 번역하여 다시 한번 영어 문서들을 검색합니다. 이렇게 함으로써 한국어로 질문을 하더라도 영어 문서까지 검색하여 더 나은 결과를 얻을 수 있습니다. 만약 두 지식저장소가 관련된 문서(Relevant documents)를 가지고 있지 않다면, Google Search API를 이용하여 인터넷에 관련된 웹페이지들이 있는지 확인하고, 이때 얻어진 결과를 RAG처럼 활용합니다. 

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/1636dcbb-611f-41fb-9051-e9fe0e4f6b29" width="800">



상세하게 단계별로 설명하면 아래와 같습니다.

단계 1: 사용자의 질문(question)은 API Gateway를 통해 Lambda에 Web Socket 방식으로 전달됩니다. Lambda는 JSON body에서 질문을 읽어옵니다. 이때 사용자의 이전 대화이력이 필요하므로 [Amazon DynamoDB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html)에서 읽어옵니다. DynamoDB에서 대화이력을 로딩하는 작업은 처음 1회만 수행합니다.

단계 2: 사용자의 대화이력을 반영하여 사용자와 Chatbot이 interactive한 대화를 할 수 있도록, 대화이력과 사용자의 질문으로 새로운 질문(Revised Question)을 생성하여야 합니다. LLM에 대화이력(chat history)를 Context로 제공하고 적절한 Prompt를 이용하면 새로운 질문을 생성할 수 있습니다.

단계 3: 새로운 질문(Revised question)으로 OpenSearch에 질문을 하여 관련된 문서(Relevant Documents)를 얻습니다. 

단계 4: 질문이 한국어인 경우에 영어 문서도 검색할 수 있도록 새로운 질문(Revised question)을 영어로 번역합니다.

단계 5: 번역된 새로운 질문(translated revised question)을 이용하여 Kendra와 OpenSearch에 질문합니다.

단계 6: 번역된 질문으로 얻은 관련된 문서가 영어 문서일 경우에, LLM을 통해 번역을 수행합니다. 관련된 문서가 여러개이므로 Multi-Region의 LLM들을 활용하여 지연시간을 최소화 합니다.

단계 7: 한국어 질문으로 얻은 N개의 관련된 문서와, 영어로 된 N개의 관련된 문서의 합은 최대 2xN개입니다. 이 문서를 가지고 Context Window 크기에 맞도록 문서를 선택합니다. 이때 관련되가 높은 문서가 Context의 상단에 가도록 배치합니다.

단계 8: 관련도가 일정 이하인 문서는 버리므로, 한개의 RAG의 문서도 선택되지 않을 수 있습니다. 이때에는 Google Seach API를 통해 인터넷 검색을 수행하고, 이때 얻어진 문서들을 Priority Search를 하여 관련도가 일정 이상의 결과를 RAG에서 활용합니다. 

단계 9: 선택된 관련된 문서들(Selected relevant documents)로 Context를 생성한 후에 새로운 질문(Revised question)과 함께 LLM에 전달하여 사용자의 질문에 대한 답변을 생성합니다.

이때의 Sequence diagram은 아래와 같습니다. 만약 RAG에서 관련된 문서를 찾지못할 경우에는 Google Search API를 통해 Query를 수행하여 RAG처럼 활용합니다. 대화이력을 가져오기 위한 DynamoDB는 첫번째 질문에만 해당됩니다. 여기서는 "us-east-1"과 "us-west-2"의 Bedrock을 사용하므로, 아래와 같이 질문마다 다른 Region의 Bedrock Claude LLM을 사용합니다.

<img src="https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/251d2666-8837-4e8b-8521-534cbd3ced53" width="1000">

파일 업로드 또는 삭제시는 아래와 같이 동작합니다.


![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/89d233cf-36df-4e5d-944d-03dce064c130)


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

### Google Search API를 이용한 검색기능

Multi-RAG로 검색하여 Relevant Document가 없는 경우에 Google API를 이용해 검색한 결과를 RAG에서 사용합니다. 상세한 내용은 [Google Search API](./GoogleSearchAPI.md)에서 확인합니다. 여기서, assessed_score는 priority search시 FAISS의 Score로 업데이트 됩니다.

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

### OpenSearch에서 Nori Plugin을 이용한 Lexical 검색

[OpenSearch에서 Lexical 검색](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/opensearch-nori-plugin.md)에 대해 이해하고 구현합니다.

### S3를 데이터 소스로 하기 위한 퍼미션

Log에 대한 퍼미션이 필요합니다.

```java
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "cloudwatch:GenerateQuery",
                "logs:*"
            ],
            "Resource": "*",
            "Effect": "Allow"
        }
    ]
}
```

개발 및 테스트를 위해 Kendra에서 추가로 S3를 등록할 수 있도록 모든 S3에 대한 읽기 퍼미션을 부여합니다. 

```java
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Action": [
				"s3:Describe*",
				"s3:Get*",
				"s3:List*"
			],
			"Resource": "*",
			"Effect": "Allow"
		}
	]
}
```

이를 CDK로 구현하면 아래와 같습니다.

```typescript
const kendraLogPolicy = new iam.PolicyStatement({
    resources: ['*'],
    actions: ["logs:*", "cloudwatch:GenerateQuery"],
});
roleKendra.attachInlinePolicy( // add kendra policy
    new iam.Policy(this, `kendra-log-policy-for-${projectName}`, {
        statements: [kendraLogPolicy],
    }),
);
const kendraS3ReadPolicy = new iam.PolicyStatement({
    resources: ['*'],
    actions: ["s3:Get*", "s3:List*", "s3:Describe*"],
});
roleKendra.attachInlinePolicy( // add kendra policy
    new iam.Policy(this, `kendra-s3-read-policy-for-${projectName}`, {
        statements: [kendraS3ReadPolicy],
    }),
);    
```

### Kendra 파일 크기 Quota

[Quota Console - File size](https://ap-northeast-1.console.aws.amazon.com/servicequotas/home/services/kendra/quotas/L-C108EA1B)와 같이 Kendra에 올릴수 있는 파일크기는 50MB로 제한됩니다. 이는 Quota 조정 요청을 위해 적절한 값으로 조정할 수 있습니다. 다만 이 경우에도 파일 한개에서 얻어낼수 있는 Text의 크기는 5MB로 제한됩니다. msg를 한국어 Speech로 변환한 후에 CloudFront URL을 이용하여 S3에 저장된 Speech를 URI로 공유할 수 있습니다.

### 결과 읽어주기

Amazon Polly를 이용하여 결과를 한국어로 읽어줍니다. [start_speech_synthesis_task](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/polly/client/start_speech_synthesis_task.html#)을 활용합니다.

```python
def get_text_speech(path, speech_prefix, bucket, msg):
    ext = "mp3"
    polly = boto3.client('polly')
    try:
        response = polly.start_speech_synthesis_task(
            Engine='neural',
            LanguageCode='ko-KR',
            OutputFormat=ext,
            OutputS3BucketName=bucket,
            OutputS3KeyPrefix=speech_prefix,
            Text=msg,
            TextType='text',
            VoiceId='Seoyeon'        
        )
        print('response: ', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create voice")
    
    object = '.'+response['SynthesisTask']['TaskId']+'.'+ext
    print('object: ', object)

    return path+speech_prefix+parse.quote(object)
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

[fsi_faq_ko.csv](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/fsi_faq_ko.csv)을 다운로드한 후에 파일 아이콘을 선택하여 업로드한후, 채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. 이때의 결과는 ＂아니오”입니다. 이때의 결과는 아래와 같습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/c7aeca05-0209-49c3-9df9-7e04026900f2)

채팅창에 "이체를 할수 없다고 나옵니다. 어떻게 해야 하나요?” 라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/56ad9192-6b7c-49c7-9289-b6a3685cb7d4)

채팅창에 "간편조회 서비스를 영문으로 사용할 수 있나요?” 라고 입력합니다. "영문뱅킹에서는 간편조회서비스 이용불가"하므로 좀더 자세한 설명을 얻었습니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/3a896488-af0c-42b2-811b-d2c0debf5462)

채팅창에 "공동인증서 창구발급 서비스는 무엇인가요?"라고 입력하고 결과를 확인합니다.

![image](https://github.com/kyopark2014/rag-chatbot-using-bedrock-claude-and-kendra/assets/52392004/2e2b2ae1-7c50-4c14-968a-6c58332d99af)

#### 잘못된 응답 유도해보기

"엔씨의 Lex 서비스는 무엇인지 설명해줘."와 같이 잘못된 단어를 조합하여 질문하였습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/78f2f2c0-cecf-43a2-98c7-843276755248)

"Amazon Varco 서비스를 Manufactoring에 적용하는 방법 알려줘."로 질문하고 응답을 확인합니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/8c484742-a294-4876-afe9-df0e0f2d96c5)

#### 한영 동시검색

"Amazon의 Athena 서비스에 대해 설명해주세요."로 검색할때 한영 동시 검색을 하면 영어 문서에서 답변에 필요한 관련문서를 추출할 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/4526c9aa-a0aa-4b23-8818-860f5376b898)

한영동시 검색을 하지 않았을때의 결과는 아래와 같습니다. 동일한 질문이지만, OpenSearch의 결과를 많이 참조하여 잘못된 답변을 할 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/b5548594-abc8-4447-8f95-d6d12d36c23e)


### Prompt Engineering 결과 예제

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


### 데이터 소스 추가

S3를 데이터 소스르 추가할때 아래와 같이 수행하면 되나, languageCode가 미지원되어서 CLI로 대체합니다.

```typescript
const cfnDataSource = new kendra.CfnDataSource(this, `s3-data-source-${projectName}`, {
    description: 'S3 source',
    indexId: kendraIndex,
    name: 'data-source-for-upload-file',
    type: 'S3',
    // languageCode: 'ko',
    roleArn: roleKendra.roleArn,
    // schedule: 'schedule',

    dataSourceConfiguration: {
        s3Configuration: {
            bucketName: s3Bucket.bucketName,
            documentsMetadataConfiguration: {
                s3Prefix: 'metadata/',
            },
            inclusionPrefixes: ['documents/'],
        },
    },
});
```

CLI 명령어 예제입니다.

```text
aws kendra create-data-source
--index-id azfbd936-4929-45c5-83eb-bb9d458e8348
--name data-source-for-upload-file
--type S3
--role-arn arn:aws:iam::123456789012:role/role-lambda-chat-ws-for-korean-chatbot-us-west-2
--configuration '{"S3Configuration":{"BucketName":"storage-for-korean-chatbot-us-west-2", "DocumentsMetadataConfiguration": {"S3Prefix":"metadata/"},"InclusionPrefixes": ["documents/"]}}'
--language-code ko
--region us-west-2
```

### OpenSearch

[Python client](https://opensearch.org/docs/latest/clients/python-low-level/)에 따라 OpenSearch를 활용합니다.

opensearch-py를 설치합니다.

```text
pip install opensearch-py
```

[Index naming restrictions](https://opensearch.org/docs/1.0/opensearch/rest-api/create-index/#index-naming-restrictions)에 따랏 index는 low case여야하고, 공백이나 ','을 가질수 없습니다.

### OpenSearch의 add_documents시 에러

문서를 충분히 많이 넣은 상태에서 아래와 같은 에러가 발생하였습니다.

```text
opensearchpy.exceptions.RequestError: RequestError(400, 'illegal_argument_exception', 'Validation Failed: 1: this action would add [15] total shards, but this cluster currently has [2991]/[3000] maximum shards open;')
```

실제로 OpenSearch DashBoard에서 2991 shards를 사용중 인것을 확인할 수 있습니다. 이때에는 cluster.max_shards_per_node를 올리거나(default 1000), node수를 늘려서 해당 문제를 해결할 수 있습니다.

![image](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/assets/52392004/e2ff5cca-3913-4d88-9697-48594fed5e4e)

### OpenSearch Embedding 에

아래와 같이 OpenSearch Embedding의 bulk_size의 기본값이 500이므로, 2000으로 아래와 같이 변경합니다. 아래의 1840은 PDF의 Text가 1540631자이므로 embedding을 해야하는 chunk의 숫자를 의미힙나다. 

```text
RuntimeError: The embeddings count, 1840 is more than the [bulk_size], 500. Increase the value of [bulk_size].
```

수정 코드는 아래와 같습니다.

```python
new_vectorstore = OpenSearchVectorSearch(
    index_name=index_name,  
    is_aoss = False,
    #engine="faiss",  # default: nmslib
    embedding_function = bedrock_embeddings,
    opensearch_url = opensearch_url,
    http_auth=(opensearch_account, opensearch_passwd),
)
response = new_vectorstore.add_documents(docs, bulk_size = 2000)
```

## Reference 

[Claude - Constructing a prompt](https://docs.anthropic.com/claude/docs/constructing-a-prompt)
