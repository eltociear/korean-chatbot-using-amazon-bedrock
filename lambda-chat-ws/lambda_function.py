import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv
import sys, traceback
import re
import base64

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from botocore.config import Config

from langchain.vectorstores import FAISS
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.retrievers import AmazonKendraRetriever

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
bedrock_region = os.environ.get('bedrock_region', 'us-west-2')
modelId = os.environ.get('model_id', 'anthropic.claude-v2')
print('model_id: ', modelId)
rag_type = os.environ.get('rag_type', 'faiss')
isReady = False   
isDebugging = True

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
enableReference = os.environ.get('enableReference', 'false')
opensearch_url = os.environ.get('opensearch_url')
path = os.environ.get('path')

kendraIndex = os.environ.get('kendraIndex')
roleArn = os.environ.get('roleArn')

# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception as ex:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")

# bedrock   
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"
def get_parameter(modelId):
    if modelId == 'amazon.titan-tg1-large' or modelId == 'amazon.titan-tg1-xlarge': 
        return {
            "maxTokenCount":1024,
            "stopSequences":[],
            "temperature":0,
            "topP":0.9
        }
    elif modelId == 'anthropic.claude-v1' or modelId == 'anthropic.claude-v2':
        return {
            "max_tokens_to_sample":8191, # 8k
            "temperature":0.1,
            "top_k":250,
            "top_p": 0.9,
            "stop_sequences": [HUMAN_PROMPT]            
        }
parameters = get_parameter(modelId)

# langchain for bedrock
llm = Bedrock(
    model_id=modelId, 
    client=boto3_bedrock, 
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    model_kwargs=parameters)

# embedding for RAG
bedrock_embeddings = BedrockEmbeddings(
    client=boto3_bedrock,
    region_name = bedrock_region,
    model_id = 'amazon.titan-embed-text-v1' 
)

map_chain = dict() # Conversation with RAG
map_chat = dict() # Conversation for normal 

kendraRetriever = AmazonKendraRetriever(index_id=kendraIndex)

def get_prompt_template(query, convType):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(query))
    print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        if (convType=='qa' and rag_type=='opensearch') or (convType=='qa' and rag_type=='kendra') or (convType=='qa' and rag_type=='faiss' and isReady):  
            # for RAG, context and question
            prompt_template = """\n\nHuman: 다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다. 여기서 Assistant의 이름은 서연입니다.
        
            <context>
            {context}
            </context>

            <question>            
            {question}
            </question>

            Assistant:"""
        elif convType == "translation":  # for translation, input
            prompt_template = """\n\nHuman: 다음의 <translation>를 영어로 번역하세요. 머리말은 건너뛰고 본론으로 바로 들어가주세요. 또한 결과는 <result> tag를 붙여주세요.

            <translation>
            {input}
            </translation>
                        
            Assistant:"""

        elif convType == "sentiment":  # for sentiment, input
            prompt_template = """\n\nHuman: 아래의 <example> review와 Extracted Topic and sentiment 인 <result>가 있습니다.
            <example>
            객실은 작지만 깨끗하고 편안합니다. 프론트 데스크는 정말 분주했고 체크인 줄도 길었지만, 직원들은 프로페셔널하고 매우 유쾌하게 각 사람을 응대했습니다. 우리는 다시 거기에 머물것입니다.
            </example>
            <result>
            청소: 긍정적, 
            서비스: 긍정적
            </result>

            아래의 <review>에 대해서 위의 <result> 예시처럼 Extracted Topic and sentiment 을 만들어 주세요..

            <review>
            {input}
            </review>
            Assistant: """

        elif convType == "extraction":  # information extraction
            prompt_template = """\n\nHuman: 다음 텍스트에서 이메일 주소를 정확하게 복사하여 한 줄에 하나씩 적어주세요. 입력 텍스트에 정확하게 쓰여있는 이메일 주소만 적어주세요. 텍스트에 이메일 주소가 없다면, "N/A"라고 적어주세요. 또한 결과는 <result> tag를 붙여주세요.

            <text>
            {input}
            </text>

            Assistant: """

        elif convType == "pii":  # removing PII(personally identifiable information) containing name, phone number, address
            prompt_template = """\n\nHuman: 아래의 <text>에서 개인식별정보(PII)를 모두 제거하여 외부 계약자와 안전하게 공유할 수 있도록 합니다. 이름, 전화번호, 주소, 이메일을 XXX로 대체합니다. 또한 결과는 <result> tag를 붙여주세요.
            
            <text>
            {input}
            </text>

            Assistant: """

        elif convType == "grammar":  # Checking Grammatical Errors
            prompt_template = """\n\nHuman: 다음의 <article>에서 문장의 오류를 찾아서 설명하고, 오류가 수정된 문장을 답변 마지막에 추가하여 주세요.

            <article>
            {input}
            </article>
            
            Assistant: """

        elif convType == "step-by-step":  # compelex question 
            prompt_template = """\n\nHuman: 다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. 아래 문맥(context)을 참조했음에도 답을 알 수 없다면, 솔직히 모른다고 말합니다. 여기서 Assistant의 이름은 서연입니다.

            Human: {input}

            Assistant: 단계별로 생각할까요?

            Human: 예, 그렇게하세요.
            
            Assistant:"""
        
        else: # for normal, history, input
            prompt_template = """\n\nHuman: 다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. 아래 문맥(context)을 참조했음에도 답을 알 수 없다면, 솔직히 모른다고 말합니다. 여기서 Assistant의 이름은 서연입니다.

            Current conversation:
            {history}

            <question>            
            {input}
            </question>
            
            Assistant:"""
    else:  # English
        if (convType=='qa' and rag_type=='opensearch') or (convType=='qa' and rag_type=='kendra') or (convType=='qa' and rag_type=='faiss' and isReady):  # for RAG
            prompt_template = """\n\nHuman: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
            {context}
                        
            <question>
            {question}
            </question>

            Assistant:"""
        elif convType=="translation": 
            prompt_template = """\n\nHuman: Here is an article, contained in <article> tags. Translate the article to Korean. Put it in <result> tags.
            
            <article>
            {input}
            </article>
                        
            Assistant:"""
        
        elif convType == "sentiment":  # for sentiment, input
            prompt_template = """\n\nHuman: Here is <example> review and extracted topics and sentiments as <result>.

            <example>
            The room was small but clean and comfortable. The front desk was really busy and the check-in line was long, but the staff were professional and very pleasant with each person they helped. We will stay there again.
            </example>

            <result>
            Cleanliness: Positive, 
            Service: Positive
            </result>

            <review>
            {input}
            </review>
            
            Assistant:"""

        elif convType == "pii":  # removing PII(personally identifiable information) containing name, phone number, address
            prompt_template = """\n\nHuman: We want to de-identify some text by removing all personally identifiable information from this text so that it can be shared safely with external contractors.
            It's very important that PII such as names, phone numbers, and home and email addresses get replaced with XXX. Put it in <result> tags.

            Here is the text, inside <text></text> XML tags.

            <text>
            {input}
            </text>

            Assistant: """

        elif convType == "extraction":  # for sentiment, input
            prompt_template = """\n\nHuman: Please precisely copy any email addresses from the following text and then write them, one per line.  Only write an email address if it's precisely spelled out in the input text.  If there are no email addresses in the text, write "N/A".  Do not say anything else.  Put it in <result> tags.

            {input}

            Assistant:"""

        elif convType == "grammar":  # Checking Grammatical Errors
            prompt_template = """\n\nHuman: Here is an article, contained in <article> tags:

            <article>
            {input}
            </article>

            Please identify any grammatical errors in the article. Also, add the fixed article at the end of answer.
            
            Assistant: """

        elif convType == "step-by-step":  # compelex question 
            rompt_template = """\n\nHuman: Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.
            
            Human: {input}

            Assistant: Can I think step by step?

            Human: Yes, please do.

            Assistant:"""

        else: # normal
            prompt_template = """\n\nHuman: Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.

            {history}
            
            Human: {input}

            Assistant:"""

            #claude_prompt = PromptTemplate.from_template("""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    return PromptTemplate.from_template(prompt_template)

# store document into Kendra
def store_document(path, s3_file_name, requestId):
    documentInfo = {
        "S3Path": {
            "Bucket": s3_bucket,
            "Key": s3_prefix+'/'+s3_file_name
        },
        "Title": s3_file_name,
        "Id": requestId        
    }

    documents = [
        documentInfo
    ]

    kendra = boto3.client("kendra")
    result = kendra.batch_put_document(
        Documents = documents,
        IndexId = kendraIndex,
        RoleArn = roleArn
    )
    print(result)

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
    
    return texts

# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    print('lins: ', len(lines))
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    print('columns: ', columns)
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'page': n+1,
                'url': path+s3_file_name
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    print('docs[0]: ', docs[0])

    return docs

def get_summary(texts):    
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+') 
    word_kor = pattern_hangul.search(str(texts))
    print('word_kor: ', word_kor)
    
    if word_kor:
        #prompt_template = """\n\nHuman: 다음 텍스트를 간결하게 요약하세오. 텍스트의 요점을 다루는 글머리 기호로 응답을 반환합니다.
        prompt_template = """\n\nHuman: 다음 텍스트를 요약해서 500자 이내로 설명하세오.

        {text}
        
        Assistant:"""        
    else:         
        prompt_template = """\n\nHuman: Write a concise summary of the following:

        {text}
        
        Assistant:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    docs = [
        Document(
            page_content=t
        ) for t in texts[:5]
    ]
    summary = chain.run(docs)
    print('summary: ', summary)

    if summary == '':  # error notification
        summary = 'Fail to summarize the document. Try agan...'
        return summary
    else:
        # return summary[1:len(summary)-1]   
        return summary
    
def load_chatHistory(userId, allowTime, convType):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            print('text: ', text)
            print('msg: ', msg)        

            #if (convType=='qa' and rag_type=='opensearch') or (convType=='qa' and rag_type=='kendra') or (convType=='qa' and #rag_type=='faiss' and isReady):
            #    memory_chain.chat_memory.add_user_message(text)
            #    memory_chain.chat_memory.add_ai_message(msg)           
            #elif convType=='qa' and rag_type=='faiss' and isReady==False:
            #    memory_chain.chat_memory.add_user_message(text)
            #    memory_chain.chat_memory.add_ai_message(msg)  

            #    memory_chat.save_context({"input": text}, {"output": msg})
            #else:
            #    memory_chat.save_context({"input": text}, {"output": msg})       

            if convType=='qa':
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)        
                
                if rag_type=='faiss' and isReady==False:
                    memory_chat.save_context({"input": text}, {"output": msg})
            else:
                memory_chat.save_context({"input": text}, {"output": msg})   
                
def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    print('msg: ', msg)
    return msg

def get_answer_using_template(query, rag_type, convType, connectionId, requestId):        
    if rag_type == 'faiss':
        query_embedding = vectorstore.embedding_function(query)
        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    elif rag_type == 'opensearch':
        relevant_documents = vectorstore.similarity_search(query)
    elif rag_type == 'kendra':
        relevant_documents = kendraRetriever.get_relevant_documents(query)

    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        print(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')
    
    print('length of relevant_documents: ', len(relevant_documents))

    PROMPT = get_prompt_template(query, convType)
    #print('PROMPT: ', PROMPT) 

    if rag_type=='kendra':
        retriever = kendraRetriever
    elif rag_type=='opensearch' or rag_type=='faiss':
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={
                #"k": 3, 'score_threshold': 0.8
                "k": 3
            }
        )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa({"query": query})    
    print('result: ', result)

    msg = readStreamMsg(connectionId, requestId, result['result'])

    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(relevant_documents)>=1 and enableReference=='true':
        reference = get_reference(source_documents, rag_type)
        #print('reference: ', reference)

        return msg+reference
    else:
        return msg

def get_reference(docs, rag_type):
    if rag_type == 'kendra':
        reference = "\n\nFrom\n"
        for doc in docs:
            name = doc.metadata['title']
            url = path+name

            if doc.metadata['document_attributes']:
                page = doc.metadata['document_attributes']['_excerpt_page_number']
                #reference = reference + (str(page)+'page in '+name+'\n')
                reference = reference + f"{page}page in <a href={url} target=_blank>{name}</a>\n"
            else:
                #reference = reference + name+'\n'
                reference = reference + f"<a href={url} target=_blank>{name}</a>\n"
    else:
        reference = "\n\nFrom\n"
        for doc in docs:
            name = doc.metadata['name']
            page = doc.metadata['page']
            url = doc.metadata['url']
        
            #reference = reference + (str(page)+'page in '+name+' ('+url+')'+'\n')
            reference = reference + f"{page}page in <a href={url} target=_blank>{name}</a>\n"
        
    return reference

_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
def extract_chat_history_from_memory():
    chat_history = []
    chats = memory_chain.load_memory_variables({})    
    print('chats: ', chats)

    for dialogue_turn in chats['chat_history']:
        role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
        chat_history.append(f"{role_prefix[2:]}{dialogue_turn.content}")

    return chat_history

def get_generated_prompt(query):    
    condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate(
        template = condense_template, input_variables = ["chat_history", "question"]
    )
    
    chat_history = extract_chat_history_from_memory()
    #print('chat_history: ', chat_history)
    
    question_generator_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    return question_generator_chain.run({"question": query, "chat_history": chat_history})

def get_answer_using_RAG(text, convType, connectionId, requestId):
    generated_prompt = get_generated_prompt(text) # generate new prompt using chat history
    print('generated_prompt: ', generated_prompt)
    msg = get_answer_using_template(text, rag_type, convType, connectionId, requestId) 
        
    if isDebugging:   # extract chat history for debug
        chat_history_all = extract_chat_history_from_memory() 
        print('chat_history_all: ', chat_history_all)

    memory_chain.chat_memory.add_user_message(text)  # append new diaglog
    memory_chain.chat_memory.add_ai_message(msg)

    return msg

def get_answer_from_conversation(text, conversation, convType, connectionId, requestId):
    conversation.prompt = get_prompt_template(text, convType)
    stream = conversation.predict(input=text)
    #print('stream: ', stream)
                        
    msg = readStreamMsg(connectionId, requestId, stream)

    if isDebugging:   # extract chat history for debug
        chats = memory_chat.load_memory_variables({})
        chat_history_all = chats['history']
        print('chat_history_all: ', chat_history_all)

    return msg

def get_answer_from_PROMPT(text, convType, connectionId, requestId):
    PROMPT = get_prompt_template(text, convType)
    #print('PROMPT: ', PROMPT)
    stream = llm(PROMPT.format(input=text))

    msg = readStreamMsg(connectionId, requestId, stream)
    return msg

def getResponse(connectionId, jsonBody):
    userId  = jsonBody['user_id']
    # print('userId: ', userId)
    requestId  = jsonBody['request_id']
    # print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    # print('requestTime: ', requestTime)
    type  = jsonBody['type']
    # print('type: ', type)
    body = jsonBody['body']
    # print('body: ', body)
    convType = jsonBody['convType']  # conversation type
    # print('convType: ', convType)

    global llm, modelId, vectorstore, enableReference, rag_type
    global parameters, map_chain, map_chat, memory_chat, memory_chain, isReady

    # create memory
    if convType=='qa':
        if userId in map_chain:  
            memory_chain = map_chain[userId]
            print('memory_chain exist. reuse it!')            
        else: 
            memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            map_chain[userId] = memory_chain
            print('memory_chain does not exist. create new one!')

        if rag_type=='faiss' and isReady==False:        
            if userId in map_chat:  
                memory_chat = map_chat[userId]
                print('memory_chat exist. reuse it!')    
            else: 
                memory_chat = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
                map_chat[userId] = memory_chat
                print('memory_chat does not exist. create new one!')        
            conversation = ConversationChain(llm=llm, verbose=False, memory=memory_chat)

    else:    # normal 
        if userId in map_chat:  
            memory_chat = map_chat[userId]
            print('memory_chat exist. reuse it!')
        else:
            memory_chat = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
            map_chat[userId] = memory_chat
            print('memory_chat does not exist. create new one!')        
        conversation = ConversationChain(llm=llm, verbose=False, memory=memory_chat)
        
    allowTime = getAllowTime()
    load_chatHistory(userId, allowTime, convType)

    # rag sources
    if convType == 'qa' and rag_type == 'opensearch':
        vectorstore = OpenSearchVectorSearch(
            #index_name = "rag-index-*", # all
            index_name = 'rag-index-'+userId+'-*',
            is_aoss = False,
            ef_search = 1024, # 512(default)
            m=48,
            #engine="faiss",  # default: nmslib
            embedding_function = bedrock_embeddings,
            opensearch_url=opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
        )
    elif convType == 'qa' and rag_type == 'faiss':
        print('isReady = ', isReady)

    start = int(time.time())    

    msg = ""
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)            
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'enableReference':
                enableReference = 'true'
                msg  = "Referece is enabled"
            elif text == 'disableReference':
                enableReference = 'false'
                msg  = "Reference is disabled"
            elif text == 'useOpenSearch':
                rag_type = 'opensearch'
                msg  = "OpenSearch is selected for Knowledge Database"
            elif text == 'useFaiss':
                rag_type = 'faiss'
                msg  = "Faiss is selected for Knowledge Database"
            elif text == 'useKendra':
                rag_type = 'kendra'
                msg  = "Kendra is selected for Knowledge Database"
            elif text == 'clearMemory':
                memory_chat = ""
                memory_chat = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
                map_chat[userId] = memory_chat
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:          
                if convType == 'qa':   # question & answering
                    if rag_type == 'faiss' and isReady==False:                               
                        msg = get_answer_from_conversation(text, conversation, convType, connectionId, requestId)      

                        memory_chain.chat_memory.add_user_message(text)  # append new diaglog
                        memory_chain.chat_memory.add_ai_message(msg)     
                    else: 
                        msg = get_answer_using_RAG(text, convType, connectionId, requestId)                

                elif convType == 'translation': 
                    msg = get_answer_from_PROMPT(text, convType, connectionId, requestId)

                elif convType == 'sentiment': 
                    msg = get_answer_from_PROMPT(text, convType, connectionId, requestId)

                elif convType == 'extraction': 
                    msg = get_answer_from_PROMPT(text, convType, connectionId, requestId)

                elif convType == 'pii': 
                    msg = get_answer_from_PROMPT(text, convType, connectionId, requestId)

                elif convType == 'grammar': 
                    msg = get_answer_from_PROMPT(text, convType, connectionId, requestId)

                elif convType == 'step-by-step': 
                    msg = get_answer_from_PROMPT(text, convType, connectionId, requestId)
                                    
                elif convType == 'none':   # no prompt
                    msg = llm(HUMAN_PROMPT+text+AI_PROMPT)
    
                else:     # normal
                    msg = get_answer_from_conversation(text, conversation, convType, connectionId, requestId)
                
        elif type == 'document':
            object = body

            file_type = object[object.rfind('.')+1:len(object)]
            print('file_type: ', file_type)
            
            if file_type == 'csv':
                docs = load_csv_document(object)
                texts = []
                for doc in docs:
                    texts.append(doc.page_content)
                print('texts: ', texts)
            else:
                texts = load_document(file_type, object)

                docs = []
                for i in range(len(texts)):
                    docs.append(
                        Document(
                            page_content=texts[i],
                            metadata={
                                'name': object,
                                'page':i+1,
                                'url': path+object
                            }
                        )
                    )        
                print('docs[0]: ', docs[0])    
                print('docs size: ', len(docs))
            
            msg = get_summary(texts)

            if convType == 'qa':
                if rag_type=='kendra':      
                    print('upload to kendra: ', object)           
                    store_document(path, object, requestId)  # store the object into kendra

                elif rag_type == 'faiss':
                    if isReady == False:   
                        vectorstore = FAISS.from_documents( # create vectorstore from a document
                            docs,  # documents
                            bedrock_embeddings  # embeddings
                        )
                        isReady = True
                    else:
                        vectorstore.add_documents(docs)

                elif rag_type == 'opensearch':    
                    new_vectorstore = OpenSearchVectorSearch(
                        index_name="rag-index-"+userId+'-'+requestId,
                        is_aoss = False,
                        #engine="faiss",  # default: nmslib
                        embedding_function = bedrock_embeddings,
                        opensearch_url = opensearch_url,
                        http_auth=(opensearch_account, opensearch_passwd),
                    )
                    new_vectorstore.add_documents(docs)      
                
        elapsed_time = int(time.time()) - start
        print("total run time(sec): ", elapsed_time)
        
        print('msg: ', msg)

        item = {
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg}
        }
        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except: 
            raise Exception ("Not able to write into dynamodb")        
        #print('resp, ', resp)

    return msg

def lambda_handler(event, context):
    print('event: ', event)
    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']
        print('connectionId: ', connectionId)
        routeKey = event['requestContext']['routeKey']
        print('routeKey: ', routeKey)
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                jsonBody = json.loads(body)
                print('body: ', jsonBody)

                requestId  = jsonBody['request_id']
                try:
                    msg = getResponse(connectionId, jsonBody)
                except Exception as ex:
                    err_msg = traceback.format_exc()
                    result = {
                        'request_id': requestId,
                        'msg': "The request was failed by the system: "+err_msg,
                        'status': 'failed'
                    }
                    sendMessage(connectionId, result)
                    print('err_msg: ', err_msg)
                    raise Exception ("Not able to send a message")
                                    
                result = {
                    'request_id': requestId,
                    'msg': msg,
                    'status': 'completed'
                }
                #print('result: ', json.dumps(result))
                sendMessage(connectionId, result)

    return {
        'statusCode': 200
    }
