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

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
bedrock_region = os.environ.get('bedrock_region', 'us-west-2')
modelId = os.environ.get('model_id', 'anthropic.claude-v2')
print('model_id: ', modelId)
rag_type = os.environ.get('rag_type', 'faiss')
isReady = False   

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
enableReference = os.environ.get('enableReference', 'false')
opensearch_url = os.environ.get('opensearch_url')
path = os.environ.get('path')

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
    except: 
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
    model_id = 'amazon.titan-embed-text-v1' # amazon.titan-e1t-medium, amazon.titan-embed-g1-text-02 amazon.titan-embed-text-v1
)

map = dict() # Conversation

def get_prompt_template(query, convType):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(query))
    print('word_kor: ', word_kor)

    if word_kor:    
        if convType=='qa':
            condense_template = """다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant는 모르는 질문을 받으면 솔직히 모른다고 말합니다.
        
            {context}
            
            Question: {question}

            Assistant:"""
        elif convType == "translation":
            condense_template = """
            
            Human: 다음을 영어로 번역해줘:{input}
            
            Assistant:"""
        
        else: # normal
            condense_template = """다음은 Human과 Assistant의 친근한 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. 아래 문맥(context)을 참조했음에도 답을 알 수 없다면, 솔직히 모른다고 말합니다.

            Current conversation:
            {history}
            
            Human: {input}
            
            Assistant:"""
    else:  # English
        if convType=="translation": 
            condense_template = """
            
            Human: 다음을 한국어로 번역해줘:{input}

            Assistant:"""
        else:
            condense_template = """Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor.

            {history}
            
            Human: {input}

            Assistant:"""

        #claude_prompt = PromptTemplate.from_template("""The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    return PromptTemplate.from_template(condense_template)

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
                'row': n+1,
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
        ) for t in texts[:3]
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

            if (convType=='qa' and rag_type=='opensearch') or (convType=='qa' and rag_type=='faise' and isReady):
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)           
            elif convType=='qa' and rag_type=='faise' and isReady==False:
                memory_chain.chat_memory.add_user_message(text)
                memory_chain.chat_memory.add_ai_message(msg)  

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
            print('event: ', event)
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    print('msg: ', msg)
    return msg

def get_answer_using_template(query, vectorstore, rag_type, convType):        
    if rag_type == 'faiss':
        query_embedding = vectorstore.embedding_function(query)
        relevant_documents = vectorstore.similarity_search_by_vector(query_embedding)
    elif rag_type == 'opensearch':
        relevant_documents = vectorstore.similarity_search(query)

    print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
    print('----')
    for i, rel_doc in enumerate(relevant_documents):
        print(f'## Document {i+1}: {rel_doc.page_content}.......')
    print('---')
    
    print('length of relevant_documents: ', len(relevant_documents))

    PROMPT = get_prompt_template(query, convType)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={
                #"k": 3, 'score_threshold': 0.8
                "k": 3
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa({"query": query})
    print('result: ', result)
    source_documents = result['source_documents']
    print('source_documents: ', source_documents)

    if len(relevant_documents)>=1 and enableReference=='true':
        reference = get_reference(source_documents)
        #print('reference: ', reference)

        return result['result']+reference
    else:
        return result['result']

def get_reference(docs):
    reference = "\n\nFrom\n"
    for doc in docs:
        name = doc.metadata['name']
        page = doc.metadata['row']
        url = doc.metadata['url']
    
        #reference = reference + (str(page)+'page in '+name+' ('+url+')'+'\n')
        reference = reference + f"({page}page in {name} ({url})\n"
        
    return reference

_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
def extract_chat_history_from_memory():
    chat_history = []
    chats = memory_chain.load_memory_variables({})    
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

def get_answer_using_RAG(text, vectorstore, convType):
    generated_prompt = get_generated_prompt(text) # generate new prompt using chat history
    print('generated_prompt: ', generated_prompt)
    msg = get_answer_using_template(text, vectorstore, rag_type, convType) 

    # extract chat history for debug
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

    # extract chat history for debug
    chats = memory_chat.load_memory_variables({})
    chat_history_all = chats['history']
    print('chat_history_all: ', chat_history_all)
    return msg

def get_answer_from_PROMPT(text, convType, connectionId, requestId):
    PROMPT = get_prompt_template(text, convType)
    print('PROMPT: ', PROMPT)
    msg = llm(PROMPT.format(input=text))

    print('msg from PROMPT: ', msg)

    # msg = readStreamMsg(connectionId, requestId, stream)
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

    global modelId, llm, parameters, map, memory_chat, memory_chain, isReady, vectorstore

    # create memory
    if (convType == 'qa' and rag_type == 'opensearch') or (convType == 'qa' and rag_type == 'faiss' and isReady):
        if userId in map:  
            memory_chain = map[userId]
            print('memory_chain exist. reuse it!')            
        else: 
            memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            map[userId] = memory_chain
            print('memory_chain does not exist. create new one!')
    else:    
        if userId in map:  
            memory_chat = map[userId]
            print('memory_chat exist. reuse it!')
        else:
            memory_chat = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
            map[userId] = memory_chat
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

            if text == 'clearMemory':
                memory_chat = ""
                memory_chat = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant')
                map[userId] = memory_chat
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:          
                if convType == 'qa':   # question & answering
                    if rag_type == 'faiss' and isReady == False: 
                        msg = get_answer_from_conversation(text, conversation, convType, connectionId, requestId)
                    else: 
                        msg = get_answer_using_RAG(text, vectorstore, convType)

                elif convType == 'translation': 
                    msg = get_answer_from_PROMPT(text, convType, connectionId, requestId)
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

            if convType == 'qa' and rag_type == 'faiss':
                if isReady == False:   
                    vectorstore = FAISS.from_documents( # create vectorstore from a document
                        docs,  # documents
                        bedrock_embeddings  # embeddings
                    )
                    isReady = True
                else:
                    vectorstore.add_documents(docs)
                    print('vector store size: ', len(vectorstore.docstore._dict))

            elif convType == 'qa' and rag_type == 'opensearch':    
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
                print("ping!.....")                
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
                        'msg': "The request was failed by the system: "+err_msg
                    }
                    sendMessage(connectionId, result)
                    print('err_msg: ', err_msg)
                    raise Exception ("Not able to send a message")
                                    
                result = {
                    'request_id': requestId,
                    'msg': msg
                }
                #print('result: ', json.dumps(result))
                sendMessage(connectionId, result)

    return {
        'statusCode': 200
    }
