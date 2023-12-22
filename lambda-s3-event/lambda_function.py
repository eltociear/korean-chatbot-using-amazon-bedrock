import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv
import traceback
import re
from urllib import parse

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from botocore.config import Config

from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from multiprocessing import Process, Pipe
from googleapiclient.discovery import build

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
meta_prefix = "metadata/"
callLogTableName = os.environ.get('callLogTableName')
kendra_region = os.environ.get('kendra_region', 'us-west-2')
profile_of_LLMs = json.loads(os.environ.get('profile_of_LLMs'))
isReady = False   
rag_method = os.environ.get('rag_method', 'RetrievalPrompt') # RetrievalPrompt, RetrievalQA, ConversationalRetrievalChain

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
enableReference = os.environ.get('enableReference', 'false')
debugMessageMode = os.environ.get('debugMessageMode', 'false')
opensearch_url = os.environ.get('opensearch_url')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'
speech_prefix = 'speech/'

useParallelUpload = os.environ.get('useParallelUpload', 'true')
useParallelRAG = os.environ.get('useParallelRAG', 'true')
kendraIndex = os.environ.get('kendraIndex')
kendra_method = os.environ.get('kendraMethod')
roleArn = os.environ.get('roleArn')
top_k = int(os.environ.get('numberOfRelevantDocs', '8'))
selected_LLM = 0
capabilities = json.loads(os.environ.get('capabilities'))
print('capabilities: ', capabilities)
MSG_LENGTH = 100
MSG_HISTORY_LENGTH = 20
speech_generation = True
history_length = 0
token_counter_history = 0

from opensearchpy import OpenSearch
def delete_index_if_exist(index_name):
    client = OpenSearch(
        hosts = [{
            'host': opensearch_url.replace("https://", ""), 
            'port': 443
        }],
        http_compress = True,
        http_auth=(opensearch_account, opensearch_passwd),
        use_ssl = True,
        verify_certs = True,
        ssl_assert_hostname = False,
        ssl_show_warn = False,
    )

    if client.indices.exists(index_name):
        print('remove index: ', index_name)
        response = client.indices.delete(
            index=index_name
        )
        print('response(remove): ', response)    
    else:
        print('no index: ', index_name)

def store_document_for_opensearch(bedrock_embeddings, docs, userId, documentId):
    index_name = "rag-index-"+userId+'-'+documentId
    index_name = index_name.replace(' ', '_') # remove spaces
    index_name = index_name.replace(',', '_') # remove commas
    index_name = index_name.lower() # change to lowercase
    print('index_name: ', index_name)
    
    delete_index_if_exist(index_name)

    try:
        new_vectorstore = OpenSearchVectorSearch(
            index_name=index_name,  
            is_aoss = False,
            #engine="faiss",  # default: nmslib
            embedding_function = bedrock_embeddings,
            opensearch_url = opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd),
        )
        response = new_vectorstore.add_documents(docs)
        print('response of adding documents: ', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                
        raise Exception ("Not able to request to LLM")

    print('uploaded into opensearch')
    
# store document into Kendra
def store_document_for_kendra(path, doc_prefix, s3_file_name, documentId):
    print('store document into kendra')
    encoded_name = parse.quote(s3_file_name)
    source_uri = path+doc_prefix+encoded_name    
    #print('source_uri: ', source_uri)
    ext = (s3_file_name[s3_file_name.rfind('.')+1:len(s3_file_name)]).upper()
    print('ext: ', ext)

    # PLAIN_TEXT, XSLT, MS_WORD, RTF, CSV, JSON, HTML, PDF, PPT, MD, XML, MS_EXCEL
    if(ext == 'PPTX'):
        file_type = 'PPT'
    elif(ext == 'TXT'):
        file_type = 'PLAIN_TEXT'         
    elif(ext == 'XLS' or ext == 'XLSX'):
        file_type = 'MS_EXCEL'      
    elif(ext == 'DOC' or ext == 'DOCX'):
        file_type = 'MS_WORD'
    else:
        file_type = ext

    kendra_client = boto3.client(
        service_name='kendra', 
        region_name=kendra_region,
        config = Config(
            retries=dict(
                max_attempts=10
            )
        )
    )

    documents = [
        {
            "Id": documentId,
            "Title": s3_file_name,
            "S3Path": {
                "Bucket": s3_bucket,
                "Key": s3_prefix+'/'+s3_file_name
            },
            "Attributes": [
                {
                    "Key": '_source_uri',
                    'Value': {
                        'StringValue': source_uri
                    }
                },
                {
                    "Key": '_language_code',
                    'Value': {
                        'StringValue': "ko"
                    }
                },
            ],
            "ContentType": file_type
        }
    ]
    print('document info: ', documents)

    result = kendra_client.batch_put_document(
        IndexId = kendraIndex,
        RoleArn = roleArn,
        Documents = documents       
    )
    # print('batch_put_document(kendra): ', result)
    print('uploaded into kendra')

# load csv documents from s3
def lambda_handler(event, context):
    print('event: ', event)
    
    return {
        'statusCode': 200
    }
