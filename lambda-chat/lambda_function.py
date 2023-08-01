import json
import boto3
import os
import time
import datetime
from io import BytesIO
import PyPDF2
import csv
import sys

from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from langchain.llms.bedrock import Bedrock
from langchain.document_loaders import CSVLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch

module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
configTableName = os.environ.get('configTableName')
endpoint_url = os.environ.get('endpoint_url')
opensearch_url = os.environ.get('opensearch_url')
bedrock_region = os.environ.get('bedrock_region')
rag_type = os.environ.get('rag_type')
opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
modelId = os.environ.get('model_id')
print('model_id: ', modelId)

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
        contents = doc.get()['Body'].read()
    elif file_type == 'csv':        
        body = doc.get()['Body'].read()
        reader = csv.reader(body)        
        contents = CSVLoader(reader)
    
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
            
    return texts

# Bedrock Contiguration
bedrock_region = bedrock_region
bedrock_config = {
    "region_name":bedrock_region,
    "endpoint_url":endpoint_url
}
    
# supported llm list from bedrock
boto3_bedrock = bedrock.get_bedrock_client(
    region=bedrock_config["region_name"],
    url_override=bedrock_config["endpoint_url"])
    
modelInfo = boto3_bedrock.list_foundation_models()    
print('models: ', modelInfo)

llm = Bedrock(model_id=modelId, client=boto3_bedrock)

# embedding
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)

vectorstore = OpenSearchVectorSearch(
    index_name = "rag-index-*",
    is_aoss = False,
    embedding_function = bedrock_embeddings,
    opensearch_url=opensearch_url,
    http_auth=(opensearch_account, opensearch_passwd),
)

def lambda_handler(event, context):
    print(event)
    userId  = event['user-id']
    print('userId: ', userId)
    requestId  = event['request-id']
    print('requestId: ', requestId)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)

    global modelId, llm, vectorstore, rag_type
    
    start = int(time.time())    

    msg = ""
    if type == 'text':
        text = body
        msg = llm(text)
            
    elif type == 'document':
        object = body
        
        file_type = object[object.rfind('.')+1:len(object)]
        print('file_type: ', file_type)
            
        # load documents where text, pdf, csv are supported
        texts = load_document(file_type, object)
        docs = [
            Document(
                page_content=t,
                metadata={
                    'name': object,
                    'page':1
                }
            ) for t in texts[:3]
        ]
        print('docs: ', docs)

        # embedding
        bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
                        
        new_vectorstore = OpenSearchVectorSearch.from_documents(
            docs, 
            bedrock_embeddings, 
            opensearch_url=opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd),
            index_name="rag-index"+userId
        )
        
        # summerization to show the document
        docs = [
            Document(
                page_content=t
            ) for t in texts[:3]
        ]
            
        prompt_template = """Write a concise summary of the following:

        {text}
                
        CONCISE SUMMARY """

        print('template: ', prompt_template)
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        summary = chain.run(docs)
        print('summary: ', summary)

        msg = summary
                
    elapsed_time = int(time.time()) - start
    print("total run time(sec): ", elapsed_time)

    print('msg: ', msg)

    return {
        'statusCode': 200,
        'msg': msg,
    }