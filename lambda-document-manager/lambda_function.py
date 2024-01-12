import json
import boto3
import os
import traceback
import PyPDF2
import time
import docx

from io import BytesIO
from urllib import parse
from botocore.config import Config
from urllib.parse import unquote_plus
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
meta_prefix = "metadata/"
kendra_region = os.environ.get('kendra_region', 'us-west-2')

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
opensearch_url = os.environ.get('opensearch_url')
kendraIndex = os.environ.get('kendraIndex')

roleArn = os.environ.get('roleArn') 
path = os.environ.get('path')

capabilities = json.loads(os.environ.get('capabilities'))
print('capabilities: ', capabilities)

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
        print('delete opensearch document index: ', index_name)
        response = client.indices.delete(
            index=index_name
        )
        print('removed index: ', response)    
    else:
        print('no index: ', index_name)

# Kendra
kendra_client = boto3.client(
    service_name='kendra', 
    region_name=kendra_region,
    config = Config(
        retries=dict(
            max_attempts=10
        )
    )
)

# embedding for RAG
bedrock_region = "us-west-2"
    
boto3_bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=bedrock_region,
    config=Config(
        retries = {
            'max_attempts': 30
        }            
    )
)

bedrock_embeddings = BedrockEmbeddings(
    client=boto3_bedrock,
    region_name = bedrock_region,
    model_id = 'amazon.titan-embed-text-v1' 
)   

def store_document_for_opensearch(bedrock_embeddings, docs, documentId):
    index_name = "rag-index-"+documentId
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
def store_document_for_kendra(path, key, documentId):
    print('store document into kendra')
    encoded_name = parse.quote(key)
    source_uri = path+encoded_name    
    #print('source_uri: ', source_uri)
    ext = (key[key.rfind('.')+1:len(key)]).upper()
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
            "Title": key,
            "S3Path": {
                "Bucket": s3_bucket,
                "Key": key
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
    print('batch_put_document(kendra): ', result)
    print('uploaded into kendra')

def create_metadata(bucket, key, meta_prefix, s3_prefix, uri, category, documentId):    
    title = key
    timestamp = int(time.time())

    metadata = {
        "Attributes": {
            "_category": category,
            "_source_uri": uri,
            "_version": str(timestamp),
            "_language_code": "ko"
        },
        "Title": title,
        "DocumentId": documentId,        
    }
    print('metadata: ', metadata)
    
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)]).upper()
    print('objectName: ', objectName)

    client = boto3.client('s3')
    try: 
        client.put_object(
            Body=json.dumps(metadata), 
            Bucket=bucket, 
            Key=meta_prefix+objectName+'.metadata.json' 
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")

# load documents from s3 for pdf and txt
def load_document(file_type, key):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, key)
    
    if file_type == 'pdf':
        Byte_contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(Byte_contents))
        
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text())
        contents = '\n'.join(texts)
        
    elif file_type == 'pptx':
        Byte_contents = doc.get()['Body'].read()
            
        from pptx import Presentation
        prs = Presentation(BytesIO(Byte_contents))

        texts = []
        for i, slide in enumerate(prs.slides):
            text = ""
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = text + shape.text
            texts.append(text)
        contents = '\n'.join(texts)
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read().decode('utf-8')

    elif file_type == 'docx':
        Byte_contents = doc.get()['Body'].read()                    
        doc_contents =docx.Document(BytesIO(Byte_contents))

        texts = []
        for i, para in enumerate(doc_contents.paragraphs):
            if(para.text):
                texts.append(para.text)
                # print(f"{i}: {para.text}")        
        contents = '\n'.join(texts)
            
    # print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 

    texts = text_splitter.split_text(new_contents) 
                
    return texts
    
# load csv documents from s3
def lambda_handler(event, context):
    print('event: ', event)    
    
    documentIds = []
    for record in event['Records']:
        bucket = record['bucket']
        # translate utf8
        key = unquote_plus(record['key']) # url decoding
        print('bucket: ', bucket)
        print('key: ', key)
        
        start_time = time.time()            
        if record['eventName'] == 'ObjectRemoved:Delete':                        
            objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)]).upper()
            print('objectName: ', objectName)
            
            # get metadata from s3
            metadata_key = meta_prefix+objectName+'.metadata.json'
            print('metadata_key: ', metadata_key)

            metadata_obj = s3.get_object(Bucket=bucket, Key=metadata_key)
            metadata_body = metadata_obj['Body'].read().decode('utf-8')
            metadata = json.loads(metadata_body)
            print('metadata: ', metadata)
            documentId = metadata['DocumentId']
            print('documentId: ', documentId)
            documentIds.append(documentId)

            # delete metadata
            print('delete metadata: ', metadata_key)
            try: 
                result = s3.delete_object(Bucket=bucket, Key=metadata_key)
                # print('result of metadata deletion: ', result)
            except Exception:
                err_msg = traceback.format_exc()
                print('err_msg: ', err_msg)
                raise Exception ("Not able to delete documents in Kendra")
    
            # delete document index of opensearch
            index_name = "rag-index-"+documentId
            # print('index_name: ', index_name)
            delete_index_if_exist(index_name)
            
            # delete kendra documents
            print('delete kendra documents: ', documentIds)
            
            try: 
                result = kendra_client.batch_delete_document(
                    IndexId = kendraIndex,
                    DocumentIdList=[
                        documentId,
                    ]
                )
                print('result: ', result)
            except Exception:
                err_msg = traceback.format_exc()
                print('err_msg: ', err_msg)
                raise Exception ("Not able to delete documents in Kendra")
        elif record['eventName'] == "ObjectCreated:Put":
            category = "upload"
            documentId = category + "-" + key
            documentId = documentId.replace(' ', '_') # remove spaces
            documentId = documentId.replace(',', '_') # remove commas
            documentId = documentId.replace('/', '_') # remove slash
            documentId = documentId.lower() # change to lowercase
            print('documentId: ', documentId)
            
            file_type = key[key.rfind('.')+1:len(key)]            
            print('file_type: ', file_type)
            
            for type in capabilities:                
                if type=='kendra':         
                    print('upload to kendra: ', key)                            
                    store_document_for_kendra(path, key, documentId)  # store the object into kendra
                    
                    create_metadata(bucket=s3_bucket, key=key, meta_prefix=meta_prefix, s3_prefix=s3_prefix, uri=path+parse.quote(key), category=category, documentId=documentId)
                
                elif type=='opensearch':
                    if file_type == 'pdf' or file_type == 'txt' or file_type == 'csv' or file_type == 'pptx' or file_type == 'docx':
                        print('upload to opensearch: ', key) 
                        texts = load_document(file_type, key)

                        docs = []
                        for i in range(len(texts)):
                            docs.append(
                                Document(
                                    page_content=texts[i],
                                    metadata={
                                        'name': key,
                                        # 'page':i+1,
                                        'uri': path+parse.quote(key)
                                    }
                                )
                            )
                        print('docs[0]: ', docs[0])    
                        print('docs size: ', len(docs))
                    
                        store_document_for_opensearch(bedrock_embeddings, docs, documentId)                        
        print('processing time: ', str(time.time() - start_time))
    return {
        'statusCode': 200
    }
