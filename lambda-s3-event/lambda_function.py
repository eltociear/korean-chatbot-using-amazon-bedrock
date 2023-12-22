import json
import boto3
import os
from io import BytesIO
import traceback
from urllib import parse

from botocore.config import Config

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
meta_prefix = "metadata/"
kendra_region = os.environ.get('kendra_region', 'us-west-2')

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
opensearch_url = os.environ.get('opensearch_url')
doc_prefix = s3_prefix+'/'

kendraIndex = os.environ.get('kendraIndex')
roleArn = os.environ.get('roleArn')

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

    print('uploaded into opensearch')
    
# load csv documents from s3
def lambda_handler(event, context):
    print('event: ', event)
    
    return {
        'statusCode': 200
    }
