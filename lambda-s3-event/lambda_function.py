import json
import boto3
import os
import traceback
from botocore.config import Config
from urllib.parse import unquote

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
#roleArn = os.environ.get('roleArn')

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

# load csv documents from s3
def lambda_handler(event, context):
    print('event: ', event)

    documentIds = []
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        # translate utf8
        key = record['s3']['object']['key'] # url decoding
        print('bucket: ', bucket)
        print('key: ', key)

        from urllib.parse import unquote_plus
        object_key = unquote_plus(key)
        print('object_key: ', object_key)

        # get metadata from s3
        metadata_key = meta_prefix+object_key+'.metadata.json'
        print('metadata_key: ', metadata_key)


        

        metadata_obj = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata_body = metadata_obj['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_body)
        print('metadata: ', metadata)
        documentId = metadata['DocumentId']
        print('documentId: ', documentId)

        documentIds.append(documentId)



        # delete metadata
        try: 
            result = s3.delete_object(Bucket=bucket, Key=metadata_key)
            print('result to delete metadata in S3: ', result)
        except Exception:
            err_msg = traceback.format_exc()
            print('err_msg: ', err_msg)
            raise Exception ("Not able to delete documents in Kendra")
  
        # delete index of opensearch
        userId = 'kyopark'
        index_name = "rag-index-"+userId+'-'+documentId
        # index_name = "rag-index-"+documentId
        index_name = index_name.replace(' ', '_') # remove spaces
        index_name = index_name.replace(',', '_') # remove commas
        index_name = index_name.lower() # change to lowercase
        print('index_name: ', index_name)

        delete_index_if_exist(index_name)

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

    
    return {
        'statusCode': 200
    }
