import json
import boto3
import os
import uuid
import traceback

sqs_client = boto3.client('sqs')
sqsUrl = os.environ.get('queueS3event')
s3 = boto3.client('s3')
max_object_size = int(os.environ.get('max_object_size'))

supportedFormat = json.loads(os.environ.get('supportedFormat'))
print('supportedFormat: ', supportedFormat)

def isSupported(type):
    for format in supportedFormat:
        if type == format:
            return True    
    return False
    
def check_supported_type(file_type, size):
    if size > 5000 and size<max_object_size and isSupported(file_type):
        return True
    if size > 0 and file_type == 'txt':
        return True
    else:
        return False

def lambda_handler(event, context):
    print('event: ', json.dumps(event))

    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        print('bucket: ', bucket)
        print('key: ', key)
        
        file_type = key[key.rfind('.')+1:len(key)].lower()
        print('file_type: ', file_type)
        
        size = 0
        try:
            s3obj = s3.get_object(Bucket=bucket, Key=key)
            print(f"Got object: {s3obj}")        
            size = int(s3obj['ContentLength'])                
            print('object size: ', size)
        except Exception:
            err_msg = traceback.format_exc()
            print('err_msg: ', err_msg)
            # raise Exception ("Not able to get object info") 
        
        if check_supported_type(file_type, size):
            eventId = str(uuid.uuid1())
            print('eventId: ', eventId)
            
            s3EventInfo = {
                'event_id': eventId,
                'event_timestamp': record['eventTime'],
                'bucket': bucket,
                'key': key,
                'type': record['eventName']
            }
                    
            # push to SQS
            try:
                sqs_client.send_message(
                    QueueUrl=sqsUrl, 
                    MessageAttributes={},
                    MessageDeduplicationId=eventId,
                    MessageGroupId="putEvent",
                    MessageBody=json.dumps(s3EventInfo)
                )
                print('Successfully push the queue message: ', json.dumps(s3EventInfo))

            except Exception as e:        
                print('Fail to push the queue message: ', e)
            
    return {
        'statusCode': 200
    }