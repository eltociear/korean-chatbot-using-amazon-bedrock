import json
import boto3
import os
import datetime
import uuid

sqs_client = boto3.client('sqs')
sqsUrl = json.loads(os.environ.get('queueUrl'))
print('sqsUrl: ', sqsUrl)

nqueue = os.environ.get('nqueue')

def lambda_handler(event, context):
    print('event: ', json.dumps(event))

    for i, record in enumerate(event['Records']):
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        print('bucket: ', bucket)
        print('key: ', key)
        
        idx = i % int(nqueue)
        print('idx: ', idx)
        
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
            print('sqsUrl: ', sqsUrl[idx])            
            #sqs_client.send_message(  # standard 
            #    DelaySeconds=0,
            #    MessageAttributes={},
            #    MessageBody=json.dumps(s3EventInfo),
            #    QueueUrl=sqsUrl
            #)
            
            sqs_client.send_message(  # fifo
                QueueUrl=sqsUrl[idx], 
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