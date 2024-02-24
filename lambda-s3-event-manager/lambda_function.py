import json
import boto3
import os
import uuid

sqs_client = boto3.client('sqs')
sqsUrl = json.loads(os.environ.get('queueUrl'))
print('sqsUrl: ', sqsUrl)

nqueue = os.environ.get('nqueue')

def lambda_handler(event, context):
    print('event: ', json.dumps(event))

    for i, record in enumerate(event['Records']):
        receiptHandle = record['receiptHandle']
        print("receiptHandle: ", receiptHandle)
        
        body = record['body']
        print("body: ", json.loads(body))
        
        idx = i % int(nqueue)
        print('idx: ', idx)
        
        eventId = str(uuid.uuid1())
        print('eventId: ', eventId)
        
        # push to SQS
        try:
            print('sqsUrl: ', sqsUrl[idx])            
            #sqs_client.send_message(  # standard 
            #    DelaySeconds=0,
            #    MessageAttributes={},
            #    MessageBody=body,
            #    QueueUrl=sqsUrl[idx])
            #)
            
            sqs_client.send_message(  # fifo
                QueueUrl=sqsUrl[idx], 
                MessageAttributes={},
                MessageDeduplicationId=eventId,
                MessageGroupId="putEvent",
                MessageBody=body
            )
            print('Successfully push the queue message: ', body)

        except Exception as e:        
            print('Fail to push the queue message: ', e)
        
    return {
        'statusCode': 200
    }