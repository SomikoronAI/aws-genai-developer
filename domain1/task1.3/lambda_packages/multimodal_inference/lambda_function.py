# Assignement Part 3
"""
Triggered by S3 ObjectCreated events for processed data artifacts.

This Lambda function performs unified inference formatting and model invocation
for multiple data modalities that have already been pre-processed and stored
as *_processed.jsonl files in Amazon S3.

Supported processed data types:
- Audio data (Amazon Transcribe output containing "transcript")
- Image data (Amazon Rekognition/Textract output containing "extracted_text")
- Text review data (Amazon Comprehend output containing "entities")
- Survey/tabular data (Amazon SageMaker output containing "summary_text")

For each supported type, the function:
- Loads the processed JSONL artifact from S3.
- Detects the content modality based on structured keys.
- Transforms the data into Anthropic Claude-compatible message format
  using helper utilities.
- Invokes the Anthropic Claude foundation model via Amazon Bedrock.
- Returns the model inference response.
"""


import os
import json
import base64
import boto3


from utils import format_audio_data, format_image_data
from utils import format_review_data, format_survey_data


# Configuration parameters
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")
MODEL_ID    = os.environ.get("ANTHROPIC_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# Initilize client 
s3_client = boto3.client(service_name="s3", region_name=REGION_NAME)
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)



def lambda_handler(event, context):
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key    = event["Records"][0]["s3"]["object"]["key"]

    if not key.endswith( ("_processed.jsonl") ):
        return {
            "statusCode": 200, 
            "body": "Not a processed data file"
            }

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")

        processed_content = json.loads( content )

        # 'processed_data' must be a python dictionary
        if isinstance(processed_content, list):
            print("[Info] Expected dictionary, got list. Using first element.")
            processed_data = processed_content[0]
        else:
            processed_data = processed_content

        # Detect content type
        # audio data processed output from transcribe 
        if "transcript" in processed_data:
            messages = format_audio_data(processed_data)

        # image data processed output from rekognition 
        elif "extracted_text" in processed_data:
            messages = format_image_data(processed_data)

        # text review data processed output from comprehend
        elif "entities" in processed_data:
            messages = format_review_data(processed_data)

        # tabular survery data processed output from sagemaker  
        elif "summary_text" in processed_data:
            messages = format_survey_data(processed_data)
        else:
            raise ValueError("Unknown processed data format")

        # Build Claude payload
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": messages
        }

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )

        result = json.loads(response["body"].read())
        print("**************************")
        print(json.dumps(result, indent=2))
        print("**************************")

        # Save model response to S3 bucket
        s3_response = s3_client.put_object(
            Bucket=bucket,
            Key=f"domain1/task3/model_response/output.json",
            Body=json.dumps( result ),
            ContentType="application/json"
        )

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }



