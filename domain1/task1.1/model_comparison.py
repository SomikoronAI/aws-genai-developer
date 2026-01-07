
"""
Compares multiple Bedrock models on extracting information from an insurance claim document.

Workflow:
1. Fetches the document text from an S3 bucket.
2. Builds a prompt using PromptTemplateManager and a payload using PayloadTemplateManager.
3. Invokes each model in the provided list via Bedrock Runtime.
4. Measures response time, output length, and returns a sample of the output.

Functions:
    get_document(bucket_name, key_name): Retrieves and decodes text from an S3 object.
    compare_models(document_text, model_list): Sends extraction requests to models and returns performance metrics.
"""

import pandas as pd
import json
import time
import boto3
from botocore.exceptions import ClientError

from prompt_template import PromptTemplateManager
from payload_template import PayloadTemplateManager
from bedrock_runtime_template import BedrockRuntimeInvokeManager


# Configuration parameters 
region_name="us-east-1"
ptm_prompt = PromptTemplateManager()
ptm_payload = PayloadTemplateManager()
bedrock_invoker = BedrockRuntimeInvokeManager(region_name)

# Initialize clients
s3_client = boto3.client(service_name="s3", region_name=region_name)
# bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)


def get_document(bucket_name, key_name):
    # Get document from S3 GP bucket
    response = s3_client.get_object(Bucket=bucket_name, Key=key_name)
    document_text = response['Body'].read().decode('utf-8')
    return document_text 


def compare_models(document_text, model_list):
    # Get data extraction prompt for the model
    extraction_prompt = ptm_prompt.get_prompt("extract_info", document_text=document_text)

    # Get payload to invoke the model
    extraction_payload = ptm_payload.get_payload_claude(
        "messages_api",
        user_prompt=extraction_prompt,
        temperature=0.0,
        max_tokens=1024
    )

    results = {}
    
    for model in model_list:
        start_time = time.time()

        response = bedrock_invoker.invoke(model, extraction_payload)        

        # Calculate metrics
        elapsed_time = time.time() - start_time
        response_body = json.loads(response['body'].read())
        output = response_body['content']

        results[model] = {
            "time_seconds": round(elapsed_time, 6),
            "output_length": len(output),
            "output_sample": output
        }
    
    return results


# Example usage
if __name__ == "__main__":
    bucket_name = "claim-documents-poc-nururrahman"
    key_name = "claims/auto_insurance_claim1.txt"

    model_list = ["us.anthropic.claude-3-sonnet-20240229-v1:0", 
                  "us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
                  "us.anthropic.claude-sonnet-4-20250514-v1:0"
                  ]
    
    document_text = get_document(bucket_name, key_name)
    result = compare_models(document_text, model_list)
    # print( json.dumps(result, indent=2) )
    # print(result)

    time_list = []
    for model in model_list:
        time = result[model]["time_seconds"]
        time_list.append(time)

df = pd.DataFrame({"models": model_list, "task_time": time_list})
print(df)