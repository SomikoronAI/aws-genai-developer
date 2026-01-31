# Assignment Part 2

import os
import json
import time
import boto3

from payload_template import PayloadTemplateManager
from tokenextract_template import TokenExtractorTemplateManager

# Configuration parameters 
region_name = os.environ["REGION_NAME"]
ptm_payload = PayloadTemplateManager()
ttm_token = TokenExtractorTemplateManager()


# Initialize Bedrock client
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)


def invoke_model(model_id, prompt, max_tokens=512):
    """Invoke a model with the given prompt and return the response and metrics."""
    start_time = time.time()
    
    try:
        # Prepare request body based on model provider
        if "claude" in model_id:
            # Get payload
            payload = ptm_payload.get_payload_claude(
                "messages_api", 
                prompt, 
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload ), 
                ) 
            # Parse response
            response_body = json.loads(response['body'].read().decode())
            output = response_body['content'][0]['text']

        elif "nova" in model_id:
            # Get payload
            payload = ptm_payload.get_payload_nova(
                prompt, 
                temperature=0.7,
                top_p=0.9, 
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload ), 
                contentType="application/json", 
                accept="application/json"
                )
            # Parse response
            response_body = json.loads(response['body'].read().decode())
            output = response_body["output"]["message"]["content"][0]["text"]

        elif "llama" in model_id:
            payload = ptm_payload.get_payload_llama(
                prompt, 
                temperature=0.7,
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload ), 
                contentType="application/json", 
                accept="application/json"
                )
            # Parse response
            response_body = json.loads(response['body'].read().decode()) 
            output = response_body["generation"]

        elif "mistral" in model_id:
            payload = ptm_payload.get_payload_mistral(
                prompt, 
                temperature=0.7, 
                top_p=0.9,
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload )
                )
            # Parse response
            response_body = json.loads(response['body'].read().decode()) 
            output = response_body["outputs"][0]["text"]

        # Calculate metrics
        latency = time.time() - start_time
        token_count = ttm_token.get_tokens(model_id, prompt, response_body)
        
        return {
            "success": True,
            "output": output,
            "latency": latency,
            "input_tokens": token_count[0],
            "output_tokens": token_count[1]
        }
    except Exception as e:
        print("Model Invocation Failed:", repr(e))
        return {
            "success": False,
            "output": False, 
            "latency": time.time() - start_time, 
            "error": str(e)
        }
