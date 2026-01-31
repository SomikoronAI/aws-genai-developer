# Assignment Part 3

"""
Handles fallback language model invocation using a lightweight model.
Processes Step Functions events and returns a generated or safe response.
Ensures continued operation when primary model calls fail.
"""

import json
import os
from invoke_model import invoke_model


# Configuration parameters 
region_name = os.environ["REGION_NAME"]

# Use a simpler, more reliable model as fallback model
model_id = "us.amazon.nova-pro-v1:0" 


def lambda_handler(event, context):
    """
    Fallback model handler that uses a simpler model.
    This function is invoked by Step Functions, not API Gateway.
    """

    # Normalize input across API Gateway and Step Functions
    if "body" in event and isinstance(event["body"], str):
        body = json.loads(event["body"])
    else:
        body = event

    # 1. Extract information from the event 
    prompt   = body.get("prompt", "")
    use_case = body.get("use_case", "general")
    context  = context
    
    safe_response = (
        "I'm currently experiencing high demand, but I can still help. "
        "Please try again if you need more detailed assistance."
    )

    # 2. Invoke the model with simplified parameters 
    try:
        response = invoke_model(model_id, prompt, temperature=0.2, top_p=0.5, max_tokens=256)

        # 3. Validate the model response
        if not isinstance(response, dict):
            raise RuntimeError("Fallback model returned non-dict response")

        success = response.get("success", False)
        output  = response.get("output")

        if not success or not output:
            raise RuntimeError("Fallback model returned invalid or empty output")


        # 4. Return successful response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'model_used': f"FALLBACK: {model_id}",
                'response': output
            })
        }

    # 5. Handle fallback failure gracefully
    except Exception as e:
        # raise
        return {
            "statusCode": 200,
            "body": json.dumps({
                "model_used": "FALLBACK_MESSAGE",
                "response": safe_response,
                "degraded": True,
                "error": str(e)
            })
        }