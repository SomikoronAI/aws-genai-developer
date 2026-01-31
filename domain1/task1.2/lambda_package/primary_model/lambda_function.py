# Assignment Part 2
 
"""
Processes API Gateway requests to generate AI responses.
Retrieves AppConfig settings to select an appropriate model per use case.
Invokes the chosen model with the request prompt and returns the result.
"""

import boto3
import json
import os

from select_model import select_model
from invoke_model import invoke_model


# Configuration parameters
REGION_NAME = os.environ["REGION_NAME"]
APPLICATION = os.environ["APPCONFIG_APPLICATION"]
ENVIRONMENT = os.environ["APPCONFIG_ENVIRONMENT"]
CONFIGURATION = os.environ["APPCONFIG_CONFIGURATION"]

# Initilize client 
appconfig_client = boto3.client( service_name="appconfig", region_name=REGION_NAME )


def lambda_handler(event, context):
    """Primary model handler invoked by API Gateway."""

    try:
        # 1. Extract information from the event 
        # event["body"] is always string
        # json.loads( event["body"] ) is always a dictionary

        # Normalize input across API Gateway and Step Functions
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event["body"])
        else:
            body = event

        prompt   = body.get("prompt", "")    
        use_case = body.get("use_case", "general")
        context  = context

    except Exception as e:
        return {
            "statusCode": 400, 
            "body": json.dumps({"error": "Invalid JSON in request body."})
        }

    try:
        # 2. Fetch AppConfig configuration
        config_response = appconfig_client.get_configuration(
            Application=APPLICATION,
            Environment=ENVIRONMENT,
            Configuration=CONFIGURATION,
            ClientId="AIAssistantLambda"
        )
        config = json.loads( config_response["Content"].read().decode("utf-8") )
    except Exception as e: 
        return { 
            "statusCode": 500, 
            "body": json.dumps({"error": "Failed to load configuration."}) 
        }

    try:
        # 3. Combine 1 & 2 to generate response 
        model_id = select_model(config, use_case)
        response = invoke_model(model_id, prompt)

        # Validate the model response
        if not isinstance(response, dict):
            raise RuntimeError("Primary model returned non-dict response")

        success = response.get("success", False)
        output  = response.get("output")

        if not success or not output:
            raise RuntimeError("Primary model returned invalid or empty output")

        return {
        "statusCode": 200,
        "body": json.dumps({
            "model_used": model_id,
            "response": output
            })
        }

    except Exception as e: 
        return { 
            "statusCode": 500, 
            "body": json.dumps({"error": "Model selection or Model response failed."}) 
        }



