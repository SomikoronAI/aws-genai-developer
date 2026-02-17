# Assignment Part 1

import os
import json
import uuid
import pathlib
from dotenv import load_dotenv, find_dotenv
from typing import Any, List, Dict, Optional, Tuple

import boto3


# Configuration parameters
try:
    env_file = os.getenv(".env")
    if env_file:
        load_dotenv(pathlib.Path(env_file).expanduser().resolve())
    else:
        load_dotenv(find_dotenv(usecwd=True) or ".env")
except Exception:
    pass

region_name     = os.environ.get("REGION_NAME", "us-east-1")
account_id      = os.environ.get("ACCOUNT_ID", "")
lambda_role_arn = os.environ.get("LAMBDA_ROLE_ARN", "")

event_rule_name     = "TextValidationCompleted"
# Lambda name
target_lambda_name  = "GlueDataQuality"
# Lambda ARN
target_lambda_arn   = f"arn:aws:lambda:{region_name}:{account_id}:function:{target_lambda_name}"


# Iniitialize client
eventbridge_client = boto3.client(service_name="events", region_name=region_name)
lambda_client = boto3.client(service_name="lambda", region_name=region_name)


event_pattern = {
    "source": ["customer-feedback.validation-lambda"],
    "detail-type": ["TextValidationCompleted"],
    "detail": {
        "status": ["SUCCESS"],
        "pipeline": ["customer-feedback"],
    }
}


# --------------------------------------
# Create a rule
# --------------------------------------
def event_put_rules(eventbridge_client, event_rule_name: str, event_pattern: Dict):
    eventbridge_client.put_rule(
        Name=event_rule_name,
        EventPattern=json.dumps(event_pattern),
        State="ENABLED",
        Description="Trigger Glue DQ evaluation run after text validation completes",
    )
    print(f"[OK] Rule created/updated: {event_rule_name}")


# --------------------------------------
# Attach a target to the rule 
# --------------------------------------
def event_put_targets(eventbridge_client, event_rule_name: str, target_lambda_arn: str):
    eventbridge_client.put_targets(
        Rule=event_rule_name,
        Targets=[
            {
                "Id": "GlueDataQualityTarget",
                "Arn": target_lambda_arn
            }
        ]
    )
    lambda_name = target_lambda_arn.split(":")[-1]
    print(f"[OK] Target attached: {lambda_name}")


# --------------------------------------
# Allow EventBridge to invoke the Lambda
# --------------------------------------
def allow_eventbridge_invoke_lambda(lambda_client, target_lambda_name: str, event_rule_name: str):
    lambda_client.add_permission(
        FunctionName=target_lambda_name,
        StatementId=f"AllowEventBridgeInvokeDataQuality-{uuid.uuid4()}",
        Action="lambda:InvokeFunction",
        Principal="events.amazonaws.com",
        SourceArn=f"arn:aws:events:{region_name}:{account_id}:rule/{event_rule_name}",
    )
    print("[OK] Permission added to the Lambda")



if __name__ == "__main__":
    event_put_rules(eventbridge_client, event_rule_name, event_pattern)
    event_put_targets(eventbridge_client, event_rule_name, target_lambda_arn)
    allow_eventbridge_invoke_lambda(lambda_client, target_lambda_name, event_rule_name)