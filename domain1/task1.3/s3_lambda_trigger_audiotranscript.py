# Assignment Part 2
"""Creates S3 trigger to AudioTranscipt lambda"""

import os
import json
import pathlib
from dotenv import load_dotenv, find_dotenv

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

# Iniitialize client
s3_client     = boto3.client(service_name="s3", region_name=region_name)
lambda_client = boto3.client(service_name="lambda", region_name=region_name)


bucket_name = "aws-genai-developer-pro"
prefix_name = "domain1/task3/raw_data/audios/"
# Lambda function name
lambda_function_name = "AudioTranscription"
# Lambda fucntion ARN
lambda_function_arn  = f"arn:aws:lambda:{region_name}:{account_id}:function:{lambda_function_name}"
# Data type extensions
extensions = [".mp3", ".mp4", ".wav", ".flac"]


# --------------------------------------
# Allow S3 to invoke the Lambda
# --------------------------------------
def allow_s3_invoke_lambda(lambda_client, lambda_function_name: str):
    statement_id = f"AllowS3Invoke{lambda_function_name}" 
    try:
        lambda_client.add_permission(
            FunctionName=lambda_function_name,
            StatementId=statement_id,
            Action="lambda:InvokeFunction",
            Principal="s3.amazonaws.com",
            SourceArn=f"arn:aws:s3:::{bucket_name}",
            SourceAccount=account_id
        )
        print("[OK] Permission added to the Lambda")
    except lambda_client.exceptions.ResourceConflictException:
        print("[INFO] Lambda permission already exists. Skipping.")


# --------------------------------------
# Add Lambda trigger to the S3 bucket
# --------------------------------------
def config_type(lambda_function_arn: str, prefix_name: str, suffix_name: str):
    return {
        "LambdaFunctionArn": lambda_function_arn,
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {
            "Key": {
                "FilterRules": [
                    {"Name": "prefix", "Value": prefix_name},
                    {"Name": "suffix", "Value": suffix_name}
                ]
            }
        }
    }    



def main():
    allow_s3_invoke_lambda(lambda_client, lambda_function_name)

    try:
        response = s3_client.get_bucket_notification_configuration(
            Bucket=bucket_name
        )

        existing_lambda_configs = response.get("LambdaFunctionConfigurations", [])

        # Deduplicate new configurations
        new_lambda_configs = [
            config_type(lambda_function_arn, prefix, suffix_name)
            for prefix in prefix_names
        ]
        
        for new in new_lambda_configs:
            if new not in existing_lambda_configs:
                existing_lambda_configs.append(new)

        # Preserve all notification types
        updated_notification_config = {
            "LambdaFunctionConfigurations": existing_lambda_configs,
            "QueueConfigurations": response.get("QueueConfigurations", []),
            "TopicConfigurations": response.get("TopicConfigurations", []),
        }

        # Preserve EventBridge if enabled
        if "EventBridgeConfiguration" in response:
            updated_notification_config["EventBridgeConfiguration"] = response["EventBridgeConfiguration"]

        # Apply updated configuration
        s3_client.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration=updated_notification_config
        )

        print("S3 trigger has been created/updated successfully!")

    except Exception as e:
        print(f"[ERROR] Failed to create/update S3 trigger: {e}")
        raise



if __name__ == "__main__":
    main()
