# Assignment Part 3

"""
Deploys or updates the AI assistant's degradation Lambda function.
Loads the packaged code, configures runtime settings, and applies environment variables.
Ensures the function is consistently provisioned whether newly created or updated.
"""

import os
import boto3

import dotenv
dotenv.load_dotenv(".env")

region_name= os.environ["REGION_NAME"]
region_name = "us-west-2"
account_id = os.environ["ACCOUNT_ID"]

lambda_function_name = "AIAssistantDegradation"
role_arn = f"arn:aws:iam::{account_id}:role/aws-lambda-execution-role"


lambda_client = boto3.client(service_name="lambda", region_name=region_name)

environment_variables = {
    "REGION_NAME": region_name,
}


base_dir = os.path.dirname(os.path.abspath("__file__"))
zip_path = os.path.join(base_dir, "lambda_package", "graceful_degradation", "degradation_lambda.zip")
with open(zip_path, "rb") as f:
    zip_content = f.read()

try:
    response = lambda_client.create_function(
        FunctionName=lambda_function_name,
        Runtime="python3.9",
        Role=role_arn,
        Handler="lambda_function.lambda_handler",
        Code={"ZipFile": zip_content},
        Timeout=30,
        MemorySize=512, 
        Environment={"Variables": environment_variables}
    )
    print("Lambda created:", response["FunctionArn"])

except lambda_client.exceptions.ResourceConflictException:
    print("Lambda exists — updating code...")
    response = lambda_client.update_function_code(
        FunctionName=lambda_function_name,
        ZipFile=zip_content
    )
    print("Lambda updated.")
