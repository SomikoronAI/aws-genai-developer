# Assignment Part 1

"""
Provision and maintain the TextValidation AWS Lambda function.

Loads configuration from environment variables via a `.env` file
Reads a pre-built deployment package (ZIP).
Ceates or updates the target Lambda function in-place. 
It is designed to be idempotent:
- If the function does not exist, it will be created.
- If the function already exists, its code and/or configuration can be updated
  in a single call, based on the supplied flags.

Usage:
python deploy_lambda_textvalidation.py
"""


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

lambda_function_name = "TextValidation"

environment_variables = {
    "REGION_NAME": region_name,
    "ACCOUNT_ID": account_id
}

# Iniitialize client
lambda_client = boto3.client(service_name="lambda", region_name=region_name)


base_dir = os.path.dirname(os.path.abspath("__file__"))
zip_path = os.path.join(base_dir, "lambda_packages", "text_validation", "textvalidation_lambda.zip")
with open(zip_path, "rb") as f:
    zip_content = f.read()


def deploy_lambda_function(
    lambda_client,
    *,
    function_name: str,
    role_arn: str,
    zip_content: bytes | None = None,
    runtime: str = "python3.12",
    handler: str = "lambda_function.lambda_handler",
    timeout: int = 30,
    memory_size: int = 512,
    environment_variables: dict | None = None,
    publish: bool = True,
    update_code: bool = True,
    update_configuration: bool = True,
):
    """
    Create or update an AWS Lambda function with optional code and
    configuration updates.
    """

    try:
        if not zip_content:
            raise ValueError(
                "zip_content must be provided when creating a new Lambda function"
                )
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime=runtime,
            Role=role_arn,
            Handler=handler,
            Code={"ZipFile": zip_content},
            Timeout=timeout,
            MemorySize=memory_size,
            Environment={"Variables": environment_variables or {}},
            Publish=publish,
        )
        print(f"Lambda created: {response['FunctionArn']}")
        return response

    except lambda_client.exceptions.ResourceConflictException:
        print("Lambda exists - updating as requested")

        if update_code and zip_content:
            lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content,
                Publish=publish,
            )
            print("Lambda code updated")

        if update_configuration:
            lambda_client.update_function_configuration(
                FunctionName=function_name,
                Role=role_arn,
                Runtime=runtime,
                Handler=handler,
                Timeout=timeout,
                MemorySize=memory_size,
                Environment={"Variables": environment_variables or {}},
            )
            print("Lambda configuration updated")

        return lambda_client.get_function(FunctionName=function_name)


if __name__ == "__main__":
    deploy_lambda_function(
        lambda_client,
        function_name=lambda_function_name,
        role_arn=lambda_role_arn,
        zip_content=zip_content, 
        environment_variables=environment_variables,
        update_code=False,
        update_configuration=False,
    )