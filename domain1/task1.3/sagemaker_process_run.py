# Assignement Part 2
# SageMaker processing job

import os
import json
import pathlib
from dotenv import load_dotenv, find_dotenv

import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput


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
sagemaker_role_arn = os.environ.get("SAGEMAKER_ROLE_ARN", "")

sagemaker_session   = sagemaker.Session()
bucket_name         = "aws-genai-developer-pro" 
input_prefix_name   = "domain1/task3/raw_data/surveys"
output_prefix_name  = "domain1/task3/processed_data/surveys" 
s3_input_data_path  = f"s3://{bucket_name}/{input_prefix_name}"
s3_output_data_path = f"s3://{bucket_name}/{output_prefix_name}"


# Initialize client
sagemaker_client = boto3.client(service_name="sagemaker", region_name=region_name) 


def run_survey_processing_job():
    # Create the processing job
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri='683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
        role=sagemaker_role_arn,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        sagemaker_session=sagemaker_session
    )

    # Run the processing job
    script_processor.run(
        code='sagemaker_process_job.py',
        inputs=[
            ProcessingInput(
                source=s3_input_data_path,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='survey_output',
                source='/opt/ml/processing/output',
                destination=s3_output_data_path
            )
        ]
    )
    
    print("Survey processing job started")



if __name__ == "__main__":
    run_survey_processing_job()
    print("Survey processing job completed")
