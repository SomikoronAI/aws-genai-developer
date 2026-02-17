# Assignement Part 2
# SageMaker processing job


import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput


def run_survey_processing_job():
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Define the processing job
    script_processor = ScriptProcessor(
        command=['python3'],
        image_uri='737474898029.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        sagemaker_session=sagemaker_session
    )
    
    # Run the processing job
    script_processor.run(
        code='survey_processing.py',
        inputs=[
            ProcessingInput(
                source='s3://customer-feedback-analysis-<your-initials>/raw-data/surveys.csv',
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='survey_output',
                source='/opt/ml/processing/output',
                destination='s3://customer-feedback-analysis-<your-initials>/processed-data/surveys/'
            )
        ]
    )
    
    print("Survey processing job started")


if __name__ == "__main__":
    run_survey_processing_job()
