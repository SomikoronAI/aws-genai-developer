# Assignement Part 3
# Lambda function to format data for Claude Foundation Models


import json
import boto3
import base64
import os


def lambda_handler(event, context):
    # Get the S3 object
    s3_client = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process processed data files
    if not key.endswith('_processed.json'):
        return {
            'statusCode': 200,
            'body': json.dumps('Not a processed data file')
        }
    
    try:
        # Get the processed data
        response = s3_client.get_object(Bucket=bucket, Key=key)
        processed_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Determine the data type and format accordingly
        if 'transcript' in processed_data:
            # Audio data
            formatted_data = format_audio_data(processed_data)
        elif 'extracted_text' in processed_data:
            # Image data
            formatted_data = format_image_data(processed_data, bucket, key)
        elif 'entities' in processed_data:
            # Text review data
            formatted_data = format_text_data(processed_data)
        elif 'summary_text' in processed_data:
            # Survey data
            formatted_data = format_survey_data(processed_data)
        else:
            pass