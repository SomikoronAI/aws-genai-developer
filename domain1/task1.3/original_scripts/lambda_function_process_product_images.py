# Assignement Part 2
# Lambda function to process product images


import json
import boto3
import os


s3_client = boto3.client('s3')
rekognition = boto3.client('rekognition')


def lambda_handler(event, context):
    # Get the S3 object
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process image files
    if not key.lower().endswith(('.png', '.jpg', '.jpeg')):
        return {
            'statusCode': 200,
            'body': json.dumps('Not an image file')
        }
    
    try:
        # Extract text from the image using Amazon Textract
        textract = boto3.client('textract')
        response = textract.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            }
        )
        
        # Extract the text
        extracted_text = ""
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + "\n"
        
        # Analyze the image using Amazon Rekognition        
        # Detect labels
        label_response = rekognition.detect_labels(
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            MaxLabels=10,
            MinConfidence=70
        )
        
        # Detect text (as a backup to Textract)
        text_response = rekognition.detect_text(
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            }
        )
        
        # Combine the results
        processed_image = {
            'image_key': key,
            'extracted_text': extracted_text,
            'labels': [label for label in label_response['Labels']],
            'detected_text': [text for text in text_response['TextDetections'] if text['Type'] == 'LINE'],
            'metadata': {
                'product_id': os.path.basename(key).split('_')[0] if '_' in os.path.basename(key) else ''
            }
        }
        
        # Save processed results
        processed_key = key.replace('raw-data', 'processed-data').replace(os.path.splitext(key)[1], '_processed.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=processed_key,
            Body=json.dumps(processed_image),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps('Successfully processed image')
        }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
