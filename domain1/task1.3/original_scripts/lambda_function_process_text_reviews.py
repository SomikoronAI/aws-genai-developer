# Assignement Part 2
# Lambda function to process text reviews


import json
import boto3
import os


def lambda_handler(event, context):
    # Get the S3 object
    s3_client = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process validated text reviews
    if not key.endswith('_validation.json'):
        return {
            'statusCode': 200,
            'body': json.dumps('Not a validated review file')
        }
    
    try:
        # Get the validation results
        response = s3_client.get_object(Bucket=bucket, Key=key)
        validation_results = json.loads(response['Body'].read().decode('utf-8'))
        
        # Check if the quality score is sufficient
        if validation_results['quality_score'] < 0.7:  # Threshold for processing
            print(f"Quality score too low: {validation_results['quality_score']}")
            return {
                'statusCode': 200,
                'body': json.dumps('Quality score too low')
            }
        
        # Get the original review text
        original_key = key.replace('validation-results', 'raw-data').replace('_validation.json', '.json')
        response = s3_client.get_object(Bucket=bucket, Key=original_key)
        review = json.loads(response['Body'].read().decode('utf-8'))
        text = review.get('review_text', '')
        
        # Use Amazon Comprehend for entity extraction and sentiment analysis
        comprehend = boto3.client('comprehend')
        # Detect entities
        entity_response = comprehend.detect_entities(
            Text=text,
            LanguageCode='en'
        )
        
        # Detect sentiment
        sentiment_response = comprehend.detect_sentiment(
            Text=text,
            LanguageCode='en'
        )
        
        # Detect key phrases
        key_phrases_response = comprehend.detect_key_phrases(
            Text=text,
            LanguageCode='en'
        )
        
        # Combine the results
        processed_review = {
            'original_text': text,
            'entities': entity_response['Entities'],
            'sentiment': sentiment_response['Sentiment'],
            'sentiment_scores': sentiment_response['SentimentScore'],
            'key_phrases': key_phrases_response['KeyPhrases'],
            'metadata': {
                'product_id': review.get('product_id', ''),
                'customer_id': review.get('customer_id', ''),
                'review_date': review.get('review_date', '')
            }
        }
        
        # Save processed results
        processed_key = key.replace('validation-results', 'processed-data').replace('_validation.json', '_processed.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=processed_key,
            Body=json.dumps(processed_review),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps('Successfully processed review')
        }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
