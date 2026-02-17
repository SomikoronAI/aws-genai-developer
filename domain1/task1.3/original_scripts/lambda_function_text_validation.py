# Assignment Part 1
# Create a Lambda function for custom text validation

import json
import re
from datetime import datetime
import boto3


def lambda_handler(event, context):
    # Get the S3 object
    s3_client = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process text reviews
    if not key.endswith('.txt') and not key.endswith('.json'):
        return {
            'statusCode': 200,
            'body': json.dumps('Not a text review file')
        }
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        
        # Parse the content (assuming JSON format)
        if key.endswith('.json'):
            review = json.loads(content)
            text = review.get('review_text', '')
        else:
            text = content
            
        # Validation checks
        validation_results = {
            'file_name': key,
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'min_length': len(text) >= 10,
                'has_product_reference': bool(re.search(r'product|item|purchase', text, re.IGNORECASE)),
                'has_opinion': bool(re.search(r'like|love|hate|good|bad|great|terrible|excellent|poor|recommend', text, re.IGNORECASE)),
                'no_profanity': not bool(re.search(r'badword1|badword2', text, re.IGNORECASE)),  # Add actual profanity list
                'has_structure': text.count('.') >= 1  # At least one sentence
            }
        }
        
        # Calculate overall quality score (simple version)
        passed_checks = sum(1 for check in validation_results['checks'].values() if check)
        total_checks = len(validation_results['checks'])
        validation_results['quality_score'] = passed_checks / total_checks
        
        # Send metrics to CloudWatch
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='CustomerFeedback/TextQuality',
            MetricData=[
                {
                    'MetricName': 'QualityScore',
                    'Value': validation_results['quality_score'],
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'Source',
                            'Value': 'TextReviews'
                        }
                    ]
                }
            ]
        )
        
        # Save validation results
        validation_key = key.replace('raw-data', 'validation-results').replace('.txt', '.json').replace('.json', '_validation.json')
        s3_client.put_object(
            Bucket=bucket,
            Key=validation_key,
            Body=json.dumps(validation_results),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(validation_results)
        }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
