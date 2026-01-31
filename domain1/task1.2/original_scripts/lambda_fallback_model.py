# Assignment Part 3

import boto3
import json


def lambda_handler(event, context):
    """Fallback model handler that uses a simpler, more reliable model."""

    prompt = event.get('prompt', '')
    use_case = event.get('use_case', 'general')
    
    # Use a simpler, more reliable model
    model_id = "amazon.titan-text-express-v1"  # Example fallback model
    
    try:
        # Invoke the model with simplified parameters
        bedrock_runtime = boto3.client('bedrock-runtime')
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 300,  # Reduced for reliability
                    "temperature": 0.5,
                    "topP": 0.9
                }
            })
        )
        
        response_body = json.loads(response['body'].read().decode())
        output = response_body['results'][0]['outputText']
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'model_used': f"FALLBACK:{model_id}",
                'response': output
            })
        }
    except Exception as e:
        # Let Step Functions catch this and move to graceful degradation
        raise Exception(f"Fallback model failed: {str(e)}")
