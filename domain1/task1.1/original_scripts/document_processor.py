import boto3
import json

# Initialize clients
s3 = boto3.client('s3')
bedrock_runtime = boto3.client('bedrock-runtime')

def process_document(bucket, key, model_id='anthropic.claude-v2'):
    # Get document from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    document_text = response['Body'].read().decode('utf-8')
    
    # Create prompt for information extraction
    prompt = f"""
    Extract the following information from this insurance claim document:
    - Claimant Name
    - Policy Number
    - Incident Date
    - Claim Amount
    - Incident Description
    
    Document:
    {document_text}
    
    Return the information in JSON format.
    """
    
    # Invoke Bedrock model
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens_to_sample": 1000
        })
    )
    
    # Parse response
    response_body = json.loads(response['body'].read())
    extracted_info = response_body['completion']
    
    # Generate summary
    summary_prompt = f"""
    Based on this extracted information:
    {extracted_info}
    
    Generate a concise summary of the claim.
    """
    
    summary_response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "prompt": summary_prompt,
            "temperature": 0.7,
            "max_tokens_to_sample": 500
        })
    )
    
    summary_body = json.loads(summary_response['body'].read())
    summary = summary_body['completion']
    
    return {
        "extracted_info": extracted_info,
        "summary": summary
    }

# Example usage
if __name__ == "__main__":
    result = process_document('claim-documents-poc-xyz', 'claims/claim1.txt')
    print(json.dumps(result, indent=2))
