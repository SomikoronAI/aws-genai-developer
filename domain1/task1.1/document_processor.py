"""
Generate insurance claim summaries using Amazon Bedrock Foundation Models and 
evaluate summary accuracy and quality.

This script performs the forllowing tasks:
1. Reads insurance claim documents (TXT or PDF) from an Amazon S3
   general-purpose bucket.
2. Uses a Bedrock Foundation Model (Claude) to extract structured claim
   information in JSON format.
3. Generates a claim summary based on the extracted data.
4. Evaluates the generated summary for:
   - Factual accuracy against the original claim document.
   - Quality metrics including clarity, completeness, conciseness, and tone.
5. Optionally applies Amazon Bedrock Guardrails during evaluation to enforce
   policy, safety, and output constraints.

Models Used:
- Primary model: Claude 3 Sonnet (for extraction and summarization)
- Judge model: Claude 3.7 Sonnet (for accuracy and quality evaluation)

Requirements:
- AWS credentials with permissions for:
  - Amazon Bedrock (bedrock-runtime)
  - Amazon S3 (read access to claim documents)
- Existing S3 bucket containing claim documents.
- Boto3 installed and configured.

Usage:
- Update configuration variables (region, model IDs, guardrail ID/version).
- Provide an S3 bucket name and document key.
- Run the script to extract claim data, generate a summary, and evaluate
  accuracy and quality with or without guardrails.

Notes:
- Designed for insurance claims and policy-related documents.
- Deterministic extraction is achieved using low temperature settings.
- Evaluation outputs are intended for human review or automated scoring pipelines.
"""

import json
import boto3
from botocore.exceptions import ClientError

from get_doc_from_s3_gp import get_document
from prompt_template import PromptTemplateManager
from payload_template import PayloadTemplateManager
from bedrock_runtime_template import BedrockRuntimeInvokeManager

# Configuration parameters
region_name="us-east-1"
model_id="anthropic.claude-3-sonnet-20240229-v1:0"
judge_model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"

# Initialize modules
ptm_prompt = PromptTemplateManager()
ptm_payload = PayloadTemplateManager()
bedrock_invoker = BedrockRuntimeInvokeManager(region_name)

# Initialize clients
s3_client = boto3.client(service_name="s3", region_name=region_name)
bedrock_client = boto3.client(service_name="bedrock", region_name=region_name)
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)



def process_document(document_text, model_id=model_id): 
    try:
        # Create extraction prompt 
        extraction_prompt = ptm_prompt.get_prompt("extract_info", document_text=document_text)
        extraction_payload = ptm_payload.get_payload_claude(
            "messages_api",
            user_prompt=extraction_prompt,
            temperature=0.0,
            max_tokens=1024
            )
        # Invoke Bedrock model
        response = bedrock_invoker.invoke( model_id, extraction_payload )
        # Parse response
        response_body = json.loads(response["body"].read())
        extracted_text= response_body["content"][0]["text"]

        # Generate summary
        summary_prompt = ptm_prompt.get_prompt("generate_summary", extracted_text=extracted_text)
        summary_payload = ptm_payload.get_payload_claude(
            "messages_api",
            user_prompt=summary_prompt,
            temperature=0.5,
            max_tokens=512
            )
        # Invoke Bedrock model
        summary_response = bedrock_invoker.invoke( model_id, summary_payload )
        # Parse response
        summary_body  = json.loads(summary_response["body"].read())
        summary_text = summary_body["content"]
        return {
            "summary_body": summary_body,
            "summary_text": summary_text
        }
    except ClientError as e:
        message = e.response.get("Error", {}).get("Message") 
        code = e.response.get("Error", {}).get("Code")
        if code=="ResourceNotFoundException":
            print(message)
        return None


# Accuracy Evaluation (fact consistency)
def evaluate_accuracy(
        claim_text, 
        summary_text, 
        judge_model_id, 
        guardrail_id=None, 
        guardrail_version=None
        ):
    try:
        system_prompt = (
            "You are astrict audidor of insurance policy information. "
            "Check factual correctness only."
        )
        user_prompt = (
            "Claim Document:\n"
            f"{claim_text}\n\n"
            "Claim Summary:\n"
            f"{summary_text}\n\n"
            "Evaluate factual accuracy. "
            "Return a score from 0-1 and list any inaccuracies."
            )
                
        payload = ptm_payload.get_payload_claude(
            "messages_api",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=512
            )
        
        response = bedrock_invoker.invoke(
            judge_model_id, 
            payload, 
            guardrail_id=guardrail_id, 
            guardrail_version=guardrail_version
            )
        accuracy_body = json.loads(response["body"].read())
        accuracy = accuracy_body["content"]
        return accuracy
    
    except ClientError as e:
        message = e.response.get("Error", {}).get("Message") 
        code = e.response.get("Error", {}).get("Code")
        if code=="ResourceNotFoundException":
            print(message)
        return None


# Quality Evaluation (clarity, completeness)
def evaluate_quality(summary_text, judge_model_id, guardrail_id=None, guardrail_version=None):
    try:
        system_prompt= (
            "You are a strict evaluator of insurance claim summaries. "
            "Score 0-5 for: clarity, completeness, organization, overall_quality. "
            "Return ONLY JSON with numeric scores and a short justification per metric."
        )
        user_prompt= (
            "Evaluate the summary using these criteria:\n"
            "- Clarity\n"
            "- Completeness\n"
            "- Conciseness\n"
            "- Professional tone\n\n"
            "Score each 1-5 and provide an overall score.\n\n"
            f"Summary:\n{summary_text}"
            )

        payload = ptm_payload.get_payload_claude(
            "messages_api",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.5,
            max_tokens=512
            )
        
        response = bedrock_invoker.invoke(
            judge_model_id, 
            payload, 
            guardrail_id=guardrail_id, 
            guardrail_version=guardrail_version
            )
        evaluate_body = json.loads(response["body"].read())
        evaluation = evaluate_body["content"]
        return evaluation 
    
    except ClientError as e:
        message = e.response.get("Error", {}).get("Message") 
        code = e.response.get("Error", {}).get("Code")
        if code=="ResourceNotFoundException":
            print(message)
        return None


# Get Guardrails information 
def get_guardrail_info(guardrail_name):
    """Retrieve Guardrail ID and version for a given guardrail name."""
    response = bedrock_client.list_guardrails()
    for gr in response.get("guardrails", []):
        if gr.get("name", "").strip() == guardrail_name:
            id = gr.get("id")
            version = gr.get("version")
            # print(f"GuardRail ID and Version: {id}, {version}", "\n")
            return id, version

    print(f"No Guardrail found with name: {guardrail_name}")
    return None



# Example usage
if __name__ == "__main__":
    bucket_name = "claim-documents-poc-nururrahman"
    key_name = "claims/auto_insurance_claim1.txt"
    key_name = "claims/auto_accident_claim_report_aflac.pdf"
    document_text = get_document(s3_client, bucket_name, key_name)
    # print("**Printing original document ... ** \n")
    # print(document_text, "\n")
    # print("="*100)

    result = process_document( document_text )
    # print( json.dumps(result, indent=2) )
    
    print("**Printing summary text ...** \n")
    summary_text = result["summary_text"][0]["text"]
    print(summary_text, "\n")
    print("="*100)


    #===============================================================
    # Without Guardrails
    #===============================================================
    print("**Printing summary accuracy ...** \n")
    result = evaluate_accuracy(document_text, summary_text, judge_model_id) 
    print( result[0]["text"] )
    print("="*100)

    print("**Printing summary quality ...** \n")
    result = evaluate_quality(summary_text, judge_model_id)
    print( result[0]["text"] )



    # #===============================================================
    # # With Guardrails applied
    # #===============================================================
    print("**Printing summary accuracy ...** \n")
    guardrail_name = "auto-policy-claim-guardrail"
    guardrail_id, guardrail_version = get_guardrail_info(guardrail_name) 
    
    result = evaluate_accuracy(
        document_text, summary_text, judge_model_id, 
        guardrail_id=guardrail_id, 
        guardrail_version=guardrail_version
        ) 
    print( result[0]["text"] )
    print("="*100)

    print("**Printing summary quality ...** \n")
    result = evaluate_quality(
        summary_text, judge_model_id, 
        guardrail_id=guardrail_id, 
        guardrail_version=guardrail_version
        )
    print( result[0]["text"] )