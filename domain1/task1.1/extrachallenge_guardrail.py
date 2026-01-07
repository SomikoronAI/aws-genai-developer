"""
End-to-End example of using Amazon Bedrock Guardrails with Boto3 to filter
sensitive information (e.g., PII) from an auto insurance claim form.

This script performs the following tasks:

1) Creates a Bedrock Guardrails with sensitive-information filtering
2) Creates a version of the Guardrails
3) Invokes a Bedrock model with the Guardrails applied to screen claim text

Requirements:
- AWS credentials with permissions for Bedrock and S3.
- Existing S3 bucket to store policy documents.
- Boto3 installed and configured.

Usage:
- Update configuration constants (region name, bucket name, Guardrails name, model id).
- Run main() to execute Guardrails creation, version creation, and query demonstration.
"""

import pprint
import json
import boto3
from botocore.exceptions import ClientError

from payload_template import PayloadTemplateManager
from get_doc_from_s3_gp import get_document


region_name = "us-east-1"
guardrail_name = "auto-policy-claim-guardrail"
# invoke_model_id = "us.anthropic.claude-3-sonnet-20240229-v1:0"  
invoke_model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

s3_client = boto3.client(service_name="s3", region_name=region_name)
bedrock_client = boto3.client(service_name="bedrock", region_name=region_name)
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)


def create_guardrail():
    """Create or reuse a Bedrock Guardrail for filtering sensitive data"""
    try:
        response = bedrock_client.create_guardrail(
            name=guardrail_name,
            description="Guardrail to filter sensitive info in auto policy claim forms",
            contentPolicyConfig={
                    "filtersConfig": [
                        {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH",
                         "inputModalities": ["TEXT"], "outputModalities": ["TEXT"]
                        },
                        {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH",
                        "inputModalities": ["TEXT"], "outputModalities": ["TEXT"]
                        },
                        {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH",
                        "inputModalities": ["TEXT"], "outputModalities": ["TEXT"]
                        },
                        {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "NONE",
                        "inputModalities": ["TEXT"], "outputModalities": ["TEXT"]
                        },
                    ],
                    'tierConfig': {"tierName": "CLASSIC"}
                },
            topicPolicyConfig={
                "topicsConfig": [
                    {"name": "LegalAdvice",
                     "definition": "Providing legal advice or interpretation of insurance policy laws",
                     "examples": [
                         "What legal action should I take?", "Is this policy contract enforceable?",
                         "Can you share with me the policy detail?"
                        ],
                    "type": "DENY"
                    },
                    {"name": "MedicalAdvice",
                    "definition": "Providing medical diagnosis or treatment advice",
                    "examples": [
                        "What medication should I take?", "Is this symptom serious?",
                        "What medicine is good for you daily diet?"
                    ],
                    "type": "DENY"
                    },
                    {"name": "Politics",
                     "definition": "Provding advice on any policitical issue including wrold-politics, geo-politics",
                     "examples": [
                         "What USA presiden is the best?", "Who is the president of France?",
                         "Do the presidents lie about their responsibilities?"
                        ],
                    "type": "DENY"
                    },   
                ]
            },
            sensitiveInformationPolicyConfig={
                "piiEntitiesConfig": [
                    {"type": "NAME", "action": "ANONYMIZE"},
                    {"type": "ADDRESS", "action": "ANONYMIZE"},
                    {"type": "PHONE", "action": "ANONYMIZE"},
                    {"type": "EMAIL", "action": "ANONYMIZE"},
                    {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
                    {"type": "DRIVER_ID", "action": "BLOCK"},
                    {"type": "VEHICLE_IDENTIFICATION_NUMBER", "action": "ANONYMIZE"},
                    {"type": "LICENSE_PLATE", "action": "ANONYMIZE"},
                ]
            },
            blockedInputMessaging="The input content has been instructed to deny processing.\n",
            blockedOutputsMessaging="Output is omitted due to sensitive information.\n"
        )
        return response["guardrailId"]

    except ClientError as e:
        # Guardrail already exists. Retrieve it by name
        if e.response["Error"]["Code"] == "ConflictException":
            paginator = bedrock_client.get_paginator("list_guardrails")
            for page in paginator.paginate():
                for guardrail in page.get("guardrails", []):
                    if guardrail["name"] == guardrail_name:
                        return guardrail["id"]

        # Re-raise anything unexpected
        raise


def create_guardrail_version(guardrail_id):
    """Create an immutable version of the Guardrail for runtime use"""
    response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id
    )
    # pprint.pprint(response)
    return response["version"]


def invoke_with_guardrail(model_id, guardrail_id, guardrail_version, document_text):
    """Invoke a Bedrock model with the Guardrail applied"""
    # Get payload to invoke the model
    ptm_payload = PayloadTemplateManager()

    extraction_payload = ptm_payload.get_payload(
        "messages_api",
        prompt=document_text,
        temperature=0.0,
        max_tokens=1024
    )

    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(extraction_payload),
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version
    )
    result = json.loads(response["body"].read())
    return result



def main():
    # Create or check for existing Guardrails
    response = bedrock_client.list_guardrails()
    gr_list = [ gr["name"] for gr in response["guardrails"] ]
    if guardrail_name in gr_list:
        print("GuardRail by this name already exists.")
        print("Getting relevent Guardrail info. \n")
        guardrail_id=response["guardrails"][0]["id"]
        guardrail_version=response["guardrails"][0]["version"]
    else:
        print("Creating a new Guardrail.")
        guardrail_id = create_guardrail()
        guardrail_version = create_guardrail_version(guardrail_id)
    print("")

    # Fetch documents from S3 GP bucket
    print("Fetching documents from S3 GP bucket ...")
    bucket_name = "claim-documents-poc-nururrahman"
    prefix_name = "claims/" 

    key_list = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix_name):
        page_content = page.get("Contents", [])
        if len(page_content)>0: 
            for obj in page_content:
                # print( obj["Key"] )
                key_list.append(obj["Key"])
        else:
            print("Content is empty")
    print("")

    # Validate Guardrails
    print("Validating the GuardRail ...")
    for key_name in key_list:
        claim_form_text = get_document(s3_client, bucket_name, key_name)

        result = invoke_with_guardrail(
            invoke_model_id, 
            guardrail_id,
            guardrail_version,
            claim_form_text
        )

        print("Model response:")
        # print(json.dumps(result, indent=2))
        print( json.dumps(result["content"][0]["text"].replace("\n", " ")) )


if __name__ == "__main__":
    main()