import json
import boto3
from botocore.exceptions import ClientError

# Configuration parameters
region_name="us-east-1"
llm_model_arn ="arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"
judge_model_id="amazon.nova-pro-v1:0"

# Initialize clients
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)
bedrock_agent_runtime = boto3.client(service_name="bedrock-agent-runtime", region_name=region_name)


# Query Knowledge Base and generate summary with GuardRails applied
def retrieve_and_generate_with_guardrail(
        kb_id, 
        question, 
        guardrail_id, 
        guardrail_version, 
        llm_model_arn, 
        max_tokens=512,
        temperature=0.5
        ):
    """Return generated summary using LLM with guardrail enforcement."""

    response = bedrock_agent_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE", 
            "knowledgeBaseConfiguration":{
                "knowledgeBaseId": kb_id,
                "modelArn": llm_model_arn,
                # 
                "generationConfiguration": {
                    "guardrailConfiguration":{
                        "guardrailId": guardrail_id, 
                        "guardrailVersion": guardrail_version
                    },
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "maxTokens": max_tokens,
                            "temperature": temperature
                            # "topP": 0.9,
                            # "stopSequences": []
                        }
                    },
                    # "promptTemplate": {"textPromptTemplate": "Answer clearly using the citations.\n\n{{text}}"},
                    "performanceConfig": {"latency": "standard"}
                },
                #
                # "orchestrationConfiguration": {},
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        # "filter": {"equals": {"key": "policy_type", "value": "AUTO"}},
                        # "implicitFilterConfiguration": {},
                        "numberOfResults": 3,
                        "overrideSearchType": "SEMANTIC", #"HYBRID",
                        # "rerankingConfiguration": {}
                    }
                }
            }
        }
    )
    summary = response["output"]["text"]
    return summary


# Accuracy Evaluation (fact consistency)
def evaluate_accuracy(claim_text, summary_text,  judge_model_id):
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
    
    payload = {
        "messages": [
            {
                "role": "system", 
                "content": [{"text": system_prompt}]
            },
            {
                "role": "user", 
                "content": [{"text": user_prompt}] 
            }
        ]
    }
    response = bedrock_runtime.converse(
        modelId=judge_model_id, 
        body=json.dumps(payload),
        inferenceConfig={"maxTokens": 512, "temperature": 0.5}
        # contentType="application/json",
        # accept="application/json"
    )
    result = json.loads(response["body"].read())
    return result


# Quality Evaluation (clarity, completeness)
def evaluate_quality(summary_text, judge_model_id):
    system_prompt = (
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

    payload = {
        "messages": [
            {
                "role": "system", 
                "content": [{"text": system_prompt}]
            },
            {
                "role": "user", 
                "content": [{"text": user_prompt}] 
            }
        ]
    }
    response = bedrock_runtime.converse(
        modelId=judge_model_id,
        body=json.dumps(payload),
        inferenceConfig={"maxTokens": 512, "temperature": 0.5}, 
        guardrailConfig={
            "guardrailIdentifier": "gr-abc123",   
            "guardrailVersion": "1",  
            "trace": "enabled" 
            },
        # contentType="application/json",
        # accept="application/json"
    )
    return json.loads(response["body"].read())
