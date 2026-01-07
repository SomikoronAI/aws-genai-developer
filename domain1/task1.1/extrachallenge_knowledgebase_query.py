"""
Retrieve documents and generate answers with an LLM from Bedrock Knowledge Base.
Optionally enforcing a Guardrails.

This script performs the following tasks:
1. Resolve Knowledge Base by name.
2. Retrieve top-N documents for a query.
3. Look up Guardrails ID/version.
4. Retrieve-and-generate with Guardrails applied.

Requirements:
- AWS credentials with permissions for Bedrock and S3.
- Existing S3 bucket to store policy documents.
- Boto3 installed and configured.

Usage:
- Update configuration constants (region name, KB name, model ARNs, guardrail name).
- Run main() to execute query demonstration.
"""

import boto3
from botocore.exceptions import ClientError

# Conbfiguration parameters
region_name = "us-east-1"
llm_model_arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"
# llm_model_arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0:256k"

# Initialize clients
bedrock_client = boto3.client(service_name="bedrock", region_name=region_name)
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)
bedrock_agent = boto3.client( service_name="bedrock-agent", region_name=region_name)
bedrock_agent_runtime = boto3.client(service_name="bedrock-agent-runtime", region_name=region_name)


# Get Knowledge Base id
def get_knowledge_base_id(kb_name):
    """Retrieve the Knowledge Base ID for a given Knowledge Base name."""
    try:
        # Paginate through knowledge bases
        paginator = bedrock_agent.get_paginator("list_knowledge_bases")
        for page in paginator.paginate():
            for kb in page.get("knowledgeBaseSummaries", []):
                if kb.get("name") == kb_name:
                    kb_id = kb.get("knowledgeBaseId")
                    print(f"Found Knowledge Base '{kb_name}' with ID: {kb_id}")
                    return kb_id
        print(f"No Knowledge Base found with name: {kb_name}")
        return None
    except Exception as e:
        print(f"Error retrieving Knowledge Base ID: {e}")
        return None


# Query the Knowledge Base (Retrieve documents without GuardRails)
def query_knowledge_base(kb_id, question, guardrail_id, guardrail_version, number_of_docs=3):
    """Return retrieved documents from the knowledge base"""
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": number_of_docs
                }
            },
            guardrailConfiguration={
                "guardrailId": guardrail_id, 
                "guardrailVersion": guardrail_version, 
            },
        )
        retrieve_docs = []
        print("\nTop Results:\n")
        for result in response["retrievalResults"]:
            text = result["content"]["text"]
            # print(text, "\n")
            retrieve_docs.append(text)

        return retrieve_docs
    
    except Exception as e:
        print(f"Error querying Knowledge Base ID: {e}")
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

# Query Knowledge Base and generate summary text with GuardRails applied
def retrieve_and_generate_with_guardrail(
        kb_id, 
        question, 
        guardrail_id, 
        guardrail_version, 
        llm_model_arn, 
        max_tokens=512,
        temperature=0.5
        ):
    """Return generated summary text and citations using LLM with guardrail enforcement."""

    response = bedrock_agent_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            # type="EXTERNAL_SOURCES",
            "type": "KNOWLEDGE_BASE", 
            # "externalSourcesConfiguration": {},
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
        },
        # sessionConfiguration={"kmsKeyArn": "string"}, 
        # sessionId="string"
    )
    summary = response["output"]["text"]
    citations = response.get("citations", []) 
    return {
        "summary": summary, 
        "citations": citations
        }




def main():
    kb_name = "kb-auto-policy-info" 
    kb_id = get_knowledge_base_id(kb_name)
    print(f"Knowledge Base Name and ID: {kb_name}, {kb_id}")
    print("")

    guardrail_name = "auto-policy-claim-guardrail"
    guardrail_id, guardrail_version = get_guardrail_info(guardrail_name) 

    question = "What does comprehensive coverage include?" 
    question = "What are the factors that affect state average expenditures and average premiums?"
    # question = "What legal action action should I take against a policy company?"
    # question = "Tell me about world politics."
 
    # Query the Knowledge Base to retrieve documents only 
    print("Query the Knowledge Base to retrieve documents.")
    response = query_knowledge_base(kb_id, question, guardrail_id, guardrail_version, number_of_docs=3)
    for retrieved_doc in response:
        print(retrieved_doc) 
        print("-"*80)   
    print("")
    
    # Query the Knowledge Base and generate text with GuardRails applied
    print("Querying Knowledge Base and generate text with GuardRails applied. \n")
    response = retrieve_and_generate_with_guardrail(kb_id, question, guardrail_id, guardrail_version, llm_model_arn) 
    print( response["summary"] )
    print( len(response["citations"]) )



if __name__ == "__main__":
    main()