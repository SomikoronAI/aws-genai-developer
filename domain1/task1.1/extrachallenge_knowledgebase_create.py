"""
Create Amazon Bedrock Knowledge Base using Boto3 for auto insurance policy information.

***Note** 
This script uses Amazon S3 vecor bucket as the vector data store for Bedrock Knowledge Base 
Options vector data store are: S3 Vector, OpenSearch, Neptune, RDS, and PineCone

This script performs the following tasks:
1. Uploads policy documents (TXT/PDF) to a specified S3 general-purpose bucket.
2. Gets S3 vector bucket and index information. 
3. Creates a Knowledge Base in Amazon Bedrock.  Poll Knowledge Base creation status.
4. Creats a data source in Knowledge Base. The data source points to S3 general-purpose bucket.
5. Ingests Data to generate vector embeddings for semantic search.  Poll data ingestion creation status.

Requirements:
- AWS credentials with permissions for Bedrock and S3.
- Existing S3 bucket to store policy documents.
- Boto3 installed and configured.

Usage:
- Update configuration constants (region name, bucket name, KB name, model ARNs, guardrail ID/version).
- Run main() to execute document upload, KB creation, and data ingestion.
"""

import os
import time
import boto3
from botocore.exceptions import ClientError

from upload_doc_to_s3_gp import upload_document


# Conbfiguration parameters
region_name = "us-east-1"
bucket_name = "claim-documents-poc-nururrahman"            # S3 general-purpose bucket name
bucket_arn  = f"arn:aws:s3:::{bucket_name}"                # S3 general-purpose bucket arn
bucket_prefix = "policies/"
vec_bucket_name = "claim-documents-poc-kb-vectors"         # S3 vector bucket name
vec_prefix="kb_vectors/"

knowledge_base_name = "kb-auto-policy-info"
knowledge_base_role_arn = "arn:aws:iam::339713184295:role/aws-bedrock-service-role"

embedding_model_arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
llm_model_arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"
# embedding_model_arn="arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-multimodal-embeddings-v1:0"
# llm_model_arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-2-lite-v1:0:256k"

s3_client = boto3.client(service_name="s3", region_name=region_name)
s3v_client = boto3.client(service_name="s3vectors", region_name=region_name)
bedrock_client = boto3.client(service_name="bedrock", region_name=region_name)
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)
bedrock_agent = boto3.client( service_name="bedrock-agent", region_name=region_name)
bedrock_agent_runtime = boto3.client(service_name="bedrock-agent-runtime", region_name=region_name)


# Get S3 vector bucket and index information
def get_vec_bucket_index_info(vec_bucket_name):
    """Get S3 vector bucket and index information that are needed to create a Knowledge Base"""
    try:
        response_bucket = s3v_client.list_vector_buckets()
        vec_bucket_list = [ bucket["vectorBucketName"] for bucket in response_bucket.get("vectorBuckets",[]) if len(bucket)>0 ]
        if vec_bucket_name in vec_bucket_list:
            idx = response_bucket["vectorBuckets"][vec_bucket_list.index(vec_bucket_name)]
            vec_bucket_arn = idx["vectorBucketArn"]
    
        response_index = s3v_client.list_indexes(vectorBucketName=vec_bucket_name)
        vec_index_arn = response_index["indexes"][0]["indexArn"]
        return vec_bucket_arn, vec_index_arn
    
    except ClientError as e:
        print(f"Error: {e}")
        return False


# Create Knowledge Base
def create_knowledge_base(
        knowledge_base_name, 
        knowledge_base_role_arn, 
        embedding_model_arn, 
        bucket_name, 
        vec_bucket_arn,
        vec_index_arn
        ):
    """Create an Amazon Bedrock Knowledge Base with a VECTOR configuration and S3 vector storage."""

    response = bedrock_agent.create_knowledge_base(
        name=knowledge_base_name,
        description="Knowledge base for auto insurance policy documents",
        roleArn=knowledge_base_role_arn,
        knowledgeBaseConfiguration= {
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embedding_model_arn,
                "supplementalDataStorageConfiguration": {
                    "storageLocations": [
                        {
                            "type": "S3",
                            "s3Location": { "uri": f"s3://{bucket_name}" }
                        }
                    ]
                }
            }
        }, 
        storageConfiguration={
            "type": "S3_VECTORS",
            "s3VectorsConfiguration": {
            "vectorBucketArn": vec_bucket_arn,
            "indexArn": vec_index_arn, 
            }
        }
    )
    kb_id = response["knowledgeBase"]["knowledgeBaseId"]
    print(f"Knowledge Base created: {kb_id}")
    return kb_id


# Poll Knowledge Base creation status 
def poll_knowledge_base_active(kb_id, poll_interval=5, timeout=300):
    """Wait until the Knowledge Base reaches ACTIVE status."""
    start_time = time.time()

    while True:
        response = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)

        status = response["knowledgeBase"]["status"]

        if status == "ACTIVE":
            return True
        elif status in ("FAILED", "STOPPED"):
            raise RuntimeError(f"Knowledge Base ended with status: {status}")
        else:
            print(f"Knowledge Base status: {status}")

        if time.time() - start_time > timeout:
            raise TimeoutError("Timed out waiting for Knowledge Base to become ACTIVE")

        time.sleep(poll_interval)


# Create Knowldege Base data source
def create_data_source(kb_id, ds_name):
    """Create a data source in Knowledge Base that points to S3 GP bucket containing the documents."""

    try:
        response = bedrock_agent.create_data_source(
            knowledgeBaseId=kb_id,
            name=ds_name,
            description="S3 source for auto insurance policy information",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": bucket_arn,
                    "inclusionPrefixes": [bucket_prefix]
                }
            },
            vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE", 
                # "chunkingStrategy": "HIERARCHICAL", 
                # "chunkingStrategy": "SEMANTIC",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 300,
                    "overlapPercentage": 5
                },
                # "hierarchicalChunkingConfiguration": {
                #           "levelConfigurations": [
                #               {"maxTokens": 1000}, # Layer 1 (larger sections)
                #               {"maxTokens": 500},  # Layer 2 (subsections)
                #             ],
                #           'overlapTokens': 50
                #       },
                # "semanticChunkingConfiguration": {
                #     'maxTokens': 300,
                #     'bufferSize': 1,
                #     'breakpointPercentileThreshold': 90
                # }
            }
        }
        )
        ds_id = response["dataSource"]["dataSourceId"]
        print(f"Data Source created: {ds_id}")
        return ds_id
    
    except ClientError as e:
        print(f"Error: {e}")
        return False


# Start the data ingestion job
def start_data_ingestion(kb_id, ds_id):
    """Data ingestion task reads docs, calls embedding model, writes vectors to S3 Vectors index"""
    response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id
    )
    job_id = response["ingestionJob"]["ingestionJobId"]
    print(f"Ingestion job started: {job_id}")
    return job_id

# Poll data ingestion creation status 
def poll_data_ingestion_status(kb_id, ds_id, job_id, poll_interval=5, timeout=300):
    """Wait until the ingestion job reaches COMPLETE status"""
    start_time = time.time()
    
    while True:
        response = bedrock_agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id
        )
        status = response["ingestionJob"]["status"]
        if status == "COMPLETE":
            return True
        elif status in ("FAILED", "STOPPED"):
            raise RuntimeError(f"Ingestion job ended with status: {status}")
        else:
            print(f"Ingestion job status: {status}")
        
        if time.time() - start_time > timeout:
            raise TimeoutError("Timed out waiting for Knowledge Base to become ACTIVE")

        time.sleep(poll_interval)


def main():
    # Upload local documents to S3 general-purpose bucket 
    local_data_path = os.path.join( os.getcwd(), "data", "policies" )
    for _, _, file_list in os.walk(local_data_path):
        for fname in file_list:
            file_name = os.path.join( os.getcwd(), "data", "policies", fname)
            s3_key_name = bucket_prefix + fname            # Desired object name in S3
            upload_document(s3_client, file_name, bucket_name, s3_key_name)
    print("")

    # Part 1
    vec_bucket_name = "claim-documents-poc-kb-vectors"         # S3 vector bucket name
    vec_bucket_arn, vec_index_arn = get_vec_bucket_index_info(vec_bucket_name)

    # Part 2
    # Create Knowledge Base and ingest data
    response = bedrock_agent.list_knowledge_bases()
    kb_list = [kb["name"] for kb in response["knowledgeBaseSummaries"]]
    if knowledge_base_name in kb_list:
        print("Knowledge Base by this name already exists.")
        print("Getting relevent Knowledge Basse info. \n")
        kb_id = response["knowledgeBaseSummaries"][0]["knowledgeBaseId"]
    else:
        print("Creating a new Knowledge Base.")
        kb_id = create_knowledge_base(
            knowledge_base_name, knowledge_base_role_arn, 
            embedding_model_arn, 
            bucket_name, 
            vec_bucket_arn, vec_index_arn
            )
    print("")
    
    # Wait for Knowledge Base status: CREATING → ACTIVE
    poll_knowledge_base_active(kb_id)

    # Part 3
    # Create Data Source
    ds_name = "auto-policy-info-s3-source"
    response = bedrock_agent.list_data_sources( knowledgeBaseId=kb_id)
    ds_list = [ ds["name"] for ds in response["dataSourceSummaries"] ]
    if ds_name in ds_list:
        ds_id = response["dataSourceSummaries"][0]["dataSourceId"]
    else:
        ds_id = create_data_source(kb_id, ds_name)
    print("")

    # Part 4
    # Start data ingestion Job
    job_id = start_data_ingestion(kb_id, ds_id)
    # Wait for data ingestion status: STARTED → COMPLETE
    poll_data_ingestion_status(kb_id, ds_id, job_id, poll_interval=5)
    print("")


if __name__ == "__main__":
    main()