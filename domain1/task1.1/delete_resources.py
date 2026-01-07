import boto3
from botocore.exceptions import ClientError


def delete_vec_bucket(vec_bucket_name, vec_index_name, region_name=None):
    """
    Delete a S3 vector bucket and S3 vector index in a specified region.

    :param vec_bucket_name: Name of the bucket to delete
    :param vec_index_name: Name of the index to delete
    :param region_name: AWS region where the bucket will be created
    :return: True if bucket gets deleted, else returns False
    """
    if region_name==None:
        region_name="us-east-1"
        
    # S3 vector bucket client
    s3v_client = boto3.client(service_name="s3vectors", region_name=region_name)

    try:
        response_index  = s3v_client.delete_index(vectorBucketName=vec_bucket_name, indexName=vec_index_name)
        response_bucket = s3v_client.delete_vector_bucket(vectorBucketName=vec_bucket_name)

        if response_index["ResponseMetadata"]["HTTPStatusCode"]==200 and response_bucket["ResponseMetadata"]["HTTPStatusCode"]==200:
            print(f"Vector bucket {vec_bucket_name} and {vec_index_name} deleted successfully from region {region_name}.")
            return True
    except ClientError as e:
        print(f"Error: {e}")
        return False



import boto3
from botocore.exceptions import ClientError
from typing import Optional


def delete_vec_bucket(vec_bucket_name: str, vec_index_name: str, region_name: Optional[str] = None) -> bool:
    """
    Delete a S3 Vector bucket and index in a specified region using S3Vectors service.

    :param vec_bucket_name: Name of the vector bucket to delete
    :param vec_index_name: Name of the vector index to delete
    :param region_name: AWS region (default: us-east-1)
    :return: True if both bucket and index are deleted, else False
    """
    if region_name is None:
        region_name = "us-east-1"

    # S3 vector bucket client
    s3v_client = boto3.client(service_name="s3vectors", region_name=region_name)

    try:
        response_index = s3v_client.delete_index(
            vectorBucketName=vec_bucket_name,
            indexName=vec_index_name
        )
        response_bucket = s3v_client.delete_vector_bucket(
            vectorBucketName=vec_bucket_name
        )

        ok_index = response_index["ResponseMetadata"]["HTTPStatusCode"] == 200
        ok_bucket = response_bucket["ResponseMetadata"]["HTTPStatusCode"] == 200
        if ok_index and ok_bucket:
            print(f"[OK] Vector bucket '{vec_bucket_name}' and index '{vec_index_name}' deleted in {region_name}.")
            return True

        print(f"[WARN] Delete returned non-200: index={response_index['ResponseMetadata']['HTTPStatusCode']}, "
              f"bucket={response_bucket['ResponseMetadata']['HTTPStatusCode']}")
        return False

    except ClientError as e:
        print(f"[ERROR] S3Vectors deletion failed: {e}")
        return False


def delete_bedrock_knowledge_base(kb_id: str, region_name: Optional[str] = None) -> bool:
    """
    Delete an Amazon Bedrock Knowledge Base.

    **Note**
    - Must pass the Knowledge Base ID (10-char alphanumeric), not the name.
    - Before deleting, must disassociate the KB from any agents, and
      be aware of the data deletion policy and vector store permissions.

    :param kb_id: Knowledge Base ID (pattern typically '[0-9A-Za-z]{10}')
    :param region_name: AWS region (default: us-east-1)
    :return: True if delete was initiated successfully, else False
    """
    if region_name is None:
        region_name = "us-east-1"

    # Bedrock agent client for Knowledge Bases
    bedrock_agent = boto3.client(service_name="bedrock-agent", region_name=region_name)

    try:
        resp = bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb_id)
        # API returns a status and KB ID; delete transitions to DELETING (or completes quickly)
        print(f"[OK] Requested deletion of Knowledge Base '{kb_id}' in {region_name}. Response: {resp}")
        return True

    except ClientError as e:
        return False


def delete_bedrock_guardrail(guardrail_identifier: str, region_name: Optional[str] = None, 
                             guardrail_version: Optional[str] = None) -> bool:
    """
    Delete an Amazon Bedrock Guardrail.

    - If only 'guardrailIdentifier' is provided, all versions are deleted.
    - To delete a specific version, also pass 'guardrailVersion'.

    :param guardrail_identifier: Guardrail ARN or ID
    :param region_name: AWS region (default: us-east-1)
    :param guardrail_version: Optional version string to delete a specific version
    :return: True if delete succeeded, else False
    """
    if region_name is None:
        region_name = "us-east-1"

    # Bedrock client for Guardrails
    bedrock_client = boto3.client(service_name="bedrock", region_name=region_name)

    try:
        kwargs = {"guardrailIdentifier": guardrail_identifier}
        if guardrail_version:
            kwargs["guardrailVersion"] = guardrail_version

        response = bedrock_client.delete_guardrail(**kwargs)
        # Successful delete returns HTTP 200 with empty dict
        print(f"[OK] Deleted Guardrail '{guardrail_identifier}'"
              f"{' version ' + guardrail_version if guardrail_version else ''} in {region_name}. Response: {response}")
        return True

    except ClientError as e:
        print(f"[ERROR] Guardrail deletion failed for '{guardrail_identifier}': {e}")
        return False


# # Example Usage
# # Delete S4 vector bucket
# delete_vec_bucket("example_bucket", "example_index", region_name="us-east-1") 

# # Delete a KB
# delete_bedrock_knowledge_base(kb_id="ab12cd34ef", region_name="us-east-1")

# # Delete a Guardrail (all versions)
# delete_bedrock_guardrail(guardrail_identifier="arn:aws:bedrock:us-east-1:123456789012:guardrail/gr-abc123")

# # Delete a specific Guardrail version
# delete_bedrock_guardrail(guardrail_identifier="arn:aws:bedrock:us-east-1:123456789012:guardrail/gr-abc123",
#                          guardrail_version="2")
