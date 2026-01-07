"""
Creates an Amazon S3 vector bucket if it does not already exist.

Functions:
    create_vec_bucket(bucket_name, region_name=None):
    Attempts to create an S3 bucket in the specified region.
    Returns True if successful, False otherwise.

Main:
    Checks if the specified bucket exists.
    If not, creates the bucket and prints status messages.
"""

import boto3
from botocore.exceptions import ClientError


# Create S3 vector bucket client
region_name='us-east-1'
s3v_client = boto3.client(service_name="s3vectors", region_name=region_name)


def create_vec_bucket(vec_bucket_name, vec_index_name, region_name=None):
    """
    Create a S3 vector bucket and S3 vector index in a specified region.

    :param vec_bucket_name: Name of the bucket to create
    :param vec_index_name: Name of the index to create
    :param region_name: AWS region where the bucket will be created
    :return: True if bucket gets created, else False
    """
    if region_name==None:
        region_name="us-east-1"

    try:
        response_bucket = s3v_client.create_vector_bucket(
            vectorBucketName=vec_bucket_name,
            # encryptionConfiguration={"sseType": "aws:kms", "kmsKeyArn": "arn:aws:kms:...:key/...."}
            )
        
        response_index = s3v_client.create_index(
             vectorBucketName=vec_bucket_name, 
             indexName=vec_index_name,
             dataType="float32",                         # vectors are float32
             dimension=1024,                             # vector dimension 
             distanceMetric="cosine",                    # "cosine" or "euclidean"
             # encryptionConfiguration={"sseType": "aws:kms", "kmsKeyArn": "arn:aws:kms:...:key/...."},
             # metadataConfiguration={"nonFilterableMetadataKeys": ["source_id", "doc_type"]}
        )

        if response_bucket["ResponseMetadata"]["HTTPStatusCode"]==200  and response_index["ResponseMetadata"]["HTTPStatusCode"]==200:
            print(f"Vector bucket {vec_bucket_name} and {vec_index_name} created successfully in region {region_name}.")
            return True
    except ClientError as e:
        print(f"Error: {e}")
        return False



def main():
   # Create a S3 vector bucket 
    vec_bucket_name = "claim-documents-poc-kb-vectors"       # S3 vector bucket name
    vec_index_name = "claim-documents-poc-kb-vectors-index"  # S3 vector index name

    response = s3v_client.list_vector_buckets()
    vec_bucket_list = [ bucket["vectorBucketName"] for bucket in response.get("vectorBuckets",[]) if len(bucket)>0 ]

    if vec_bucket_name in vec_bucket_list:
        print("The vector bucket and the vector index already exists")
        print("Skipping the bucket creation.")
    else:
        print(f"There is no vector bucket by the name {vec_bucket_name}.")
        print("Creating a new vector bucket.")
        create_vec_bucket(vec_bucket_name, vec_index_name, region_name)



if __name__ == "__main__":
    main()

     # Delete vector index and vector bucket to save cost 
    vec_bucket_name = "claim-documents-poc-kb-vectors"      
    vec_index_name = "claim-documents-poc-kb-vectors-index" 
    # delete_vec_bucket(vec_bucket_name, vec_index_name)