"""
Creates an Amazon S3 general-purpose bucket if it does not already exist.

Functions:
    create_gp_bucket(bucket_name, region_name=None):
    Attempts to create an S3 bucket in the specified region.
    Returns True if successful, False otherwise.

Main:
    Checks if the specified bucket exists.
    If not, creates the bucket and prints status messages.
"""

import boto3
from botocore.exceptions import ClientError


# Create S3 general-purpose bucket client
region_name='us-east-1'
s3_client = boto3.client(service_name='s3', region_name=region_name)


def create_gp_bucket(bucket_name, region_name=None):
    """
    Create an S3 bucket in a specified region.

    :param bucket_name: Name of the bucket to create
    :param region_name: AWS region where the bucket will be created
    :return: True if bucket gets created, else False
    """
    if region_name==None:
        region_name="us-east-1"

    try:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"General-Purpose bucket {bucket_name} created successfully in region {region_name}.")
        return True
    except ClientError as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Create a S3 general purpose bucket 
    gp_bucket_name = "claim-documents-poc-nururrahman"   # S3 GP bucket name

    response = s3_client.list_buckets()
    bucket_list = [ bucket["Name"] for bucket in response.get("Buckets",[]) if len(bucket)>0 ]

    if gp_bucket_name in bucket_list:
        print("The general purpose bucket already exists.")
        print("Skipping the bucket creation.")
    else:
        print("There is no bucket by this name.")
        print("Creating a new bucket.")
        create_gp_bucket(gp_bucket_name, region_name)

