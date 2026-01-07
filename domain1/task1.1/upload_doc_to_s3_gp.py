"""
Uploads local files to an Amazon S3 general-purpose bucket.

This script performs the following tasks:
- Uploads the specified file to the given S3 bucket.
- If s3_object_name is not provided, the local file name is used as the object key.
- Returns True if upload succeeds, otherwise False.

Requirements:
- AWS credentials with permissions for Bedrock and S3.
- Existing S3 bucket to store policy documents.
- Boto3 installed and configured.

Usage:
import boto3
from upload_doc_to_s3_gp import upload_document
"""

import boto3
from botocore.exceptions import ClientError


def upload_document(s3_client, file_name, bucket_name, s3_object_name=None):
    # If S3 object_name was not specified, use file_name
    if s3_object_name is None:
        s3_object_name = file_name

    try:
        s3_client.upload_file(file_name, bucket_name, s3_object_name)
        print(f"File {file_name} uploaded to bucket {bucket_name} as {s3_object_name}.")
        return True
    except FileNotFoundError:
        print("Error: The file was not found. There is no upload to S3.")
        return False
    except ClientError as e:
        print(f"Error: {e}")
        return False


def main():
    region_name="us=east-1"
    s3_client = boto3.client(service_name="s3", region_name=region_name)

    # Upload file to S3
    bucket_name = "claim-documents-poc-nururrahman"           # S3 bucket name
    file_path = "./data/claims/auto_insurance_claim1.txt"     # Path to the local file
    s3_object_key = "claims/auto_insurance_claim1.txt"        # Desired object name in S3
    upload_document(s3_client, file_path, bucket_name, s3_object_key)



if __name__ == "__main__":
    main()
