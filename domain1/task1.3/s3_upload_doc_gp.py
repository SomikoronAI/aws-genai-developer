"""
Upload structured project data files to an Amazon S3 general-purpose bucket.

This script:
- Validates the file type based on the provided --source-type
  (reviews, surveys, images, audios).
- Enforces allowed file extensions per source type.
- Uploads the local file from the ./data directory to S3.
- Organizes objects using the key structure:
  domain1/raw_data/<source_type>/<file_name>

If --bucket-name is not provided, the default bucket
"aws-genai-developer-pro" is used.

Requirements:
- AWS credentials configured (via environment variables, AWS CLI, or IAM role).
- S3 bucket must already exist.
- Boto3 installed and aws CLI confifured.
- Optional: .env file containing REGION_NAME and ACCOUNT_ID.

Example CLI usage in Windows PS:
python s3_upload_doc_gp.py `
--source-type reviews `
--file-name product_reviews.jsonl

Example import usage:
from upload_doc_to_s3_gp import upload_document
"""


import os
import sys
import json
import argparse
import pathlib
from dotenv import load_dotenv, find_dotenv


import boto3
from botocore.exceptions import ClientError


# Configuration parameters
try:
    env_file = os.getenv(".env")
    if env_file:
        load_dotenv(pathlib.Path(env_file).expanduser().resolve())
    else:
        load_dotenv(find_dotenv(usecwd=True) or ".env")
except Exception:
    pass

region_name = os.environ.get("REGION_NAME", "us-east-1")
account_id  = os.environ.get("ACCOUNT_ID", "")


def upload_document(file_name, bucket_name, s3_object_name=None):
    # If S3 object_name was not specified, use file_name
    if s3_object_name is None:
        s3_object_name = file_name

    try:
        s3_client = boto3.client(service_name="s3", region_name=region_name)
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
    # Upload file to S3
    parser = argparse.ArgumentParser(
        prog="s3_upload_doc_gp.py", 
        description="Upload file to S3", 
        epilog="Upload data made easier"
        )

    parser.add_argument("--bucket-name", default="aws-genai-developer-pro")
    parser.add_argument("--source-type", required=True, choices=["reviews", "surveys", "images", "audios"])
    parser.add_argument("--file-name", required=True)

    args = parser.parse_args()

    bucket_name = args.bucket_name
    source_type = args.source_type
    file_name = args.file_name

    file_lower = file_name.lower()

    # Validate file extension
    if source_type == "reviews":
        valid_ext = (".json", ".jsonl", ".txt")
    elif source_type == "surveys":
        valid_ext = (".csv", ".txt")
    elif source_type == "images":
        valid_ext = (".jpg", ".jpeg", ".png")
    elif source_type == "audios":
        valid_ext = (".mp3", ".mp4", ".wav", ".flac")
    else:
        print("[Error] Unknown source type")
        print("Valide source type ['reviews', 'surveys', 'images', 'audios']")
        sys.exit(1)

    if not file_lower.endswith(valid_ext):
        print(f"[Error] Invalid file format for {source_type}")
        print("File format extension must be in lower case")
        sys.exit(1)

    key_name = file_lower
    file_path = f"./data/{file_name}"
    s3_key_name = f"domain1/raw_data/{source_type}/{key_name}"

    print("Bucket:", bucket_name)
    print("File path:", file_path)
    print("S3 key:", s3_key_name)

    upload_document(file_path, bucket_name, s3_key_name)



if __name__ == "__main__":
    main()
