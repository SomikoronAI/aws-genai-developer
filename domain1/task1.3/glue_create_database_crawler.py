# Assignment Part 1

"""
Create a Glue database and a crawler, then start the crawler.

Equivalent to:
1. aws glue create-database --database-input Name=customer_feedback_db
2. aws glue create-crawler --name customer-feedback-crawler `
--role AWSGlueServiceRole-CustomerFeedback `
--database-name customer_feedback_db `
--targets '{"S3Targets": [{"Path": "s3://<bucket>/<prefix>/raw-data/"}]}'
  
3. aws glue start-crawler --name customer-feedback-crawler
"""

import os
import sys
import pathlib
from dotenv import load_dotenv, find_dotenv

import boto3
import botocore


# Configuration parameters
try:
    env_file = os.getenv(".env")
    if env_file:
        load_dotenv(pathlib.Path(env_file).expanduser().resolve())
    else:
        load_dotenv(find_dotenv(usecwd=True) or ".env")
except Exception:
    pass

region_name   = os.environ.get("REGION_NAME", "us-east-1")
account_id    = os.environ.get("ACCOUNT_ID", "")
glue_role_arn = os.environ.get("GLUE_ROLE_ARN", "")

bucket_name   = "aws-genai-developer-pro"
prefix_name   = "domain1/task3"
database_name = "customer_feedback_db"
crawler_name  = "customer_feedback_crawler"


# Initialize clients
glue_client = boto3.client(service_name="glue", region_name=region_name)


crawler_list = [
    {
        "name": "customer_feedback_reviews_crawler",
        "s3_path": f"s3://{bucket_name}/{prefix_name}/raw_data/reviews/",
    },
    {
        "name": "customer_feedback_surveys_crawler",
        "s3_path": f"s3://{bucket_name}/{prefix_name}/raw_data/surveys/",
    }
]


def ensure_database(glue_client, name: str):
    try:
        print("")
        glue_client.create_database(DatabaseInput={"Name": name})
        print(f"[OK] Created Glue database: {name}")
    except glue_client.exceptions.AlreadyExistsException:
        print(f"[OK] Glue database already exists: {name}")


def ensure_crawler(glue_client, role: str, database: str, crawler_name: str,  s3_path: str):
    # Check if crawler exists
    try:
        glue_client.get_crawler(Name=crawler_name)
        exists = True
    except glue_client.exceptions.EntityNotFoundException:
        exists = False

    if exists:
        # Update the crawler targets/role/db if needed
        glue_client.update_crawler(
            Name=crawler_name,
            Role=role,
            DatabaseName=database,
            Targets={
                "S3Targets": [{
                    "Path": s3_path,
                     "Exclusions": ["**/*.png", "**/*.jpg", "**/*.mp3"]
                    }]
                },
        )
        print(f"[OK] Updated existing crawler: {crawler_name}")
    else:
        glue_client.create_crawler(
            Name=crawler_name,
            Role=role,
            DatabaseName=database,
            Targets={
                "S3Targets": [{
                    "Path": s3_path, 
                     "Exclusions": ["**/*.png", "**/*.jpg", "**/*.mp3"]
                    }]
                },
        )
        print(f"[OK] Created crawler: {crawler_name}")


def start_crawler(glue_client, crawler_name: str):
    try:
        glue_client.start_crawler(Name=crawler_name)
        print(f"[OK] Started crawler: {crawler_name}")
    except glue_client.exceptions.CrawlerRunningException:
        print(f"[INFO] Crawler already running: {crawler_name}")
    except botocore.exceptions.ClientError as e:
        print(f"[ERROR] Failed to start crawler: {e}", file=sys.stderr)
        raise


def main():
    ensure_database(glue_client, database_name)

    for crawler in crawler_list:
        ensure_crawler(
            glue_client, 
            glue_role_arn, 
            database_name, 
            crawler["name"], 
            crawler["s3_path"]
            )
        start_crawler(glue_client, crawler["name"])

    print("\nDone. Monitor progress in the AWS Glue console --> Crawlers.")



if __name__ == "__main__":
    main()

