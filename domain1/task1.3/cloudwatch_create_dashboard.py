# Assignment Part 1

"""
Creates and manages the CloudWatch dashboard for the customer feedback
data quality pipeline.

The dashboard visualizes text validation metrics emitted by the
S3-triggered Lambda function and data quality metrics automatically
published by AWS Glue Data Quality evaluations. It is used for
operational monitoring and trend analysis across the pipeline.
"""


import os
import json
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

# Initialize client
cloudwatch_client = boto3.client(service_name="cloudwatch", region_name=region_name)


dashboard_name = "CustomerFeedbackQuality"
lambda_aggregation = 300
glue_aggregation   = 900

dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [
                        "CustomerFeedback/TextQuality", "QualityScore", "Source", "TextReviews"]
                    ],
                "period": lambda_aggregation,
                "stat": "Average",
                "region": region_name,
                "title": "Text Review: Quality Score"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [
                        "CustomerFeedback/TextQuality", "FilesProcessed", "Source", "TextReviews"]
                    ],
                "period": lambda_aggregation,
                "stat": "Sum",
                "region": region_name,
                "title": "Text Review: Files Processed"
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [
                        "CustomerFeedback/TextQuality", "ReviewsFailed", "Source", "TextReviews"]
                    ],
                "period": lambda_aggregation,
                "stat": "Sum",
                "region": region_name,
                "title": "Text Review: Reviews Failed"
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [
                        "CustomerFeedback/TextQuality", "ReviewsProcessed", "Source", "TextReviews"]
                    ],
                "period": lambda_aggregation, 
                "stat": "Sum",
                "region": region_name,
                "title": "Text Review: Reviews Processed"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [
                        "AWS/GlueDataQuality",
                        "dq.rules.passed",
                        "RulesetName",
                        "CustomerFeedbackRuleset"
                    ],
                    [
                        "AWS/GlueDataQuality",
                        "dq.rules.failed",
                        "RulesetName",
                        "CustomerFeedbackRuleset"
                    ]
                ],
                "period": glue_aggregation,
                "stat": "Sum",
                "region": region_name,
                "title": "Glue Data Quality: Rules Passed vs Failed"
            }
        }
    ]
}


def ensure_dashboard(name: str):
    try:
        cloudwatch_client.get_dashboard(DashboardName=name)
        print(f"[OK] Cloudwatch dashboard already exists: {name}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFound":
            return False
        else:
            raise


def check_or_create_dashboard(name: str, body: dict):
    if ensure_dashboard(name):
        print(f"Dashboard: {name} already exisits")
    else:
        response = cloudwatch_client.put_dashboard(
            DashboardName=name,
            DashboardBody=json.dumps(body)
        )
        print("Cloudwatch dashboard has been created")
        print("Printing response meta data ... ...  ")
        print( response['ResponseMetadata'] )


def get_dashboard_arn(name: str):
    response = cloudwatch_client.list_dashboards(DashboardNamePrefix=name)
    dashboard_arn = response["DashboardEntries"][0]["DashboardArn"]
    print(f"Created dashboard arn: {dashboard_arn}")




if __name__ == "__main__":
    check_or_create_dashboard(dashboard_name, dashboard_body)
    get_dashboard_arn(dashboard_name)