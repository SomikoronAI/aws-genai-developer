# Assignment Part 1

"""
Triggered by EventBridge after validation results are written to S3.

This Lambda starts AWS Glue Data Quality ruleset evaluation runs for
configured Glue tables that source data from the raw reviews S3 folder.
Glue automatically publishes data quality metrics to CloudWatch, which
are visualized in an existing dashboard.

This function acts as an orchestrator only; it does not scan data or
compute metrics locally.
"""


import json
import os
import boto3
from datetime import datetime


# Configuration parameters
REGION_NAME    = os.environ.get("AWS_REGION", "us-east-1")
GLUE_ROLE_ARN  = os.environ["GLUE_ROLE_ARN"]

TABLE_LIST = [
    {
        "database_name": "customer_feedback_db",
        "table_name"   : "reviews",
        "ruleset_name" : "CustomerFeedbackRuleset",
    },
    {
        "database_name": "customer_feedback_db",
        "table_name"   : "surveys",
        "ruleset_name" : "CustomerFeedbackRuleset",
    }
]

# Initialize clients 
glue_client = boto3.client(service_name="glue", region_name=REGION_NAME)


def start_ruleset_evaluation(
    glue_client,
    role_arn: str,
    database: str,
    table: str,
    ruleset: str
):
    """
    Starts a Glue Data Quality ruleset evaluation run.
    CloudWatch metrics are emitted automatically by Glue.
    """
    response = glue_client.start_data_quality_ruleset_evaluation_run(
        DataSource={
            "GlueTable": {
                "DatabaseName": database,
                "TableName": table
            }
        },
        Role=role_arn,
        NumberOfWorkers=5,
        Timeout=900,
        RulesetNames=[ruleset],
        AdditionalRunOptions={
            "CloudWatchMetricsEnabled": True,
            "CompositeRuleEvaluationMethod": "COLUMN"
        }
    )

    return response["RunId"]


def lambda_handler(event, context):
    """
    Triggered by EventBridge after validation results are written to S3.
    Starts Glue Data Quality evaluations for configured tables.
    """

    print("*****************************************")
    print("Glue Data Quality Lambda Triggered")
    print(f"Event: {json.dumps(event)}")
    print("*****************************************")

    run_results = []

    for table in TABLE_LIST:
        try:
            run_id = start_ruleset_evaluation(
                glue_client=glue_client,
                role_arn=GLUE_ROLE_ARN,
                database=table["database_name"],
                table=table["table_name"],
                ruleset=table["ruleset_name"]
            )

            print(
                f"Started Data Quality run for "
                f"{table['database_name']}.{table['table_name']} "
                f"(RunId={run_id})"
            )

            run_results.append({
                "database": table["database_name"],
                "table": table["table_name"],
                "ruleset": table["ruleset_name"],
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat()
            })

        except Exception as e:
            print(
                f"ERROR starting DQ run for "
                f"{table['database_name']}.{table['table_name']}: {str(e)}"
            )
            raise e  # fail fast so EventBridge retries if needed

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Glue Data Quality runs started",
            "runs": run_results
        })
    }
