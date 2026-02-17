# Assignment Part 1

import os
import json
import pathlib
from dotenv import load_dotenv, find_dotenv

import boto3
from botocore.exceptions import ClientError
# from awsglue.data_quality import DataQualityRule, DataQualityRulesetEvaluator


# Configuration parameters
try:
    env_file = os.getenv(".env")
    if env_file:
        load_dotenv(pathlib.Path(env_file).expanduser().resolve())
    else:
        load_dotenv(find_dotenv(usecwd=True) or ".env")
except Exception:
    pass

region_name   = os.environ["region_name"]
account_id    = os.environ.get("account_id", "")
glue_role_arn = os.environ.get("glue_role_arn", "")


# Iniitialize client
glue_client = boto3.client(service_name="glue", region_name=region_name)


table_list = [
    {
        "database_name" : "customer_feedback_db",
        "table_name"    : "reviews",
        "ruleset_name"  : "CustomerFeedbackRuleset",
    },
    {
        "database_name" : "customer_feedback_db",
        "table_name"    : "surveys",
        "ruleset_name"  : "CustomerFeedbackRuleset",
    }
]


def attach_ruleset(glue_client, role_arn: str, database: str, table: str, ruleset: str):
    try:
        glue_client.start_data_quality_ruleset_evaluation_run(
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
                "CompositeRuleEvaluationMethod": "COLUMN", #"COLUMN"|"ROW"
            }
        )
        print(f"Rule set {ruleset} is attahced to table {table}")
    except Exception as e:
        print(e)


def main():
    for tabel in tabel_list:
        attach_ruleset(
            glue_client, 
            glue_role_arn, 
            database = table["database_name"],
            table = tabel["table_name"],
            ruleset = tabel["ruleset_name"]
        )



if __name__ == "__main__":
    main()
