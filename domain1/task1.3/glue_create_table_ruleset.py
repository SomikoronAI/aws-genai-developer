# Assignment Part 1

"""
Glue Data Quality ruleset for customer feedback reviews.

"""


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

region_name     = os.environ.get("REGION_NAME", "us-east-1")
account_id      = os.environ.get("ACCOUNT_ID", "")

# Iniitialize client
glue_client = boto3.client(service_name="glue", region_name=region_name)


RULESET_NAME = "CustomerFeedbackRuleset"
RULESET_DESC = "Data quality rules for customer feedback reviews"
RULESET_TAGS = {"Project": "CustomerFeedbackAnalysis"}

# # Define rules for customer reviews
# rules = [
#     # Check for completeness of required fields
#     DataQualityRule.is_complete("review_text"),
#     DataQualityRule.is_complete("product_id"),
#     DataQualityRule.is_complete("customer_id"),
    
#     # Check for valid values
#     DataQualityRule.column_values_match_pattern(
#     "review_text", ".{10,}"
#     ),  # At least 10 chars
    
#     DataQualityRule.column_values_match_pattern(
#     "rating", "^[1-5]$"
#     ),  # Rating 1-5
    
#     # Check for data consistency
#     DataQualityRule.column_values_match_pattern(
#     "review_date", "\\d{4}-\\d{2}-\\d{2}"
#     ),  # YYYY-MM-DD
    
#     # Check for statistical properties
#     DataQualityRule.column_length_distribution_match(
#         "review_text", 
#         min_length=10, 
#         max_length=1000
#     )
# ]

# Minimal DQDL builder helpers (string-based)
def dq_is_complete(column: str):
    # DQDL uses IsComplete "col"
    return f'IsComplete "{column}"'  

def dq_matches_regex(column: str, pattern: str):
    # DQDL uses: ColumnValues "col" matches "regex"  (double quotes)
    # Escape double quotes inside the pattern.
    esc = pattern.replace('"', '\\"')
    return f'ColumnValues "{column}" matches "{esc}"'  

def dq_length_between(column: str, min_len: int, max_len: int):
    # ColumnLength "col" between a and b  (expression grammar supports between)
    return f'ColumnLength "{column}" between {min_len} and {max_len}'  


def build_ruleset(rules: list[str]):
    inner = ",\n  ".join(rules)
    return f"Rules = [\n  {inner}\n]"


# The rules translated to DQDL 
rules = [
    dq_is_complete("review_id"),
    dq_is_complete("product_id"),
    dq_is_complete("customer_id"),
    dq_is_complete("rating"),
    dq_is_complete("review_date"),
    dq_is_complete("review_text"),

    dq_matches_regex("review_text", r"^[\s\S]{10,}$"),
    dq_matches_regex("rating", r"^[1-5]$"),
    dq_matches_regex("review_date", r"^\d{4}-\d{2}-\d{2}$"),

    dq_length_between("review_date", 10, 100),
    dq_length_between("review_text", 10, 100),
]
DQDL = build_ruleset(rules)



def get_ruleset_arn(name: str):
    try:
        response = glue_client.get_data_quality_ruleset(Name=name)
        return response["RulesetArn"]
    except glue_client.exceptions.EntityNotFoundException:
        return None


def create_or_update_ruleset(name: str, description: str, dqdl: str, tags: dict | None = None):
    ruleset_arn = get_ruleset_arn(name)

    if ruleset_arn:
        glue_client.update_data_quality_ruleset(
            Name=name,
            Description=description,
            Ruleset=dqdl,
        )
        print(f"[OK] Updated ruleset: {name}")
    else:
        resp = glue_client.create_data_quality_ruleset(
            Name=name,
            Description=description,
            Ruleset=dqdl,
            Tags={},  # tags applied explicitly below
        )
        ruleset_arn = resp["RulesetArn"]
        print(f"[OK] Created ruleset: {name}")

    # Apply tags
    if tags and account_id:
        glue_client.tag_resource(
            ResourceArn=ruleset_arn,
            TagsToAdd=tags,
        )
        print(f"[OK] Tagged ruleset: {tags}")


if __name__ == "__main__":
    create_or_update_ruleset(
        RULESET_NAME,
        RULESET_DESC,
        DQDL,
        RULESET_TAGS,
    )

    print("\nFinal DQDL:\n")
    print(DQDL)
