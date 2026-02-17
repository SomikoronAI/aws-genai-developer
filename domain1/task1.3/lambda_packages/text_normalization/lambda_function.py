# Assignment Part 4
"""
Lambda entry point to normalize a file.
Expects: event['file_path'] pointing to S3 local path or Lambda temp file.
"""
    

import json
from core_build_docs import build_docs


def lambda_handler(event, context):
    file_path = event.get("file_path")
    if not file_path:
        return {"statusCode": 400, "body": "Missing file_path in event"}

    doc = build_docs(file_path, id_strategy="content")
    if doc is None:
        return {"statusCode": 500, "body": f"Failed to process file {file_path}"}

    return {
        "statusCode": 200, 
        "body": json.dumps(doc)
        }
