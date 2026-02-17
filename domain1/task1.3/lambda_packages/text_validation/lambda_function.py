# Assignment Part 1
"""
Triggered by S3 ObjectCreated events for raw customer review files.

This Lambda performs lightweight text and semantic validation checks on
uploaded review data, computes per-file and aggregate quality metrics,
and writes validation statistics to an S3 validation results folder.
It also publishes text quality metrics to CloudWatch for monitoring and
dashboard visualization.

This function processes only supported text-based formats and skips
derived or validation artifacts.
"""



import os
import json 
import uuid
import re
from datetime import datetime, timezone
from urllib.parse import unquote_plus
import posixpath

import boto3


# Configuration parameters
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")

# Initilize client 
s3_client          = boto3.client(service_name="s3", region_name=REGION_NAME)
cloudwatch_client  = boto3.client(service_name="cloudwatch", region_name=REGION_NAME)
eventbridge_client = boto3.client(service_name="events", region_name=REGION_NAME)



def make_s3_key(current_key, folder_from, folder_to, suffix):
    """
    Input : domain1/task3/raw_data/reviews/reviews.jsonl
    Output: domain1/task3/validation_results/reviews/reviews_validation.jsonl
    """
    parts = current_key.split("/")

    try:
        index = parts.index(folder_from)
    except ValueError:
        raise ValueError(f"{folder_from} not found in key: {current_key}")

    # Replace existing folder with expected folder
    parts[index] = folder_to

    # Filename handling
    filename = parts[-1]
    stem, ext = filename.rsplit(".", 1)
    
    if stem.endswith("_validation"):
        stem = stem.replace("_validation", "")
    elif stem.endswith("_processed"):
        stem = stem.replace("_processed", "")
        
    if suffix:
        new_filename = f"{stem}_{suffix}.jsonl"
    else:
        new_filename = f"{stem}.jsonl"

    # Final key 
    parts[-1] = new_filename
    final_key = "/".join(parts)
    return final_key



def lambda_handler(event, context):
    """
    Handle S3 ObjectCreated events for raw text file uploads.

    Extracts key metrics from the uploaded files and writes to 
    a designated S3 prefix.
    """

    # Get the S3 object
    bucket  = event["Records"][0]["s3"]["bucket"]["name"]
    raw_key = event["Records"][0]["s3"]["object"]["key"]
    key     = unquote_plus(raw_key)

    # Skip any validation results 
    if "_validation" in key or "validation_results/" in key:
        print(f"Skipping validation file: {key}")
        return {
            "statusCode": 200,
            "body": json.dumps("Validation data file skipped")
        }

    # Only process text reviews
    if not key.lower().endswith( (".txt",".json",".jsonl") ):
        return {
            "statusCode": 200,
            "body": json.dumps("Text review file has unknown format")
        }
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content  = response["Body"].read().decode("utf-8")
        
        # Parse the content from different formats
        reviews = []
        if key.lower().endswith(".jsonl"):
            for line in content.splitlines():
                if line.strip():  # skip empty lines
                    reviews.append(json.loads(line))
        elif key.lower().endswith(".json"):
            reviews.append(json.loads(content))
        elif key.lower().endswith(".txt"):
            reviews.append({"review_text": content})

        # Get validation checks
        validation_results = []
        for review in reviews:
            text = review.get("review_text", "")
            result = {
                "file_name": key,
                "review_id": review.get("review_id"), 
                "timestamp": datetime.now().isoformat(),
                "checks": {
                    "min_length": len(text) >= 10,
                    "has_product_reference": bool(re.search(r"product|item|purchase", text, re.IGNORECASE)),
                    "has_opinion": bool(re.search(r"like|love|hate|good|bad|great|terrible|excellent|poor|recommend", text, re.IGNORECASE)),
                    "no_profanity": not bool(re.search(r"bloody|bastard|crap|damn|shit", text, re.IGNORECASE)),  # Profanity list
                    "has_structure": text.count(".") >= 1  # At least one sentence
                }
            }
            
            # Calculate overall quality score (simple version)
            passed_checks = sum(1 for check in result["checks"].values() if check)
            total_checks  = len(result["checks"])
            result["quality_score"] = passed_checks / total_checks
            validation_results.append(result)
        
        print("*****************************************")
        print(f"Validation results: {validation_results}")        
        print("*****************************************")

        # Send metrics to CloudWatch
        average_quality_score = sum(r["quality_score"] for r in validation_results) / len(validation_results)
        print( f"Average Quality Score: {average_quality_score}" )
        cloudwatch_client.put_metric_data(
            Namespace="CustomerFeedback/TextQuality",
            MetricData=[
                {
                    "MetricName": "QualityScore",
                    "Value": average_quality_score,
                    "Unit": "None",
                    "Dimensions": [
                        {"Name": "Source", "Value": "TextReviews"},
                    ]
                }
            ]
        )
        #
        print( f"Number of files processed: {1}" )
        cloudwatch_client.put_metric_data(
            Namespace="CustomerFeedback/TextQuality",
            MetricData=[
                {
                    "MetricName": "FilesProcessed",
                    "Value": 1,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "Source", "Value": "TextReviews"}
                    ]
                }
            ]
        )
        #
        reviews_processed = len(validation_results)
        print( f"Number of reviews processed: {reviews_processed}" )
        cloudwatch_client.put_metric_data(
            Namespace="CustomerFeedback/TextQuality",
            MetricData=[
                {
                    "MetricName": "ReviewsProcessed",
                    "Value": reviews_processed,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "Source", "Value": "TextReviews"}
                    ]
                }
            ]
        )
        #
        reviews_failed = sum(1 for r in validation_results if r["quality_score"] < 0.5)
        print( f"Number of failed reviews: {reviews_failed}" )
        cloudwatch_client.put_metric_data(
            Namespace="CustomerFeedback/TextQuality",
            MetricData=[
                {
                    "MetricName": "ReviewsFailed",
                    "Value": reviews_failed,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "Source", "Value": "TextReviews"}
                    ]
                }
            ]
        )

        # Save validation results
        validation_key = make_s3_key(key, "raw_data", "validation_results", "validation")
        print("*****************************************")
        print(f"Validation results key: {validation_key}")        
        print("*****************************************")
        s3_client.put_object(
            Bucket=bucket,
            Key=validation_key,
            Body=json.dumps(validation_results),
            ContentType="application/json"
        )
        
        # Send event results to the Eventbridge
        response = eventbridge_client.put_events(
            Entries=[
                {
                    "Source": "customer-feedback.validation-lambda",
                    "DetailType": "TextValidationCompleted",
                    "Detail": json.dumps({
                        "status": "SUCCESS",
                        "pipeline": "customer-feedback"
                    }),
                     "EventBusName": "default",
                     "Time": datetime.now(timezone.utc)
                }
            ]
        )
        print("*****************************************" )
        print("EventBridge put_events response:", response)
        print("*****************************************" )

        return {
            "statusCode": 200,
            "body": json.dumps("Successfully processed reviews")
        }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }

