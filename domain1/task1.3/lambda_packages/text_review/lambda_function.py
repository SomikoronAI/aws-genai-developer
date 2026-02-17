# Assignement Part 2
"""
Triggered by S3 ObjectCreated events for validated customer review files.

This Lambda performs entity extraction and sentiment analysis of the review 
data using Amazon Comprehend.

This function processes only supported text-based formats and skips raw text 
review data or artifacts.
"""


import os
import json
import boto3


# Configuration parameters
REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")

# Initilize client 
s3_client         = boto3.client(service_name="s3", region_name=REGION_NAME)
comprehend_client = boto3.client(service_name="comprehend", region_name=REGION_NAME)



def make_s3_key(current_key, folder_from, folder_to, suffix):
    """
    Input : domain1/task3/validation_results/reviews/reviews_validation.jsonl 
    Output: domain1/task3/raw_data/reviews/reviews.jsonl
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
    # Get the S3 object
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key    = event["Records"][0]["s3"]["object"]["key"]
    
    # Only process validated text reviews
    if "validation" not in key or "validation_results/" not in key:
        return {
            "statusCode": 200,
            "body": json.dumps("Not a validation data file")
        }
    
    try:
        # Get the validation results
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        
        if isinstance(content, str):
            validation_results = json.loads(content)
        else:
            print("[Error] Unknown content type")
        
        # Check if the quality score is sufficient
        if isinstance(validation_results, list):
            scores = [x["quality_score"] for x in validation_results]
            quality_score = sum(scores)/len(scores)
        elif isinstance(validation_results, dict):
            quality_score = validation_results["quality_score"]
        else:
            print("[Error] Unknown valdation result")

        print("*****************************************")
        print(f"Text quality score : {quality_score}"    )
        print("*****************************************")

        if quality_score < 0.7:  # Threshold for processing
            print(f"Quality score too low: {quality_score}")
            return {
                "statusCode": 200,
                "body": json.dumps("Quality score too low")
            }

     
        # Get the original review text
        # original_key = key.replace("validation_results", "raw_data").replace("_validation.json", ".json")
        original_key = make_s3_key(key, "validation_results", "raw_data", "")
        response = s3_client.get_object(Bucket=bucket, Key=original_key)
        content  = response["Body"].read().decode("utf-8")
        
        reviews = []
        if key.lower().endswith(".jsonl"):
            for line in content.splitlines():
                if line.strip():  # skip empty lines
                    reviews.append(json.loads(line))
        elif key.lower().endswith(".json"):
            reviews.append(json.loads(content))
        elif key.lower().endswith(".txt"):
            reviews.append({"review_text": content})

        # Extract entity and analyze sentiment using Amazon Comprehend   
        processed_results = []      
        for review in reviews:
            text  = review.get("review_text", "")

            # Detect entities
            entity_response = comprehend_client.detect_entities(
                Text=text,
                LanguageCode="en"
            )
            
            # Detect sentiment
            sentiment_response = comprehend_client.detect_sentiment(
                Text=text,
                LanguageCode="en"
            )
            
            # Detect key phrases
            key_phrases_response = comprehend_client.detect_key_phrases(
                Text=text,
                LanguageCode="en"
            )
            
            # Combine the results
            result = {
                "original_text": text,
                "entities": entity_response["Entities"],
                "sentiment": sentiment_response["Sentiment"],
                "sentiment_scores": sentiment_response["SentimentScore"],
                "key_phrases": key_phrases_response["KeyPhrases"],
                "metadata": {
                    "product_id" : review.get("product_id",  ""),
                    "customer_id": review.get("customer_id", ""),
                    "review_date": review.get("review_date", "")
                }
            }

            processed_results.append(result)

        print("*****************************************")
        print(f"Processed results: {processed_results}"  )        
        print("*****************************************")

        # Save processed results
        # processed_key = key.replace("validation_results", "processed_data").replace("_validation.json", "_processed.json")
        processed_key =  make_s3_key(key, "validation_results", "processed_data", "processed")

        s3_client.put_object(
            Bucket=bucket,
            Key=processed_key,
            Body=json.dumps(processed_results),
            ContentType="application/json"
        )
        
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
