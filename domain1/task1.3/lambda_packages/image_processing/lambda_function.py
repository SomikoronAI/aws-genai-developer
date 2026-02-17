# Assignement Part 2
"""
S3-triggered Lambda for image-based text extraction and visual analysis.

This function is invoked by S3 ObjectCreated events for raw product image
files stored in a designated input prefix. It processes supported image
formats (PNG, JPG, JPEG) and performs the following steps:

- Extracts readable text from images using Amazon Textract
- Detects objects, scenes, and concepts using Amazon Rekognition
- Performs secondary text detection via Rekognition as a fallback
- Aggregates extracted text, detected labels, and basic metadata into
  a structured JSON payload
- Writes the processed results to a corresponding S3 processed-data
  prefix using a deterministic key transformation

Non-image files and derived artifacts are safely ignored. The function
is designed as a lightweight, stateless preprocessing step for downstream
document understanding, search, or analytics pipelines.
"""



import os
import json
import boto3


# Configuration parameters
REGION_NAME = os.environ["REGION_NAME"]

# Initilize client 
s3_client         = boto3.client(service_name="s3", region_name=REGION_NAME)
textract_client   = boto3.client(service_name="textract", region_name=REGION_NAME)
rekognition_client= boto3.client(service_name="rekognition", region_name=REGION_NAME)



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
    """
    Handle S3 ObjectCreated events for raw image uploads.

    Extracts text and visual labels from the uploaded image and writes
    a processed JSON representation to the processed-data S3 prefix.
    """
    # Get the S3 object
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key    = event["Records"][0]["s3"]["object"]["key"]
    
    # Only process image files
    if not key.lower().endswith((".png", ".jpg", ".jpeg")):
        return {
            "statusCode": 200,
            "body": json.dumps("Not an image file")
        }
    
    try:
        # Extract text from the image using Amazon Textract
        textract_response = textract_client.detect_document_text(
            Document={
                "S3Object": {
                    "Bucket": bucket,
                    "Name": key
                }
            }
        )
        # Extract the text
        extracted_text = ""
        for item in textract_response["Blocks"]:
            if item["BlockType"] == "LINE":
                extracted_text += item["Text"] + "\n"
        

        # Analyze the image using Amazon Rekognition        
        # Detect labels
        label_response = rekognition_client.detect_labels(
            Image={
                "S3Object": {
                    "Bucket": bucket,
                    "Name": key
                }
            },
            MaxLabels=10,
            MinConfidence=70
        )
        # Detect text (as a backup to Textract)
        text_response = rekognition_client.detect_text(
            Image={
                "S3Object": {
                    "Bucket": bucket,
                    "Name": key
                }
            }
        )
        detected_text = ""
        for item in text_response["TextDetections"]:
            if item["Type"]=="LINE":
                detected_text += item["DetectedText"] + "\n"

        # Combine the results
        processed_image = {
            "bucket": bucket, 
            "image_key": key,
            "extracted_text": extracted_text,
            "labels": [label for label in label_response["Labels"]],
            "detected_text": detected_text,
            "metadata": {
                "product_id": os.path.basename(key).split("_")[0] if "_" in os.path.basename(key) else ""
            }
        }
        
        # Save processed results
        # processed_key = key.replace("raw_data", "processed_data").replace(os.path.splitext(key)[1], "_processed.json")
        processed_key = make_s3_key(key, "raw_data", "processed_data", "processed")
        s3_client.put_object(
            Bucket=bucket,
            Key=processed_key,
            Body=json.dumps(processed_image),
            ContentType="application/json"
        )
        
        return {
            "statusCode": 200,
            "body": json.dumps("Successfully processed image")
        }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }
