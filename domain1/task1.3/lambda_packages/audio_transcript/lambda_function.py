# Assignement Part 2
"""
S3-triggered Lambda for audio transcription and language analysis.

This function is invoked by S3 ObjectCreated events for raw audio files
(MP3, WAV, FLAC). It submits the audio to Amazon Transcribe, waits for
job completion, retrieves the transcription output from S3, and enriches
the transcript using Amazon Comprehend for sentiment and key phrase
analysis.

The processed results - including transcript text, speaker segments,
sentiment scores, and extracted key phrases - are written to a designated
processed-data S3 prefix using a deterministic key transformation.

Non-audio files are ignored. This function is intended as a preprocessing
step for conversational analytics and downstream NLP workflows.
"""



import os
import json
import uuid
import time
import boto3


# Configuration parameters
REGION_NAME = os.environ["REGION_NAME"]

# Initilize client 
s3_client         = boto3.client(service_name="s3", region_name=REGION_NAME)
comprehend_client = boto3.client(service_name="comprehend", region_name=REGION_NAME)
transcribe_client = boto3.client(service_name="transcribe", region_name=REGION_NAME)


def make_s3_key(current_key, folder_from, folder_to, suffix):
    """
    Input : domain1/task3/raw_data/audios/product_reviews.mp3 
    Output: domain1/task3/processed_data/audios/product_reviews.jsonl
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
    
    # Only process audio files
    if not key.lower().endswith((".mp3", ".mp4", ".wav", ".flac")):
        return {
            "statusCode": 200,
            "body": json.dumps("Not an audio file")
        }
    
    try:
        # Start a transcription job
        job_name   = f"transcribe-{uuid.uuid4()}"

        # output_key = key.replace("raw_data", "transcriptions").replace(os.path.splitext(key)[1], ".json")
        output_key = make_s3_key(key, "raw_data", "transcriptions", "")
        output_uri = f"s3://{bucket}/{output_key}"
        
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={
                "MediaFileUri": f"s3://{bucket}/{key}"
            },
            MediaFormat=os.path.splitext(key)[1][1:],  # Remove the dot
            LanguageCode="en-US",
            OutputBucketName=bucket,
            OutputKey=output_key,
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 2  # Assuming customer and agent
            }
        )
        
        # Wait for the transcription job to complete (in production, use Step Functions or EventBridge)
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
                break
            time.sleep(5)
        
        if status["TranscriptionJob"]["TranscriptionJobStatus"] == "COMPLETED":
            # Process the transcription with Comprehend
            # First, get the transcription file
            response = s3_client.get_object(Bucket=bucket, Key=output_key)
            transcription = json.loads(response["Body"].read().decode("utf-8"))
            
            # Extract the transcript text
            transcript = transcription["results"]["transcripts"][0]["transcript"]
            
            # Use Amazon Comprehend for sentiment analysis
            sentiment_response = comprehend_client.detect_sentiment(
                Text=transcript,
                LanguageCode="en"
            )
            
            # Detect key phrases
            key_phrases_response = comprehend_client.detect_key_phrases(
                Text=transcript,
                LanguageCode="en"
            )
            
            # Combine the results
            processed_call = {
                "audio_key": key,
                "transcript": transcript,
                "speakers": transcription["results"].get("speaker_labels", {}).get("segments", []),
                "sentiment": sentiment_response["Sentiment"],
                "sentiment_scores": sentiment_response["SentimentScore"],
                "key_phrases": key_phrases_response["KeyPhrases"],
                "metadata": {
                    "call_id": os.path.basename(key).split(".")[0],
                    "duration": status["TranscriptionJob"]["MediaFormat"]
                }
            }
            
            # Save processed results
            # processed_key = key.replace("raw-data", "processed-data").replace(os.path.splitext(key)[1], "_processed.json")
            processed_key = make_s3_key(key, "raw_data", "processed_data", "processed")

            s3_client.put_object(
                Bucket=bucket,
                Key=processed_key,
                Body=json.dumps(processed_call),
                ContentType="application/json"
            )
            
            return {
                "statusCode": 200,
                "body": json.dumps("Successfully processed audio")
            }
        else:
            return {
                "statusCode": 500,
                "body": json.dumps(f"Transcription failed: {status["TranscriptionJob"]["FailureReason"]}")
            }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }
