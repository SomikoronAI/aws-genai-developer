# Assignement Part 2
# Lambda function to  process audio recordings


import json
import os
import uuid
import time
import boto3


s3_client = boto3.client('s3')
transcribe = boto3.client('transcribe')


def lambda_handler(event, context):
    # Get the S3 object
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process audio files
    if not key.lower().endswith(('.mp3', '.wav', '.flac')):
        return {
            'statusCode': 200,
            'body': json.dumps('Not an audio file')
        }
    
    try:
        # Start a transcription job
        job_name = f"transcribe-{uuid.uuid4()}"
        output_key = key.replace('raw-data', 'transcriptions').replace(os.path.splitext(key)[1], '.json')
        output_uri = f"s3://{bucket}/{output_key}"
        
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={
                'MediaFileUri': f"s3://{bucket}/{key}"
            },
            MediaFormat=os.path.splitext(key)[1][1:],  # Remove the dot
            LanguageCode='en-US',
            OutputBucketName=bucket,
            OutputKey=output_key,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2  # Assuming customer and agent
            }
        )
        
        # Wait for the transcription job to complete (in production, use Step Functions or EventBridge)
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
        
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            # Process the transcription with Comprehend
            # First, get the transcription file
            response = s3_client.get_object(Bucket=bucket, Key=output_key)
            transcription = json.loads(response['Body'].read().decode('utf-8'))
            
            # Extract the transcript text
            transcript = transcription['results']['transcripts'][0]['transcript']
            
            # Use Amazon Comprehend for sentiment analysis
            comprehend = boto3.client('comprehend')
            sentiment_response = comprehend.detect_sentiment(
                Text=transcript,
                LanguageCode='en'
            )
            
            # Detect key phrases
            key_phrases_response = comprehend.detect_key_phrases(
                Text=transcript,
                LanguageCode='en'
            )
            
            # Combine the results
            processed_call = {
                'audio_key': key,
                'transcript': transcript,
                'speakers': transcription['results'].get('speaker_labels', {}).get('segments', []),
                'sentiment': sentiment_response['Sentiment'],
                'sentiment_scores': sentiment_response['SentimentScore'],
                'key_phrases': key_phrases_response['KeyPhrases'],
                'metadata': {
                    'call_id': os.path.basename(key).split('.')[0],
                    'duration': status['TranscriptionJob']['MediaFormat']
                }
            }
            
            # Save processed results
            processed_key = key.replace('raw-data', 'processed-data').replace(os.path.splitext(key)[1], '_processed.json')
            s3_client.put_object(
                Bucket=bucket,
                Key=processed_key,
                Body=json.dumps(processed_call),
                ContentType='application/json'
            )
            
            return {
                'statusCode': 200,
                'body': json.dumps('Successfully processed audio')
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps(f"Transcription failed: {status['TranscriptionJob']['FailureReason']}")
            }
        
    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }
