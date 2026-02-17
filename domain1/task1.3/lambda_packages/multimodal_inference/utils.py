import os
import json
import base64
import boto3

REGION_NAME = os.environ.get("REGION_NAME", "us-east-1")

def format_audio_data(processed_data):
    transcript = processed_data.get("transcript","")
    if not transcript:
        raise ValueError("No audio transcript available!")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                        You are an expert AI analyst.

                        Analyze the following customer call transcript.

                        Audio Transcript:
                        {transcript}

                        Provide response and recommendations regarding the call transcript strictly 
                        in valid JSON format with the following structure:
                        
                        {{
                        "Sentiment": "...",
                        "Key_issues": [],
                        "Summary": "...",
                        "Recommensations": []
                        }}
                        """
                }
            ]
        }
    ]


def format_image_data(processed_data):
    bucket_name = processed_data.get("bucket") 
    image_key   = processed_data.get("image_key")
        
    if not all([bucket_name, image_key]):
        raise ValueError("Missing image metadata!")

    # Detect media type from extension
    if image_key.lower().endswith(".png"):
        media_type = "image/png"
    else:
        media_type = "image/jpeg"

    # Fetch original image in bytes
    s3_client = boto3.client(service_name="s3", region_name=REGION_NAME)
    response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
    image_bytes = response["Body"].read()
    base64_string = base64.b64encode(image_bytes).decode("utf-8")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_string
                    },
                },
                {
                    "type": "text",
                    "text": f"""
                        Extract insights from this image, summarize the content, and provide 
                        recommendations if you have any.

                        Return output in valid JSON format:

                        {{
                        "Visible_text": "...",
                        "Detected_objects": [],
                        "Sentiment": "...",
                        "Risk_level": "...",
                        "Recommendations": []
                        }}
                        """
                }
            ]
        }
    ]


def format_review_data(processed_data):
    text = processed_data.get("original_text", "")
    if not text:
        raise ValueError("No review text available")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                        You are an expert AI analyst.

                        Analyze this text review:

                        Text Review:
                        {text}

                        Provide response and recommendations regarding the review strictly 
                        in valid JSON format with the following structure:
                        
                        {{
                        "Sentiment": "...",
                        "Key issues": [],
                        "Summary": "...",
                        "Recommensations": []
                        }}
                        """
                }
            ]
        }
    ]


def format_survey_data(processed_data):
    summary = processed_data.get("summary_text", "")
    if not summary:
        raise ValueError("No survey summary available!")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                        You are an expert AI analyst.

                        Analyze this survey summary, extract trends, and provide 
                        recommendations if you have any:

                        Survey Summary:
                        {summary}

                        Return output in valid JSON format:

                        {{
                        "Dominant_sentiment": "...",
                        "Top_trends": [],
                        "Risk_areas": [],
                        "Recommendions": []
                        }}
                        """
                }
            ]
        }
    ]

