import os
import json
import pathlib
from dotenv import load_dotenv, find_dotenv

import boto3


try:
    env_file = os.getenv(".env")
    if env_file:
        load_dotenv(pathlib.Path(env_file).expanduser().resolve())
    else:
        load_dotenv(find_dotenv(usecwd=True) or ".env")
except Exception:
    pass

region_name = os.environ["region_name"]

bucket_name = "aws-genai-developer-pro"
prefix_name = "domain1/task3/raw_data/images/"


rekognition_client = boto3.client(service_name="rekognition", region_name=region_name)

image_list = ["p_1001_1.png", "p_1002_1.png", "p_1003_1.png", "p_1004_1.png", "p_1005_1.png", "p_1006_1.png",]  

results = []
for img in image_list:
    response = rekognition_client.detect_labels(
        Image={"S3Object":{"Bucket":bucket_name,"Name":prefix_name+img}}, 
        MaxLabels=10, 
        MinConfidence=70
        )
    results.append({"image": img, "labels": response.get("Labels", [])})


pathlib.Path("image_labels.json").write_text( json.dumps(results, indent=2) )

