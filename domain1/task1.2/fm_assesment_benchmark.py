# Assignment Part 1

import os
import json
import time
import pandas as pd
import boto3
from concurrent.futures import ThreadPoolExecutor

from payload_template import PayloadTemplateManager
from tokenextract_template import TokenExtractorTemplateManager

import dotenv
dotenv.load_dotenv(".env")

# Configuration parameters 
region_name= os.environ["REGION_NAME"]
ptm_payload = PayloadTemplateManager()
ttm_token = TokenExtractorTemplateManager()

# Initialize client
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)


# Models to evaluate
model_list = [
    "us.anthropic.claude-3-sonnet-20240229-v1:0", 
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
    "us.anthropic.claude-sonnet-4-20250514-v1:0", 
    # "us.amazon.nova-micro-v1:0", 
    "us.amazon.nova-2-lite-v1:0",
    "us.amazon.nova-pro-v1:0", 
    "meta.llama3-8b-instruct-v1:0",
    "mistral.mistral-small-2402-v1:0"
]


def get_test_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f: 
        data_list = json.loads(f.read())
    return data_list


def calculate_similarity(output, ground_truth):
    """Calculate similarity between model output and ground truth (simplified)."""
    # In a real implementation, use more sophisticated NLP techniques
    # This is a very simplified version
    output_words = set(output.lower().split())
    truth_words  = set(ground_truth.lower().split())
    
    if not truth_words:
        return 0.0
        
    common_words = output_words.intersection(truth_words)
    return len(common_words) / len(truth_words)


def invoke_model(model_id, prompt, max_tokens=512):
    """Invoke a model with the given prompt and return the response and metrics."""
    start_time = time.time()
    
    try:
        # Prepare request body based on model provider
        if "claude" in model_id:
            # Get payload
            payload = ptm_payload.get_payload_claude(
                "messages_api", 
                prompt, 
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload ), 
                ) 
            # Parse response
            response_body = json.loads(response['body'].read().decode())
            output = response_body['content'][0]['text']

        elif "nova" in model_id:
            # Get payload
            payload = ptm_payload.get_payload_nova(
                prompt, 
                temperature=0.7,
                top_p=0.9, 
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload ), 
                contentType="application/json", 
                accept="application/json"
                )
            # Parse response
            response_body = json.loads(response['body'].read().decode())
            output = response_body["output"]["message"]["content"][0]["text"]

        elif "llama" in model_id:
            payload = ptm_payload.get_payload_llama(
                prompt, 
                temperature=0.7,
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload ), 
                contentType="application/json", 
                accept="application/json"
                )
            # Parse response
            response_body = json.loads(response['body'].read().decode()) 
            output = response_body["generation"]

        elif "mistral" in model_id:
            payload = ptm_payload.get_payload_mistral(
                prompt, 
                temperature=0.7, 
                top_p=0.9,
                max_tokens=max_tokens
                )
            # Invoke model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps( payload )
                )
            # Parse response
            response_body = json.loads(response['body'].read().decode()) 
            output = response_body["outputs"][0]["text"]

        # Calculate metrics
        latency = time.time() - start_time
        # token_count = len(output.split())  # Rough estimate
        token_count = ttm_token.get_tokens(model_id, prompt, response_body)
        
        return {
            "success": True,
            "output": output,
            "latency": latency,
            "input_tokens": token_count[0],
            "output_tokens": token_count[1]
        }
    except Exception as e:
        print("Model Invocation Failed:", repr(e))
        return {
            "success": False,
            "output": False, 
            "latency": time.time() - start_time, 
            "error": str(e)
        }


def evaluate_models(data_path, n=None):
    """Evaluate all models on all or selected test cases and return results."""
    test_cases = get_test_data(data_path)
    if n: test_cases = test_cases[0:n]

    results = []
    for test_case in test_cases:
        prompt = f"Question: {test_case['question']}\nContext: {test_case['context']}"
        
        for model_id in model_list:
            print(f"Evaluating {model_id} on: {test_case['question']}")
            response = invoke_model(model_id, prompt)
            
            if response["success"]:
                # Calculate similarity score with ground truth (simplified)
                similarity = calculate_similarity(response["output"], test_case["ground_truth"])
                
                results.append({
                    "model_id": model_id,
                    "question": test_case["question"],
                    "output": response["output"],
                    "latency": response["latency"],
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                    "similarity_score": similarity
                })
            else:
                results.append({
                    "model_id": model_id,
                    "question": test_case["question"],
                    "error": response["error"],
                    "latency": response["latency"]
                })
    return pd.DataFrame(results)




def main():
    data_files = ["data_fm_finance_qa.json","data_fm_general_qa.json"]

    for data_file in data_files:
        base_dir = os.path.dirname(os.path.abspath("__file__"))
        data_path = os.path.join(base_dir, "data", data_file)

        # Evaluate models
        results_df = evaluate_models(data_path, n=25)
        print(results_df.shape)

        # Save results to CSV
        if "finance" in data_file:
            result_file = "results_fm_finance_evaluation.csv"
        elif "general" in data_file:
            result_file = "results_fm_general_evaluation.csv"
        
        result_path = os.path.join(base_dir, "data", result_file)
        results_df.to_csv(result_path, index=False)
    
        # Print summary
        print("\nEvaluation Summary:")
        summary = results_df.groupby("model_id").agg({
            "latency": "mean",
            "similarity_score": "mean",
            "input_tokens": "mean",
            "output_tokens": "mean"
        }).reset_index()
        
        print(summary)




if __name__ == "__main__":
    main()