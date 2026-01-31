# Assignment Part 1

import boto3
import json
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# Configuration parameters 
region_name = "us-east-1"

# Initialize Bedrock client
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)


# Models to evaluate
models = [
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-instant-v1",
    "amazon.titan-text-express-v1"
]

# Test cases with ground truth answers
test_cases = [
    {
        "question": "What is a 401(k) retirement plan?",
        "context": "Financial services",
        "ground_truth": "A 401(k) is a tax-advantaged retirement savings plan offered by employers."
    },
    # Add more test cases...
]

def invoke_model(model_id, prompt, max_tokens=500):
    """Invoke a model with the given prompt and return the response and metrics."""
    start_time = time.time()
    
    # Prepare request body based on model provider
    if "anthropic" in model_id:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        })
    elif "amazon" in model_id:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": 0.7,
                "topP": 0.9
            }
        })
    # Add more model providers as needed
    
    try:
        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read().decode())
        
        if "anthropic" in model_id:
            output = response_body['content'][0]['text']
        elif "amazon" in model_id:
            output = response_body['results'][0]['outputText']
        
        # Calculate metrics
        latency = time.time() - start_time
        token_count = len(output.split())  # Rough estimate
        
        return {
            "success": True,
            "output": output,
            "latency": latency,
            "token_count": token_count
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.time() - start_time
        }

def evaluate_models():
    """Evaluate all models on all test cases and return results."""
    results = []
    
    for test_case in test_cases:
        prompt = f"Question: {test_case['question']}\nContext: {test_case['context']}"
        
        for model_id in models:
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
                    "token_count": response["token_count"],
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


def calculate_similarity(output, ground_truth):
    """Calculate similarity between model output and ground truth (simplified)."""
    # In a real implementation, use more sophisticated NLP techniques
    # This is a very simplified version
    output_words = set(output.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    if not truth_words:
        return 0.0
        
    common_words = output_words.intersection(truth_words)
    return len(common_words) / len(truth_words)


# Run evaluation
def main():
    # Evaluate models
    results_df = evaluate_models()
    
    # Save results to CSV
    results_df.to_csv("model_evaluation_results.csv", index=False)
    
    # Print summary
    print("\nEvaluation Summary:")
    summary = results_df.groupby("model_id").agg({
        "latency": "mean",
        "similarity_score": "mean",
        "token_count": "mean"
    }).reset_index()
    
    print(summary)



if __name__ == "__main__":
    main()