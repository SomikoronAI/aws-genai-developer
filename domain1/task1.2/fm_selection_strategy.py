# Assignment Part 1

"""
Generates a model selection strategy for multiple use cases by processing
evaluation result CSVs, computing weighted scores (similarity and latency),
selecting primary and fallback models, and exporting the final strategy as 
a JSON file for AppConfig consumption.
"""

import os
import json 
import pandas as pd 


def create_model_selection_strategy(results_df):
    """Create a model selection strategy based on evaluation results."""
    
    # Calculate overall scores
    model_scores = results_df.groupby("model_id").agg({
        "latency": "mean",
        "similarity_score": "mean"
    }).reset_index()
    
    # Normalize scores (lower latency is better, higher similarity is better)
    max_latency = model_scores["latency"].max()
    model_scores["latency_score"] = 1 - (model_scores["latency"] / max_latency)
    
    # Calculate weighted score (adjust weights based on priorities)
    model_scores["overall_score"] = (
        0.7 * model_scores["similarity_score"] + 
        0.3 * model_scores["latency_score"]
    )
    
    # Sort by overall score
    model_scores = model_scores.sort_values("overall_score", ascending=False)
    
    # Create strategy
    strategy = {
        "primary_model": model_scores.iloc[0]["model_id"],
        "fallback_models": model_scores.iloc[1:]["model_id"].tolist(),
        "model_scores": model_scores.to_dict(orient="records")
    }
    
    return strategy


def main():
    strategies = {"use_cases": {}}
    
    file_to_use_case = {
        "finance": "results_fm_finance_evaluation.csv",
        "general": "results_fm_general_evaluation.csv",
        "default": "none"
    }


    base_dir = os.path.dirname(os.path.abspath("__file__"))

    for use_case, result_file in file_to_use_case.items():
        if use_case=="default":
            selected={"primary_model":"us.amazon.nova-micro-v1:0"}
            strategies["use_cases"].update({use_case: selected})
        else:
            result_path = os.path.join(base_dir, "data", result_file)
            results_df  = pd.read_csv(result_path)
            strategy    = create_model_selection_strategy(results_df)
            strategies["use_cases"][use_case] = strategy

    # Validate required use cases
    required_use_cases = {"finance", "general", "default"}
    missing = required_use_cases - strategies["use_cases"].keys()

    if missing:
        raise ValueError(f"Missing strategies for use cases: {missing}")

    print(json.dumps(strategies, indent=2))

    # Save strategy to file for AppConfig
    output_file = "fm_selection_strategy.json"
    output_path = os.path.join(base_dir, "data", output_file)

    with open(output_path, "w") as f:
        json.dump(strategies, f, indent=2)



if __name__ == "__main__":
    main()