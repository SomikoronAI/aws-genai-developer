# Assignment Part 1

import pandas as pd 
import json 

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


# Generate strategy
result_file = "model_evaluation_results.csv"
results_df = pd.read_csv(result_file)

strategy = create_model_selection_strategy(results_df)
print(json.dumps(strategy, indent=2))

# Save strategy to file for AppConfig
with open("model_selection_strategy.json", "w") as f:
    json.dump(strategy, f, indent=2)
