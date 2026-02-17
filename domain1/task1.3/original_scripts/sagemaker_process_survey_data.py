# Assignement Part 2
# SageMaker processing script for survey data


import pandas as pd
import numpy as np
import argparse
import os
import json


def process_survey_data(input_path, output_path):
    # Read the survey data
    df = pd.read_csv(f"{input_path}/surveys.csv")
    
    # Basic data cleaning
    # Drop rows with missing key fields
    df = df.dropna(subset=['customer_id', 'survey_date'])  
    
    # Convert categorical ratings to numerical
    rating_map = {'Very Dissatisfied': 1, 'Dissatisfied': 2, 'Neutral': 3, 'Satisfied': 4, 'Very Satisfied': 5}
    for col in df.columns:
        if 'rating' in col.lower() or 'satisfaction' in col.lower():
            df[col] = df[col].map(rating_map).fillna(df[col])
    
    # Calculate summary statistics
    summary_stats = {
        'total_surveys': len(df),
        'avg_satisfaction': df['overall_satisfaction'].mean(),
        'sentiment_distribution': df['overall_satisfaction'].value_counts().to_dict(),
        'top_issues': df['improvement_area'].value_counts().head(3).to_dict()
    }
    
    # Generate natural language summaries for each survey
    summaries = []
    for _, row in df.iterrows():
        summary = {
            'customer_id': row['customer_id'],
            'survey_date': row['survey_date'],
            'summary_text': generate_summary(row),
            'ratings': {col: row[col] for col in df.columns if 'rating' in col.lower() or 'satisfaction' in col.lower()},
            'comments': row.get('comments', '')
        }
        summaries.append(summary)
    
    # Save the processed data
    with open(f"{output_path}/survey_summaries.json", 'w') as f:
        json.dump(summaries, f)
    
    with open(f"{output_path}/survey_statistics.json", 'w') as f:
        json.dump(summary_stats, f)


def generate_summary(row):
    """Generate a natural language summary of a survey response"""
    satisfaction_level = "satisfied" if row['overall_satisfaction'] >= 4 else \
                        "neutral" if row['overall_satisfaction'] == 3 else "dissatisfied"
    
    summary = f"Customer {row['customer_id']} was {satisfaction_level} overall with their experience. "
    
    # Add details about specific ratings
    if 'product_rating' in row:
        summary += f"They rated the product {row['product_rating']}/5. "
    
    if 'service_rating' in row:
        summary += f"They rated the customer service {row['service_rating']}/5. "
    
    # Add improvement area if available
    if 'improvement_area' in row and pd.notna(row['improvement_area']):
        summary += f"They suggested improvements in the area of {row['improvement_area']}. "
    
    # Add comments if available
    if 'comments' in row and pd.notna(row['comments']) and len(str(row['comments'])) > 0:
        summary += f"Their comments: '{row['comments']}'"
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",  type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
    args = parser.parse_args()
    
    process_survey_data(args.input_path, args.output_path)
