# Assignment Part 1

# Create a Glue Data Quality ruleset for customer reviews:


import boto3
from awsglue.data_quality import DataQualityRule, DataQualityRulesetEvaluator


# Define rules for customer reviews
rules = [
    # Check for completeness of required fields
    DataQualityRule.is_complete("review_text"),
    DataQualityRule.is_complete("product_id"),
    DataQualityRule.is_complete("customer_id"),
    
    # Check for valid values
    DataQualityRule.column_values_match_pattern(
    "review_text", ".{10,}"
    ),  # At least 10 chars
    
    DataQualityRule.column_values_match_pattern(
    "rating", "^[1-5]$"
    ),  # Rating 1-5
    
    # Check for data consistency
    DataQualityRule.column_values_match_pattern(
    "review_date", "\\d{4}-\\d{2}-\\d{2}"
    ),  # YYYY-MM-DD
    
    # Check for statistical properties
    DataQualityRule.column_length_distribution_match(
        "review_text", 
        min_length=10, 
        max_length=5000
    )
]

# Create ruleset
glue_client = boto3.client(service_name='glue', region_name=region_name)
response = glue_client.create_data_quality_ruleset(
    Name='customer_reviews_ruleset',
    Description='Data quality rules for customer reviews',
    Ruleset='\n'.join([str(rule) for rule in rules]),
    Tags={'Project': 'CustomerFeedbackAnalysis'}
)

print(f"Created ruleset: {response['Name']}")
