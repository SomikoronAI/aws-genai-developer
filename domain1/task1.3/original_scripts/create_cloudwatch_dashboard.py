# Assignment Part 1
# Create a CloudWatch Dashboard for monitoring data quality


import boto3


# Create CloudWatch dashboard
cloudwatch_client = boto3.client('cloudwatch')
cloudwatch_client = boto3.client(service_name='cloudwatch', region_name=REGION_NAME)


dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["CustomerFeedback/TextQuality", "QualityScore", "Source", "TextReviews"]
                ],
                "period": 86400,
                "stat": "Average",
                "region": "us-east-1",
                "title": "Text Review Quality Score"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["CustomerFeedback/DataQuality", "RulesetPassRate", "Ruleset", "customer_reviews_ruleset"]
                ],
                "period": 86400,
                "stat": "Average",
                "region": "us-east-1",
                "title": "Glue Data Quality Pass Rate"
            }
        }
    ]
}

response = cloudwatch_client.put_dashboard(
    DashboardName='CustomerFeedbackQuality',
    DashboardBody=json.dumps(dashboard_body)
)

print(f"Created dashboard: {response['DashboardArn']}")