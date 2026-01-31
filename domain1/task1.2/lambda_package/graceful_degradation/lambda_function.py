# Assignment Part 3

"""
AWS Lambda handler that performs graceful degradation. 
Returns a predefined fallback message based on the use case. 
Ensures a consistent response when normal processing is unavailable.
"""
    
import json


def lambda_handler(event, context):
    """Graceful degradation handler that returns a predefined response."""

    # Normalize input across API Gateway and Step Functions
    if "body" in event and isinstance(event["body"], str):
        body = json.loads(event["body"])
    else:
        body = event

    prompt   = body.get("prompt", " ")
    use_case = body.get("use_case", "general")
    context  = context 


    # Provide a graceful response based on the use case
    responses = {
        "general": "I'm sorry, but I'm currently experiencing technical difficulties. Please try again later or contact customer service at 1-800-555-1234 for immediate assistance.",
        "financial": "I'm unable to access financial information or process financial requests at this time. For immediate assistance, please contact our billing department at 1-800-555-1234.", 
        "product_question": "I apologize, but I can't access product information right now. Please refer to our product documentation or contact customer service at 1-800-555-1234.",
        "account_inquiry": "I'm unable to process account inquiries at the moment. For urgent matters, please call our customer service line at 1-800-555-1234."
    }
    
    default_response = "I'm sorry, but I'm currently experiencing technical difficulties. Please try again later."
    response_text = responses.get(use_case, default_response)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'model_used': "DEGRADED_SERVICE",
            'response': response_text
        })
    }
