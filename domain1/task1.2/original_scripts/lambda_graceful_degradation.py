# Assignment Part 3

import json


def lambda_handler(event, context):
    """Graceful degradation handler that returns a predefined response."""

    prompt = event.get('prompt', '')
    use_case = event.get('use_case', 'general')
    
    # Provide a graceful response based on the use case
    responses = {
        "general": "I'm sorry, but I'm currently experiencing technical difficulties. Please try again later or contact customer service for immediate assistance.",
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
