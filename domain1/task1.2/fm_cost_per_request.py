# Assignment Part 1
"""
Calculate estimated cost of a single Bedrock model request.

Parameters:
- prompt (str): The input question or text sent to the model.
- response (str): The model's output text.
- model_rates (dict): Dictionary containing input/output rates per model.
Example:
{
"model_id_1": {"input": 0.0001, "output": 0.002},
"model_id_2": {"input": 0.0002, "output": 0.004}
}
- model_id (str): The model identifier used for this request.

Returns:
- float: Estimated cost in USD.

Notes:
- Rates are assumed per 1,000 characters of input/output text.
- Rates vary by the AWS regions and tier types.
- Current price dictionary is valid for us-east-1 and standard tier.
- Created on Jan 10, 2026.
- Source: https://aws.amazon.com/bedrock/pricing/
"""


model_rates = {
    "us.anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015}, 
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 0.003, "output": 0.015}, 
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {"input": 0.003, "output": 0.0015}, 
    "us.amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014}, 
    "us.amazon.nova-2-lite-v1:0": {"input": 0.00033, "output": 0.00275},
    "us.amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032}, 
    "meta.llama3-8b-instruct-v1:0": {"input": 0.00022, "output": 0.00022}, 
    "mistral.mistral-small-2402-v1:0": {"input": 0.00050, "output": 0.0015}
    }


def calculate_cost(input_tokens, output_tokens, model_rates, model_id) :
    input_rate = model_rates.get(model_id,{}).get("input", 0.0)
    output_rate = model_rates.get(model_id,{}).get("output", 0.0)
 
    input_cost  = (input_tokens / 1000) * input_rate
    output_cost = (output_tokens / 1000) * output_rate

    return input_cost + output_cost




if __name__ == "__main__":
    prompt = "What is a 401(k) retirement plan?"
    response = "A 401(k) is a tax-advantaged retirement savings plan offered by employers."
    input_chars  = len(prompt)
    output_chars = len(response)

    model_id = "us.amazon.nova-micro-v1:0"

    cost = calculate_cost(input_chars, output_chars, model_rates, model_id)
    print(f"Estimated cost: ${cost:.6f}")