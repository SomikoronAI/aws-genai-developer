"""
Lightweight token-extraction utility for multiple FM providers.

Inspects the model_id to determine the provider (Claude, Nova, Llama, Mistral)
and extracts (input_tokens, output_tokens, total_tokens) from the model's
response_body using provider - specific field patterns. Falls back to
(None, None, None) when the provider is unknown.
"""

class TokenExtractorTemplateManager:
    def get_tokens(self, model_id, prompt, response_body):
        # Claude
        if "claude" in model_id:
            input_tokens  = response_body["usage"].get("input_tokens", None)
            output_tokens = response_body["usage"].get("output_tokens", None) 
            total_tokens  = response_body["usage"].get("total_tokens", None)
            return input_tokens, output_tokens, total_tokens
        # Nova
        elif "nova" in model_id:
            input_tokens  = response_body["usage"].get("inputTokens", None)
            output_tokens = response_body["usage"].get("outputTokens", None) 
            total_tokens  = response_body["usage"].get("totalTokens", None)
            return input_tokens, output_tokens, total_tokens
        # Llama
        elif "llama" in model_id:
            input_tokens  = response_body.get("prompt_token_count", None)
            output_tokens = response_body.get("generation_token_count", None) 
            total_tokens  = response_body.get("total_tokens", None)
            return input_tokens, output_tokens, total_tokens
        # Mistral
        elif "mistral" in model_id: 
            text = response_body['outputs'][0]['text'].strip()
            input_tokens  = len(prompt.split())
            output_tokens = len(text.split())
            total_tokens  = None
            return input_tokens, output_tokens, total_tokens
        else:
            # fallback generic
            return None,None,None