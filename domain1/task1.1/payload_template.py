from typing import List, Optional, Dict, Any, Tuple

class PayloadTemplateManager:
    """    
    Provides structured payload templates for invoking Amazon Bedrock models using
    'invoke_model' api.

    With the 'bedrock-runtime' client, there are two main options:

    - 'invoke_model': Direct invocation using the model's native schema (e.g., 
    Anthropic Claude, Amazon Nova, OpenAI GPT-4, etc.).
    - 'converse': A unified API that standardizes interactions across models and 
    supports features like guardrails, multi-turn context, and streaming.
    
    This manager supports building request bodies for different Bedrock API styles,
    such as the 'messages' API (Claude 3/4) and legacy text-completion API (Claude v2).
    It centralizes template definitions and allows dynamic substitution of parameters
    like prompt text, temperature, and token limits.
    """

    def get_payload_claude(
            self, 
            template_name: str, 
            user_prompt: str, 
            system_prompt: Optional[str] = None, 
            temperature: float =0.2, 
            top_p: float=0.9, 
            top_k: int =None, 
            max_tokens: int =512, 
            max_tokens_to_sample: int=None
        ) -> Dict[str, Any]:
        """Build a payload for invoking Anthopic Calude models using invoke_model."""

        if template_name=="messages_api": 
        # Bedrock messages API
            payload: Dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": user_prompt}]
                    }
                ]
            }
            if system_prompt==None:
                system_prompt="""
                You are Claude, an AI assistant created by Anthropic to be helpful, harmless, 
                and honest. Your goal is to provide informative and substantive response to 
                queries while avoiding potential harms.""".replace('\n', ' ').strip()
                payload["system"] = system_prompt
            else:
                payload["system"] = system_prompt
            

        if template_name=="text_completion_api":
        # Bedrock text completion API to use with Claude v2, Claude v2.1
            payload: Dict[str, Any] = {
                "prompt": f"\n\nHuman:{user_prompt}\n\nAssistant:", 
                "temperature": temperature, 
                "top_p": top_p, 
                "top_k": top_k, 
                "max_tokens_to_sample": max_tokens_to_sample, 
                "stop_sequences": ["\n\nHuman:"]
                }
            
        return payload


    def get_payload_nova(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
        """ Build a payload for invoking Amazon Nova models using invoke_model."""

        payload: Dict[str, Any] = {
            "schemaVersion": "messages-v1",
            "messages": [],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        }

        # User message
        payload["messages"].append({
            "role": "user",
            "content": [{"text": user_prompt}]
        })
        if system_prompt==None:
            system_prompt="""
            You are Nova, an AI assistant created by Amazon to be helpful, harmless, 
            and honest. Your goal is to provide informative and substantive response to 
            queries while avoiding potential harms.""".replace('\n', ' ').strip()
            payload["system"] = [{"text": system_prompt}]
        else:
            payload["system"] = [{"text": system_prompt}]

        if stop_sequences:
            payload["generationConfig"]["stopSequences"] = stop_sequences

        return payload



    def build_payload(
        self,
        model_id: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
        guardrail_id: Optional[str] = None,
        guardrail_version: Optional[str] = None,
        enable_guardrail_trace: bool = False,
        ) -> Dict[str, Any]:
        """Construct a valid Converse payload for any Bedrock model."""

        payload: Dict[str, Any] = {
            "modelId": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        }
        # System prompt
        if system_prompt:
            payload["system"] = [{"text": system_prompt}]

        if stop_sequences:
            payload["inferenceConfig"]["stopSequences"] = stop_sequences

        # Guardrail configuration
        if guardrail_id and guardrail_version:
            payload["guardrailConfig"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version,
            }
            if enable_guardrail_trace:
                payload["guardrailConfig"]["trace"] = "enabled"

        return payload

 