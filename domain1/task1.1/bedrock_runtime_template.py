import json
import boto3
from botocore.exceptions import ClientError


class BedrockRuntimeInvokeManager:
    """
    Standardized wrapper for invoking Amazon Bedrock Foundation Models.

    This class centralizes model invocation logic, including payload handling,
    optional guardrail configuration, and response parsing.
    """

    def __init__(self, region_name):
        """
        Initialize the Bedrock Runtime client.

        :param region_name: AWS region where Bedrock is available
        """
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name
        )

    def invoke(
        self,
        model_id,
        payload,
        guardrail_id= None,
        guardrail_version= None
    ):
        """
        Invoke a Bedrock Foundation Model.

        :param model_id: Bedrock model ID or ARN
        :param payload: Model-specific request payload
        :param guardrail_id: Optional Bedrock Guardrail ID
        :param guardrail_version: Optional Bedrock Guardrail version
        :return: Parsed JSON response body
        """
        try:
            kwargs = {
                "modelId": model_id,
                "body": json.dumps(payload),
                "contentType": "application/json",
                "accept": "application/json"
            }

            if guardrail_id and guardrail_version:
                kwargs["guardrailIdentifier"] = guardrail_id
                kwargs["guardrailVersion"] = guardrail_version

            response = self.client.invoke_model(**kwargs)
            return response

        except ClientError as e:
            error = e.response.get("Error", {})
            raise RuntimeError(
                f"Bedrock invocation failed: {error.get('Code')} - {error.get('Message')}"
            ) from e
