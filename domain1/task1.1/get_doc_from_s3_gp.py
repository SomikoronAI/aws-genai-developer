"""
Retrieve and extract textual content from an object stored in Amazon S3.

This script performs the following tasks:
- Fetch the object (document) from S3
- Determine object (document) type
- if the obect is PDF, extracts text from the PDF
- if the obect is text, decodes the text safely

The function supports:
- Plain text files (e.g., .txt, .csv, .json, .log)
- Text-based PDF files (not scanned/image-only PDFs)

Requirements:
- AWS credentials with permissions for Bedrock and S3.
- Existing S3 bucket to store policy documents.
- Boto3 installed and configured.

Usage:
import boto3
from get_doc_from_s3_gp import get_document
"""

from PyPDF2 import PdfReader
from io import BytesIO
import boto3
from botocore.exceptions import ClientError


def get_document(s3_client, bucket_name, key_name, encoding="utf-8"):
    try:
        # Fetch the object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key_name)
        body = response["Body"].read()

        content_type = response.get("ContentType", "").lower()
        key_lower = key_name.lower()

        # Determine object type
        is_pdf = content_type == "application/pdf" or key_lower.endswith(".pdf")
        is_text_like = (
            content_type.startswith("text/")
            or content_type in {"application/json", "text/csv"}
            or key_lower.endswith((".txt", ".csv", ".json", ".log"))
        )

        if is_pdf:
            # Extract text from PDF
            try:
                reader = PdfReader(BytesIO(body))
            except Exception as e:
                raise RuntimeError(f"Failed to open PDF document: {key_name}") from e

            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

            return "\n".join(text_parts).strip()

        if is_text_like:
            # Decode text safely
            try:
                return body.decode( encoding )
            except UnicodeDecodeError:
                return body.decode("latin-1")

        # Unsupported type
        raise ValueError(
            f"Unsupported content type for text extraction: "
            f"{content_type or 'unknown'} (key: {key_name})"
        )

    except ClientError as e:
        error = e.response.get("Error", {})
        code = error.get("Code", "ClientError")
        message = error.get("Message", str(e))
        raise RuntimeError(f"S3 error {code}: {message}") from e

    except Exception as e:
        # Catch-all for unexpected parsing errors
        raise RuntimeError( 
            f"Failed to extract document content from {key_name}: {e}"
            ) from e



def main():
    region_name="us=east-1"
    s3_client = boto3.client(service_name="s3", region_name=region_name)

    # Upload file to S3
    bucket_name = "claim-documents-poc-nururrahman"       # S3 bucket name
    key_name = "claims/auto_insurance_claim1.txt"         # Desired object name in S3
    get_document(s3_client, bucket_name, key_name)


if __name__ == "__main__":
    main()
