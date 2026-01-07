Task 1.1: Analyze Requirements and Design GenAI Solutions

Bonus Assignment : Build a proof-of-concept document processing solution that extracts information from insurance claim documents and generates summaries using Amazon Bedrock.

Scenario : An insurance company wants to automate processing of claim documents to reduce manual effort and improve consistency.

Step 1. Design the architecture (Skill 1.1.1)
Create a simple architecture diagram showing the following:
Document storage (Amazon S3)
Processing workflow
Foundation model integration
Response generation
Select appropriate Amazon Bedrock models for the following:
Document understanding
Information extraction
Summary generation


Step 2. Implement proof-of-Concept (Skill 1.1.2)
Set up AWS environment:
Create a S3 bucket
You can use Python SDK to create a S3 bucket.
You can also use AWS CLI to create a bucket. For example : aws s3 mb s3://claim-documents-poc-nururrahman

Create a Python application with the following:
Document upload functionality
Amazon Bedrock integration
Simple RAG component using policy information
Claim summary generation


Step 3. Create reusable components (Skill 1.1.3)
Develop standardized for the following:
Prompt template manager
Model invoker
Basic content validator


Step 4. Test and evaluate
Test with 2-3 sample documents
Compare performance of different models
Document findings and recommendations

Test the solution based on:
Create sample claim documents (or use public datasets)
Upload to your S3 bucket
Run the processor on different document types
Compare results from different models
Document the findings

Evaluate the solution based on:
Accuracy of information extraction
Quality of generated summaries
Processing time and efficiency
Code organization and reusability


Extra challenging steps:
Add a simple web interface using Flask
Implement a knowledge base with insurance policy information
Add content filtering for sensitive information
Create a simple feedback mechanism

