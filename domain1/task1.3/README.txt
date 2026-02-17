Task 1.3: Implement data validation and processing pipelines for FM consumption

Bonus Assignment:
Build a comprehensive data validation and processing pipeline for analyzing customer feedback data from 
multiple sources (text reviews, product images, customer service call recordings, and survey responses). 

The pipeline will prepare this diverse data for consumption by foundation models to generate actionable 
business insights.

** Project Architecture and Components **

Part 1: Data validation workflow
Set up AWS Glue Data Quality for validating structured customer feedback data
Create Lambda functions for custom validation of unstructured text reviews
Implement CloudWatch metrics to monitor data quality over time

Part 2. Multimodal data processing
Process text reviews using Amazon Comprehend for entity extraction and sentiment analysis
Extract text from product images using Amazon Textract
Transcribe customer service calls using Amazon Transcribe
Transform tabular survey data into natural language summaries

Part 3. Data formatting for FMs
Format processed data for Anthropic Claude in Amazon Bedrock
Create conversation templates for dialog-based analysis
Implement multimodal request formatting for image and text analysis

Part 4. Data quality enhancement
Use Amazon Comprehend to extract key entities and themes
Implement text normalization with Lambda functions
Create a feedback loop to improve data quality based on model responses

