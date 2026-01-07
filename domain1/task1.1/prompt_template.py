
class PromptTemplateManager:
    """
    Manages prompt templates for generating AI instructions.

    This class provides predefined templates for tasks such as:
    - Extracting key information from insurance claim documents.
    - Generating concise summaries based on extracted data.

    Methods:
    get_prompt(template_name, **kwargs): 
    Returns a formatted prompt by substituting placeholders in the specified template.
    """
        
    def __init__(self):
        self.templates = {
            "extract_info": """
            Extract the following information from this insurance claim document:
            - Claimant Name
            - Patient Name if different than Claimant Name
            - Name of the person who signs the claim form 
            - Policy Number
            - Claim Amount
            - Incident Date
            - Incident Description
            
            Document:
            {document_text}
            
            Return the information in JSON format.
            """,
            
            "generate_summary": """
            Based on this extracted information:
            {extracted_text}
            
            Generate a concise summary of the claim.
            """
        }

    def get_prompt(self, template_name, **kwargs):
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        return template.format(**kwargs)
