import PyPDF2
from typing import List, Dict
from openai import OpenAI
import google.generativeai as genai

def extract_text_from_pdfs(pdf_files) -> str:
    """Extract text from multiple PDF files"""
    combined_text = ""
    try:
        for pdf_file in pdf_files:
            # Verify the file is not empty
            if pdf_file.size == 0:
                raise ValueError(f"File {pdf_file.name} is empty")
                
            # Reset file pointer to beginning
            pdf_file.seek(0)
            
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                if len(pdf_reader.pages) == 0:
                    raise ValueError(f"No pages found in {pdf_file.name}")
                    
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        combined_text += text + "\n"
                    
            except PyPDF2.PdfReadError as e:
                raise ValueError(f"Error reading PDF {pdf_file.name}: {str(e)}")
                
        if not combined_text.strip():
            raise ValueError("No text could be extracted from the PDF files")
            
        return combined_text
        
    except Exception as e:
        # Log the error for debugging
        print(f"PDF parsing error: {str(e)}")
        raise ValueError("Failed to parse PDF file(s). Please ensure valid PDF files are uploaded.") from e


def get_compliance_summary(pdf_text: str, model_provider: str, **kwargs) -> str:
    """Get an enhanced summary of compliance documentation using the selected LLM."""
    summary_prompt = f"""
Analyze the following compliance documentation and provide a clear, structured summary.
Focus on extracting and organizing compliance requirements by their codes and measures.

COMPLIANCE DOCUMENTATION:
{pdf_text}

Provide a comprehensive analysis that:

1. COMPLIANCE CODES AND MEASURES:
   - Extract and list all compliance codes (e.g., XX.1.2.3) with their associated measures
   - Group related measures together
   - Format each entry as "CODE: Measure Description"
   - Maintain the exact codes as they appear in the documentation

2. KEY SECURITY REQUIREMENTS:
   - List the main security requirements
   - Reference the specific compliance codes that apply
   - Highlight critical security controls

3. CONTROL CATEGORIES:
   - Group controls by their primary security function
   - Include the relevant compliance codes for each category
   - Explain the purpose of each control category

4. TRUST BOUNDARIES:
   - Identify system trust boundaries
   - List applicable security zones
   - Map compliance requirements to trust boundaries

5. DATA PROTECTION REQUIREMENTS:
   - Detail data security requirements
   - Specify data classification levels
   - List data handling controls with their compliance codes

Format the output in clear sections with:
- Hierarchical organization
- Bullet points for readability
- Clear references to compliance codes
- Explicit connections between requirements and controls

IMPORTANT:
- Only include codes and measures that explicitly exist in the documentation
- Maintain exact code formatting (XX.1.2.3)
- Do not invent or assume requirements
- Highlight any critical or high-priority requirements
"""

    try:
        if model_provider == "OpenAI API":
            client = OpenAI(api_key=kwargs.get('openai_api_key'))
            response = client.chat.completions.create(
                model=kwargs.get('selected_model'),
                messages=[
                    {"role": "system", "content": "You are a compliance analysis expert. Provide detailed, structured analysis of compliance documentation with precise reference to compliance codes."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        elif model_provider == "Google AI API":
            genai.configure(api_key=kwargs.get('google_api_key'))
            model = genai.GenerativeModel(kwargs.get('google_model'))
            # Modified this part to use the correct format
            response = model.generate_content(summary_prompt)
            return response.text
            
        elif model_provider == "Vertex AI API":
            from vertex_ai import get_vertex_response
            if not all([kwargs.get('vertex_project_id'), kwargs.get('vertex_model'), kwargs.get('vertex_location')]):
                return "Error: Missing required Vertex AI parameters. Please ensure Project ID, Model, and Location are provided."
            
            # Add system message for Vertex AI
            enhanced_prompt = """You are a compliance analysis expert. Provide detailed, structured analysis of compliance documentation with precise reference to compliance codes.

""" + summary_prompt
            
            return get_vertex_response(
                project_id=kwargs.get('vertex_project_id'),
                model_name=kwargs.get('vertex_model'),
                location=kwargs.get('vertex_location'),
                prompt=enhanced_prompt
            )
            
    except Exception as e:
        print(f"Error generating compliance summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def get_compliance_titles(pdf_text: str, model_provider: str, **kwargs) -> List[str]:
    """Extract compliance requirement titles and IDs from the documentation using LLM."""
    titles_prompt = f"""
Analyze the following compliance documentation and extract all requirement titles and their IDs.
Focus on finding patterns like 'XX.1.2.3' followed by their titles.

COMPLIANCE DOCUMENTATION:
{pdf_text}

Extract and list all requirement IDs and their titles in this format:
ID: Title
Example:
AA.1.2.3: Password Complexity Requirements
BB.3.4.5: Access Control Mechanisms

Only include requirements that are explicitly present in the documentation.
Do not invent or assume any requirements.
"""

    try:
        if model_provider == "OpenAI API":
            client = OpenAI(api_key=kwargs.get('openai_api_key'))
            response = client.chat.completions.create(
                model=kwargs.get('selected_model'),
                messages=[{"role": "user", "content": titles_prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        elif model_provider == "Google AI API":
            genai.configure(api_key=kwargs.get('google_api_key'))
            model = genai.GenerativeModel(kwargs.get('google_model'))
            response = model.generate_content(titles_prompt)
            return response.text
            
        elif model_provider == "Vertex AI API":
            from vertex_ai import get_vertex_response
            if not all([kwargs.get('vertex_project_id'), kwargs.get('vertex_model'), kwargs.get('vertex_location')]):
                return "Error: Missing required Vertex AI parameters."
            return get_vertex_response(
                project_id=kwargs.get('vertex_project_id'),
                model_name=kwargs.get('vertex_model'),
                location=kwargs.get('vertex_location'),
                prompt=titles_prompt
            )
            
    except Exception as e:
        print(f"Error extracting compliance titles: {str(e)}")
        return f"Error extracting titles: {str(e)}"

def format_compliance_context(pdf_text: str, model_provider: str = None, **kwargs) -> str:
    """Format compliance information for LLM prompt"""
    
    # Get compliance titles if model provider is specified
    titles = ""
    if model_provider:
        titles = get_compliance_titles(pdf_text, model_provider, **kwargs)
        if titles and not titles.startswith("Error"):
            titles = "\nCOMPLIANCE REQUIREMENTS INDEX:\n" + titles + "\n"
    
    return f"""
COMPANY COMPLIANCE RULES:
{pdf_text}
{titles}
Note: Any identified things above are extracted from company documentation.
The LLM should analyze the full context to determine security implications and trust levels.
"""
