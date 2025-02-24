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
Focus on key security requirements, controls, and trust boundaries.

COMPLIANCE DOCUMENTATION:
{pdf_text}

Provide a concise summary that highlights:
1. Key security requirements
2. Important controls and safeguards
3. Critical compliance points
4. Trust boundaries and security zones
5. Data protection requirements

Format the summary in clear sections with bullet points for readability.
"""

    try:
        if model_provider == "OpenAI API":
            client = OpenAI(api_key=kwargs.get('openai_api_key'))
            response = client.chat.completions.create(
                model=kwargs.get('selected_model'),
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content
            
        elif model_provider == "Google AI API":
            genai.configure(api_key=kwargs.get('google_api_key'))
            model = genai.GenerativeModel(kwargs.get('google_model'))
            response = model.generate_content(summary_prompt)
            return response.text
            
        elif model_provider == "Vertex AI API":
            from vertex_ai import get_vertex_response
            return get_vertex_response(
                kwargs.get('vertex_project_id'),
                kwargs.get('vertex_model'),
                kwargs.get('vertex_location'),
                summary_prompt
            )
            
    except Exception as e:
        print(f"Error generating compliance summary: {str(e)}")
        return "Error generating summary. Please check the logs for details."

def format_compliance_context(pdf_text: str) -> str:
    """Format compliance information for LLM prompt"""
    return f"""
COMPANY COMPLIANCE RULES:
{pdf_text}

Note: Any identified things above are extracted from company documentation.
The LLM should analyze the full context to determine security implications and trust levels.
"""
