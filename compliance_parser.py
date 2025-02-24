import PyPDF2
from typing import List, Dict

def extract_text_from_pdfs(pdf_files) -> str:
    """Extract text from multiple PDF files"""
    combined_text = ""
    for pdf_file in pdf_files:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            combined_text += page.extract_text() + "\n"
    return combined_text

def format_compliance_context(pdf_text: str, used_components: List[str]) -> str:
    """Format compliance information for LLM prompt"""
    return f"""
COMPANY COMPLIANCE RULES AND TRUSTED COMPONENTS:
{pdf_text}

TRUSTED COMPONENTS USED IN THIS APPLICATION:
{', '.join(used_components)}

Note: The above components are officially approved and thoroughly tested company components. 
Their security controls are verified and should be considered trusted. Do not generate threats 
or attack vectors for vulnerabilities that these components are designed to prevent.
"""
