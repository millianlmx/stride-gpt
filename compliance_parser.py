import PyPDF2
from typing import List, Dict

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

def extract_components_from_text(pdf_text: str) -> List[str]:
    """
    Extract trusted components from PDF text using common indicators.
    Returns a list of identified components.
    """
    components = []
    
    # Split text into lines for analysis
    lines = pdf_text.split('\n')
    
    # Keywords that might indicate approved components
    indicators = [
        'approved component',
        'trusted component',
        'verified component',
        'approved software',
        'authorized component',
        'certified component'
    ]
    
    for line in lines:
        line = line.lower()
        # Check if line contains any indicators
        if any(indicator in line for indicator in indicators):
            # Extract the component name - typically follows the indicator
            for indicator in indicators:
                if indicator in line:
                    # Get text after the indicator, clean it up
                    component = line.split(indicator)[1].strip(' :.,-')
                    if component:
                        components.append(component)
                        break

    return list(set(components))  # Remove duplicates

def format_compliance_context(pdf_text: str) -> str:
    """Format compliance information for LLM prompt"""
    # Automatically extract components
    components = extract_components_from_text(pdf_text)
    
    components_text = "No specific trusted components identified." if not components else \
                     "IDENTIFIED TRUSTED COMPONENTS:\n" + "\n".join(f"- {comp}" for comp in components)
    
    return f"""
COMPANY COMPLIANCE RULES AND COMPONENTS:
{pdf_text}

{components_text}

Note: Any identified components above are extracted from company documentation. 
The LLM should analyze the full context to determine security implications and trust levels.
"""
