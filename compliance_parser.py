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


def format_compliance_context(pdf_text: str) -> str:
    """Format compliance information for LLM prompt"""
    return f"""
COMPANY COMPLIANCE RULES:
{pdf_text}
"""
