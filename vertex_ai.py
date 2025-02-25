from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai
import json
from typing import Dict

def init_vertex_ai(project_id: str, location: str = "us-central1"):
    """Initialize Vertex AI client"""
    vertexai.init(project=project_id, location=location)

def get_vertex_response(
    project_id: str,
    model_name: str,
    location: str,
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    top_p: float = 0.8,
    top_k: int = 40,
) -> str:
    """Get response from Vertex AI model"""
    try:
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Initialize the model
        model = GenerativeModel(model_name)
        
        # Generate content
        response = model.generate_content(
            contents=prompt,  # Use 'contents' instead of just prompt
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        
        # Check if response has content
        if response and hasattr(response, 'text'):
            return response.text
        else:
            print("No text attribute in Vertex AI response")
            return ""
            
    except Exception as e:
        print(f"Error in Vertex AI response: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""  # Return empty string instead of raising

def get_image_analysis_vertex(project_id: str, model_name: str, location: str, prompt: str, base64_image: str) -> str:
    """Get image analysis from Vertex AI."""
    vertexai.init(project=project_id, location=location)
    
    try:
        # Initialize the model
        model = GenerativeModel(model_name)
        
        if "gemini" in model_name.lower():
            # For Gemini models
            response = model.generate_content(
                contents=[prompt, Part.from_data(data=base64_image, mime_type="image/jpeg")],
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                }
            )
            return response.text
        elif "claude" in model_name.lower():
            # For Claude models
            response = model.generate_content(
                contents=[Part.from_text(prompt), Part.from_data(data=base64_image, mime_type="image/jpeg")],
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                }
            )
            return response.text
        else:
            raise ValueError(f"Model {model_name} does not support image analysis")
            
    except Exception as e:
        print(f"Error in Vertex AI image analysis: {str(e)}")
        raise

def get_threat_model_vertex(project_id: str, model_name: str, location: str, prompt: str) -> Dict:
    """Get threat model from Vertex AI"""
    try:
        response = get_vertex_response(project_id, model_name, location, prompt)
        
        # Debug output
        print(f"Vertex AI response type: {type(response)}")
        print(f"Response preview: {response[:200] if response else 'Empty response'}")
        
        if not response:
            print("Empty response from Vertex AI")
            return {"threat_model": [], "improvement_suggestions": []}
            
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response if 'response' in locals() else 'No response'}")
        return {"threat_model": [], "improvement_suggestions": []}
    except Exception as e:
        print(f"Error in Vertex AI threat model generation: {str(e)}")
        return {"threat_model": [], "improvement_suggestions": []}

def get_attack_tree_vertex(project_id: str, model_name: str, location: str, prompt: str) -> str:
    """Get attack tree from Vertex AI"""
    response = get_vertex_response(project_id, model_name, location, prompt)
    try:
        tree_data = json.loads(response)
        from attack_tree import convert_tree_to_mermaid
        return convert_tree_to_mermaid(tree_data)
    except json.JSONDecodeError:
        from utils import extract_mermaid_code
        return extract_mermaid_code(response)

def get_mitigations_vertex(project_id: str, model_name: str, location: str, prompt: str) -> str:
    """Get mitigations from Vertex AI"""
    return get_vertex_response(project_id, model_name, location, prompt)

def get_test_cases_vertex(project_id: str, model_name: str, location: str, prompt: str) -> str:
    """Get test cases from Vertex AI"""
    return get_vertex_response(project_id, model_name, location, prompt)

def get_dread_assessment_vertex(project_id: str, model_name: str, location: str, prompt: str) -> Dict:
    """Get DREAD assessment from Vertex AI"""
    response = get_vertex_response(project_id, model_name, location, prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        raise
