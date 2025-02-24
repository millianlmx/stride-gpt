from google.cloud import aiplatform
import json

def init_vertex_ai(project_id: str, location: str = "us-central1"):
    """Initialize Vertex AI client"""
    aiplatform.init(project=project_id, location=location)

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
    init_vertex_ai(project_id)
    
    # Handle different model types
    if "gemini" in model_name.lower():
        model = aiplatform.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        return response.text
    else:
        # For other models
        model = aiplatform.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": top_p,
                "top_k": top_k
            }
        )
        return response.text

def get_image_analysis_vertex(project_id: str, model_name: str, location: str, prompt: str, base64_image: str) -> str:
    """Get image analysis from Vertex AI."""
    init_vertex_ai(project_id)
    
    try:
        if "gemini" in model_name.lower():
            # For Gemini models
            model = aiplatform.GenerativeModel(model_name)
            response = model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": base64_image}],
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                }
            )
            return response.text
        elif "claude" in model_name.lower():
            # For Claude models
            model = aiplatform.GenerativeModel(model_name)
            response = model.generate_content(
                [{"text": prompt}, {"image": base64_image}],
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
    response = get_vertex_response(project_id, model_name, location, prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        raise

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
