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
        # Increase max_output_tokens to avoid truncation
        response = get_vertex_response(
            project_id, 
            model_name, 
            location, 
            prompt,
            max_output_tokens=4096  # Increase this value
        )
        
        # Debug output
        print(f"Vertex AI response type: {type(response)}")
        print(f"Response preview: {response[:200] if response else 'Empty response'}")
        
        if not response:
            print("Empty response from Vertex AI")
            return {"threat_model": [], "improvement_suggestions": []}
        
        # Clean up Markdown code blocks if present
        if response.strip().startswith("```json") or response.strip().startswith("```"):
            # Extract content between triple backticks
            import re
            code_block_match = re.search(r'```(?:json)?\s*\n(.*?)(?:```|$)', response, re.DOTALL)
            if code_block_match:
                response = code_block_match.group(1).strip()
            else:
                # If regex fails, try a simpler approach: remove starting and ending backticks
                response = response.replace("```json", "").replace("```", "").strip()
        
        # Try to parse the JSON, handling potential truncation
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Initial JSON parse error: {e}")
            # Try to extract as much valid JSON as possible
            # Add closing brackets to truncated JSON
            import re
            
            # Count opened brackets
            open_braces = response.count('{')
            close_braces = response.count('}')
            open_brackets = response.count('[')
            close_brackets = response.count(']')
            
            # Add missing closing brackets
            if open_braces > close_braces:
                response += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                response += ']' * (open_brackets - close_brackets)
            
            # Try to salvage threat_model array even if JSON is corrupted
            threat_model = []
            try:
                # Look for valid threats in the response
                threat_pattern = r'{\s*"Threat Type":\s*"([^"]+)",\s*"Scenario":\s*"([^"]+)",\s*"Potential Impact":\s*"([^"]+)"\s*}'
                matches = re.findall(threat_pattern, response)
                
                for match in matches:
                    threat_model.append({
                        "Threat Type": match[0],
                        "Scenario": match[1],
                        "Potential Impact": match[2]
                    })
                
                # If we found threats, return them
                if threat_model:
                    print(f"Recovered {len(threat_model)} threats from incomplete JSON")
                    return {
                        "threat_model": threat_model,
                        "improvement_suggestions": []
                    }
            except Exception as e2:
                print(f"Error during recovery: {e2}")
            
            # If all recovery attempts fail, return empty model
            return {"threat_model": [], "improvement_suggestions": []}
                    
    except Exception as e:
        print(f"Error in Vertex AI threat model generation: {str(e)}")
        return {"threat_model": [], "improvement_suggestions": []}

def get_attack_tree_vertex(project_id: str, model_name: str, location: str, prompt: str) -> str:
    """Get attack tree from Vertex AI"""
    # Increase max output tokens to avoid truncation
    response = get_vertex_response(
        project_id, 
        model_name, 
        location, 
        prompt,
        max_output_tokens=4096
    )
    
    # Debug output
    print(f"Vertex AI attack tree response type: {type(response)}")
    print(f"Response preview: {response[:200] if response else 'Empty response'}")
    
    if not response:
        print("Empty response from Vertex AI")
        return "graph TD\nA[Empty Response] --> B[Please try again]"
    
    # Try different parsing approaches
    try:
        # Check if response is in a code block (```json or ```mermaid)
        if "```" in response:
            # Extract the content from inside code blocks
            import re
            
            # First try to find JSON code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response, re.DOTALL)
            if json_match:
                try:
                    tree_data = json.loads(json_match.group(1).strip())
                    from attack_tree import convert_tree_to_mermaid
                    return convert_tree_to_mermaid(tree_data)
                except json.JSONDecodeError:
                    # Not valid JSON, might be direct Mermaid code
                    pass
            
            # If not valid JSON, look for Mermaid blocks or extract using utility
            from utils import extract_mermaid_code
            return extract_mermaid_code(response)
        
        # If no code blocks, try parsing as raw JSON
        tree_data = json.loads(response)
        from attack_tree import convert_tree_to_mermaid
        return convert_tree_to_mermaid(tree_data)
    
    except json.JSONDecodeError:
        # If all JSON parsing fails, try extracting Mermaid
        from utils import extract_mermaid_code
        return extract_mermaid_code(response)
    except Exception as e:
        print(f"Error processing attack tree: {str(e)}")
        # Return a simple fallback diagram
        return "graph TD\nA[Error] --> B[Processing failed]"

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
