import json
import requests
from anthropic import Anthropic
from mistralai import Mistral, UserMessage
from openai import OpenAI, AzureOpenAI
import streamlit as st

import google.generativeai as genai
from groq import Groq
from utils import process_groq_response, create_reasoning_system_prompt

# Function to convert JSON to Markdown for display.    
def json_to_markdown(threat_model, improvement_suggestions):
    markdown_output = "## Threat Model\n\n"
    
    # Start the markdown table with headers
    markdown_output += "| Threat Type | Scenario | Potential Impact |\n"
    markdown_output += "|-------------|----------|------------------|\n"
    
    # Fill the table rows with the threat model data
    for threat in threat_model:
        markdown_output += f"| {threat['Threat Type']} | {threat['Scenario']} | {threat['Potential Impact']} |\n"
    
    markdown_output += "\n\n## Improvement Suggestions\n\n"
    for suggestion in improvement_suggestions:
        markdown_output += f"- {suggestion}\n"
    
    return markdown_output

# Function to create a prompt for generating a threat model
def create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input, compliance_context=""):
    prompt = f"""
{compliance_context}
Act as a cyber security expert with more than 20 years experience of using the STRIDE threat modelling methodology to produce comprehensive threat models for a wide range of applications. Your task is to analyze the provided code summary, README content, application description, and compliance documentation to produce a list of specific threats for the application.

Pay special attention to:
1. The README content as it provides valuable context about the project's purpose and architecture
2. The compliance documentation which defines security requirements and controls that must be considered
3. Any potential conflicts between the application design and compliance requirements

For each of the STRIDE categories (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, and Elevation of Privilege), list multiple (3 or 4) credible threats if applicable. Each threat scenario should:
- Provide a credible scenario in which the threat could occur in the context of the application
- Consider how the threat might violate specific compliance requirements
- Identify potential compliance gaps or conflicts

When providing the threat model, use a JSON formatted response with the keys "threat_model" and "improvement_suggestions". Under "threat_model", include an array of objects with the keys "Threat Type", "Scenario", and "Potential Impact". 

Under "improvement_suggestions", include an array of strings that suggest what additional information or details the user could provide to make the threat model more comprehensive and accurate in the next iteration. Focus on identifying gaps in:
- Missing architectural details that would help identify more specific threats
- Unclear authentication flows that need more detail
- Incomplete data flow descriptions
- Missing technical stack information
- Unclear system boundaries or trust zones
- Incomplete description of sensitive data handling
- Areas where compliance requirements need clarification

Do not provide general security recommendations - focus only on what additional information would help create a better threat model.

APPLICATION TYPE: {app_type}
AUTHENTICATION METHODS: {authentication}
INTERNET FACING: {internet_facing}
SENSITIVE DATA: {sensitive_data}
CODE SUMMARY, README CONTENT, AND APPLICATION DESCRIPTION:
{app_input}

Example of expected JSON response format:
  
    {{
      "threat_model": [
        {{
          "Threat Type": "Spoofing",
          "Scenario": "Example Scenario 1",
          "Potential Impact": "Example Potential Impact 1"
        }},
        {{
          "Threat Type": "Spoofing",
          "Scenario": "Example Scenario 2",
          "Potential Impact": "Example Potential Impact 2"
        }},
        // ... more threats
      ],
      "improvement_suggestions": [
        "Please provide more details about the authentication flow between components to better analyze potential authentication bypass scenarios.",
        "Consider adding information about how sensitive data is stored and transmitted to enable more precise data exposure threat analysis.",
        // ... more suggestions for improving the threat model input
      ]
    }}
"""
    return prompt

def create_image_analysis_prompt():
    prompt = """
    You are a Senior Solution Architect tasked with explaining the following architecture diagram to 
    a Security Architect to support the threat modelling of the system.

    In order to complete this task you must:

      1. Analyse the diagram
      2. Explain the system architecture to the Security Architect. Your explanation should cover the key 
         components, their interactions, and any technologies used.
    
    Provide a direct explanation of the diagram in a clear, structured format, suitable for a professional 
    discussion.
    
    IMPORTANT INSTRUCTIONS:
     - Do not include any words before or after the explanation itself. For example, do not start your
    explanation with "The image shows..." or "The diagram shows..." just start explaining the key components
    and other relevant details.
     - Do not infer or speculate about information that is not visible in the diagram. Only provide information that can be
    directly determined from the diagram itself.
    """
    return prompt

# Function to get analyse uploaded architecture diagrams.
def get_image_analysis(api_key, model_name, prompt, base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 4000
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Log the response for debugging
    try:
        response.raise_for_status()  # Raise an HTTPError for bad responses
        response_content = response.json()
        return response_content
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # HTTP error
    except Exception as err:
        print(f"Other error occurred: {err}")  # Other errors

    print(f"Response content: {response.content}")  # Log the response content for further inspection
    return None

def get_image_analysis_vertex(project_id: str, model_name: str, location: str, prompt: str, base64_image: str) -> str:
    """Get image analysis from Vertex AI."""
    aiplatform.init(project=project_id, location=location)
    
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

def get_image_analysis_google(api_key, model_name, prompt, base64_image):
    """Get image analysis from Google AI API."""
    genai.configure(api_key=api_key)
    
    # Initialize the model
    model = genai.GenerativeModel(model_name)
    
    try:
        # Create the image part from base64
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64_image
        }
        
        # Generate content with both text prompt and image
        response = model.generate_content(
            [prompt, image_part],
            safety_settings={
                'DANGEROUS': 'block_only_high'
            }
        )
        
        # Extract the text content from the response
        if response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            raise ValueError("No response received from Google AI API")
            
    except Exception as e:
        print(f"Error in Google AI image analysis: {str(e)}")
        raise

# Function to get threat model from the GPT response.
def get_threat_model(api_key, model_name, prompt):
    client = OpenAI(api_key=api_key)

    # For reasoning models (o1, o3-mini), use a structured system prompt
    if model_name in ["o1", "o3-mini"]:
        system_prompt = create_reasoning_system_prompt(
            task_description="Analyze the provided application description and generate a comprehensive threat model using the STRIDE methodology.",
            approach_description="""1. Carefully read and understand the application description
2. For each component and data flow:
   - Identify potential Spoofing threats
   - Identify potential Tampering threats
   - Identify potential Repudiation threats
   - Identify potential Information Disclosure threats
   - Identify potential Denial of Service threats
   - Identify potential Elevation of Privilege threats
3. For each identified threat:
   - Describe the specific scenario
   - Analyze the potential impact
4. Generate improvement suggestions based on identified threats
5. Format the output as a JSON object with 'threat_model' and 'improvement_suggestions' arrays"""
        )
        # Create completion with max_completion_tokens for o1/o3-mini
        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=4000
        )
    else:
        system_prompt = "You are a helpful assistant designed to output JSON."
        # Create completion with max_tokens for other models
        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000
        )

    # Convert the JSON string in the 'content' field to a Python dictionary
    response_content = json.loads(response.choices[0].message.content)

    return response_content


# Function to get threat model from the Azure OpenAI response.
def get_threat_model_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, prompt):
    client = AzureOpenAI(
        azure_endpoint = azure_api_endpoint,
        api_key = azure_api_key,
        api_version = azure_api_version,
    )

    response = client.chat.completions.create(
        model = azure_deployment_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    # Convert the JSON string in the 'content' field to a Python dictionary
    response_content = json.loads(response.choices[0].message.content)

    return response_content


# Function to get threat model from the Google response.
def get_threat_model_google(google_api_key, google_model, prompt):
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(
        google_model,
        generation_config={"response_mime_type": "application/json"})
    response = model.generate_content(
        prompt,
        safety_settings={
            'DANGEROUS': 'block_only_high' # Set safety filter to allow generation of threat models
        })
    try:
        # Access the JSON content from the 'parts' attribute of the 'content' object
        response_content = json.loads(response.candidates[0].content.parts[0].text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {str(e)}")
        print("Raw JSON string:")
        print(response.candidates[0].content.parts[0].text)
        return None

    return response_content

# Function to get threat model from the Mistral response.
def get_threat_model_mistral(mistral_api_key, mistral_model, prompt):
    client = Mistral(api_key=mistral_api_key)

    response = client.chat.complete(
        model = mistral_model,
        response_format={"type": "json_object"},
        messages=[
            UserMessage(content=prompt)
        ]
    )

    # Convert the JSON string in the 'content' field to a Python dictionary
    response_content = json.loads(response.choices[0].message.content)

    return response_content

# Function to get threat model from Ollama hosted LLM.
def get_threat_model_ollama(ollama_endpoint, ollama_model, prompt):
    """
    Get threat model from Ollama hosted LLM.
    
    Args:
        ollama_endpoint (str): The URL of the Ollama endpoint (e.g., 'http://localhost:11434')
        ollama_model (str): The name of the model to use
        prompt (str): The prompt to send to the model
        
    Returns:
        dict: The parsed JSON response from the model
        
    Raises:
        requests.exceptions.RequestException: If there's an error communicating with the Ollama endpoint
        json.JSONDecodeError: If the response cannot be parsed as JSON
    """
    if not ollama_endpoint.endswith('/'):
        ollama_endpoint = ollama_endpoint + '/'
    
    url = ollama_endpoint + "api/generate"

    system_prompt = "You are a helpful assistant designed to output JSON."
    full_prompt = f"{system_prompt}\n\n{prompt}"

    data = {
        "model": ollama_model,
        "prompt": full_prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(url, json=data, timeout=60)  # Add timeout
        response.raise_for_status()  # Raise exception for bad status codes
        outer_json = response.json()
        
        try:
            # Parse the JSON response from the model's response field
            inner_json = json.loads(outer_json['response'])
            return inner_json
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing model response as JSON: {str(e)}")
            print("Raw response:", outer_json)
            raise
            
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama endpoint: {str(e)}")
        raise

# Function to get threat model from the Claude response.
def get_threat_model_anthropic(anthropic_api_key, anthropic_model, prompt):
    client = Anthropic(api_key=anthropic_api_key)
    response = client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        system="You are a helpful assistant designed to output JSON.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Combine all text blocks into a single string
    full_content = ''.join(block.text for block in response.content)
    
    # Parse the combined JSON string
    response_content = json.loads(full_content)
    return response_content

# Function to get threat model from LM Studio Server response.
def get_threat_model_lm_studio(lm_studio_endpoint, model_name, prompt):
    client = OpenAI(
        base_url=f"{lm_studio_endpoint}/v1",
        api_key="not-needed"  # LM Studio Server doesn't require an API key
    )

    # Define the expected response structure
    threat_model_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "threat_model_response",
            "schema": {
                "type": "object",
                "properties": {
                    "threat_model": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "Threat Type": {"type": "string"},
                                "Scenario": {"type": "string"},
                                "Potential Impact": {"type": "string"}
                            },
                            "required": ["Threat Type", "Scenario", "Potential Impact"]
                        }
                    },
                    "improvement_suggestions": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["threat_model", "improvement_suggestions"]
            }
        }
    }

    response = client.chat.completions.create(
        model=model_name,
        response_format=threat_model_schema,
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
    )

    # Convert the JSON string in the 'content' field to a Python dictionary
    response_content = json.loads(response.choices[0].message.content)

    return response_content

# Function to get threat model from the Groq response.
def get_threat_model_groq(groq_api_key, groq_model, prompt):
    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model=groq_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    # Process the response using our utility function
    reasoning, response_content = process_groq_response(
        response.choices[0].message.content,
        groq_model,
        expect_json=True
    )
    
    # If we got reasoning, display it in an expander in the UI
    if reasoning:
        with st.expander("View model's reasoning process", expanded=False):
            st.write(reasoning)

    return response_content
