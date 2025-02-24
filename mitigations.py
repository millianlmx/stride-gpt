import requests
from anthropic import Anthropic
from mistralai import Mistral
from openai import OpenAI, AzureOpenAI
import streamlit as st

import google.generativeai as genai
from groq import Groq
from utils import process_groq_response, create_reasoning_system_prompt

# Function to create a prompt to generate mitigating controls
def create_mitigations_prompt(threats, compliance_context=""):
    prompt = f"""
{compliance_context}
Act as a cyber security expert with more than 20 years experience of implementing security controls for a wide range of applications. Your task is to analyze the provided threats, compliance requirements, and codebase to suggest appropriate security controls and mitigations.

For each threat, analyze:
1. If mitigations are already implemented in the codebase
2. The completeness and effectiveness of existing controls
3. What additional mitigations are needed
4. How mitigations align with compliance requirements

IDENTIFIED THREATS:
{threats}

YOUR RESPONSE (in markdown format):
Please provide a detailed table with the following columns:
- Threat Type
- Scenario
- Implementation Status (choose one: "Implemented", "Partially Implemented", "Not Implemented", "Cannot Determine")
- Code Analysis (describe any relevant code findings, patterns, or missing controls)
- Additional Mitigations Needed
- Compliance Alignment (explicitly reference the compliance requirement IDs from the provided documentation, e.g., AA.1.2.3)

IMPORTANT: 
- For Implementation Status:
  * "Implemented" - Clear evidence in code of proper security controls
  * "Partially Implemented" - Some controls exist but are incomplete
  * "Not Implemented" - No evidence of required controls
  * "Cannot Determine" - Insufficient code context to assess
- When referencing compliance requirements:
  * Use the exact requirement IDs from the compliance documentation (format: AA.1.2.3)
  * Multiple requirements should be separated by commas
  * Only reference requirements that are actually present in the provided compliance documentation
- For Code Analysis:
  * Reference specific code patterns, functions, or security mechanisms found
  * Identify gaps or potential weaknesses in implementations
  * Note any security-relevant code comments or documentation
"""
    return prompt


# Function to get mitigations from the GPT response.
def get_mitigations(api_key, model_name, prompt):
    client = OpenAI(api_key=api_key)

    # For reasoning models (o1, o3-mini), use a structured system prompt
    if model_name in ["o1", "o3-mini"]:
        system_prompt = create_reasoning_system_prompt(
            task_description="Generate effective security mitigations for the identified threats using the STRIDE methodology.",
            approach_description="""1. Analyze each threat in the provided threat model
2. For each threat:
   - Understand the threat type and scenario
   - Consider the potential impact
   - Identify appropriate security controls and mitigations
   - Ensure mitigations are specific and actionable
3. Format the output as a markdown table with columns for:
   - Threat Type
   - Scenario
   - Suggested Mitigation(s)
4. Ensure mitigations follow security best practices and industry standards"""
        )
    else:
        system_prompt = "You are a helpful assistant that provides threat mitigation strategies in Markdown format."

    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the content directly as the response will be in text format
    mitigations = response.choices[0].message.content

    return mitigations


# Function to get mitigations from the Azure OpenAI response.
def get_mitigations_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, prompt):
    client = AzureOpenAI(
        azure_endpoint = azure_api_endpoint,
        api_key = azure_api_key,
        api_version = azure_api_version,
    )

    response = client.chat.completions.create(
        model = azure_deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the content directly as the response will be in text format
    mitigations = response.choices[0].message.content

    return mitigations

# Function to get mitigations from the Google model's response.
def get_mitigations_google(google_api_key, google_model, prompt):
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(
        google_model,
        system_instruction="You are a helpful assistant that provides threat mitigation strategies in Markdown format.",
    )
    response = model.generate_content(prompt)
    try:
        # Extract the text content from the 'candidates' attribute
        mitigations = response.candidates[0].content.parts[0].text
        # Replace '\n' with actual newline characters
        mitigations = mitigations.replace('\\n', '\n')
    except (IndexError, AttributeError) as e:
        print(f"Error accessing response content: {str(e)}")
        print("Raw response:")
        print(response)
        return None

    return mitigations

# Function to get mitigations from the Mistral model's response.
def get_mitigations_mistral(mistral_api_key, mistral_model, prompt):
    client = Mistral(api_key=mistral_api_key)

    response = client.chat.complete(
        model = mistral_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the content directly as the response will be in text format
    mitigations = response.choices[0].message.content

    return mitigations

# Function to get mitigations from Ollama hosted LLM.
def get_mitigations_ollama(ollama_endpoint, ollama_model, prompt):
    """
    Get mitigations from Ollama hosted LLM.
    
    Args:
        ollama_endpoint (str): The URL of the Ollama endpoint (e.g., 'http://localhost:11434')
        ollama_model (str): The name of the model to use
        prompt (str): The prompt to send to the model
        
    Returns:
        str: The generated mitigations in markdown format
        
    Raises:
        requests.exceptions.RequestException: If there's an error communicating with the Ollama endpoint
        KeyError: If the response doesn't contain the expected fields
    """
    if not ollama_endpoint.endswith('/'):
        ollama_endpoint = ollama_endpoint + '/'
    
    url = ollama_endpoint + "api/chat"

    data = {
        "model": ollama_model,
        "stream": False,
        "messages": [
            {
                "role": "system", 
                "content": """You are a cyber security expert with more than 20 years experience of implementing security controls for a wide range of applications. Your task is to analyze the provided application description and suggest appropriate security controls and mitigations.

Please provide your response in markdown format with appropriate headings and bullet points."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(url, json=data, timeout=60)  # Add timeout
        response.raise_for_status()  # Raise exception for bad status codes
        outer_json = response.json()
        
        try:
            # Access the 'content' attribute of the 'message' dictionary
            mitigations = outer_json["message"]["content"]
            return mitigations
            
        except KeyError as e:
            print(f"Error accessing response fields: {str(e)}")
            print("Raw response:", outer_json)
            raise
            
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama endpoint: {str(e)}")
        raise

# Function to get mitigations from the Anthropic model's response.
def get_mitigations_anthropic(anthropic_api_key, anthropic_model, prompt):
    client = Anthropic(api_key=anthropic_api_key)
    response = client.messages.create(
        model=anthropic_model,
        max_tokens=4096,
        system="You are a helpful assistant that provides threat mitigation strategies in Markdown format.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Access the text content from the first content block
    mitigations = response.content[0].text

    return mitigations

# Function to get mitigations from LM Studio Server response.
def get_mitigations_lm_studio(lm_studio_endpoint, model_name, prompt):
    client = OpenAI(
        base_url=f"{lm_studio_endpoint}/v1",
        api_key="not-needed"  # LM Studio Server doesn't require an API key
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the content directly as the response will be in text format
    mitigations = response.choices[0].message.content

    return mitigations

# Function to get mitigations from the Groq model's response.
def get_mitigations_groq(groq_api_key, groq_model, prompt):
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {"role": "user", "content": prompt}
        ]
    )

    # Process the response using our utility function
    reasoning, mitigations = process_groq_response(
        response.choices[0].message.content,
        groq_model,
        expect_json=False
    )
    
    # If we got reasoning, display it in an expander in the UI
    if reasoning:
        with st.expander("View model's reasoning process", expanded=False):
            st.write(reasoning)

    return mitigations
