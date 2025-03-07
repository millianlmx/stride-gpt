#main.py

import base64
import streamlit as st
import streamlit.components.v1 as components
from github import Github
from collections import defaultdict
import re
import os
from compliance_parser import extract_text_from_pdfs, format_compliance_context, get_compliance_summary
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json
from google.cloud import aiplatform

from threat_model import create_threat_model_prompt, get_threat_model, get_threat_model_azure, get_threat_model_google, get_threat_model_mistral, get_threat_model_ollama, get_threat_model_anthropic, get_threat_model_lm_studio, get_threat_model_groq, json_to_markdown, get_image_analysis, get_image_analysis_google, create_image_analysis_prompt
from utils import clean_markdown_for_display
from vertex_ai import (
    get_image_analysis_vertex, 
    get_threat_model_vertex, 
    get_attack_tree_vertex,
    get_mitigations_vertex,
    get_test_cases_vertex,
    get_dread_assessment_vertex
)
from attack_tree import create_attack_tree_prompt, get_attack_tree, get_attack_tree_azure, get_attack_tree_mistral, get_attack_tree_ollama, get_attack_tree_anthropic, get_attack_tree_lm_studio, get_attack_tree_groq, get_attack_tree_google
from mitigations import create_mitigations_prompt, get_mitigations, get_mitigations_azure, get_mitigations_google, get_mitigations_mistral, get_mitigations_ollama, get_mitigations_anthropic, get_mitigations_lm_studio, get_mitigations_groq
from test_cases import create_test_cases_prompt, get_test_cases, get_test_cases_azure, get_test_cases_google, get_test_cases_mistral, get_test_cases_ollama, get_test_cases_anthropic, get_test_cases_lm_studio, get_test_cases_groq
from dread import create_dread_assessment_prompt, get_dread_assessment, get_dread_assessment_azure, get_dread_assessment_google, get_dread_assessment_mistral, get_dread_assessment_ollama, get_dread_assessment_anthropic, get_dread_assessment_lm_studio, get_dread_assessment_groq, dread_json_to_markdown

# ------------------ Helper Functions ------------------ #

# Function to get available models from LM Studio Server
def get_lm_studio_models(endpoint):
    try:
        client = OpenAI(
            base_url=f"{endpoint}/v1",
            api_key="not-needed"
        )
        models = client.models.list()
        return [model.id for model in models.data]
    except requests.exceptions.ConnectionError:
        st.error("""Unable to connect to LM Studio Server. Please ensure:
1. LM Studio is running and the local server is started
2. The endpoint URL is correct (default: http://localhost:1234)
3. No firewall is blocking the connection""")
        return ["local-model"]
    except Exception as e:
        st.error(f"""Error fetching models from LM Studio Server: {e}
        
Please check:
1. LM Studio is properly configured and running
2. You have loaded a model in LM Studio
3. The server is running in local inference mode""")
        return ["local-model"]

def get_ollama_models(ollama_endpoint):
    """
    Get list of available models from Ollama.
    
    Args:
        ollama_endpoint (str): The URL of the Ollama endpoint (e.g., 'http://localhost:11434')
        
    Returns:
        list: List of available model names
        
    Raises:
        requests.exceptions.RequestException: If there's an error communicating with the Ollama endpoint
    """
    if not ollama_endpoint.endswith('/'):
        ollama_endpoint = ollama_endpoint + '/'
    
    url = ollama_endpoint + "api/tags"
    
    try:
        response = requests.get(url, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise exception for bad status codes
        models_data = response.json()
        
        # Extract model names from the response
        model_names = [model['name'] for model in models_data['models']]
        if not model_names:
            st.warning("""No models found in Ollama. Please ensure you have:
1. Pulled at least one model using 'ollama pull <model_name>'
2. The model download completed successfully""")
            return ["local-model"]
        return model_names
            
    except requests.exceptions.ConnectionError:
        st.error("""Unable to connect to Ollama. Please ensure:
1. Ollama is installed and running
2. The endpoint URL is correct (default: http://localhost:11434)
3. No firewall is blocking the connection""")
        return ["local-model"]
    except requests.exceptions.Timeout:
        st.error("""Request to Ollama timed out. Please check:
1. Ollama is responding and not overloaded
2. Your network connection is stable
3. The endpoint URL is accessible""")
        return ["local-model"]
    except (KeyError, json.JSONDecodeError):
        st.error("""Received invalid response from Ollama. Please verify:
1. You're running a compatible version of Ollama
2. The endpoint URL is pointing to Ollama and not another service""")
        return ["local-model"]
    except Exception as e:
        st.error(f"""Unexpected error fetching Ollama models: {str(e)}
        
Please check:
1. Ollama is properly installed and running
2. You have pulled at least one model
3. You have sufficient system resources""")
        return ["local-model"]

# Function to get user input for the application description and key details
def get_input(model_provider=None, selected_model=None, google_model=None):
    # Initialize combined analysis
    combined_analysis = ""

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        # GitHub repositories
        github_urls = st.text_area(
            label="Enter GitHub repository URLs (one per line)",
            placeholder="https://github.com/owner/repo1\nhttps://github.com/owner/repo2",
            key="github_url",
            help="Enter the URLs of the GitHub repositories you want to analyze (one per line).",
        )
        repo_urls = [url.strip() for url in github_urls.split('\n') if url.strip()]

        # Process GitHub URLs
        github_analysis = ""
        for repo_url in repo_urls:
            if repo_url and repo_url != st.session_state.get('last_analyzed_url', ''):
                if 'github_api_key' not in st.session_state or not st.session_state['github_api_key']:
                    st.warning("Please enter a GitHub API key to analyze the repositories.")
                else:
                    with st.spinner(f'Analyzing GitHub repository: {repo_url}'):
                        system_description = analyze_github_repo(repo_url)
                        github_analysis += system_description + "\n\n"
                        st.session_state['last_analyzed_url'] = repo_url

        # Add GitHub analysis to combined analysis if available
        if github_analysis:
            combined_analysis += "GITHUB REPOSITORY ANALYSIS:\n" + github_analysis

    with col2:
        # Allow diagram uploads for OpenAI, Google AI API, and Vertex AI API
        if model_provider in ["OpenAI API", "Google AI API", "Vertex AI API"]:
            if model_provider == "OpenAI API" and selected_model not in ["gpt-4o", "gpt-4o-mini"]:
                st.info("⚠️ Image analysis is only available with GPT-4 Vision models (gpt-4o or gpt-4o-mini)")
            elif model_provider == "Vertex AI API" and not ("gemini" in vertex_model.lower() or "claude" in vertex_model.lower()):
                st.info("⚠️ Image analysis is only available with Gemini and Claude models")
            else:
                uploaded_file = st.file_uploader("Upload architecture diagram", type=["jpg", "jpeg", "png"], key="diagram_uploader")

                if uploaded_file is not None:
                    if (model_provider == "OpenAI API" and not openai_api_key) or \
                       (model_provider == "Google AI API" and not google_api_key) or \
                       (model_provider == "Vertex AI API" and not vertex_project_id):
                        st.error(f"Please enter your {model_provider.split()[0]} credentials to analyse the image.")
                    else:
                        if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file:
                            st.session_state.uploaded_file = uploaded_file
                            with st.spinner("Analysing the uploaded image..."):
                                def encode_image(uploaded_file):
                                    return base64.b64encode(uploaded_file.read()).decode('utf-8')

                                base64_image = encode_image(uploaded_file)
                                image_analysis_prompt = create_image_analysis_prompt()

                                try:
                                    if model_provider == "OpenAI API":
                                        image_analysis_output = get_image_analysis(openai_api_key, selected_model, image_analysis_prompt, base64_image)
                                        if image_analysis_output and 'choices' in image_analysis_output and image_analysis_output['choices'][0]['message']['content']:
                                            image_analysis_content = image_analysis_output['choices'][0]['message']['content']
                                    elif model_provider == "Google AI API":
                                        image_analysis_output = get_image_analysis_google(google_api_key, google_model, image_analysis_prompt, base64_image)
                                        if image_analysis_output:
                                            image_analysis_content = image_analysis_output
                                    elif model_provider == "Vertex AI API":
                                        image_analysis_output = get_image_analysis_vertex(vertex_project_id, vertex_model, vertex_location, image_analysis_prompt, base64_image)
                                        if image_analysis_output:
                                            image_analysis_content = image_analysis_output

                                    # Add image analysis to combined analysis
                                    if 'image_analysis_content' in locals():
                                        combined_analysis += "\nARCHITECTURE DIAGRAM ANALYSIS:\n" + image_analysis_content + "\n\n"
                                        st.session_state.image_analysis_content = image_analysis_content
                                except Exception as e:
                                    st.error(f"An error occurred while analyzing the image: {str(e)}")
                                    print(f"Error: {e}")

    # Compliance documentation
    compliance_files = st.file_uploader(
        "Upload compliance documentation",
        type=["pdf"],
        accept_multiple_files=True,
        key="compliance"
    )
    
    # Process compliance files and show summary
    compliance_text = ""
    if compliance_files:
        compliance_text = extract_text_from_pdfs(compliance_files)
        st.session_state['compliance_text'] = compliance_text
        
        # Show summary of uploaded compliance documentation
        with st.expander("View Compliance Documentation Summary", expanded=True):
            st.markdown("### Uploaded Compliance Documentation")
            for file in compliance_files:
                st.markdown(f"- {file.name}")
            st.markdown("### Compliance Documentation Summary")
            with st.spinner("Generating compliance summary..."):
                # Prepare kwargs based on model provider
                kwargs = {
                    'openai_api_key': st.session_state.get('openai_api_key'),
                    'selected_model': selected_model if 'selected_model' in locals() else None,
                    'google_api_key': st.session_state.get('google_api_key'),
                    'google_model': google_model if 'google_model' in locals() else None,
                }
                
                # Add Vertex AI specific parameters if it's the selected provider
                if model_provider == "Vertex AI API":
                    kwargs.update({
                        'vertex_project_id': st.session_state.get('vertex_project_id'),
                        'vertex_model': st.session_state.get('selected_model'),
                        'vertex_location': vertex_location
                    })

                summary = get_compliance_summary(
                    compliance_text,
                    model_provider,
                    **kwargs
                )
                # Store the summary in session state
                st.session_state['compliance_summary'] = summary
                st.markdown(summary)

    # Get any existing manual input that's not from analysis
    existing_input = st.session_state.get('app_input', '')

    # Text input for additional description
    input_text = st.text_area(
        label="Describe the application to be modelled",
        value=combined_analysis or existing_input,  # Use combined_analysis if available, otherwise use existing input
        placeholder="Enter your application details...",
        height=300,
        key="app_desc",
        help="Please provide a detailed description of the application, including the purpose of the application, the technologies used, and any other relevant information.",
    )

    # Update session state with the complete input
    st.session_state['app_input'] = input_text

    return input_text

def analyze_github_repo(repo_url):
    # Extract owner and repo name from URL
    parts = repo_url.split('/')
    owner = parts[-2]
    repo_name = parts[-1]

    # Initialize PyGithub
    g = Github(st.session_state.get('github_api_key', ''))

    # Get the repository
    repo = g.get_repo(f"{owner}/{repo_name}")

    # Get the default branch
    default_branch = repo.default_branch

    # Get the tree of the default branch
    tree = repo.get_git_tree(default_branch, recursive=True)

    # Analyze files
    file_summaries = defaultdict(list)
    total_chars = 0
    char_limit = 100000  # Adjust this based on your model's token limit
    readme_content = ""

    for file in tree.tree:
        if file.path.lower() == 'readme.md':
            content = repo.get_contents(file.path, ref=default_branch)
            readme_content = base64.b64decode(content.content).decode()
        elif file.type == "blob" and file.path.endswith(('.py', '.js', '.ts', '.html', '.css', '.java', '.go', '.rb')):
            content = repo.get_contents(file.path, ref=default_branch)
            decoded_content = base64.b64decode(content.content).decode()
            
            # Summarize the file content
            summary = summarize_file(file.path, decoded_content)
            file_summaries[file.path.split('.')[-1]].append(summary)
            
            total_chars += len(summary)
            if total_chars > char_limit:
                break

    # Compile the analysis into a system description
    system_description = f"Repository: {repo_url}\n\n"
    
    if readme_content:
        system_description += "README.md Content:\n"
        # Truncate README if it's too long
        if len(readme_content) > 5000:
            system_description += readme_content[:5000] + "...\n(README truncated due to length)\n\n"
        else:
            system_description += readme_content + "\n\n"

    for file_type, summaries in file_summaries.items():
        system_description += f"{file_type.upper()} Files:\n"
        for summary in summaries:
            system_description += summary + "\n"
        system_description += "\n"

    return system_description

def summarize_file(file_path, content):
    # Extract important parts of the file
    imports = re.findall(r'^import .*|^from .* import .*', content, re.MULTILINE)
    functions = re.findall(r'def .*\(.*\):', content)
    classes = re.findall(r'class .*:', content)

    summary = f"File: {file_path}\n"
    if imports:
        summary += "Imports:\n" + "\n".join(imports[:5]) + "\n"  # Limit to first 5 imports
    if functions:
        summary += "Functions:\n" + "\n".join(functions[:5]) + "\n"  # Limit to first 5 functions
    if classes:
        summary += "Classes:\n" + "\n".join(classes[:5]) + "\n"  # Limit to first 5 classes

    return summary

# Function to render Mermaid diagram
def mermaid(code: str, height: int = 500) -> None:
    components.html(
        f"""
        <div style="height: 100%; width: 100%;">
            <pre class="mermaid" style="height: 100%;">
                {code}
            </pre>
        </div>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=height,
    )

def load_env_variables():
    # Try to load from .env file
    if os.path.exists('.env'):
        load_dotenv('.env')
    
    # Load GitHub API key from environment variable
    github_api_key = os.getenv('GITHUB_API_KEY')
    if github_api_key:
        st.session_state['github_api_key'] = github_api_key

    # Load other API keys if needed
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        st.session_state['openai_api_key'] = openai_api_key

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if anthropic_api_key:
        st.session_state['anthropic_api_key'] = anthropic_api_key

    azure_api_key = os.getenv('AZURE_API_KEY')
    if azure_api_key:
        st.session_state['azure_api_key'] = azure_api_key

    azure_api_endpoint = os.getenv('AZURE_API_ENDPOINT')
    if azure_api_endpoint:
        st.session_state['azure_api_endpoint'] = azure_api_endpoint

    azure_deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')
    if azure_deployment_name:
        st.session_state['azure_deployment_name'] = azure_deployment_name

    google_api_key = os.getenv('GOOGLE_API_KEY')
    if google_api_key:
        st.session_state['google_api_key'] = google_api_key

    mistral_api_key = os.getenv('MISTRAL_API_KEY')
    if mistral_api_key:
        st.session_state['mistral_api_key'] = mistral_api_key

    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        st.session_state['groq_api_key'] = groq_api_key

    # Add Ollama endpoint configuration
    ollama_endpoint = os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434')
    st.session_state['ollama_endpoint'] = ollama_endpoint

    # Add LM Studio Server endpoint configuration
    lm_studio_endpoint = os.getenv('LM_STUDIO_ENDPOINT', 'http://localhost:1234')
    st.session_state['lm_studio_endpoint'] = lm_studio_endpoint

# Call this function at the start of your app
load_env_variables()

# ------------------ Streamlit UI Configuration ------------------ #

st.set_page_config(
    page_title="STRIDE GPT",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Sidebar ------------------ #

st.sidebar.image("logo.png")

# Add instructions on how to use the app to the sidebar
st.sidebar.header("How to use STRIDE GPT")

with st.sidebar:
    # Add model selection input field to the sidebar
    model_provider = st.selectbox(
        "Select your preferred model provider:",
        ["OpenAI API", "Anthropic API", "Azure OpenAI Service", "Google AI API", "Vertex AI API", "Mistral API", "Groq API", "Ollama", "LM Studio Server"],
        key="model_provider",
        help="Select the model provider you would like to use. This will determine the models available for selection.",
    )

    if model_provider == "OpenAI API":
        st.markdown(
        """
    1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) and chosen model below 🔑
    2. Provide details of the application that you would like to threat model  📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
    )
        # Add OpenAI API key input field to the sidebar
        openai_api_key = st.text_input(
            "Enter your OpenAI API key:",
            value=st.session_state.get('openai_api_key', ''),
            type="password",
            help="You can find your OpenAI API key on the [OpenAI dashboard](https://platform.openai.com/account/api-keys).",
        )
        if openai_api_key:
            st.session_state['openai_api_key'] = openai_api_key

        # Add model selection input field to the sidebar
        selected_model = st.selectbox(
            "Select the model you would like to use:",
            ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
            key="selected_model",
            help="GPT-4o and GPT-4o mini are OpenAI's latest models and are recommended."
        )

    if model_provider == "Anthropic API":
        st.markdown(
        """
    1. Enter your [Anthropic API key](https://console.anthropic.com/settings/keys) and chosen model below 🔑
    2. Provide details of the application that you would like to threat model  📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
    )
        # Add Anthropic API key input field to the sidebar
        anthropic_api_key = st.text_input(
            "Enter your Anthropic API key:",
            value=st.session_state.get('anthropic_api_key', ''),
            type="password",
            help="You can find your Anthropic API key on the [Anthropic console](https://console.anthropic.com/settings/keys).",
        )
        if anthropic_api_key:
            st.session_state['anthropic_api_key'] = anthropic_api_key

        # Add model selection input field to the sidebar
        anthropic_model = st.selectbox(
            "Select the model you would like to use:",
            ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
            key="selected_model",
        )

    if model_provider == "Azure OpenAI Service":
        st.markdown(
        """
    1. Enter your Azure OpenAI API key, endpoint and deployment name below 🔑
    2. Provide details of the application that you would like to threat model  📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
    )

        # Add Azure OpenAI API key input field to the sidebar
        azure_api_key = st.text_input(
            "Azure OpenAI API key:",
            value=st.session_state.get('azure_api_key', ''),
            type="password",
            help="You can find your Azure OpenAI API key on the [Azure portal](https://portal.azure.com/).",
        )
        if azure_api_key:
            st.session_state['azure_api_key'] = azure_api_key
        
        # Add Azure OpenAI endpoint input field to the sidebar
        azure_api_endpoint = st.text_input(
            "Azure OpenAI endpoint:",
            value=st.session_state.get('azure_api_endpoint', ''),
            help="Example endpoint: https://YOUR_RESOURCE_NAME.openai.azure.com/",
        )
        if azure_api_endpoint:
            st.session_state['azure_api_endpoint'] = azure_api_endpoint

        # Add Azure OpenAI deployment name input field to the sidebar
        azure_deployment_name = st.text_input(
            "Deployment name:",
            value=st.session_state.get('azure_deployment_name', ''),
        )
        if azure_deployment_name:
            st.session_state['azure_deployment_name'] = azure_deployment_name
        
        st.info("Please note that you must use an 1106-preview model deployment.")

        azure_api_version = '2023-12-01-preview' # Update this as needed

        st.write(f"Azure API Version: {azure_api_version}")

    if model_provider == "Google AI API":
        st.markdown(
        """
    1. Enter your [Google AI API key](https://makersuite.google.com/app/apikey) and chosen model below 🔑
    2. Provide details of the application that you would like to threat model  📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
    )
        # Add OpenAI API key input field to the sidebar
        google_api_key = st.text_input(
            "Enter your Google AI API key:",
            value=st.session_state.get('google_api_key', ''),
            type="password",
            help="You can generate a Google AI API key in the [Google AI Studio](https://makersuite.google.com/app/apikey).",
        )
        if google_api_key:
            st.session_state['google_api_key'] = google_api_key

        # Add model selection input field to the sidebar
        google_model = st.selectbox(
            "Select the model you would like to use:",
            ["gemini-2.0-flash", "gemini-1.5-pro"],
            key="selected_model",
        )

    if model_provider == "Vertex AI API":
        st.markdown(
        """
    1. Enter your [Google Cloud Project ID](https://console.cloud.google.com/project) and chosen model below 🔑
    2. Provide details of the application that you would like to threat model  📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
    )
        # Add Project ID input field
        vertex_project_id = st.text_input(
            "Enter your Google Cloud Project ID:",
            value=st.session_state.get('vertex_project_id', ''),
            help="You can find your Project ID in the Google Cloud Console.",
            type="password"
        )
        if vertex_project_id:
            st.session_state['vertex_project_id'] = vertex_project_id

        # Add model selection input field
        vertex_model = st.selectbox(
            "Select the model you would like to use:",
            [
                "gemini-2.0-flash-001",
                "claude-3-5-sonnet-v2@20241022",
                "mistralai/mistral-large-2411@001"
            ],
            key="selected_model",
            help="Select from available Vertex AI models. Gemini models are recommended for best results."
        )

        # Add location selection based on selected model
        if "mistral" in vertex_model.lower():
            vertex_location = st.selectbox(
                "Select the location:",
                ["us-central1", "europe-west4"],
                help="Select the region where your Vertex AI resources are located. These regions are available for Mistral models."
            )
        elif "claude" in vertex_model.lower():
            vertex_location = st.selectbox(
                "Select the location:",
                ["us-east5", "europe-west1"],
                help="Select the region where your Vertex AI resources are located. These regions are available for Claude models."
            )
        else:  # Gemini models
            vertex_location = st.selectbox(
                "Select the location:",
                [
                    "us-central1", "us-east1", "us-west1", 
                    "europe-west1", "europe-west2", "europe-west3", "europe-west4",
                    "asia-east1", "asia-southeast1"
                ],
                help="Select the region where your Vertex AI resources are located. All regions are available for Gemini models."
            )

    if model_provider == "Mistral API":
        st.markdown(
        """
    1. Enter your [Mistral API key](https://console.mistral.ai/api-keys/) and chosen model below 🔑
    2. Provide details of the application that you would like to threat model  📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
    )
        # Add OpenAI API key input field to the sidebar
        mistral_api_key = st.text_input(
            "Enter your Mistral API key:",
            value=st.session_state.get('mistral_api_key', ''),
            type="password",
            help="You can generate a Mistral API key in the [Mistral console](https://console.mistral.ai/api-keys/).",
        )
        if mistral_api_key:
            st.session_state['mistral_api_key'] = mistral_api_key

        # Add model selection input field to the sidebar
        mistral_model = st.selectbox(
            "Select the model you would like to use:",
            ["mistral-large-latest", "mistral-small-latest"],
            key="selected_model",
        )

    if model_provider == "Ollama":
        st.markdown(
        """
    1. Configure your Ollama endpoint below (defaults to http://localhost:11434) 🔧
    2. Provide details of the application that you would like to threat model 📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
        )
        # Add Ollama endpoint configuration field
        ollama_endpoint = st.text_input(
            "Enter your Ollama endpoint:",
            value=st.session_state.get('ollama_endpoint', 'http://localhost:11434'),
            help="The URL of your Ollama instance. Default is http://localhost:11434 for local installations.",
        )
        if ollama_endpoint:
            # Basic URL validation
            if not ollama_endpoint.startswith(('http://', 'https://')):
                st.error("Endpoint URL must start with http:// or https://")
            else:
                st.session_state['ollama_endpoint'] = ollama_endpoint
                # Fetch available models from Ollama
                available_models = get_ollama_models(ollama_endpoint)

        # Add model selection input field
        selected_model = st.selectbox(
            "Select the Ollama model you would like to use:",
            available_models if ollama_endpoint and ollama_endpoint.startswith(('http://', 'https://')) else ["local-model"],
            key="selected_model",
            help="Select the model you have pulled into your Ollama instance."
        )

    if model_provider == "LM Studio Server":
        st.markdown(
        """
    1. Configure your LM Studio Server endpoint below (defaults to http://localhost:1234) 🔧
    2. Provide details of the application that you would like to threat model 📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
        )
        # Add LM Studio Server endpoint configuration field
        lm_studio_endpoint = st.text_input(
            "Enter your LM Studio Server endpoint:",
            value=st.session_state.get('lm_studio_endpoint', 'http://localhost:1234'),
            help="The URL of your LM Studio Server instance. Default is http://localhost:1234 for local installations.",
        )
        if lm_studio_endpoint:
            # Basic URL validation
            if not lm_studio_endpoint.startswith(('http://', 'https://')):
                st.error("Endpoint URL must start with http:// or https://")
            else:
                st.session_state['lm_studio_endpoint'] = lm_studio_endpoint
                # Fetch available models from LM Studio Server
                available_models = get_lm_studio_models(lm_studio_endpoint)

        # Add model selection input field
        selected_model = st.selectbox(
            "Select the LM Studio Server model you would like to use:",
            available_models if lm_studio_endpoint and lm_studio_endpoint.startswith(('http://', 'https://')) else ["local-model"],
            key="selected_model",
            help="Select the model you have loaded in your LM Studio Server instance."
        )

    if model_provider == "Groq API":
        st.markdown(
        """
    1. Enter your [Groq API key](https://console.groq.com/keys) and chosen model below 🔑
    2. Provide details of the application that you would like to threat model  📝
    3. Generate a threat list, attack tree and/or mitigating controls for your application 🚀
    """
    )
        # Add Groq API key input field to the sidebar
        groq_api_key = st.text_input(
            "Enter your Groq API key:",
            value=st.session_state.get('groq_api_key', ''),
            type="password",
            help="You can find your Groq API key in the [Groq console](https://console.groq.com/keys).",
        )
        if groq_api_key:
            st.session_state['groq_api_key'] = groq_api_key

        # Add model selection input field to the sidebar
        groq_model = st.selectbox(
            "Select the model you would like to use:",
            [
                "deepseek-r1-distill-llama-70b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            key="selected_model",
            help="Select from Groq's supported models. The Llama 3.3 70B Versatile model is recommended for best results."
        )

    # Add GitHub API key input field to the sidebar
    github_api_key = st.sidebar.text_input(
        "Enter your GitHub API key (optional):",
        value=st.session_state.get('github_api_key', ''),
        type="password",
        help="You can find or create your GitHub API key in your GitHub account settings under Developer settings > Personal access tokens.",
    )

    # Store the GitHub API key in session state
    if github_api_key:
        st.session_state['github_api_key'] = github_api_key

    st.markdown("""---""")

# Add "About" section to the sidebar
st.sidebar.header("About")

with st.sidebar:
    st.markdown(
        "Welcome to STRIDE GPT, an AI-powered tool designed to help teams produce better threat models for their applications."
    )
    st.markdown(
        "Threat modelling is a key activity in the software development lifecycle, but is often overlooked or poorly executed. STRIDE GPT aims to help teams produce more comprehensive threat models by leveraging the power of Large Language Models (LLMs) to generate a threat list, attack tree and/or mitigating controls for an application based on the details provided."
    )
    st.markdown("Created by [Matt Adams](https://www.linkedin.com/in/matthewrwadams/).")
    # Add "Star on GitHub" link to the sidebar
    st.sidebar.markdown(
        "⭐ Star on GitHub: [![Star on GitHub](https://img.shields.io/github/stars/mrwadams/stride-gpt?style=social)](https://github.com/mrwadams/stride-gpt)"
    )
    st.markdown("""---""")


# Add "Example Application Description" section to the sidebar
st.sidebar.header("Example Application Description")

with st.sidebar:
    st.markdown(
        "Below is an example application description that you can use to test STRIDE GPT:"
    )
    st.markdown(
        "> A web application that allows users to create, store, and share personal notes. The application is built using the React frontend framework and a Node.js backend with a MongoDB database. Users can sign up for an account and log in using OAuth2 with Google or Facebook. The notes are encrypted at rest and are only accessible by the user who created them. The application also supports real-time collaboration on notes with other users."
    )
    st.markdown("""---""")

# Add "FAQs" section to the sidebar
st.sidebar.header("FAQs")

with st.sidebar:
    st.markdown(
        """
    ### **What is STRIDE?**
    STRIDE is a threat modeling methodology that helps to identify and categorise potential security risks in software applications. It stands for **S**poofing, **T**ampering, **R**epudiation, **I**nformation Disclosure, **D**enial of Service, and **E**levation of Privilege.
    """
    )
    st.markdown(
        """
    ### **How does STRIDE GPT work?**
    When you enter an application description and other relevant details, the tool will use a GPT model to generate a threat model for your application. The model uses the application description and details to generate a list of potential threats and then categorises each threat according to the STRIDE methodology.
    """
    )
    st.markdown(
        """
    ### **Do you store the application details provided?**
    No, STRIDE GPT does not store your application description or other details. All entered data is deleted after you close the browser tab.
    """
    )
    st.markdown(
        """
    ### **Why does it take so long to generate a threat model?**
    If you are using a free OpenAI API key, it will take a while to generate a threat model. This is because the free API key has strict rate limits. To speed up the process, you can use a paid API key.
    """
    )
    st.markdown(
        """
    ### **Are the threat models 100% accurate?**
    No, the threat models are not 100% accurate. STRIDE GPT uses GPT Large Language Models (LLMs) to generate its output. The GPT models are powerful, but they sometimes makes mistakes and are prone to 'hallucinations' (generating irrelevant or inaccurate content). Please use the output only as a starting point for identifying and addressing potential security risks in your applications.
    """
    )
    st.markdown(
        """
    ### **How can I improve the accuracy of the threat models?**
    You can improve the accuracy of the threat models by providing a detailed description of the application and selecting the correct application type, authentication methods, and other relevant details. The more information you provide, the more accurate the threat models will be.
    """
    )


# ------------------ Main App UI ------------------ #

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Threat Model", "Attack Tree", "Mitigations", "DREAD", "Test Cases"])

with tab1:
    st.markdown("""
A threat model helps identify and evaluate potential security threats to applications / systems. It provides a systematic approach to 
understanding possible vulnerabilities and attack vectors. Use this tab to generate a threat model using the STRIDE methodology.
""")
    st.markdown("""---""")
    
    # Two column layout for the main app content
    col1, col2 = st.columns([1, 1])

    # Initialize app_input in the session state if it doesn't exist
    if 'app_input' not in st.session_state:
        st.session_state['app_input'] = ''

    with col1:

        # Use the get_input() function to get the application description and GitHub URL
        app_input = get_input(
            model_provider=model_provider,
            selected_model=selected_model if 'selected_model' in locals() else None,
            google_model=google_model if 'google_model' in locals() else None
        )
        # Update session state only if the text area content has changed
        if app_input != st.session_state['app_input']:
            st.session_state['app_input'] = app_input

    # Ensure app_input is always up to date in the session state
    app_input = st.session_state['app_input']



        # Create input fields for additional details
    with col2:
            app_type = st.selectbox(
                label="Select the application type",
                options=[
                    "Web application",
                    "Mobile application",
                    "Desktop application",
                    "Cloud application",
                    "IoT application",
                    "Other",
                ],
                key="app_type",
            )

            sensitive_data = st.selectbox(
                label="What is the highest sensitivity level of the data processed by the application?",
                options=[
                    "Top Secret",
                    "Secret",
                    "Confidential",
                    "Restricted",
                    "Unclassified",
                    "None",
                ],
                key="sensitive_data",
            )

        # Create input fields for internet_facing and authentication
            internet_facing = st.selectbox(
                label="Is the application internet-facing?",
                options=["Yes", "No"],
                key="internet_facing",
            )

            authentication = st.multiselect(
                "What authentication methods are supported by the application?",
                ["SSO", "MFA", "OAUTH2", "Basic", "None"],
                key="authentication",
            )



    # ------------------ Threat Model Generation ------------------ #

    # Create a submit button for Threat Modelling
    threat_model_submit_button = st.button(label="Generate Threat Model")

    # If the Generate Threat Model button is clicked and the user has provided an application description
    if threat_model_submit_button and st.session_state.get('app_input'):
        app_input = st.session_state['app_input']  # Retrieve from session state
        # Create compliance context with stored summary
        compliance_context = format_compliance_context(
            st.session_state.get('compliance_text', ''),
            model_provider,
            compliance_summary=st.session_state.get('compliance_summary', '')
        )
    
        # Generate the prompt using the create_prompt function with compliance context
        threat_model_prompt = create_threat_model_prompt(
            app_type, authentication, internet_facing, sensitive_data, 
            app_input, compliance_context
        )

        # Show a spinner while generating the threat model
        with st.spinner("Analysing potential threats..."):
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Call the relevant get_threat_model function with the generated prompt
                    if model_provider == "Azure OpenAI Service":
                        model_output = get_threat_model_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, threat_model_prompt)
                    elif model_provider == "OpenAI API":
                        model_output = get_threat_model(openai_api_key, selected_model, threat_model_prompt)
                    elif model_provider == "Google AI API":
                        model_output = get_threat_model_google(google_api_key, google_model, threat_model_prompt)
                    elif model_provider == "Mistral API":
                        model_output = get_threat_model_mistral(mistral_api_key, mistral_model, threat_model_prompt)
                    elif model_provider == "Ollama":
                        model_output = get_threat_model_ollama(st.session_state['ollama_endpoint'], selected_model, threat_model_prompt)
                    elif model_provider == "Anthropic API":
                        model_output = get_threat_model_anthropic(anthropic_api_key, anthropic_model, threat_model_prompt)
                    elif model_provider == "LM Studio Server":
                        model_output = get_threat_model_lm_studio(st.session_state['lm_studio_endpoint'], selected_model, threat_model_prompt)
                    elif model_provider == "Groq API":
                        model_output = get_threat_model_groq(groq_api_key, groq_model, threat_model_prompt)
                    elif model_provider == "Vertex AI API":
                        model_output = get_threat_model_vertex(
                            vertex_project_id,
                            vertex_model,
                            vertex_location,
                            threat_model_prompt
                        )

                    # Check if we have a valid response
                    if model_output is None:
                        raise ValueError("Received None response from model")
                
                    # Access the threat model and improvement suggestions from the parsed content
                    threat_model = model_output.get("threat_model", [])
                    improvement_suggestions = model_output.get("improvement_suggestions", [])
            
                    # Check if we have valid content
                    if not threat_model:
                        print("Warning: Empty threat model received")
                        if retry_count < max_retries - 1:
                            retry_count += 1
                            st.warning(f"Received empty threat model. Retrying attempt {retry_count+1}/{max_retries}...")
                            continue
                        else:
                            # Use default values on last attempt
                            threat_model = []
                            improvement_suggestions = []

                    # Save the threat model to the session state for later use in mitigations
                    st.session_state['threat_model'] = threat_model
                    break  # Exit the loop if successful
                except Exception as e:
                    retry_count += 1
                    # Print detailed error for debugging
                    import traceback
                    print(f"Error generating threat model: {str(e)}")
                    print(traceback.format_exc())
            
                    if retry_count == max_retries:
                        st.error(f"Error generating threat model after {max_retries} attempts: {e}")
                        threat_model = []
                        improvement_suggestions = []
                    else:
                        st.warning(f"Error generating threat model. Retrying attempt {retry_count+1}/{max_retries}...")

        # Convert the threat model JSON to Markdown
        markdown_output = json_to_markdown(threat_model, improvement_suggestions)

        # Display the threat model in Markdown
        st.markdown(markdown_output)

        # Add a button to allow the user to download the output as a Markdown file
        st.download_button(
            label="Download Threat Model",
            data=markdown_output,  # Use the Markdown output
            file_name="stride_gpt_threat_model.md",
            mime="text/markdown",
       )

# If the submit button is clicked and the user has not provided an application description
if threat_model_submit_button and not st.session_state.get('app_input'):
    st.error("Please enter your application details before submitting.")



# ------------------ Attack Tree Generation ------------------ #

with tab2:
    st.markdown("""
Attack trees are a structured way to analyse the security of a system. They represent potential attack scenarios in a hierarchical format, 
with the ultimate goal of an attacker at the root and various paths to achieve that goal as branches. This helps in understanding system 
vulnerabilities and prioritising mitigation efforts.
""")
    st.markdown("""---""")
    if model_provider == "Mistral API" and mistral_model == "mistral-small-latest":
        st.warning("⚠️ Mistral Small doesn't reliably generate syntactically correct Mermaid code. Please use the Mistral Large model for generating attack trees, or select a different model provider.")
    else:
        if model_provider in ["Ollama", "LM Studio Server"]:
            st.warning("⚠️ Users may encounter syntax errors when generating attack trees using local LLMs. Experiment with different local LLMs to assess their output quality, or consider using a hosted model provider to generate attack trees.")
        
        # Create a submit button for Attack Tree
        attack_tree_submit_button = st.button(label="Generate Attack Tree")
        
        # If the Generate Attack Tree button is clicked and the user has provided an application description
        if attack_tree_submit_button and st.session_state.get('app_input'):
            app_input = st.session_state.get('app_input')
            # Create compliance context with full PDF text
            compliance_context = format_compliance_context(
                st.session_state.get('compliance_text', ''),
                model_provider,
                vertex_project_id=st.session_state.get('vertex_project_id'),
                vertex_model=vertex_model if 'vertex_model' in locals() else None,
                vertex_location=vertex_location if 'vertex_location' in locals() else None,
                openai_api_key=st.session_state.get('openai_api_key'),
                selected_model=selected_model if 'selected_model' in locals() else None,
                google_api_key=st.session_state.get('google_api_key'),
                google_model=google_model if 'google_model' in locals() else None
            )
            
            # Generate the prompt using the create_attack_tree_prompt function
            attack_tree_prompt = create_attack_tree_prompt(
                app_type, authentication, internet_facing, sensitive_data, 
                app_input, compliance_context
            )

            # Show a spinner while generating the attack tree
            with st.spinner("Generating attack tree..."):
                try:
                    # Call the relevant get_attack_tree function with the generated prompt
                    if model_provider == "Azure OpenAI Service":
                        mermaid_code = get_attack_tree_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, attack_tree_prompt)
                    elif model_provider == "OpenAI API":
                        mermaid_code = get_attack_tree(openai_api_key, selected_model, attack_tree_prompt)
                    elif model_provider == "Google AI API":
                        mermaid_code = get_attack_tree_google(google_api_key, google_model, attack_tree_prompt)
                    elif model_provider == "Mistral API":
                        mermaid_code = get_attack_tree_mistral(mistral_api_key, mistral_model, attack_tree_prompt)
                    elif model_provider == "Ollama":
                        mermaid_code = get_attack_tree_ollama(st.session_state['ollama_endpoint'], selected_model, attack_tree_prompt)
                    elif model_provider == "Anthropic API":
                        mermaid_code = get_attack_tree_anthropic(anthropic_api_key, anthropic_model, attack_tree_prompt)
                    elif model_provider == "LM Studio Server":
                        mermaid_code = get_attack_tree_lm_studio(st.session_state['lm_studio_endpoint'], selected_model, attack_tree_prompt)
                    elif model_provider == "Groq API":
                        mermaid_code = get_attack_tree_groq(groq_api_key, groq_model, attack_tree_prompt)
                    elif model_provider == "Vertex AI API":
                        mermaid_code = get_attack_tree_vertex(
                            vertex_project_id,
                            vertex_model,
                            vertex_location,
                            attack_tree_prompt
                        )


                    # Display the generated attack tree code
                    st.write("Attack Tree Code:")
                    st.code(mermaid_code)

                    # Visualise the attack tree using the Mermaid custom component
                    st.write("Attack Tree Diagram Preview:")
                    mermaid(mermaid_code, height=800)  # Increased height to 800px
                    
                    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
                    
                    with col1:              
                        # Add a button to allow the user to download the Mermaid code
                        st.download_button(
                            label="Download Diagram Code",
                            data=mermaid_code,
                            file_name="attack_tree.md",
                            mime="text/plain",
                            help="Download the Mermaid code for the attack tree diagram."
                        )

                    with col2:
                        # Add a button to allow the user to open the Mermaid Live editor
                        mermaid_live_button = st.link_button("Open Mermaid Live", "https://mermaid.live")
                    
                    with col3:
                        # Blank placeholder
                        st.write("")
                    
                    with col4:
                        # Blank placeholder
                        st.write("")
                    
                    with col5:
                        # Blank placeholder
                        st.write("")

                except Exception as e:
                    st.error(f"Error generating attack tree: {e}")


# ------------------ Mitigations Generation ------------------ #

with tab3:
    st.markdown("""
Use this tab to generate potential mitigations for the threats identified in the threat model. Mitigations are security controls or
countermeasures that can help reduce the likelihood or impact of a security threat. The generated mitigations can be used to enhance
the security posture of the application and protect against potential attacks.
""")
    st.markdown("""---""")
    
    # Create a submit button for Mitigations
    mitigations_submit_button = st.button(label="Suggest Mitigations")

    # If the Suggest Mitigations button is clicked and the user has identified threats
    if mitigations_submit_button:
        # Check if threat_model data exists
        if 'threat_model' in st.session_state and st.session_state['threat_model']:
            # Convert the threat_model data into a Markdown list
            threats_markdown = json_to_markdown(st.session_state['threat_model'], [])
            
            # Create compliance context with stored summary
            compliance_context = format_compliance_context(
                st.session_state.get('compliance_text', ''),
                model_provider,
                compliance_summary=st.session_state.get('compliance_summary', '')
            )
            
            # Generate the prompt using the create_mitigations_prompt function with compliance context
            mitigations_prompt = create_mitigations_prompt(threats_markdown, compliance_context)

            # Show a spinner while suggesting mitigations
            with st.spinner("Suggesting mitigations..."):
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Call the relevant get_mitigations function with the generated prompt
                        if model_provider == "Azure OpenAI Service":
                            mitigations_markdown = get_mitigations_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, mitigations_prompt)
                        elif model_provider == "OpenAI API":
                            mitigations_markdown = get_mitigations(openai_api_key, selected_model, mitigations_prompt)
                        elif model_provider == "Google AI API":
                            mitigations_markdown = get_mitigations_google(google_api_key, google_model, mitigations_prompt)
                        elif model_provider == "Mistral API":
                            mitigations_markdown = get_mitigations_mistral(mistral_api_key, mistral_model, mitigations_prompt)
                        elif model_provider == "Ollama":
                            mitigations_markdown = get_mitigations_ollama(st.session_state['ollama_endpoint'], selected_model, mitigations_prompt)
                        elif model_provider == "Anthropic API":
                            mitigations_markdown = get_mitigations_anthropic(anthropic_api_key, anthropic_model, mitigations_prompt)
                        elif model_provider == "LM Studio Server":
                            mitigations_markdown = get_mitigations_lm_studio(st.session_state['lm_studio_endpoint'], selected_model, mitigations_prompt)
                        elif model_provider == "Groq API":
                            mitigations_markdown = get_mitigations_groq(groq_api_key, groq_model, mitigations_prompt)
                        elif model_provider == "Vertex AI API":
                            mitigations_markdown = get_mitigations_vertex(
                                vertex_project_id,
                                vertex_model,
                                vertex_location,
                                mitigations_prompt
                            )

                        # Display the suggested mitigations in Markdown
                        cleaned_markdown = clean_markdown_for_display(mitigations_markdown)
                        st.markdown(cleaned_markdown)
                        break  # Exit the loop if successful
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"Error suggesting mitigations after {max_retries} attempts: {e}")
                            mitigations_markdown = ""
                        else:
                            st.warning(f"Error suggesting mitigations. Retrying attempt {retry_count+1}/{max_retries}...")
            
            st.markdown("")

            # Add a button to allow the user to download the mitigations as a Markdown file
            st.download_button(
                label="Download Mitigations",
                data=cleaned_markdown,
                file_name="mitigations.md",
                mime="text/markdown",
            )
        else:
            st.error("Please generate a threat model first before suggesting mitigations.")

# ------------------ DREAD Risk Assessment Generation ------------------ #
with tab4:
    st.markdown("""
DREAD is a method for evaluating and prioritising risks associated with security threats. It assesses threats based on **D**amage potential, 
**R**eproducibility, **E**xploitability, **A**ffected users, and **D**iscoverability. This helps in determining the overall risk level and 
focusing on the most critical threats first. Use this tab to perform a DREAD risk assessment for your application / system.
""")
    st.markdown("""---""")
    
    # Create a submit button for DREAD Risk Assessment
    dread_assessment_submit_button = st.button(label="Generate DREAD Risk Assessment")
    # If the Generate DREAD Risk Assessment button is clicked and the user has identified threats
    if dread_assessment_submit_button:
        # Check if threat_model data exists
        if 'threat_model' in st.session_state and st.session_state['threat_model']:
            # Convert the threat_model data into a Markdown list
            threats_markdown = json_to_markdown(st.session_state['threat_model'], [])
            # Generate the prompt using the create_dread_assessment_prompt function
            dread_assessment_prompt = create_dread_assessment_prompt(threats_markdown)
            # Show a spinner while generating DREAD Risk Assessment
            with st.spinner("Generating DREAD Risk Assessment..."):
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Call the relevant get_dread_assessment function with the generated prompt
                        if model_provider == "Azure OpenAI Service":
                            dread_assessment = get_dread_assessment_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, dread_assessment_prompt)
                        elif model_provider == "OpenAI API":
                            dread_assessment = get_dread_assessment(openai_api_key, selected_model, dread_assessment_prompt)
                        elif model_provider == "Google AI API":
                            dread_assessment = get_dread_assessment_google(google_api_key, google_model, dread_assessment_prompt)
                        elif model_provider == "Mistral API":
                            dread_assessment = get_dread_assessment_mistral(mistral_api_key, mistral_model, dread_assessment_prompt)
                        elif model_provider == "Ollama":
                            dread_assessment = get_dread_assessment_ollama(st.session_state['ollama_endpoint'], selected_model, dread_assessment_prompt)
                        elif model_provider == "Anthropic API":
                            dread_assessment = get_dread_assessment_anthropic(anthropic_api_key, anthropic_model, dread_assessment_prompt)
                        elif model_provider == "LM Studio Server":
                            dread_assessment = get_dread_assessment_lm_studio(st.session_state['lm_studio_endpoint'], selected_model, dread_assessment_prompt)
                        elif model_provider == "Groq API":
                            dread_assessment = get_dread_assessment_groq(groq_api_key, groq_model, dread_assessment_prompt)
                        elif model_provider == "Vertex AI API":
                            dread_assessment = get_dread_assessment_vertex(
                                vertex_project_id,
                                vertex_model,
                                vertex_location,
                                dread_assessment_prompt
                            )
                        
                        # Save the DREAD assessment to the session state for later use in test cases
                        st.session_state['dread_assessment'] = dread_assessment
                        break  # Exit the loop if successful
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"Error generating DREAD risk assessment after {max_retries} attempts: {e}")
                            dread_assessment = []
                        else:
                            st.warning(f"Error generating DREAD risk assessment. Retrying attempt {retry_count+1}/{max_retries}...")
            # Convert the DREAD assessment JSON to Markdown
            dread_assessment_markdown = dread_json_to_markdown(dread_assessment)
            # Clean and display the DREAD assessment in Markdown
            cleaned_markdown = clean_markdown_for_display(dread_assessment_markdown)
            st.markdown(cleaned_markdown)
            # Add a button to allow the user to download the test cases as a Markdown file
            st.download_button(
                label="Download DREAD Risk Assessment",
                data=cleaned_markdown,
                file_name="dread_assessment.md",
                mime="text/markdown",
            )
        else:
            st.error("Please generate a threat model first before requesting a DREAD risk assessment.")


# ------------------ Test Cases Generation ------------------ #

with tab5:
    st.markdown("""
Test cases are used to validate the security of an application and ensure that potential vulnerabilities are identified and 
addressed. This tab allows you to generate test cases using Gherkin syntax. Gherkin provides a structured way to describe application 
behaviours in plain text, using a simple syntax of Given-When-Then statements. This helps in creating clear and executable test 
scenarios.
""")
    st.markdown("""---""")
                
    # Create a submit button for Test Cases
    test_cases_submit_button = st.button(label="Generate Test Cases")

    # If the Generate Test Cases button is clicked and the user has identified threats
    if test_cases_submit_button:
        # Check if threat_model data exists
        if 'threat_model' in st.session_state and st.session_state['threat_model']:
            # Convert the threat_model data into a Markdown list
            threats_markdown = json_to_markdown(st.session_state['threat_model'], [])
            # Generate the prompt using the create_test_cases_prompt function
            test_cases_prompt = create_test_cases_prompt(threats_markdown)

            # Show a spinner while generating test cases
            with st.spinner("Generating test cases..."):
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Call to the relevant get_test_cases function with the generated prompt
                        if model_provider == "Azure OpenAI Service":
                            test_cases_markdown = get_test_cases_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, test_cases_prompt)
                        elif model_provider == "OpenAI API":
                            test_cases_markdown = get_test_cases(openai_api_key, selected_model, test_cases_prompt)
                        elif model_provider == "Google AI API":
                            test_cases_markdown = get_test_cases_google(google_api_key, google_model, test_cases_prompt)
                        elif model_provider == "Mistral API":
                            test_cases_markdown = get_test_cases_mistral(mistral_api_key, mistral_model, test_cases_prompt)
                        elif model_provider == "Ollama":
                            test_cases_markdown = get_test_cases_ollama(st.session_state['ollama_endpoint'], selected_model, test_cases_prompt)
                        elif model_provider == "Anthropic API":
                            test_cases_markdown = get_test_cases_anthropic(anthropic_api_key, anthropic_model, test_cases_prompt)
                        elif model_provider == "LM Studio Server":
                            test_cases_markdown = get_test_cases_lm_studio(st.session_state['lm_studio_endpoint'], selected_model, test_cases_prompt)
                        elif model_provider == "Groq API":
                            test_cases_markdown = get_test_cases_groq(groq_api_key, groq_model, test_cases_prompt)
                        elif model_provider == "Vertex AI API":
                            test_cases_markdown = get_test_cases_vertex(
                                vertex_project_id,
                                vertex_model,
                                vertex_location,
                                test_cases_prompt
                            )

                        # Display the suggested mitigations in Markdown
                        cleaned_markdown = clean_markdown_for_display(test_cases_markdown)
                        st.markdown(cleaned_markdown)
                        break  # Exit the loop if successful
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"Error generating test cases after {max_retries} attempts: {e}")
                            test_cases_markdown = ""
                        else:
                            st.warning(f"Error generating test cases. Retrying attempt {retry_count+1}/{max_retries}...")
            
            st.markdown("")

            # Add a button to allow the user to download the test cases as a Markdown file
            st.download_button(
                label="Download Test Cases",
                data=cleaned_markdown,
                file_name="test_cases.md",
                mime="text/markdown",
            )
        else:
            st.error("Please generate a threat model first before requesting test cases.")
