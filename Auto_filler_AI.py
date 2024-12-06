# Import required libraries from various modules
# langchain library for embeddings, text splitting, and conversational retrieval
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate

# Document loader and vector store modules for processing PDFs
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from g4f.client import Client

# BeautifulSoup for parsing HTML content
from bs4 import BeautifulSoup

# Flask for web server and CORS for cross-origin resource sharing
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
from google.api_core.exceptions import ResourceExhausted
from langchain.llms.base import LLM
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Pydantic for typing in LangChain
from pydantic import PrivateAttr

# Utility function for exponential backoff
import time
import random
from g4f import Provider, models
from langchain.llms.base import LLM

import os
import platform
import json
import logging
import base64
import requests

import tiktoken  # For token counting
from time import time  # For response timing


# Function to clear the terminal
def clear_terminal():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

# Clear terminal at the start of the script
clear_terminal()

# Print a startup message
print("AI Form Auto Filler is started...")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Counts the number of tokens in a given text using the specified model.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Retry mechanism with logging
def retry_with_backoff(api_call, max_retries=5):
    for retry in range(max_retries):
        try:
            return api_call()
        except ResourceExhausted:
            wait_time = (2 ** retry) + random.uniform(0, 1)
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("API retries exceeded. Quota may be exhausted.")


# Custom Language Model Wrapper
class GeminiGenAIModel(LLM):
    _model: any = PrivateAttr()

    def __init__(self, model):
        super().__init__()
        self._model = model

    def _call(self, prompt: str, stop=None) -> str:
        try:
            response = retry_with_backoff(lambda: self._model.generate_content(prompt))
            return response.text
        except AttributeError:
            raise ValueError("The GenerativeModel object does not support 'generate_content'.")
        except ResourceExhausted:
            raise ValueError("Quota exhausted. Please try again later.")

    @property
    def _llm_type(self):
        return "custom_genai"
from typing import Optional, List

# Retrieve the custom Google GenAI LLM
def get_gemini_llm():
    os.environ["API_KEY"] = "AIzaSyDihkQrXCbVsaRb_4lkrIy7FmIulrVD77s"
    genai.configure(api_key=os.environ["API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    return GeminiGenAIModel(model)

from langchain.llms.base import LLM
from pydantic import BaseModel, PrivateAttr
from typing import Optional, List

class OpenAIModel(LLM, BaseModel):
    api_key: str
    llm_model: str

    _chat_model: any = PrivateAttr()

    def __init__(self, api_key: str, llm_model: str, **kwargs):
        # Initialize the Pydantic BaseModel
        super().__init__(api_key=api_key, llm_model=llm_model, **kwargs)
        from langchain_openai import ChatOpenAI
        # Initialize the private ChatOpenAI model
        self._chat_model = ChatOpenAI(
            model_name=self.llm_model, openai_api_key=self.api_key, temperature=0.4
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Call the underlying ChatOpenAI model
        response = self._chat_model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    @property
    def _llm_type(self) -> str:
        return "openai"

# Function to retrieve OpenAI LLM
def get_open_ai_llm():
    api_key = "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"
    llm_model = "gpt-4o-mini"  # Specify the desired OpenAI model
    return OpenAIModel(api_key, llm_model)
    
def process_data():
    loader = PyPDFDirectoryLoader("info")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    db = FAISS.from_documents(texts, embeddings)
    return db

from bs4 import BeautifulSoup

def get_form_field_descriptions(html_content):
    """
    Extracts form field descriptions from generic HTML content using BeautifulSoup.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Attempt to locate containers with form fields
    containers = soup.find_all(['div', 'section', 'li'], class_=lambda x: x and 'question' in x.lower())

    field_info = []

    for container in containers:
        # Find form-related elements (input, textarea, select)
        form_fields = container.find_all(['input', 'textarea', 'select'])

        for field in form_fields:
            field_data = {}

            # Attempt to find label or descriptive text
            label = container.find(['label', 'div', 'span'], class_=lambda x: x and 'text' in x.lower())
            if label and label.get_text(strip=True):
                field_data['label'] = label.get_text(strip=True)
            elif field.get('placeholder'):
                field_data['label'] = field.get('placeholder')
            elif field.get('name'):
                field_data['label'] = field.get('name')
            elif field.get('aria-label'):
                field_data['label'] = field.get('aria-label')
            else:
                field_data['label'] = "Unknown Field"

            # Assign unique ID or name
            field_data['id'] = field.get('id') or field.get('name') or "unknown"

            # Determine field type
            field_type = field.get('type', 'text')  # Default to 'text'
            if field.name == 'textarea':
                field_type = 'textarea'
            elif field.name == 'select':
                field_type = 'select'

            field_data['type'] = field_type

            
            # Extract dropdown options
            if field.name == 'select':
                options = field.find_all('option')
                field_data['options'] = [option.get_text(strip=True) for option in options if option.get_text(strip=True)]

            # Extract radio button values
            if field_type == 'radio':
                radios = container.find_all('input', {'type': 'radio'})
                field_data['options'] = [radio.get('value') for radio in radios if radio.get('value')]


            # Append valid field data
            if 'label' in field_data and 'id' in field_data:
                field_info.append(field_data)

    return field_info

# Initialize Flask app
app = Flask(__name__)
CORS(app)

def get_json_request(form_fields_info):
    """
    Creates a JSON object where keys are field labels, and values include options for applicable fields.
    """
    json_request = {}
    
    for field in form_fields_info:
        field_label = field['label']
        json_request[field_label] = {"value": ""}  # Default empty value for the field

        # Include options if the field has them (for dropdowns, radios, etc.)
        if 'options' in field:
            json_request[field_label]['options'] = field['options']

    print()
    print()
    logger.info(f"Generated JSON request with options: {json.dumps(json_request, indent=2)}")
    return json_request

def filling_form_single_request(form_fields_info):
    try:
        llm = get_open_ai_llm()
        db = process_data()
        json_request = get_json_request(form_fields_info)

        prompt = (
            f"You are an AI assistant helping to fill out a job application form using the provided documents. I am sharing my resume, "
            f"which contains my name, email, phone number, LinkedIn URL, GitHub URL, and portfolio URL (https://varun-sahni.com/), but these details might not have explicit labels. "
            f"Extract this information directly from the text in the resume and map it appropriately to the form fields. "
            f"Here is the form structure and its valid options (if applicable):\n\n"
            f"{json.dumps(json_request, indent=2)}\n\n"
            f"For the field with the key 'No location found. Try entering a different locationLoading', if a location is present in the document, write the location from the document. "
            f"For each question, generate an appropriate response based on the user's resume, documents, and job application context. "
            f"Ensure that fields such as LinkedIn URL, GitHub URL, Twitter URL, and portfolio URL are filled using the provided information. "
            f"If specific data for these fields is missing in the documents, infer or generate placeholder URLs (e.g., 'https://www.linkedin.com/in/username', "
            f"'https://github.com/username', 'https://twitter.com/username', 'https://yourportfolio.com') based on standard formats, ensuring relevance and professionalism. "
            f"If a question asks about motivations, company-specific enthusiasm, or open-ended responses (e.g., 'What gets you excited about joining this team?'), "
            f"generate a thoughtful answer based on common professional aspirations and values. "
            f"For any questions not explicitly covered in the document, provide a response based on your knowledge about me and the context of the job application. "
            f"Ensure that all responses strictly conform to the options provided (if any). "
            f"Return the completed JSON object strictly in JSON format without any additional text, code blocks, or comments."
        )

        # Measure tokens in the prompt
        prompt_tokens = count_tokens(prompt, model="gpt-4")

        # Start timer
        start_time = time()

        # Create a conversational retrieval chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={'k': 4}),
        )

        # Call the API with retry mechanism
        def api_call():
            response = conversation_chain.invoke({"question": prompt, "chat_history": []})
            return response

        result = retry_with_backoff(api_call)

        # End timer and calculate elapsed time
        end_time = time()
        response_time = end_time - start_time

        llm_response = result['answer'].strip() if result['answer'] else "{}"

        # Measure tokens in the response
        response_tokens = count_tokens(llm_response, model="gpt-4")

        print()
        # Print stats for form filling
        print("\n--- GPT Call: Form Filling ---")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Response Tokens: {response_tokens}")
        print(f"Total Tokens: {prompt_tokens + response_tokens}")
        print(f"Response Time: {response_time:.2f} seconds\n")

        print()
        logger.info(f"Raw LLM response: {llm_response}")

        # Clean up the LLM response
        if llm_response.startswith("```") and llm_response.endswith("```"):
            llm_response = llm_response.strip("```").strip()

        # Validate and clean up JSON response
        try:
            llm_response = llm_response.strip()
            
            # Remove unwanted prefixes or suffixes (e.g., 'json\n')
            if llm_response.lower().startswith("json"):
                llm_response = llm_response[len("json"):].strip()

            # Remove any surrounding code block indicators (e.g., triple backticks)
            if llm_response.startswith("```") and llm_response.endswith("```"):
                llm_response = llm_response.strip("```").strip()

            # Attempt to parse JSON
            filled_data = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {llm_response}")
            raise ValueError("Invalid JSON response from LLM.") from e

        #print()
        #logger.info(f"Parsed JSON response: {filled_data}")

        for field in form_fields_info:
            label = field.get('label', '').strip()
            if label in filled_data:
                response_value = filled_data[label].get('value', "")

                # Handle radio button responses
                if field['type'] == 'radio' and 'options' in field:
                    if response_value in field['options']:
                        field['response'] = response_value  # Match response with options
                    else:
                        field['response'] = ""  # Set to empty if no match

                # Handle dropdown responses
                elif field['type'] == 'select' and 'options' in field:
                    if response_value in field['options']:
                        field['response'] = response_value  # Match response with options
                    else:
                        field['response'] = ""  # Set to empty if no match

                else:
                    # For other field types, simply assign the value
                    field['response'] = response_value
            else:
                field['response'] = ""

        #print()
        #logger.info(f"Final filled form fields: {form_fields_info}")
        return form_fields_info

    except Exception as e:
        logger.error(f"Error in filling_form_single_request: {e}")
        raise

@app.route('/api/auto_fill', methods=['POST'])
def auto_fill():
    if request.is_json:
        html_content = request.json.get('html_content', '')
        try:
            form_fields_info = get_form_field_descriptions(html_content)
            #logger.info(f"Extracted Form Fields Info: {form_fields_info}")

            # Fill form with a single LLM call
            structured_responses = filling_form_single_request(form_fields_info)

            # Build the response data to fill form fields
            response_data = {}
            for field in structured_responses:
                field_id = field.get('id', '')
                response_data[field_id] = field.get('response', '')

            #print()
            #logger.info(f"Final Response Data: {response_data}")
            
            # At this point, form filling is completed
            print("Form has been filled.")

            return jsonify(response_data), 200
        except Exception as e:
            logger.error(f"Error in auto_fill: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid request format"}), 400

# Add the new endpoint for generating cover letters
@app.route('/api/generate_cover_letter', methods=['POST'])
def generate_cover_letter():
    """
    Flask endpoint for generating a cover letter.
    """
    if request.is_json:
        job_description = request.json.get('job_description', '')
        try:
            # Generate the cover letter using the LLM
            cover_letter = create_cover_letter(job_description)
            return jsonify({'cover_letter': cover_letter}), 200
        except Exception as e:
            logger.error(f"Error in generate_cover_letter: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid request format"}), 400
# Function to create a cover letter using the LLM
def create_cover_letter(job_description):
    try:
        llm = get_open_ai_llm()
        # Load user's information from documents
        db = process_data()

        # Create a prompt for the LLM
        prompt = (
            f"Using the following job description and the provided documents, write the body of a fully tailored, polished, and professional cover letter. "
            f"Ensure the output is a complete, ready-to-submit cover letter body that focuses on aligning my experience, skills, and education with the job's requirements and the company's mission and values. "
            f"End the letter with 'Sincerely,' followed by my full name, which you will extract directly from the provided document. "
            f"Do not include any placeholders or guess my name if it is missing. If the name is not found in the document, do not write closing salution."
            f"Do not mention the platform where the job was found, specific hiring manager names, or any placeholders. "
            f"If specific job details are missing, craft a general professional cover letter body suitable for a technical software engineering role, emphasizing my key strengths and achievements. "
            f"Job Description:\n{job_description}\n\n"
            f"Cover Letter Body:"
        )


        # Token measurement for prompt
        prompt_tokens = count_tokens(prompt, model="gpt-4")

        # Start timer for cover letter generation
        start_time = time()

        # Create a conversational retrieval chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={'k': 4}),
        )

        # Define an API call function for retries
        def api_call():
            response = conversation_chain.invoke({"question": prompt, "chat_history": []})
            #logger.info(f"LLM Response: {response.get('answer', '')}")
            return response

        # Call the API with retry mechanism
        result = retry_with_backoff(api_call)

        # End timer for cover letter generation
        end_time = time()
        response_time = end_time - start_time

        cover_letter = result['answer'].strip() if result['answer'] else ""
        
        #logger.info(f"Generated Cover Letter: {cover_letter}")

        # Token measurement for response
        response_tokens = count_tokens(cover_letter, model="gpt-4")

        # Print stats for cover letter generation
        print()
        print(f"Response Time: {response_time:.2f} seconds\n")

        print("\n--- GPT Call: Cover Letter Generation ---")
        print(f"Prompt Tokens: {prompt_tokens}")
        print(f"Response Tokens: {response_tokens}")
        print(f"Total Tokens: {prompt_tokens + response_tokens}")

        return cover_letter

    except Exception as e:
        logger.error(f"Error in create_cover_letter: {e}")
        raise

@app.route('/api/fetch_resume', methods=['POST'])
def fetch_resume():
    if request.is_json:
        resume_url = request.json.get('resume_url', '')
        if not resume_url:
            return jsonify({"error": "No resume URL provided"}), 400
        try:
            resp = requests.get(resume_url)
            if resp.status_code != 200:
                return jsonify({"error": f"Failed to fetch resume from URL. Status: {resp.status_code}"}), 500

            file_content = base64.b64encode(resp.content).decode('utf-8')
            # Return a data URL like: data:application/pdf;base64,<content>
            data_url = f"data:application/pdf;base64,{file_content}"
            return jsonify({"fileContent": data_url}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid request format"}), 400


      
# Flask app entry point
if __name__ == '__main__':
    app.run(debug=True, port=5055)