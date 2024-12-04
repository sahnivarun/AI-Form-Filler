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

import json
import logging

from langchain_g4f import G4FLLM

def retry_with_backoff(api_call, max_retries=5):
    for retry in range(max_retries):
        try:
            return api_call()
        except ResourceExhausted:
            wait_time = (2 ** retry) + random.uniform(0, 1)  # Exponential backoff with jitter
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("API retries exceeded. Quota may be exhausted.")


# Custom Language Model Wrapper
class CustomGenAIModel(LLM):
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

class G4FLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "g4f"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = Client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.choices[0].message.content
        if stop:
            stop_indexes = (output.find(s) for s in stop if s in output)
            min_stop = min(stop_indexes, default=-1)
            if min_stop > -1:
                output = output[:min_stop]
        return output

# Initialize and retrieve the language model from IBM Watson
def get_llm_gpt():
    llm: LLM = G4FLLM(
    model=models.gpt_35_turbo,
    provider=Provider.Aichat,
    )
    return llm
# Retrieve the custom Google GenAI LLM
def get_llm():
    os.environ["API_KEY"] = "AIzaSyDihkQrXCbVsaRb_4lkrIy7FmIulrVD77s"
    genai.configure(api_key=os.environ["API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    return CustomGenAIModel(model)

def process_data():
    loader = PyPDFDirectoryLoader("info")
    docs = loader.load()
    logger.info(f"Loaded Documents: {docs}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    logger.info(f"Split Texts: {texts}")
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_form_field_descriptions(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    form_fields = soup.find_all(['input', 'select', 'textarea'])
    field_info = []

    for field in form_fields:
        field_data = {}

        # Try to find the label using the 'for' attribute
        label = None
        if field.get('id'):
            label = soup.find('label', {'for': field.get('id')})

        # If no label is found, check parent or previous siblings for associated text
        if not label:
            parent = field.find_parent('div', class_='form-group')
            if parent:
                label = parent.find('label')

        # Extract label text or fallback to placeholder/name attributes
        if label and label.get_text(strip=True):
            field_data['label'] = label.get_text(strip=True)
        elif field.get('placeholder'):
            field_data['label'] = field.get('placeholder')
        elif field.get('name'):
            field_data['label'] = field.get('name')
        else:
            continue  # Skip fields without meaningful labels

        # Use 'id' if available, otherwise use 'name'
        field_data['id'] = field.get('id') or field.get('name')

        # Determine field type
        field_type = field.get('type', 'text')  # Default to 'text' if no type is specified
        if field.name == 'textarea':
            field_type = 'textarea'
        elif field.name == 'select':
            field_type = 'select'
        elif field_type == 'file':
            field_data['type'] = 'file'
            field_data['file_path'] = "C:/Users/varun/OneDrive/Documents/docs/Sahni_Varun_d.pdf"  # Specify the local path to the resume

        field_data['type'] = field_type

        # Ensure both label and id are present
        if 'label' in field_data and 'id' in field_data:
            field_info.append(field_data)

    return field_info


def get_dynamic_html(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    #chrome_service = Service('C:\Users\varun\Downloads\chrome-win64')  # Update this path
    chrome_service = Service('chromedriver.exe')  # Escaping backslashes
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    driver.get(url)
    time.sleep(2)  # Wait for dynamic content to load
    page_source = driver.page_source
    driver.quit()

    return page_source

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_json_request(form_fields_info):
    """
    Creates a JSON object where keys are field labels and values are empty strings.
    """
    json_request = {field['label']: "" for field in form_fields_info if field['type'] == 'text'}
    logger.info(f"Generated JSON request: {json_request}")
    return json_request


def filling_form_single_request(form_fields_info):
    try:
        llm = get_llm()
        db = process_data()
        json_request = get_json_request(form_fields_info)

        prompt = (
            f"Using the provided document content, fill in the following JSON object strictly in JSON format. "
            f"Keys are the questions, and values are the answers based on the document. "
            f"If the document does not contain an answer for a key, return an empty string for that key. "
            f"Return only the JSON object and no additional text or explanation.\n\n{json.dumps(json_request)}"
        )
        logger.info(f"Prompt sent to LLM: {prompt}")

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={'k': 4}),
        )

        def api_call():
            response = conversation_chain.invoke({"question": prompt, "chat_history": []})
            logger.info(f"Retrieved Context: {response.get('context', '')}")
            return response

        result = retry_with_backoff(api_call)
        llm_response = result['answer'].strip() if result['answer'] else "{}"
        logger.info(f"Raw LLM response: {llm_response}")

        if llm_response.startswith("```") and llm_response.endswith("```"):
            llm_response = llm_response.strip("```json").strip()
        logger.info(f"Cleaned LLM response: {llm_response}")

        try:
            filled_data = json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response from LLM: {e}")
            raise ValueError("Invalid JSON response from LLM.")

        for field in form_fields_info:
            if field['label'] in filled_data:
                field['response'] = filled_data[field['label']]
            if field['type'] == 'file':
                field['response'] = field.get('file_path', "")
                #field['file_path'] = "C:/Users/varun/OneDrive/Documents/docs/Sahni_Varun_d.pdf"

        logger.info(f"Final filled form fields: {form_fields_info}")
        return form_fields_info

    except Exception as e:
        logger.error(f"Error in filling_form_single_request: {e}")
        raise


@app.route('/api/auto_fill', methods=['POST'])
def auto_fill():
    """
    Flask endpoint for auto-filling the form.
    """
    if request.is_json:
        html_content = request.json.get('html_content', '')
        try:
            form_fields_info = get_form_field_descriptions(html_content)
            logger.info(f"Extracted Form Fields Info: {form_fields_info}")

            # Fill form with a single LLM call
            structured_responses = filling_form_single_request(form_fields_info)
            logger.info(f"Structured Responses: {structured_responses}")

            # Create a response mapping field IDs to their responses
            # response_data = {
            #     field['id']: field.get('response', "")
            #     for field in structured_responses if field['type'] == 'text'
            # }
            response_data = {
                field['id']: field.get('response', "") if field['type'] != 'file' else field.get('file_path', "")
                for field in structured_responses
            }
            logger.info(f"Final Response Data: {response_data}")

            return jsonify(response_data), 200
        except Exception as e:
            logger.error(f"Error in auto_fill: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid request format"}), 400

# Flask app entry point
if __name__ == '__main__':
    app.run(debug=True, port=5055)