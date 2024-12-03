# Import required libraries from various modules
# from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
# from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watson_machine_learning.foundation_models import Model

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
def get_llm_old():
    my_credentials = {"url": "https://us-south.ml.cloud.ibm.com"}
    params = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.0,
    }
    LLAMA2_model = Model(
        model_id='meta-llama/llama-2-70b-chat',
        credentials=my_credentials,
        params=params,
        project_id="skills-network"
    )
    llm = WatsonxLLM(model=LLAMA2_model)
    return llm

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


# Process and index PDF documents
def process_data():
    loader = PyPDFDirectoryLoader("info")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
    db = FAISS.from_documents(texts, embeddings)
    return db


# Extract form field descriptions from HTML
# def get_form_field_descriptions(html_content):
#     soup = BeautifulSoup(html_content, 'html.parser')
#     form_fields = soup.find_all(['input', 'select', 'textarea'])
#     field_info = []

#     for field in form_fields:
#         field_data = {}
#         label = soup.find('label', {'for': field.get('id')})
#         if label:
#             field_data['label'] = label.get_text().strip().rstrip(':')
#         else:
#             placeholder = field.get('placeholder')
#             name = field.get('name')
#             description = placeholder if placeholder else name
#             if description:
#                 field_data['label'] = description.strip()

#         field_id = field.get('id') or field.get('name')
#         if field_id:
#             field_data['id'] = field_id

#         if 'label' in field_data and 'id' in field_data:
#             field_info.append(field_data)

#     return field_info
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

        # Ensure both label and id are present
        if 'label' in field_data and 'id' in field_data:
            field_info.append(field_data)

    return field_info


# Automate form filling
def filling_form(form_fields_info):
    try:
        llm = G4FLLM()
        db = process_data()

        structured_responses = []
        for field in form_fields_info:
            # Updated intelligent prompt
            prompt = (
                f"Extract the value for '{field['label']}' that matches the form field with ID '{field['id']}'. "
                f"Only provide the value. If no relevant value is found in the document, respond with an empty string."
            )

            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={'k': 4}),
                condense_question_prompt=PromptTemplate(input_variables=[], template=prompt),
            )

            def api_call():
                return conversation_chain.invoke({"question": prompt, "chat_history": []})

            result = retry_with_backoff(api_call)
            response_text = result['answer'].strip() if result['answer'] else ""

            # Post-process the response to ensure it is clean
            if any(
                phrase in response_text.lower()
                for phrase in ["i'm sorry", "no relevant value", "not found", "lo siento"]
            ):
                response_text = ""  # Leave the field blank

            # Append the cleaned response for the current field
            structured_responses.append({**field, "response": response_text})

        return structured_responses
    except Exception as e:
        print(f"Error in filling_form: {str(e)}")
        raise


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


# API endpoint for form auto-fill
@app.route('/api/auto_fill', methods=['POST'])
def auto_fill():
    if request.is_json:
        html_content = request.json.get('html_content', '')
        #url = request.json.get('url', '')
        # print("url :")
        # print(url)
        # if not url:
        #     return jsonify({"error": "Missing 'url' in request"}), 400
        #print(html_content)
        # if not html_content:
        #     return jsonify({"error": "Missing 'html_content' in request"}), 400
        try:
            #html_content = get_dynamic_html(url)
            #print(html_content)
            form_fields_info = get_form_field_descriptions(html_content)
            print(form_fields_info)
            structured_responses = filling_form(form_fields_info)
            print("structured_responses")
            print(structured_responses)
            #response_data = {field['id']: field['response'] for field in structured_responses}
            response_data = {
                field['id'] if field['id'] else field['label']: field['response']
                for field in structured_responses
            }
            print("print_responses")
            print(response_data)
            return jsonify(response_data), 200
        except Exception as e:
            print(f"Error in auto_fill: {str(e)}")
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid request format"}), 400


# Flask app entry point
if __name__ == '__main__':
    app.run(debug=True, port=5055)
