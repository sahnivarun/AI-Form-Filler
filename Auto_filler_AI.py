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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# class G4FLLM(LLM):
#     @property
#     def _llm_type(self) -> str:
#         return "g4f"

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         client = Client()
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#         )
#         output = response.choices[0].message.content
#         if stop:
#             stop_indexes = (output.find(s) for s in stop if s in output)
#             min_stop = min(stop_indexes, default=-1)
#             if min_stop > -1:
#                 output = output[:min_stop]
#         return output

# Retrieve the custom Google GenAI LLM
def get_llm():
    os.environ["API_KEY"] = "AIzaSyDihkQrXCbVsaRb_4lkrIy7FmIulrVD77s"
    genai.configure(api_key=os.environ["API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    return CustomGenAIModel(model)

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

# def get_json_request(form_fields_info):
#     """
#     Creates a JSON object where keys are field labels and values are empty strings.
#     """
#     json_request = {field['label']: "" for field in form_fields_info }
#     logger.info(f"Generated JSON request: {json_request}")
#     return json_request

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

    logger.info(f"Generated JSON request with options: {json.dumps(json_request, indent=2)}")
    return json_request


def filling_form_single_request(form_fields_info):
    try:
        llm = get_llm()
        db = process_data()
        json_request = get_json_request(form_fields_info)

        # # Create a prompt for the LLM
        # prompt = (
        #     f"You are an AI assistant helping to fill out a job application form using the provided documents. "
        #     f"For each question in the form, generate an appropriate response based on the user's resume, documents, "
        #     f"and job application context. If a question asks about motivations, company-specific enthusiasm, or "
        #     f"open-ended responses (e.g., 'What gets you excited about joining this team?'), "
        #     f"generate a thoughtful answer based on common professional aspirations and values. "
        #     f"If the documents do not contain an answer for a question, provide a general but contextually relevant response. "
        #     f"Return the completed JSON object strictly in JSON format. Respond strictly in valid JSON format without any additional text, code blocks, or comments. "
        #     f"JSON object:\n\n{json.dumps(json_request)}"
        # )
        prompt = (
            f"You are an AI assistant helping to fill out a job application form using the provided documents. "
            f"Here is the form structure and its valid options (if applicable):\n\n"
            f"{json.dumps(json_request, indent=2)}\n\n"
            f"For each question, generate an appropriate response based on the user's resume, documents, "
            f"and job application context. If a question asks about motivations, company-specific enthusiasm, or "
            f"open-ended responses (e.g., 'What gets you excited about joining this team?'), "
            f"generate a thoughtful answer based on common professional aspirations and values. "
            f"Ensure that all responses strictly conform to the options provided (if any). "
            f"Return the completed JSON object strictly in JSON format. Respond strictly in valid JSON format without any additional text, code blocks, or comments."
        )


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
        llm_response = result['answer'].strip() if result['answer'] else "{}"

        logger.info(f"Raw LLM response: {llm_response}")

        # #debug
        # llm_response = {
        #     "resume": "Varun Sahni",
        #     "name": "Varun Sahni",
        #     "email": "varunsahni@tamu.edu",
        #     "phone": "(979) 344-3430",
        #     "location": "3803 Wellborn Rd, Apt 1221A, Bryan, Texas, 77801",
        #     "citizenship": "Indian",
        #     "org": "",
        #     "urls[LinkedIn]": "https://www.linkedin.com/in/varun-sahni-10134v786/",
        #     "urls[Other (GitHub, Portfolio, etc.)]": "",
        #     "eeo[gender]": "Male",
        #     "Hispanic or Latino": "No",
        #     "eeo[veteran]": "No",
        #     "eeo[disability]": "No",
        #     "Enter your full name": "Varun Sahni",
        #     "MM/DD/YYYY": "12/04/2024",
        #     "education": {
        #         "current_program": {
        #         "institution": "Texas A&M University",
        #         "degree": "Master of Science in Computer Science",
        #         "GPA": "4.0/4.0",
        #         "duration": "Aug 2023 – Aug 2025"
        #         },
        #         "previous_program": {
        #         "institution": "Delhi Technological University",
        #         "degree": "Bachelor of Engineering in Electrical Engineering",
        #         "GPA": "4.0/4.0",
        #         "duration": "Aug 2015 – June 2019"
        #         }
        #     },
        #     "professional_experience": [
        #         {
        #         "title": "Business Intelligence Developer",
        #         "organization": "NVIDIA",
        #         "duration": "Nov 2023 – Aug 2024",
        #         "technologies": ["Python", "React", "SQL", "Power BI"],
        #         "key_contributions": [
        #             "Automated tasks using Python and created insightful Power BI dashboards."
        #         ]
        #         },
        #         {
        #         "title": "Product Engineer",
        #         "organization": "Bharti Airtel Limited",
        #         "duration": "Jun 2019 – Jul 2023",
        #         "technologies": ["Java", "SQL", "Figma", "Power BI"],
        #         "key_contributions": [
        #             "Reduced customer handling time by 13% through application enhancements.",
        #             "Streamlined customer journeys, reducing service requests by 22%."
        #         ]
        #         }
        #     ],
        #     "technical_skills": {
        #         "programming_languages": ["Python", "SQL", "JavaScript", "C++", "R", "Java", "HTML", "CSS"]
        #     },
        #     "legal_authorization": "Yes",
        #     "visa_sponsorship": "Yes",
        #     "acknowledgements": "Acknowledged",
        #     "available_to_work": "Yes",
        #     "available_for_full_time": "Yes",
        #     "family_relationships_in_org": "No"
        #     }
        
        # logger.info(f"Updated LLM response: {llm_response}")

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

        # # Validate and clean up JSON response
        # try:
        #     if isinstance(llm_response, str):  # Ensure response is a string before parsing
        #         # Handle string responses (if any)
        #         if llm_response.startswith("```") and llm_response.endswith("```"):
        #             llm_response = llm_response.strip("```").strip()
        #         if llm_response.lower().startswith("json"):
        #             llm_response = llm_response[len("json"):].strip()

        #         # Parse JSON if it's still a string
        #         filled_data = json.loads(llm_response)
        #     elif isinstance(llm_response, dict):
        #         # If already a dictionary, use it directly
        #         filled_data = llm_response
        #     else:
        #         raise ValueError("Unexpected response format from LLM.")
        # except json.JSONDecodeError as e:
        #     logger.error(f"Error parsing JSON response: {llm_response}")
        #     raise ValueError("Invalid JSON response from LLM.") from e

        # logger.info(f"Cleaned LLM response: {llm_response}")

        logger.info(f"Parsed JSON response: {filled_data}")

        # Map JSON responses to form fields
        for field in form_fields_info:
            label = field.get('label', '').strip()
            if label in filled_data:
                field['response'] = filled_data[label]
            else:
                field['response'] = ""

        logger.info(f"Final filled form fields: {form_fields_info}")
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
            logger.info(f"Extracted Form Fields Info: {form_fields_info}")

            # Fill form with a single LLM call
            structured_responses = filling_form_single_request(form_fields_info)

            # Build the response data to fill form fields
            response_data = {}
            for field in structured_responses:
                field_id = field.get('id', '')
                response_data[field_id] = field.get('response', '')

            logger.info(f"Final Response Data: {response_data}")
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
        llm = get_llm()
        # Load user's information from documents
        db = process_data()

        # Create a prompt for the LLM
        prompt = (
            f"Using the following job description and your knowledge about me from my documents, "
            f"please write a personalized cover letter for this job application. "
            f"Job Description:\n{job_description}\n\n"
            f"Cover Letter:"
        )
        #logger.info(f"Prompt sent to LLM: {prompt}")

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
        cover_letter = result['answer'].strip() if result['answer'] else ""
        #logger.info(f"Generated Cover Letter: {cover_letter}")

        return cover_letter

    except Exception as e:
        logger.error(f"Error in create_cover_letter: {e}")
        raise
    
      
# Flask app entry point
if __name__ == '__main__':
    app.run(debug=True, port=5055)