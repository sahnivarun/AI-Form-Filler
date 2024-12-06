Auto Filler AI
Auto Filler AI is an automation tool that assists in generating cover letters and filling out forms on various websites by leveraging AI technology. This tool integrates with a Google Chrome extension and a Python backend to simplify tasks like filling out application forms using your resume data or generating tailored cover letters based on job descriptions.

Features
Cover Letter Generator: Automatically generates a cover letter by parsing the job description URL.
Form Filler: Automatically fills out application forms on websites using your resume (either local file or Google Drive link).
Easy-to-use Chrome Extension: Integrates seamlessly with Google Chrome for quick and efficient form filling.
Prerequisites
Before getting started, ensure you have the following installed:

Python 3.x (preferably 3.8+)
Google Chrome (for the extension)
pip (Python package manager)

Installation Guide
Follow these steps to set up the Auto Filler AI system and Chrome extension.
________________________________________
1. Extract the ZIP Folder
•	Download the project ZIP file and extract it to your desired location on your computer.
2. Create Your Info Document
•	Inside the extracted folder, you’ll find a Word document named document.docx in the home directory. This document contains the template that was used for input. You can edit this document, enter your details, generate a PDF, and then place it into the info directory inside the main project directory.
•	Once done, make sure to remove document.pdf from the info directory, as it is the previous document that contains my details. The LLM will use the information in your new document.pdf that contains your details to process and generate responses.
3. Set Up Virtual Environment
•	Open a terminal or command prompt in the root directory of the extracted folder.
•	Create a new Python virtual environment by running the following command:
python -m venv venv
•	Activate the virtual environment:
o	On Windows:
venv\Scripts\activate
o	On macOS/Linux:
source venv/bin/activate
4. Install Dependencies
•	Once the virtual environment is activated, install the required Python dependencies:
pip install -r requirements.txt
•	If any additional dependencies are needed, install them manually using pip:
pip install <package-name>
5. Run the Backend
•	Now that your environment is set up, run the backend server by executing:
python Auto_filler_AI.py
•	This will start the backend process and keep it running to support the Chrome extension.
________________________________________
6. Install the Chrome Extension
•	Open Google Chrome and go to the Extensions page by navigating to chrome://extensions/:
o	Chrome > Settings > Extensions
•	Enable "Developer mode" in the top-right corner.
•	Click "Load Unpacked" and select the chrome-extension folder located inside the root directory of the project.
•	The extension will now be installed and available in your browser. You can pin the extension icon to the address bar for easier access.
________________________________________

7. Using the Extension
•	Filling a Cover Letter:
o	Navigate to a job description webpage.
o	Copy the URL of the page containing the job description.
o	Open the extension and paste the URL into the provided field.
o	Click "Generate Cover Letter." The extension will generate a cover letter based on the job description.
•	Filling Out a Form:
o	To fill out a form on a website, you can either upload your resume from your local computer or provide a Google Drive link to your resume.
o	Once the resume is attached, or the URL is entered, click on the "Fill Form" button.
o	The extension will autofill the form fields using the resume data.
o	After filling out the form, you can manually review the fields and make any necessary adjustments.
o	Once satisfied, click "Submit" to submit the form.
