import streamlit as st # Import the Streamlit library for building web applications
import PyPDF2 # Import PyPDF2 for extracting text from PDF files
import io # Import io for handling in-memory binary streams (e.g., uploaded PDF files)
import os # Import os for interacting with the operating system, specifically for environment variables
from openai import OpenAI # Import the OpenAI client library to interact with OpenAI's API
from dotenv import load_dotenv # Import load_dotenv to load environment variables from a .env file

# Load environment variables from the .env file.
# This should be called at the very beginning of your script,
# before attempting to access any environment variables.
load_dotenv()

# Set Streamlit page configuration for better aesthetics and layout.
# page_title: Sets the title displayed in the browser tab.
# page_icon: Sets the favicon for the browser tab.
# layout: Sets the overall layout of the app; "centered" keeps content in the middle, "wide" uses full width.
st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃", layout="centered")

# Display the main title and a brief description for the app.
st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

# Retrieve the OpenAI API key from environment variables.
# It's crucial that OPENAI_API_KEY is set either in your system's environment
# or in a .env file in the project root, which `load_dotenv()` will then load.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create a file uploader widget.
# 'type' specifies accepted file extensions.
uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

# Create a text input for the user to specify a job role. This is optional.
job_role = st.text_input("Enter the job role you're targeting (optional)")

# Create a button to trigger the resume analysis.
analyze = st.button("Analyze Resume")

# --- Function Definitions ---

def create_resume_critique_prompt(resume_text: str, job_context: str) -> str:
    """
    Generates a detailed, structured prompt for the AI to critique a resume.

    Args:
        resume_text (str): The extracted text content of the resume.
        job_context (str): The job role the user is targeting, or an empty string if not provided.

    Returns:
        str: The complete prompt string formatted for the OpenAI API.
    """
    # If no job context is provided, set a default message for the AI.
    if not job_context:
        job_context = "a general professional role based on the resume's content"

    # This is the powerful, structured prompt template.
    # It guides the AI to provide feedback in a specific, actionable, and comprehensive format.
    prompt = f"""
**TASK:**
Provide a comprehensive, constructive, and actionable critique of the user's resume.
Analyze the resume against the provided job context.

**INPUTS:**
1.  **RESUME_TEXT:**
    ```
    {resume_text}
    ```
2.  **JOB_CONTEXT:** The resume should be evaluated for a candidate applying for: **{job_context}**

**OUTPUT STRUCTURE:**
Present your feedback in Markdown format, following this exact structure:

---

### **Overall Score: [Provide a score out of 100]**

* **Clarity & Readability:** [Score / 25]
* **Impact & Achievements:** [Score / 35]
* **Relevance to Job:** [Score / 25]
* **Formatting & ATS Friendliness:** [Score / 15]

---

### **Executive Summary**
(A 2-3 sentence high-level overview. Start with the strongest aspect of the resume and then state the single most important area for improvement.)

---

### **Section-by-Section Analysis**

**1. General Formatting & Readability**
* **✅ What's Good:** (Comment on clean layout, good use of white space, clear fonts, etc.)
* **🔍 Areas for Improvement:** (Comment on inconsistent formatting, dense text blocks, unprofessional fonts, length, or typos/grammar.)
* **💡 Actionable Suggestion:** (Provide a specific, actionable tip.)

**2. Professional Summary / Objective**
* **✅ What's Good:** (Comment on its impact and clarity.)
* **🔍 Areas for Improvement:** (Analyze if it's generic, passive, or not tailored to the **JOB_CONTEXT**.)
* **💡 Actionable Suggestion:** (Provide a rewritten example tailored to the job context.)

**3. Skills Section**
* **✅ What's Good:** (Comment on relevant skills and good organization.)
* **🔍 Areas for Improvement:** (Are the skills relevant? Are they just buzzwords? Are they backed up by experience?)
* **💡 Actionable Suggestion:** (Suggest grouping skills or adding specific keywords from the job context.)

**4. Professional Experience**
* **✅ What's Good:** (Praise the use of strong action verbs or quantified results.)
* **🔍 Areas for Improvement:** (Identify bullet points that describe duties instead of achievements. Point out where metrics are missing.)
* **💡 Actionable Suggestion (Rewrite Example):** Show how to transform a weak bullet point into an impactful one using the STAR method (Situation, Task, Action, Result). For example, turn "Responsible for social media" into "Grew organic social media engagement by 45% over 6 months by executing a new content strategy."

**5. Projects Section**
* **✅ What's Good:** (Comment on relevant project choices.)
* **🔍 Areas for Improvement:** (Do descriptions state the problem, tech stack, and outcome? Are there links to GitHub/live demos?)
* **💡 Actionable Suggestion:** (Suggest adding quantifiable outcomes or a link to the repository.)

---

### **Keyword & Skill Alignment for: {job_context}**

* **Keywords Present:** (List key skills/technologies from the job context that you found in the resume.)
* **Keywords Missing:** (List important skills/technologies from the job context that are missing and suggest where they could be added.)
* **Overall Recommendation:** (A final sentence on how well the resume is tailored.)
"""
    return prompt

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts text from an uploaded PDF file.

    Args:
        pdf_file: A file-like object (from Streamlit's file_uploader) representing the PDF.

    Returns:
        str: The concatenated text from all pages of the PDF.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n" # Extract text from each page and add a newline
    return text

def extract_text_from_file(uploaded_file) -> str:
    """
    Extracts text from an uploaded file, handling both PDF and TXT types.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        str: The extracted text content.
    """
    if uploaded_file.type == "application/pdf":
        # For PDF, read the file into an in-memory binary stream and then process with PyPDF2.
        # uploaded_file.read() returns bytes, which io.BytesIO can wrap.
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    # For TXT files, directly read and decode the bytes into a UTF-8 string.
    return uploaded_file.read().decode("utf-8")

# --- Main Application Logic ---

# This block executes when the "Analyze Resume" button is clicked AND a file has been uploaded.
if analyze and uploaded_file:
    # Use a Streamlit spinner to indicate that analysis is in progress.
    with st.spinner("Analyzing your resume... This might take a moment."):
        try:
            # Extract text content from the uploaded file.
            file_content = extract_text_from_file(uploaded_file)

            # Check if the extracted content is empty after stripping whitespace.
            if not file_content.strip():
                st.error("Error: Could not extract any readable content from the uploaded file. Please ensure it's a valid PDF or TXT.")
                st.stop() # Stop further execution if no content is found.

            # Generate the specific prompt for the AI using the extracted resume text and job role.
            prompt = create_resume_critique_prompt(file_content, job_role)

            # Initialize the OpenAI client.
            # The api_key parameter is explicitly passed here, ensuring it uses the key
            # loaded from the environment variable.
            client = OpenAI(api_key=OPENAI_API_KEY)

            # Make the API call to OpenAI's chat completions endpoint.
            # model: Specifies the AI model to use (e.g., "gpt-4o" for advanced capabilities).
            # messages: A list of message objects defining the conversation.
            #   - "system" message sets the AI's persona.
            #   - "user" message contains the prompt with the resume text and job context.
            # temperature: Controls the randomness of the AI's response (0.7 is a good balance).
            # max_tokens: Limits the length of the AI's response.
            response = client.chat.completions.create(
                model="gpt-4o",  # You could also use "gpt-3.5-turbo" if preferred or for cost efficiency.
                messages=[
                    {"role": "system", "content": "You are an expert resume reviewer with millions of years in reviewing resumes. Provide detailed, actionable, and structured feedback following the user's requested output format. Be critical yet constructive."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500 # Increased max_tokens slightly for a more comprehensive critique.
            )

            # Display the analysis results in Markdown format.
            # The AI's response is expected to be in Markdown as per the prompt structure.
            st.markdown("### 📋 Analysis Results")
            st.markdown(response.choices[0].message.content)

        except Exception as e:
            # Catch any exceptions that occur during file processing or API calls.
            st.error(f"An unexpected error occurred: {str(e)}")
