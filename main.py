import streamlit as st
import PyPDF2
import io
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃", layout="centered")

st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you're targeting (optional)")
analyze = st.button("Analyze Resume")
def create_resume_critique_prompt(resume_text: str, job_context: str) -> str:
    """
    Generates a detailed, structured prompt for the AI to critique a resume.
    """
    # If no job context is provided, set a default message for the AI.
    if not job_context:
        job_context = "a general professional role based on the resume's content"

    # This is the powerful, structured prompt template.
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

*   **Clarity & Readability:** [Score / 25]
*   **Impact & Achievements:** [Score / 35]
*   **Relevance to Job:** [Score / 25]
*   **Formatting & ATS Friendliness:** [Score / 15]

### **Executive Summary**
(A 2-3 sentence high-level overview. Start with the strongest aspect of the resume and then state the single most important area for improvement.)

---

### **Section-by-Section Analysis**

**1. General Formatting & Readability**
*   **✅ What's Good:** (Comment on clean layout, good use of white space, clear fonts, etc.)
*   **🔍 Areas for Improvement:** (Comment on inconsistent formatting, dense text blocks, unprofessional fonts, length, or typos/grammar.)
*   **💡 Actionable Suggestion:** (Provide a specific, actionable tip.)

**2. Professional Summary / Objective**
*   **✅ What's Good:** (Comment on its impact and clarity.)
*   **🔍 Areas for Improvement:** (Analyze if it's generic, passive, or not tailored to the **JOB_CONTEXT**.)
*   **💡 Actionable Suggestion:** (Provide a rewritten example tailored to the job context.)

**3. Skills Section**
*   **✅ What's Good:** (Comment on relevant skills and good organization.)
*   **🔍 Areas for Improvement:** (Are the skills relevant? Are they just buzzwords? Are they backed up by experience?)
*   **💡 Actionable Suggestion:** (Suggest grouping skills or adding specific keywords from the job context.)

**4. Professional Experience**
*   **✅ What's Good:** (Praise the use of strong action verbs or quantified results.)
*   **🔍 Areas for Improvement:** (Identify bullet points that describe duties instead of achievements. Point out where metrics are missing.)
*   **💡 Actionable Suggestion (Rewrite Example):** Show how to transform a weak bullet point into an impactful one using the STAR method (Situation, Task, Action, Result). For example, turn "Responsible for social media" into "Grew organic social media engagement by 45% over 6 months by executing a new content strategy."

**5. Projects Section**
*   **✅ What's Good:** (Comment on relevant project choices.)
*   **🔍 Areas for Improvement:** (Do descriptions state the problem, tech stack, and outcome? Are there links to GitHub/live demos?)
*   **💡 Actionable Suggestion:** (Suggest adding quantifiable outcomes or a link to the repository.)

---

### **Keyword & Skill Alignment for: {job_context}**

*   **Keywords Present:** (List key skills/technologies from the job context that you found in the resume.)
*   **Keywords Missing:** (List important skills/technologies from the job context that are missing and suggest where they could be added.)
*   **Overall Recommendation:** (A final sentence on how well the resume is tailored.)
"""
    return prompt
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))  # ✅ Call read()
    return uploaded_file.read().decode("utf-8")  # ✅ Works for .txt

if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)

        if not file_content.strip():
            st.error("File does not have any content.")
            st.stop()

        prompt =create_resume_critique_prompt(file_content,job_role)

        client = OpenAI(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-3.5-turbo" if gpt-4o is unavailable
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer with millions of years in reviewing resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        st.markdown("### 📋 Analysis Results")
        st.markdown(response.choices[0].message.content)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
