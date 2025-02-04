#This block of code imports the required libraries and dependencies
import streamlit as st
from groq import Groq
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


#This function handles pre-processing of pdfs 
def pre_processing(pdf_content):
    document = pdf_content.lower()
    document = re.sub(r'[^\w\s]',"",document)
    
    
    stopwords_set = set(stopwords.words("english")) #Stopword removal
    lemmatizer = WordNetLemmatizer()
    
    #Tokenization and Lemmatization
    tokens = word_tokenize(document)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]
    return ' '.join(tokens)


#This function prompts the llama3.3 model to tailor the resume
def ResumeBuilder(resume, job_desc, api_key):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": f"Please take in this job description {job_desc} and tailor make this resume {resume} to fit it. Don't suggest any recommendations, just do them and make sure the resume is the only thing in the output"
        }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True
    )
    
    return ''.join(chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content)


#The following block of streamlit code is for deployment of the application
st.title("Resume Builder")

api_key = "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"
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
job_description = st.text_area("Enter the job description:")

if st.button("Generate Tailored Resume"):
    if not all([api_key, uploaded_file, job_description]):
        st.error("Please provide all required inputs")
    else:
        with st.spinner("Processing..."):
            resume_text = extract_text(uploaded_file)
            processed_resume = pre_processing(resume_text)
            new_resume = ResumeBuilder(processed_resume, job_description, api_key)
            
            st.download_button(
                "Download Tailored Resume",
                new_resume,
                "tailored_resume.txt",
                "text/plain"
            )
            st.success("Resume generated successfully!")