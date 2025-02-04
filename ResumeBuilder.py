#This block of code imports the required libraries and dependencies
import streamlit as st
from groq import Groq
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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



#This function calls llama3.3 and prompts it to take in a resume and makes suggestions on how to improve the resume 
def ResumeAnalysis(resume):
    client = Groq(api_key= api_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"Please analyze this resume {resume} and make suggestions on what you think should be done to enchance it, don't neglect to mention any additional skills you think should be acquired and added to the resume. Don't acknowledge this prompt or ask if you can offer any more help, just return strictly the recommendations"
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    response = ""
    for chunk in completion:
         if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
    return response



#The following block of streamlit code is for deployment of the application
st.title("Resume Builder")

api_key = "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"
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file:
    resume_text = extract_text(uploaded_file)
    processed_resume = pre_processing(resume_text)


choice = st.selectbox("What do you want to do?", ["",'Analyze your resume', 'Tailor your resume to a job'])


#The following block of code gives users the choice between tailoriing the resume or analyzing it
if choice == "":
    pass

elif choice  == 'Analyze your resume':
    with st.spinner("Processing..."):
        resume_analysis = ResumeAnalysis(processed_resume)
        st.write(resume_analysis)   
    
    
else:
    job_description = st.text_area("Enter the job description:")
    
    if st.button("Generate Tailored Resume"):
        if not all([api_key, uploaded_file, job_description]):
            st.error("Please provide all required inputs")
        else:
            with st.spinner("Processing..."):
                
                new_resume = ResumeBuilder(processed_resume, job_description, api_key)
                
                st.download_button(
                    "Download Tailored Resume",
                    new_resume,
                    "tailored_resume.txt",
                    "text/plain"
                )
                st.success("Resume generated successfully!")