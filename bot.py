import streamlit as st
import pdfplumber
import openai
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API client with your API key
openai.api_key = st.secrets['OPENAI_KEY']

# Define the function to extract text from PDF



def extract_text(feed):
    text = ""
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    chunks = text_splitter.split_text(text)        
    return chunks




def summarize_text(chunks):
    summarizer = load_summarize_chain('t5-base')
    summary = summarizer.run(chunks)
    return summary


st.title("PDF Text Extractor and Summarizer")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    chunks = extract_text(uploaded_file)
    st.subheader("Extracted Text Chunks:")
    for chunk in chunks:
        st.write(chunk)
    
    st.subheader("Summarized Text:")
    summary = summarize_text(chunks)
    st.write(summary)

else:
    st.write("No PDF file uploaded.")



           

