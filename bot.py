import streamlit as st
import pdfplumber
import openai
from dotenv import load_dotenv
from openai import OpenAI, Document, CharacterTextSplitter, load_summarize_chain

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
    return text

# Define the function to generate a summary using OpenAI's GPT-3 API
def generate_summary(text):
    # Instantiate the LLM model
    llm = OpenAI(temperature=0, openai_api_key=openai.api_key)

    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)

    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]

    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Streamlit app code
st.title("PDF Text Extractor and Summarizer")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
    extracted_text = extract_text(uploaded_file)
    
    st.subheader("Summarized Text:")
    summarized_text = generate_summary(extracted_text)
    st.text(summarized_text)
else:
    st.write("No text found in the PDF file.")
