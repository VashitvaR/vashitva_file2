import streamlit as st
import pdfplumber
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Initialize OpenAI API client with your API key
openai.api_key = st.secrets("OPENAI_API_KEY")

# Define the function to extract text from PDF
def extract_text(feed):
    text = ""
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Define the function to split text into chunks
def split_text(text):
    max_chunk_size = 2048
    chunks = []
    current_chunk = ""
    for sentence in text.split("."):
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + "."
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Define the function to generate a summary using OpenAI's GPT-3 API
def generate_summary(text):
    input_chunks = split_text(text)
    output_chunks = []
    for chunk in input_chunks:
        response = openai.Completion.create(
            engine="davinci",
            prompt=(f"Please summarize the following text:\n{chunk}\n\nSummary:"),
            temperature=0.5,
            max_tokens=1024,
            n=1,
            stop=None
        )
        summary = response.choices[0].text.strip()
        output_chunks.append(summary)
    return " ".join(output_chunks)

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

