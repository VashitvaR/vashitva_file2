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
    llm = OpenAI(temperature=0, openai_api_key=openai.api_key )
  
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    # Loop through each chunk and summarize it
    for chunk in chunks:
        with get_openai_callback() as cb:
            response = chain.run(chunk)
            st.write(response)

  

    

st.title("PDF Text Extractor and Summarizer")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    chunks = extract_text(uploaded_file)
   
    st.subheader("Summarized Text:")
    summarize_text(chunks)
    

else:
    st.write("No PDF file uploaded.")



           

