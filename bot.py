import streamlit as st
import pdfplumber
import openai
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

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

def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    if len(embeddings) != 0:
        knowledgeBase = FAISS.from_texts(chunks, embeddings)
        return knowledgeBase


def main():
    st.title("📄PDF Summarizer")
    st.divider()

    uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

    if uploaded_file:
        text = extract_text(uploaded_file)
        # Create the knowledge base object
        knowledgeBase = process_text(text)

        query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

        if query:
            docs = knowledgeBase.similarity_search(query)
            OpenAIModel = "gpt-3.5-turbo-16k"
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('Summary Results:')
            st.write(response)

if __name__ == "__main__":
    main()
