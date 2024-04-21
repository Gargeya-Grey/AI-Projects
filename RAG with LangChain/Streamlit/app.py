from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain_openai.llms import OpenAI # It could be any LLM that is already integrated with Langchain

import os
import streamlit as st


with st.sidebar:
    openai_api_key = st.text_input("OPENAI API Key", key="pdf_qa_api_key", type="password")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Read and process the input data that is recieved from the pdf file
st.title("Question your Document")

uploaded_file = st.file_uploader("Uplaod your PDF file", type=("pdf"))

question = st.text_input(
    "Ask Something about the uploaded document",
    placeholder="Can you give me the summary of the document?",
    disabled=not uploaded_file,
)

if uploaded_file and question and openai_api_key:
    if not os.path.exists("tempDir"): os.mkdir("tempDir")
    with open(os.path.join("tempDir", uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
    file_path = str(os.path.join("tempDir", uploaded_file.name))
    pdfreader = PdfReader(file_path)

    # reading text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    
    # For the first step, the input needs to be splitted with respect to token length constrictions
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 800, # Can change according to the LLM that we one might use
        chunk_overlap = 200,
        length_function = len,
    )

    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = document_search.similarity_search(question)
    answer = chain.run(input_documents = docs, question=question)

    st.write("### Answer")
    st.write(answer)
