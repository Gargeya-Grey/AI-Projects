from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain_openai.llms import OpenAI # It could be any LLM that is already integrated with Langchain

import os
import chainlit as cl
from utils import OPENAI_API_KEY
import textwrap
from typing_extensions import Concatenate

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

## Read and Preprocess the input PDF
@cl.on_chat_start
async def on_chat_start():
    files = None

    while files == None:
        files = await cl.AskFileMessage(
        content="Please upload a PDF to talk to!",
        accept=['pdf'],
        max_size_mb=30,
        timeout=180,
      ).send()

    file = files[0]

    msg = cl.Message(content=f"Reading the passed pdf:>> {file.name}", disable_feedback=True)
    await msg.send()
    # Reading your given file
    pdfreader = PdfReader(f"{file.path}")

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
    document_search = await cl.make_async(FAISS.from_texts)(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    msg.content = f"PDF>> {file.name} Loaded and processed successfully!"
    await msg.update()

    cl.user_session.set("chain", chain)
    cl.user_session.set("document_search", document_search)

@cl.on_message
async def main(query: cl.Message):
    chain = cl.user_session.get("chain")
    document_search = cl.user_session.get("document_search")

    docs = document_search.similarity_search(query.content)
    answer = chain.run(input_documents = docs, question=query.content)

    await cl.Message(content=answer).send()



