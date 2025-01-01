!pip install langchain langchain-experimental langchain-community langchain-openai openai chromadb pypdf sentence_transformers gradio langchain-together

import os
from typing import List, Tuple, Dict, Any


from langchain_community.document_loaders import PyPDFLoader

# Vector store
from langchain_community.vectorstores import Chroma

# LLM
from langchain_openai import OpenAI

# Load the PDF document
loader = PyPDFLoader("/content/blabla.pdf")  # Upload the document you wish to use for the chatbot
pages: List[Any] = loader.load()

# Check the number of pages
print(len(pages))

# Display content of a specific page
print(pages[16])

# Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents: List[Any], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs: List[Any] = text_splitter.split_documents(documents)
    return docs

# Split the pages into smaller chunks
new_pages: List[Any] = split_docs(pages)
print(len(new_pages))

# Display content of specific chunks
print(new_pages[500].page_content)
print(new_pages[499].page_content)

# Embedding function
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store from the documents
db = Chroma.from_documents(new_pages, embedding_function)

# LLM setup
from langchain_together import Together

llm = Together(
    model="meta-llama/Llama-2-70b-chat-hf",
    max_tokens=256,
    temperature=0,
    top_k=1,
    together_api_key="Need API KEY here"  # Replace with your actual API key(can be generated with subscription)
)

# Retriever setup
retriever = db.as_retriever(similarity_score_threshold=0.9)

# Prompt template
from langchain.prompts import PromptTemplate

prompt_template: str = """Please answer question- Try explaining in simple words. Answer in less than 100 words. If you don't know the answer simply respond as "Don't know man!"
CONTEXT: {context}
QUESTION: {question}"""

PROMPT = PromptTemplate(template=f"[INST] {prompt_template} [/INST]", input_variables=["context", "question"])

# RetrievalQA chain setup
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    input_key='query',
    return_source_documents=True,
    chain_type_kwargs={"prompt":PROMPT},
    verbose=True
)

# Query input and response
query: str = input("Enter your query: ")
response: Dict[str, Any] = chain(query)
print(response['result'])

## UI to be done later