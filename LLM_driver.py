import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

os.environ["OPENAI_API_KEY"] = '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'

documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir("/Users/benstager/Desktop/LLM_documents/new_docs/"):
    if file.endswith(".pdf"):
        pdf_path = "/Users/benstager/Desktop/LLM_documents/new_docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())


# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()

# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

st.title('Buyers Products Demo')
st.write('Hello')
chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print("Hi! I'm the Buyers Bot. How can I assist you today?. If you have a specific question, please go ahead and ask. Otherwise, I'll suggest some information that may be helpful.")
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa.invoke(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))