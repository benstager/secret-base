import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.chains import RetrievalQA 
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from PyPDF2 import PdfReader
from tempfile import NamedTemporaryFile
import os

def generate_response_txt_file(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
      # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

def generate_response_pdf_file(uploaded_file, openai_api_key, query_text):

    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)

    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_texts(texts, embeddings)

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    
    return qa.run(query_text)

def generate_response_docx_file(uploaded_file, openai_api_key, query_text):

    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
        tmp.write(bytes_data)                      # write data from the uploaded file into it
        documents = Docx2txtLoader(tmp.name).load()        # <---- now it works!       

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key='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')
        # Create a vectorstore from documents
        db = Chroma.from_documents(documents, embeddings)

        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key='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'), chain_type='map_reduce', retriever=retriever)
        return qa.run(query_text)
    os.remove(tmp.name)
# Page title
st.set_page_config(page_title='🥷🏻 Agent E 🥷🏻')
st.title('🥷🏻 Agent E 🥷🏻')

#create dropdown menu here
choice = st.selectbox("Choose your file extension", options=['txt','pdf','docx'])

# footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by<a style='display: block; text-align: center;' href="https://github.com/benstager" target="_blank">Benjamin Stager</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader('Upload an article', type = choice)
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            if choice == 'txt':
                response = generate_response_txt_file(uploaded_file, openai_api_key, query_text)
            if choice == 'pdf':
                response = generate_response_pdf_file(uploaded_file, openai_api_key, query_text)
            if choice == 'docx':
                response = generate_response_docx_file(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key
            
if len(result):
    st.info(response)