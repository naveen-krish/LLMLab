import os
from constants import openai_key

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
import streamlit as st
from langchain.llms import OpenAI
from io import BytesIO


# Set page title and background image
st.set_page_config(page_title="DocuBot", page_icon=":robot:", layout="wide")
st.markdown(
    """
    <style>
     .title {
        color: red;  /* Change color of page title */
    }
    .reportview-container {
        background: url("./docs/q&a1.png");
        background-position: center top;
        background-repeat: no-repeat;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add image on the right corner
sidebar_image = "./docs/q&a.png"
st.sidebar.image(sidebar_image, use_column_width=True)

# Add text below the image
st.sidebar.write("Contact us:")
st.sidebar.write("Email: naveenkrishiv@gmail.com")

# Main content
st.title("DocuBot")

# Add your existing code here
#uploaded_file = st.file_uploader("Upload The File for Model Analysis")

os.environ["OPENAI_API_KEY"]=openai_key

embeddings = OpenAIEmbeddings()

# Use TextLoader for a single text file or DirectoryLoader for a directory of text files

st.header(" Upload The File for Model Analysis ")
uploaded_file = st.file_uploader("")

if uploaded_file:
   print("Filename: ", uploaded_file.name)
   file_contents = uploaded_file.read()  # Read the file contents as bytes
   current_dir = os.getcwd()
   file_path = os.path.join(current_dir, uploaded_file.name)
   with open(file_path, "wb") as file:
        file.write(file_contents)
   
#filePath = st.text_input('File Path')


#loader = TextLoader('docs/Info.txt')
#loader = DirectoryLoader('docs', glob="**/*.txt")

if uploaded_file:
    loader = TextLoader(uploaded_file.name)
    documents = loader.load()
 
 
    print(len(documents))
    input_text=st.text_input("Type your Query ")

    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(texts)
    llm=OpenAI()
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 1})
    )




    def query(q):
        #st.write("Query: ", q)
        st.write("Query Response : ", qa.run(q))
        
    if input_text:
        st.write(query(input_text))


    
    
    
#query(" What is the DOB of Naveen?")