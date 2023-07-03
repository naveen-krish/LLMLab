import os
#from constants import openai_key

from langchain.document_loaders import DirectoryLoader, TextLoader,PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
import streamlit as st
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
from io import BytesIO
import base64
from PIL import Image
from langchain.document_loaders import PyPDFLoader 


# Set page title and background image
st.set_page_config(page_title="DocuBot", page_icon="bot1.jpeg", layout="centered")
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bg5.jpeg') 

# Add image on the right corner
sidebar_image = "./docs/q&a.png"
st.sidebar.image(sidebar_image, use_column_width=True)

# Add text below the image
st.sidebar.header("Author : Naveen ")
st.sidebar.write("Email: naveenkrishiv@gmail.com")

# Main content
#st.markdown("<h1 style='text-align: center; color: white;'>DocuBot</h1>", unsafe_allow_html=True)
#image_path = "docs/q&a1.png"
#image = Image.open(image_path)
#resized_image = image.resize((50, 200))
#st.image(resized_image, caption="DocuBot", use_column_width=True)

# Define the CSS style for the bot icon
css = """
.bot-icon {
    display: inline-block;
    vertical-align: middle;
    margin-right: 10px;
}
"""

# Add the custom CSS style
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


import streamlit as st

# Add the Font Awesome CSS
st.markdown('<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">', unsafe_allow_html=True)

# Render the icon and title
st.markdown("<h1 style='text-align: center; color: white;'> DocuBot </h1>", unsafe_allow_html=True)

#st.title("DocuBot")<i class="fa-regular fa-robot" style="color: #0c0c0d;"></i>

# Add your existing code here
#uploaded_file = st.file_uploader("Upload The File for Model Analysis")

#os.environ["OPENAI_API_KEY"]=openai_key

embeddings = OpenAIEmbeddings()

# Use TextLoader for a single text file or DirectoryLoader for a directory of text files

st.markdown("<h5 style='text-align: center; color: yellow;'>Upload File for Model Analysis</h5>", unsafe_allow_html=True)

#st.text(" Upload File for Model Analysis ")
#uploaded_file = st.file_uploader("")
#uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
pdf_dir_path=st.text_input(" Enter the Dir Path: ")


#if uploaded_file:
 #  print("Filename: ", uploaded_file.name)
 #  file_contents = uploaded_file.read()  # Read the file contents as bytes
  # current_dir = os.getcwd()
 #  file_path = os.path.join(current_dir, uploaded_file.name)
 #  with open(file_path, "wb") as file:
  #      file.write(file_contents)
   
#filePath = st.text_input('File Path')


#loader = TextLoader('docs/Info.txt')
#loader = DirectoryLoader('docs', glob="**/*.txt")

loader = PyPDFDirectoryLoader(pdf_dir_path)
docs = loader.load()

#if uploaded_file:
    #loader = TextLoader(uploaded_file.name)
    #documents = loader.load()
 #   loader = PyPDFLoader(file_path)
  #  pages = loader.load_and_split()
 
 #   pdf_reader = PdfReader(uploaded_file)
 #   text = ""
#  for page in pdf_reader.pages:
#        text += page.extract_text()
 
print(len(docs))
input_text=st.text_input("Type your Query ")


#print(pages)
llm=OpenAI()
    
    
docsearch = Chroma.from_documents(docs, embeddings)
#knowledge_base = FAISS.from_texts(chunks, embeddings)
print(' after docsearch chroma...')

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 1})
)




def query(q):
    st.write("Query: ", q)
    st.write("Query Response : ", qa.run(q))
    
if input_text:
    st.write(query(input_text))


    
    
    
#query(" What is the DOB of Naveen?")
