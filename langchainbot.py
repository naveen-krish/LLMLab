import os


from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI
import streamlit as st
from langchain.llms import OpenAI
from io import BytesIO
import base64
from PIL import Image

# Set page title and background image


# Set page title and background image
st.set_page_config(page_title="DocuBot", page_icon="bot1.jpeg", layout="centered")

# Add image on the right corner
sidebar_image = "./docs/q&a.png"
st.sidebar.image(sidebar_image, use_column_width=True)

# Add text below the image
st.sidebar.header("Author : Naveen ")
st.sidebar.write("Email: naveenkrishiv@gmail.com")

# Main content
#st.markdown("<h1 style='text-align: center; color: black;'>DocuBot</h1>", unsafe_allow_html=True)
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
st.markdown("<h1 style='text-align: center; color: green;'> DocuBot </h1>", unsafe_allow_html=True)

#st.title("DocuBot")<i class="fa-regular fa-robot" style="color: #0c0c0d;"></i>

# Add your existing code here
#uploaded_file = st.file_uploader("Upload The File for Model Analysis")

os.environ["OPENAI_API_KEY"]=st.secrets["openai_key"]
embeddings = OpenAIEmbeddings()

# Use TextLoader for a single text file or DirectoryLoader for a directory of text files

st.markdown("<h5 style='text-align: center; color: coral;'>File Up and Let's Query </h5>", unsafe_allow_html=True)

#st.text(" Upload File for Model Analysis ")
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
    input_text=st.text_input(" Type your Query 👇")

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
        st.write('<p style="color:DarkSlateGray;">Query Response : </p>',qa.run(q),unsafe_allow_html=True) 
         
    if input_text:
        st.write(query(input_text))
