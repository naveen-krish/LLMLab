import streamlit as st
from PIL import Image
import pytesseract
import re
import json
from sqlalchemy import create_engine
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
import os
import sys

# Assuming 'constants' is a module where 'openai_key' is defined.
#from constants import openai_key

# Function to parse the schema file
def parse_schema_file(schema_file):
    """Parse the schema file and extract tables, columns, and relationships."""
    schema_data = json.load(schema_file)
    tables = schema_data['tables']
    relationships = schema_data['relationships']
    return tables, relationships

# Function to generate a dynamic prompt based on schema
def generate_prompt(tables, relationships):
    table_descriptions = []
    for table in tables:
        columns = ', '.join(table['columns'])
        table_descriptions.append(f"{table['name']} table has columns {columns}.")

    relationship_descriptions = []
    for rel in relationships:
        relationship_descriptions.append(f"{rel['from_table']} is related to {rel['to_table']} through {rel['column']} column.")

    dynamic_prompt = """
    you are a very intelligent AI assistant who is expert in identifying relevant questions from user and converting them into SQL queries to generate correct answer.
    Please use the below context to write the MySQL queries.
    context:
    """ + ' '.join(table_descriptions + relationship_descriptions) + """
    As an expert you must use joins whenever required.
    """
    return dynamic_prompt

# Function to parse text and extract relationships (for PNG processing)
def parse_extracted_text(text):
    relationships = re.findall(r'(\w+)\s+[\-\>\<]+\s+(\w+)', text)
    return [{"from_table": rel[0], "to_table": rel[1], "column": "N/A"} for rel in relationships]

# Streamlit UI
st.title("SQL Query BOT")
schema_file = st.file_uploader("Upload schema diagram file", type=['json', 'png'])

tables = []
relationships = []

if schema_file is not None:
    # Process JSON file
    if schema_file.name.endswith('.json'):
        tables, relationships = parse_schema_file(schema_file)
    # Process PNG file
    elif schema_file.name.endswith('.png'):
        image = Image.open(schema_file)
        extracted_text = pytesseract.image_to_string(image)
        relationships = parse_extracted_text(extracted_text)
      #  st.text_area("Extracted Text (for debugging)", extracted_text, height=150)  # Optional: for debugging

    prompt_template = generate_prompt(tables, relationships)
else:
    st.warning("Please upload a schema diagram file.")
    prompt_template = ""  # Default or empty prompt template

user_query = st.text_input("Enter your question:", "")

if st.button("Get Response") and prompt_template:
    # Setup connection to the database
    connection_string = "mysql+pymysql://root:secret@localhost:3306/genai"
    db_engine = create_engine(connection_string)
    db = SQLDatabase(db_engine)
    os.environ["OPENAI_API_KEY"]=st.secrets["openai_key"]
    #os.environ["OPENAI_API_KEY"] = openai_key

    # Initialize LLM and SQL toolkit
    llm = ChatOpenAI(temperature=0.0, model="gpt-4")
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create SQL Agent for querying
    sql_agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_execution_time=100, max_iterations=1000)
    
    # Format the prompt with the user's question
    formatted_prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("user", f"{user_query} ai: ")
    ])
    
    # Run SQL Agent
    response = sql_agent.run(formatted_prompt.format_prompt(question=user_query))
    st.text("SQL Query Generated:")
    st.code(response)  # Display the generated SQL query
