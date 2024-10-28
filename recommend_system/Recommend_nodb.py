import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the language model
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7,api_key=openai_api_key)

# Define custom prompt
template = """
You are a recommender system that helps users find similar items/products/movies/books based on their queries and preferences. 
For each query, suggest three similar items/products/movies/books, providing a short description of each and explaining why the user might like it. 
If you don't have enough information to make a recommendation, simply state that you don't know, and do not fabricate an answer.

Query: {question}
Your Response:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit app layout
st.title("Demo Recommendation System")
st.write("Enter the product you are looking for, and we will recommend similar items.")

# Get user input
item = st.text_input("Enter the product name:")

if item:
    response = llm_chain.run({"question": item})
    # Display the results
    st.write(response)