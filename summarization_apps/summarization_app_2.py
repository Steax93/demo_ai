import openai
import streamlit as st
import os
import torch
from langchain.prompts import PromptTemplate 
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import LatexTextSplitter
from langchain.schema import Document


# Make sure you have the OPENAI API KEY set in your environment
# Load environment variables
load_dotenv()
api_key = os.environ['OPENAI_API_KEY']
#Assigned to torch available devices




#The setup_documents function is responsible for loading a PDF file, extracting text from it, 
#And then splitting the text into smaller chunks based on the given chunk size and overlap.
@st.cache_data
def setup_documents(pdf_file_path, chunk_size, chunk_overlap):
    #Extracting text from pdf
    elements = partition_pdf(pdf_file_path,)
    titles = [elem for elem in elements if elem.category == "Title"]
    raw_text_unst = ''
    for title in titles:
        raw_text_unst += str(title)
    #LaTextSplitter  used to split the text into smaller chunks.
    latex_splitter = LatexTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = latex_splitter.split_text(raw_text_unst)
    #Converting our chunked tex into document class so we can insert it into our vector store
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs




#The custom_summaryfunction takes the documents (chunks of text), language model (llm), custom summary prompt, chain type, 
#and the number of summaries the user wants.
def custom_summary(docs,llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n\n {text}"""#Define the user prompt
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n\n{text}", input_variables=["text"])#Define the system prompt
    if chain_type == "map_reduce":#Chain type 
        chain = load_summarize_chain(llm, chain_type=chain_type, 
                                    map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)
    summaries = []
    for i in range(num_summaries):
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
    
    return summaries

#It then creates a summarization chain based on the specified chain type and the language model. 
#The function subsequently loops through the documents, generating summaries based on the given chain.
#Map_reduce chain type: This takes all the chunks, passes them along with the query to a language model, gets back a response, 
#and then uses another language model call to summarize all of the individual responses into a final answer.
#Stuff chain type: The stuff method is really nice because it’s pretty simple. We just put all of it into one prompt and send that to the language model and get back one response.
#Refine chain type: the refine documents chain constructs a response by looping over the input documents and iteratively updating its answer.


@st.cache_data
def color_chunks(text: str, chunk_size: int, overlap_size: int) -> str:
    overlap_color = "#808080" # Light gray for the overlap
    chunk_colors = ["#a8d08d", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2"] # Different shades of green for chunks

    colored_text = ""
    overlap = ""
    color_index = 0

    for i in range(0, len(text), chunk_size-overlap_size):
        chunk = text[i:i+chunk_size]
        if overlap:
            colored_text += f'<mark style="background-color: {overlap_color};">{overlap}</mark>'
        chunk = chunk[len(overlap):]
        colored_text += f'<mark style="background-color: {chunk_colors[color_index]};">{chunk}</mark>'
        color_index = (color_index + 1) % len(chunk_colors)
        overlap = text[i+chunk_size-overlap_size:i+chunk_size]

    return colored_text

#The color_chunks function is responsible for creating a visually appealing HTML representation of text chunks with overlaps. 
#This function will be useful for debugging chunk size and overlap when visualizing how the text will be split.
# Now we can Create a responsive user interface with Streamlit.


def main():
    st.set_page_config(layout="wide")
    st.title("Summarization App")
    chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=10000, step=100, value=1900)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=100, max_value=10000, step=100, value=200)
    
    if st.sidebar.checkbox("Debug chunk size"):
        st.header("Interactive Text Chunk Visualization")

        text_input = st.text_area("Input Text", "This is a test text to showcase the functionality of the interactive text chunk visualizer.")

        # Set the minimum to 1, the maximum to 5000 and default to 100
        html_code = color_chunks(text_input, chunk_size, chunk_overlap)
        st.markdown(html_code, unsafe_allow_html=True)
    
    else:
        user_prompt = st.text_input("Enter the user prompt")
        pdf_file_path = st.text_input("Enter the pdf file path")
        
        temperature = st.sidebar.number_input("Temperature for your LLM", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
        num_summaries = st.sidebar.number_input("Number of Summaries", min_value=1, max_value=10, step=1, value=1)
        
        #make the choice of llm to select from a selectbox
        llm = ChatOpenAI(model_name="gpt-4",temperature=temperature,api_key=api_key)

        
        if pdf_file_path != "":
            docs = setup_documents(pdf_file_path, chunk_size, chunk_overlap)
            st.write("Pdf was loaded successfully")
            
            if st.button("Summarize"):
                result = custom_summary(docs,llm, user_prompt, chain_type, num_summaries)
                st.write("Summaries:")
                for summary in result:
                    st.write(summary)



#In the main() function, we implement the user interface of the app using Streamlit. 
#We set the page configuration, create titles, and provide options for users to select the language model, chain type, chunk size, and chunk overlap values.

if __name__ == "__main__":
    main()