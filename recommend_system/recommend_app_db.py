import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize the language model
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7, api_key=openai_api_key)

# Define custom prompt
template = """
You are a recommender system that helps users find similar items/products/movies/books based on their queries and preferences.
Use the following pieces of context to answer the question at the end. 
For each question, suggest three similar items/products/movies/books, providing a short description of each and explaining why the user might like it. 
If you don't have enough information to make a recommendation, simply state that you don't know, and do not fabricate an answer.

Context: {context}
Question: {question}
Response:
"""

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def load_vector_store(file_path, parent_chunk, child_chunk):
    # Define what will be loaded
    loader = PyPDFLoader(file_path)
    docs = []
    docs.extend(loader.load())
    # Split information in the documents
    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk)
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="split_parents", embedding_function=embeddings
    )
    # The storage layer for the parent documents
    store = InMemoryStore()
    vector_db = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    vector_db.add_documents(docs)

    return vector_db

prompt = PromptTemplate(template=template, input_variables=["question"])

chain_type_kwargs = {"prompt": prompt}

def qa(vectordb):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=vectordb,
                                           return_source_documents=True,
                                           chain_type_kwargs=chain_type_kwargs)
    return qa_chain

@st.cache_data
def color_chunks(text: str, chunk_size: int, overlap_size: int) -> str:
    overlap_color = "#808080"  # Light gray for the overlap
    chunk_colors = ["#a8d08d", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2"]  # Different shades of green for chunks

    colored_text = ""
    overlap = ""
    color_index = 0

    for i in range(0, len(text), chunk_size - overlap_size):
        chunk = text[i:i + chunk_size]
        if overlap:
            colored_text += f'<mark style="background-color: {overlap_color};">{overlap}</mark>'
        chunk = chunk[len(overlap):]
        colored_text += f'<mark style="background-color: {chunk_colors[color_index]};">{chunk}</mark>'
        color_index = (color_index + 1) % len(chunk_colors)
        overlap = text[i + chunk_size - overlap_size:i + chunk_size]

    return colored_text

def main():
    st.set_page_config(layout="wide")
    st.title("Recommendation App")
    parent_chunk = st.sidebar.slider("Parent chunk size", min_value=100, max_value=10000, step=100, value=1900)
    child_chunk = st.sidebar.slider("Child chunk size", min_value=100, max_value=10000, step=100, value=200)
    if st.sidebar.checkbox("Debug chunk size"):
        st.header("Interactive Text Chunk Visualization")

        text_input = st.text_area("Input Text", "This is a test text to showcase the functionality of the interactive text chunk visualizer.")

        # Set the minimum to 1, the maximum to 5000 and default to 100
        html_code = color_chunks(text_input, parent_chunk, child_chunk)
        st.markdown(html_code, unsafe_allow_html=True)
    else:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with open(os.path.join("tmp", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_path = os.path.join("tmp", uploaded_file.name)
            docs = load_vector_store(file_path, parent_chunk, child_chunk)
            st.write("PDF was loaded successfully")
            
            item = st.text_input("Enter your preferences:")
            if st.button("Find your preferences"):
                chain = qa(docs)
                response = chain({"query": item})
                st.write("Recommendation:")
                st.write(response['result'])

if __name__ == "__main__":
    main()