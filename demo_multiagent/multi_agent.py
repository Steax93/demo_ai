import os
import streamlit as st
import logging
import json

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model_name='gpt-4o', temperature=0.7,api_key=openai_api_key)
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
#nltk.download("all")

logger = logging.getLogger(__name__)


def create_agent_subfolder(user_name, agent_type):
    user_folder = f'data_{user_name}'
    agent_folder = f'{user_folder}/{agent_type.lower().replace(" ", "_")}'
    os.makedirs(agent_folder, exist_ok=True)
    return agent_folder

def save_uploaded_file(uploaded_file, folder_path):
    try:
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving file {uploaded_file.name}: {str(e)}")
        return False

def load_chat_history(user_name, agent_type):
    user_folder = f'user_{user_name}'
    agent_history_folder = os.path.join(user_folder, agent_type.lower().replace(" ", "_"))
    os.makedirs(agent_history_folder, exist_ok=True)
    
    chat_files = sorted([f for f in os.listdir(agent_history_folder) if f.startswith('chathistory_') and f.endswith('.json')])
    
    if chat_files:
        latest_file = chat_files[-1]
        file_path = os.path.join(agent_history_folder, latest_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return []
    return []

def save_chat_history(user_name, agent_type, history):
    user_folder = f'user_{user_name}'
    agent_history_folder = os.path.join(user_folder, agent_type.lower().replace(" ", "_"))
    os.makedirs(agent_history_folder, exist_ok=True)
    
    file_name = 'chathistory_001.json'
    file_path = os.path.join(agent_history_folder, file_name)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

def display_chat_history(history):
    for message in history:
        if message['role'] == 'user':
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")


def set_qa_prompt(qa_template):
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context','history', 'question'])
    return prompt


def vectordb_load(embeddings, folder_path):
    # Modify this function to load documents from the specific folder_path
    loader = DirectoryLoader(path=folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = []
    docs.extend(loader.load())
    logger.info(f'Extracted text from pdf for folder {folder_path}: {str(docs)}')
    
        
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        
    vectorstore = FAISS.from_texts([""], embeddings)
    store = InMemoryStore()
        
    vector_db = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
    vector_db.search_kwargs = {"k": 5}
    vector_db.add_documents(docs)
    return vector_db

memory = ConversationBufferWindowMemory(k=1, memory_key='history', input_key='question')



def ga_function(question, rag, user_name, agent_type):
    chat_history = load_chat_history(user_name, agent_type)
    
    history_messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in chat_history]
    
    memory.chat_memory.messages = history_messages
    
    response = rag({"query": question})
    final_response = response['result']
    
    if not final_response:
        final_response = "I'm sorry, I couldn't generate a response based on the available information. Could you please rephrase your question or provide more context?"
    
    logger.info(f"Full response: {response}")
    logger.info(f"Final cleaned response: {final_response}")
    
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": final_response})
    logger.info(f"Updated history: {chat_history}")

    save_chat_history(user_name, agent_type, chat_history)
    
    return final_response

qa_medical_template = """You are HealthAssist, an AI assistant providing general health information and guidance. 
Your role is to assist users with health-related inquiries, offer information on common health topics, and provide general wellness advice.
Be empathetic, informative, and responsible in your interactions.

Key Guidelines:
1. Provide general health information only. Do not attempt to diagnose specific conditions or prescribe treatments.
2. Always advise users to consult with a qualified healthcare professional for personalized medical advice, diagnosis, or treatment.
3. You must only use information that you can find inside database, do not invent your own, only use provided.
4. Respect user privacy and confidentiality. Do not ask for or store personal health information.
5. Be clear about your limitations as an AI system. Explain that you cannot access real-time medical data or provide personalized medical advice.
6. Offer information from reputable health organizations when possible, citing sources when appropriate.
7. Encourage healthy lifestyle choices and preventive care, but avoid making guarantees about health outcomes.
8. Be sensitive to the emotional aspects of health concerns, offering empathy and support where appropriate.
9. If a question is not related to the medical field, politely inform the user that you're designed to assist with health-related queries. Suggest they rephrase their question if it has a health component.
IMPORTANT: Only provide a response to the question asked. Do not generate new question/response pairs. If you're unsure about any aspect of the inquiry, express your limitations clearly and advise consulting a healthcare professional.

###################
Database: {context}
Chat history: {history}
Question: {question}
Response:"""

qa_company_template = """You are a helpful assistant. Your goal is to respond to client questions based on the documentation you can find in database.
If the client is greeted you, you should respond: "Hello, I'm your helpful assistant. I'm here to help you analyze your documents. Please upload them to the database so I can get started."
If you do not understand the client's question, ask them to rephrase it.
If the client is grateful, you should respond: "I'm here to help, please upload the documents so I can analyze them". 
Respond only to the user's questions with information found in the database.
If you cannot find relevant information in the database, politely inform the client that you don't have the necessary information to answer their question.
Keep your responses concise, no longer than 50 words or 3 sentences.

IMPORTANT: Only provide a response to the question asked. Do not generate new question/response pairs.

###################
Database: {context}
Chat history: {history}
Question: {question}
Response:"""


qa_recomend_template =  """
You are a recommender system that helps users find similar items/products/movies/books based on their queries and preferences.
Use the following pieces of context to answer the question at the end. 
For each question, suggest three similar items/products/movies/books, providing a short description of each and explaining why the user might like it. 
If you don't have enough information to make a recommendation, simply state that you don't know, and do not fabricate an answer.

Context: {context}
Chat history: {history}
Question: {question}
Response:
"""

def create_rag(vectordb,prompt):
    rag = RetrievalQA.from_chain_type(
            llm,
            chain_type='stuff',
            retriever=vectordb,
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory
            }
        )
    return rag





medical_prompt = set_qa_prompt(qa_medical_template)
manegar_prompt = set_qa_prompt(qa_company_template)
recommend_prompt = set_qa_prompt(qa_recomend_template)


def main():
    st.title("Document Q&A Assistant")

    user_name = st.text_input("Enter your username:")
    if not user_name:
        st.warning("Please enter a username to get started.")
        return

    st.sidebar.header(f"Welcome, {user_name}!")

    assistant_type = st.sidebar.selectbox(
        "Choose your assistant:",
        ("Medical Assistant", "Company Manager Assistant", "Recommendation Assistant")
    )

    # Create agent-specific subfolder for documents
    agent_folder = create_agent_subfolder(user_name, assistant_type)

    # File uploader
    uploaded_files = st.sidebar.file_uploader(f"Upload documents for {assistant_type}", accept_multiple_files=True, type=['pdf', 'txt'])

    if uploaded_files:
        successfully_saved = [file.name for file in uploaded_files if save_uploaded_file(file, agent_folder)]
        if successfully_saved:
            st.sidebar.success(f"Successfully saved files: {', '.join(successfully_saved)}")
        else:
            st.sidebar.error("Failed to save any uploaded files. Please try again.")

    # Display files in the agent's folder
    files_in_folder = os.listdir(agent_folder)
    if files_in_folder:
        st.sidebar.info(f"Files for {assistant_type}: {', '.join(files_in_folder)}")
    else:
        st.sidebar.warning(f"No files found for {assistant_type}. Please upload some documents.")

    # Load chat history
    chat_history = load_chat_history(user_name, assistant_type)

    # Display chat history
    st.subheader("Chat History")
    chat_container = st.empty()
    with chat_container:
        display_chat_history(chat_history)

    # User input
    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.button("Send"):
        try:
            # Load vector database and create RAG
            vector_db = vectordb_load(embeddings, agent_folder)
            if vector_db is None or not vector_db.vectorstore:
                st.warning("No documents found or error loading documents. Please upload some documents to get started.")
                return

            if assistant_type == "Medical Assistant":
                rag = create_rag(vector_db, medical_prompt)
            elif assistant_type == "Company Manager Assistant":
                rag = create_rag(vector_db, manegar_prompt)
            else:
                rag = create_rag(vector_db, recommend_prompt)

            # Generate response
            response = ga_function(user_question, rag, user_name, assistant_type)
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_question})
            chat_history.append({"role": "assistant", "content": response})
            
            # Save updated chat history
            save_chat_history(user_name, assistant_type, chat_history)
            
            # Clear the container and redisplay the updated chat history
            chat_container.empty()
            with chat_container:
                display_chat_history(chat_history)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            st.error("An error occurred while generating the response. Please try again.")

if __name__ == "__main__":
    main()