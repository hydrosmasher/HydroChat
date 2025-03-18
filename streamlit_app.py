iimport streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pandas as pd
import json

# Load OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# Initialize OpenAI and embeddings
llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def load_dataset(file_path):
    """Load dataset from CSV or JSON."""
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

# Sidebar for selecting functionality
st.sidebar.title("Hydro Chat Services")
service = st.sidebar.selectbox("Select a service:", [
    "Swim Workout Generator", "Dryland Workout Generator", "Pace & Workout Analysis",
    "Injury Advice", "Nutritional Advice", "General Knowledge"
])

# Option to upload dataset
st.sidebar.subheader("Upload dataset (optional)")
uploaded_file = st.sidebar.file_uploader("Upload a dataset (CSV/JSON)", type=["csv", "json"])
dataset = load_dataset(uploaded_file.name) if uploaded_file else None

# Function to generate responses using RAG
def get_rag_response(query, dataset):
    """Retrieve answers using RAG if dataset is provided."""
    if dataset is not None:
        vectorstore = FAISS.from_texts(dataset, embeddings)
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        return qa_chain.run(query)
    return llm(query)

# Handle each service
st.title(f"💬 {service}")
user_query = st.text_input("Enter your question:")
if st.button("Generate Response"):
    if user_query:
        response = get_rag_response(user_query, dataset)
        st.write(response)
    else:
        st.warning("Please enter a question to proceed.")
