import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="NJIT Highlander",
    page_icon=":school_satchel:",
    layout="wide"
)

st.title("NJIT QNA Bot")
st.write("Welcome to the NJIT QNA Bot! Ask any question about NJIT, and I'll do my best to provide a helpful answer.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def setup_rag():
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    urls = ["https://www.njit.edu/admissions"]
    web_data = []
    for url in urls:
        loader = WebBaseLoader(url)
        web_data.extend(loader.load())
    pdf_files=["Graduate Student Guidebook 10-3-24.pdf"]
    pdf_data = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        pdf_data.extend(loader.load())

    data = web_data + pdf_data

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    llm = ChatGroq(temperature=0.3,model_name="llama-3.1-8b-instant")

    message = """
    Answer the question based on the given context.

    Question:
    {question}

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain = ({"context": retriever, "question": RunnablePassthrough()}| prompt| llm)

    return rag_chain

rag_chain = setup_rag()

for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])


user_question = st.chat_input("Ask a question about NJIT...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            response = rag_chain.invoke(user_question)
            st.write(response.content)


    st.session_state.chat_history.append({
        "question": user_question,
        "answer": response.content
    })
