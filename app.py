import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory

from dotenv import load_dotenv
import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("RAG QnA with uploaded PDF and chat history")
st.write("Upload a PDF file and ask questions about its content. The chatbot will remember your previous questions and answers.")

api_key = st.text_input("Enter your Groq API Key", type="password")

if api_key:
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=api_key,
    )
    
    ## stateful history
    session_id = st.text_input("Enter a session ID", value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store = {}
        
    uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)
    ## process the file
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        
        ## split and create embeddings
        ## create a retriever
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        contextualise_q_sytem_prompt = (
            "Given a chat history and latest user question"
            "which might refer context in the chat history"
            "formulate a standalone which can be understood"
            "without the chat history. DO NOT answer the question."
            "just reformulate the question if needed or return as it is."
        )
        
        contextualise_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualise_q_sytem_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=contextualise_q_prompt)
        
        sytem_prompt = (
            "You are a helpful assistant that helps answer user's questions."
            "You will be provided with the context and the question."
            "Answer the question based on the context."
            "If the question is not related to the context, say 'I don't know'."
            "\n\n"
            "Context: {context}"
        )
        qna_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sytem_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        qna_chain = create_stuff_documents_chain(llm=llm, prompt=qna_prompt)
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=qna_chain
        )
        
        
        def get_session_history(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        retrieval_chain_with_history = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        user_input = st.text_input("Ask a question about the PDF content")  
        if user_input:
            session_history= get_session_history(session_id)
            response = retrieval_chain_with_history.invoke(
                {"input": user_input},
                message_history=session_history,
                config={
                    "configurable" : {"session_id": session_id},
                }
            )
            st.write(st.session_state.store)
            st.success(f"Answer: {response['answer']}")
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter your Groq API Key to use the application.")