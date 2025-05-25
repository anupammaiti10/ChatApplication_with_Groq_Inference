from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import Dict,Any
import streamlit as st
import time
import os

from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.environ["GROQ_API_KEY"]
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY") 
if not groq_api_key:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()                          

llm=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",groq_api_key=groq_api_key)
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_tool=TavilySearchResults(
    max_results=10,
    include_images=True,
    search_depth="advanced",
)
# If you not added just below line then you have to spend more time every refresh and This prevents the application from re-creating or re-loading this data every time the page reloads, which can be computationally expensive.
st.title("LangChain + Groq + Ollama + FAISS + Streamlit")
st.write("This is a demo of LangChain with Groq, Ollama, FAISS and Streamlit.")
input_text=st.text_input("Enter your question:")
if "vectorstore" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    st.session_state.loader=WebBaseLoader("https://github.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)   
    st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:10])                       
    st.session_state.vectorstore=FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)


if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store=InMemoryChatMessageHistory()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # For a single-user Streamlit app, we can simply return the pre-initialized
    # InMemoryChatMessageHistory from session state.
    # The session_id is required by RunnableWithMessageHistory but not used to differentiate users here
    return st.session_state.chat_history_store

db_retrieval=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
# 1. Convert Tavily Tool into a Retriever-like function
def tavily_retrive(query: str)->list[Document]:
    raw_results=tavily_tool.invoke({"query": query})  
    documents=[                      
        Document(
        page_content=result.get("content"),
        metadata={
            "url":result.get("url",""),
            "title":result.get("title",""),
            "image":result.get("images",""),
            }
        )
        for result in raw_results
    ]   
    return documents   
# 2. Wrap in RunnableLambda for compatibility
tavily_retriver = RunnableLambda(tavily_retrive)
# print(tavily_retriver)
# 3. Create EnsembleRetriever
ensemble_retriever=EnsembleRetriever(
    retrievers=[tavily_retriver,db_retrieval]
)

prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question within atleast 500 words based on the context.If asking questions related to the programming language then provide the code snippets. If you don't know the answer, say 'I don't know'.\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"), # Placeholder for conversational history
    ("human", "{input}")
])

#To give the LLM relevant information and Create the document stuffing chain
chaining=create_stuff_documents_chain(
    llm,
    prompt=prompt
)

def format_documents(inputs:Dict[str,Any])->Dict[str, Any]:
    # Get retrieved documents (from context)        
    context_docs=inputs.get("context")
    #Generate the LLM response
    answer=chaining.invoke({"input": inputs["input"], "context": context_docs, "chat_history": inputs.get("chat_history", [])})
    #Extract the metadata from the documents
    url=[doc.metadata.get("url") for doc in context_docs]
    images=[doc.metadata.get("image") for doc in context_docs]
    # Return the final answer and sources
    return {"answer": answer, "url": url,"images": images}

#Retrieving relevant documents ,Stuffing those documents i,Prompting the LLM with the user's question,Generating a final answer and Create the retrieval chain (combines retriever and document chain)
# retrieval_chain=create_retrieval_chain(ensemble_retriever,chaining)

full_chain = (
    RunnablePassthrough.assign(context=lambda x: ensemble_retriever.invoke(x["input"]),chat_history=lambda x: x["chat_history"]) # Pass history explicitly)
    | RunnableLambda(format_documents)
)
# Create a conversational chain that includes the chat history
# and the retrieval chain             

conversational_chain=RunnableWithMessageHistory(
    full_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output"
)
# The conversational_chain.invoke() might return:
# A string (e.g., the raw LLM response).
if input_text:
    start=time.process_time()
    with st.spinner("Generating response..."):
        response=conversational_chain.invoke(
            {
            "input": input_text,
            "chat_history": st.session_state.chat_history_store.messages
            },                        
            config={"configurable": {"session_id": "streamlit_user_session"}}
        )
        end=time.process_time()
        st.success(f"Response generated in {end-start} seconds")
       
        # st.write(response)
        st.write(response["answer"])
        
        if response.get("url") or response.get("images"):
            with st.expander("Sources"):
                # for image in response["images"]:
                #     if image != None:
                #         st.image(image)
                for url in response["url"]:
                    if url != None:
                        st.markdown(f"- [{url}]({url})")
                # for image in response["images"]:
                #     if image != None:
                #         st.image(image)
                 