import streamlit as st
import chromadb
from Invoke_OpenAI import get_open_ai_response
from prompt.RAG_prompt import prompt
from pdf_reader import read_pdf, extract_text_from_pdf
from chunking_strategy import invoke_text_splitter
from chromadb_function import create_collection, add_to_collection

client = chromadb.PersistentClient(".mycollection")

st.title("RAG Application(PDF)")
st.markdown("This app uses Chroma to perform similarity searches on a collection of documents.")

st.sidebar.title("Configurations")
st.sidebar.markdown("Adjust the settings for your query.")
upload_pdf = st.file_uploader("Upload File", type="pdf")
user_question = st.text_input("Ask a question:", key="user_question")
n_results = st.sidebar.number_input("Number of results:", min_value=1, max_value=10, value=1)
open_ai_key = st.sidebar.text_input("OpenAI Key", type="password")
collection_name = st.sidebar.text_input("Collection Name")

if not open_ai_key:
    st.error("Please provide the OpenAI Key.")
else:
    if st.button("Get Answers"):
        if not upload_pdf:
            st.error("Please provide the pdf file.")
        else:
            collection = create_collection(collection_name, client)
            reader = read_pdf(upload_pdf)
            pdf_content = extract_text_from_pdf(reader)
            text_chunks = invoke_text_splitter(
                separators=["\n\n", "\n", " ", ". ", " "],
                chunk_size=2000,
                chunk_overlap=250,
                content=pdf_content
            )
            add_to_collection(text_chunks, collection=collection)

results = collection.query(query_texts=[user_question], n_results=n_results, include=["documents", "metadatas"])
result_text = "".join(results["documents"][0])


response = get_open_ai_response(
    OpenAI_Key=open_ai_key,
    prompt=prompt.format(user_question=user_question, search_text=result_text)
)
st.write("Search results:", response)
st.success("Got the response successfully 😊")


if st.button("Delete Collection"):
    client.delete_collection(name=collection_name)
    st.success("Collection deleted successfully.")