__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from openai import OpenAI

openai_api_key = st.secrets["OPENAI_API_KEY"]
client = chromadb.PersistentClient("./mycollection")

collection = client.get_or_create_collection(name = "RAG_Assistant", metadata = {"hnsw:space" : "cosine"})

# Checkpoint 2: Update previous titles and markdown text
st.title("RAG Assistant")
st.markdown("This app uses Chroma to perform similarity searches on a collection of documents.")
st.sidebar.title("Enter suitable text")
st.sidebar.markdown("Adjust the settings for your query.")

#Checkpoint 3
#a: Add input text widget for user question
input_text = st.text_area("Add some text", key="input_key")

#b: Add number of results to the sidebar
n_results = st.sidebar.number_input("Number of results", min_value=1, max_value=10, value=1)

# Checkpoint 4: Create a button that triggers the action of querying the Chroma Collection
if st.button("Get Answers"):
    st.write(f"Question: {input_text}")
    st.write(f"Number of Results: {n_results}")
    results = collection.query(query_texts=[input_text], n_results=n_results)
    for res in results["documents"]:
        for txt in res:
            st.write(txt)
    st.json(results)