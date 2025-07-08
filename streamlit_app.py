# UNCOMMENT TO PUSH TO STREAMLIT CLOUD
#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb

import os
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# Step 1: Load text from file
with open("transcript_clean.txt", "r", encoding="utf-8") as file:
    text_to_chunk = file.read()

# Step 2: Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=300,
    chunk_overlap=50,
)

# Step 3: Split the document into chunks
chunks = text_splitter.create_documents([text_to_chunk])

# Step 4: Create Chroma client and collection
chroma_client = chromadb.PersistentClient(path="./Data")
collection = chroma_client.get_or_create_collection(
    name="test_bizint_chunks",
    metadata={"hnsw:space": "cosine"}
)

# Step 5: Add chunks to the collection
for idx, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk.page_content],
        ids=[f"chunk_{idx}"],
        metadatas=[{
            "chunk_index": idx,
            "source": "transcript_clean.txt"
        }]
    )

client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

# Step 6: Query the collection
query_text = "Chi era Devens?"
results = collection.query(query_texts=[query_text], n_results=3)

# Step 7: Print the results
print(f"\nTop results for query: '{query_text}'\n")
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}:\n{doc}\n")

def get_completion(user_prompt, system_prompt, model="gpt-4"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion.choices[0].message.content


system_prompt = "You are a helpful RAG search assistant who uses results from a text document to answer user queries."


# Prompt the model with a user query

def make_rag_prompt(query, result_str):
  return f"""
Instructions:
Your task is to answer the following user question. The search results of a document search have been included to give you more context. Use the information in Search Results to help you answer the question accurately.
Not all information in the search results will be useful. However, if you find any information that's useful for answering the user's question, draw from it in your answer. At the end of your answer, cite the URL of the search result your answer draws from. Use the following format: <Your answer here>. Source: <URL of the search result your answer comes from here>



User question:
{query}


Search Results:
{result_str}


Your answer:
"""

def get_RAG_completion(query, n_results=3):
  
    search_results = collection.query(query_texts=[query], n_results=n_results)
    result_str = ""
    for result in search_results["documents"][0]:
        result_str += result

    formatted_query = make_rag_prompt(query, result_str)
    print("\n********This is the RAG prompt********\n")
    print(formatted_query)
    print("\n*********************************\n")
    return get_completion(formatted_query, system_prompt)

get_RAG_completion("Come le aziende possono usare i dati per prendere decisioni?")


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