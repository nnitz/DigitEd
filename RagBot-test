# %%
# FULL CODE

import chromadb
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv('.env')


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
chroma_client = chromadb.PersistentClient(path="/Users/natalienitz/Desktop/DigitEd/Data")
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

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Step 6: Query the collection
query_text = "Chi era Devens?"
results = collection.query(query_texts=[query_text], n_results=3)

# Step 7: Print the results
print(f"\nTop results for query: '{query_text}'\n")
for i, doc in enumerate(results['documents'][0]):
    print(f"Result {i+1}:\n{doc}\n")

# %%
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

# %%
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

# %%
# Custom CSS
st.markdown("""
    <style>
        .main {
            font-family: 'Segoe UI', sans-serif;
        }
        .logo {
            font-weight: 700;
            font-size: 32px;
            color: #FF00FF; /* bright magenta */
        }
        .login {
            position: absolute;
            top: 1.5rem;
            right: 2rem;
            font-weight: 500;
            font-size: 18px;
            color: #1F1F1F;
        }
        .heading {
            font-size: 36px;
            font-weight: 600;
            text-align: center;
            margin-top: 2rem;
        }
        .subheading {
            font-size: 20px;
            text-align: center;
            color: #444444;
            margin-bottom: 2.5rem;
        }
        .chat-box {
            background-color: #f9f9f9;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
        }
        .chat-text {
            font-size: 20px;
            font-weight: 500;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# %%
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


