import streamlit as st
from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

configuration = {
    "client": "PersistentClient",
    "path": "/tmp/.chroma"
}

collection_name = "documents_collection"

conn = st.connection("chromadb",
                     type=ChromaDBConnection,
                     **configuration)
documents_collection_df = conn.get_collection_data(collection_name)
st.dataframe(documents_collection_df)