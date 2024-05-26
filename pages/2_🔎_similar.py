from pathlib import Path
from typing import Union
from dotenv import load_dotenv
import streamlit as st
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

st.title("Similar Patents")

if "answers" not in st.session_state:
    st.session_state["answers"] = ""

# Configure parameters
load_dotenv()

mistral_api_key = os.environ.get("MISTRAL_API_KEY")


def process_documents(folder: Union[str, Path] = 'data'):
    all_docs = []
    files = glob.glob(f"{str(folder)}/*")
    Loader = PyPDFLoader

    for d in files:

        loader = Loader(d)

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            separators=[" ", ",", "\n"], chunk_size=2000, chunk_overlap=200, length_function=len
        )
        docs = text_splitter.split_documents(documents)
        all_docs.extend(docs)
    return all_docs


if __name__ == "__main__":
    texts = process_documents()

    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        texts, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    if st.session_state["answers"]:

        st.header("Client Letter")
        st.write(st.session_state["answers"])
        print(st.session_state["answers"])

        user_query = st.session_state["answers"]
        query_vector = embedding_model.embed_query(user_query)

        retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=1)
        unique_matadata = []
        unique_similars = []
        for s in retrieved_docs:
            if s.metadata["source"] not in unique_matadata:
                unique_matadata.append(s.metadata["source"])
                unique_similars.append(s)

        st.header("Relevant Documents:")
        for i, doc in enumerate(unique_similars):
            source = doc.metadata["source"]
            page = doc.metadata["page"]

            st.write(source)
            st.write(doc.page_content)

            model = "mistral-large-latest"

            client = MistralClient(api_key=mistral_api_key)
            chat_response = client.chat(
                model=model,
                messages=[
                    ChatMessage(role="system", content="""Can you give me all the differences between the user_patent and similar_patent,
                                can you also give me at the end a score in % on how much the user_patent is close to similar_patent """),
                    ChatMessage(role="user", content=f"""
                                user_patent : {user_query} 
                                
                                similar_patent: {doc.page_content}
                                
                                """),
                ]
            )
            st.header("key differences")
            st.write(chat_response.choices[0].message.content)
