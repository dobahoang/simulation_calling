

import chromadb
from langchain.chains import OpenAIModerationChain
from langchain_community.vectorstores import Chroma
from beapp.config.settings import PERSIST_DIRECTORY_CONTACT, COLLECTION_NAME_CONTACT
from langchain_openai import OpenAIEmbeddings

moderate = OpenAIModerationChain()

persistent_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY_CONTACT)

def get_retriever(embeddings: OpenAIEmbeddings, persist_directory: str) -> Chroma:
    if persist_directory == PERSIST_DIRECTORY_CONTACT:
        vectordb = Chroma(
            client=persistent_client,
            collection_name=COLLECTION_NAME_CONTACT,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    return retriever

def save_documents_to_chromadb(embeddings: OpenAIEmbeddings, persist_directory: str, texts: list) -> None:
    docsearch = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    docsearch.persist()
