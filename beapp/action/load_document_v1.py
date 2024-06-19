import json
import os
from typing import List, Optional

import chromadb
from langchain.indexes import index
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from beapp.action.record import record_manager_1
from beapp.config.settings import EMBEDDING_MODEL, DATABASE_DIRECTORIES

RECORD_MANAGERS = {
    "contact_db": record_manager_1,
}
def create_chroma_vectorstore(database_path: str, collection_name: str, model_name: str = EMBEDDING_MODEL) -> Chroma:
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(collection_name)
    embedding = OpenAIEmbeddings(model=model_name)
    return Chroma(client=client, collection_name=collection_name, embedding_function=embedding,
                  persist_directory=database_path)


def clear_vectorstore(vectorstore: Chroma) -> None:
    print("-------- start clear --------")
    index([], None, vectorstore, cleanup="full", source_id_key="source")


def load_text_documents(embedded_file_path: str, metadata_source: Optional[str] = None) -> List[Document]:
    loader = DirectoryLoader(path=embedded_file_path, use_multithreading=True, loader_cls=TextLoader)
    docs = loader.load()
    if metadata_source is None:
        metadata_source = embedded_file_path
    documents = [Document(page_content=page.strip(), metadata={"source": metadata_source}) for doc in docs for page in
                 doc.page_content.strip().split("--")]
    return documents


def load_json_documents(embedded_file_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(embedded_file_path):
        if filename.endswith(".json"):
            with open(os.path.join(embedded_file_path, filename), "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        documents.append(Document(page_content=json.dumps(item), metadata={"source": filename}))
                else:
                    documents.append(Document(page_content=json.dumps(data), metadata={"source": filename}))
    return documents


def load_documents(embedded_file_path: str) -> List[Document]:
    if "data" in embedded_file_path:
        return load_json_documents(embedded_file_path)
    return load_text_documents(embedded_file_path)


class CustomLoader(BaseLoader):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def lazy_load(self):
        yield from load_documents(self.folder_path)

    def load(self):
        return list(self.lazy_load())


def setup_vectorstores() -> dict:
    vectorstores = {}
    for domain, database_path in DATABASE_DIRECTORIES.items():
        collection_name = f"index_document_{domain}"
        vectorstores[domain] = create_chroma_vectorstore(database_path, collection_name)
    return vectorstores


def clear_all_vectorstores(vectorstores: dict) -> None:
    for vectorstore in vectorstores.values():
        clear_vectorstore(vectorstore)


def index_documents(vectorstores: dict) -> None:
    for domain, vectorstore in vectorstores.items():
        loader = CustomLoader("action/data")
        print(f"Indexing for domain: {domain}")
        record_manager = RECORD_MANAGERS[domain]
        print(index(loader, record_manager, vectorstore, cleanup="full", source_id_key="source"))
        print(f"There are {vectorstore._collection.count()} documents in the collection {domain}")


def embed_documents() -> None:
    vectorstores = setup_vectorstores()
    # clear_all_vectorstores(vectorstores)  # Uncomment to clear existing content
    index_documents(vectorstores)


if __name__ == "__main__":
    embed_documents()
