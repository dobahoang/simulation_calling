from langchain.indexes import SQLRecordManager

def create_record_manager(collection_name: str, suffix: str) -> SQLRecordManager:
    namespace = f"chromadb/{collection_name}"
    db_url = f"sqlite:///record_manager_cache_{suffix}.sql"
    record_manager = SQLRecordManager(namespace, db_url=db_url)
    print(f"Creating schema {suffix}")
    record_manager.create_schema()
    return record_manager

record_manager_1 = create_record_manager("index_document_contact", "1")
