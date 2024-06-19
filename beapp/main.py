from beapp.action.rag import process_user_instruction
from beapp.action.load_document_v1 import embed_documents

embed_documents()

if __name__ == "__main__":
    process_user_instruction("I want to call Mom")
    process_user_instruction("Give Dad a call")
    process_user_instruction("Call my friend John")