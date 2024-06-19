from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from beapp.action.chroma_db import get_retriever
from beapp.config.settings import EMBEDDING_MODEL, PERSIST_DIRECTORY_CONTACT
import re

# Template for the prompt to identify the relevant phone number
template = """Given a user query (Question) and a context containing potential contact information, 
identify the most relevant phone number for the user's request. If no suitable phone number exists within the context, return "None".

potential contact information : 
{context}

Question: {question}
Answer: 
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
retriever = get_retriever(embeddings, PERSIST_DIRECTORY_CONTACT)

def format_documents(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": retriever | format_documents, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

def handle_query(query: str) -> str:
    return chain.invoke(query)

def extract_phone_number(response: str) -> str:
    # Extract contiguous string of digits
    match = re.search(r'\b\d+\b', response)
    return match.group(0) if match else "None"

def simulate_call(phone_number: str) -> None:
    if phone_number != "None":
        print(f"Calling {phone_number}...")
    else:
        print("Contact not found.")

def process_user_instruction(instruction: str) -> None:
    response = handle_query(instruction)
    phone_number = extract_phone_number(response)
    simulate_call(phone_number)

if __name__ == "__main__":
    process_user_instruction("I want to call Mom")
    process_user_instruction("Give Dad a call")
    process_user_instruction("Call my friend John")