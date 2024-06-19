### README.
Hi, I'm Hoang. My email: bahoangwork@gmail.com

# Contact Calling Simulation Application

This application simulates a basic interaction between a user and an LLM (Language Model) assistant to handle phone calls. The user provides a simple instruction (e.g., "Call Mom"), and the application processes the instruction, identifies the intended action, accesses a simulated contact list, and initiates a call. The application is built using RAG (Retrieval-Augmented Generation), embeddings, LangChain, and LLM models.

## Part 1: Installation and Running on Windows

### Prerequisites

- Python 3.9 or higher


### Steps

1. **Clone the repository (optional):**

   If you have Git installed, you can clone the repository using the following command. Otherwise, you can download the ZIP file from the repository and extract it.


2. **Set up a virtual environment:**

   Open a command prompt or PowerShell window and navigate to the project directory. Then, create and activate a virtual environment.

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   Make sure your virtual environment is activated, then install the dependencies listed in the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   Execute the main file to test the application.

   ```bash
   python beapp/main.py
   ```

   You should see output indicating the simulated actions based on the user instructions.

## Part 2: Technical Details

### Technologies Used

- **RAG (Retrieval-Augmented Generation)**: Combines retrieval of relevant documents with generative models to provide accurate and contextually relevant responses.
- **Embeddings**: Converts text into numerical vectors to capture semantic meaning, enabling efficient similarity search and matching.
- **LangChain**: A framework for building applications with LLMs, providing tools for chaining together multiple processing steps.
- **LLM (Large Language Models)**: Used for natural language understanding and generation to interpret user instructions and generate appropriate responses.

### Benefits

1. **Accurate Contextual Responses**: By using RAG, the application retrieves relevant context(action such á: call mom... from user query) before generating a response, ensuring high accuracy and relevance.
2. **Semantic Understanding**: Embeddings allow the application to understand and match the semantic meaning of user instructions, handling variations in phrasing effectively.
3. **Modularity and Scalability**: LangChain provides a modular approach to building LLM applications, making it easy to extend and scale the application with additional functionalities.
4. **Natural Language Processing**: Leveraging LLMs enables the application to process and understand natural language instructions, providing a more intuitive and user-friendly interaction.

### How It Works

1. **User Instruction Processing**: The application receives a user instruction (e.g., "Call Mom").
2. **Context Retrieval**: Uses embeddings and a retriever to find the most relevant contact information from the simulated contact list.
3. **Action Identification**: The LLM identifies the intended action from the instruction.
4. **Simulated Call**: Extracts the phone number and simulates a call by printing a message.

### Example Usage
please watch beapp/action/data/contact.json to check phone number 

- **Instruction**: "I want to call Mom"
  - **Output**: "Calling [Mom's Phone Number]..."

- **Instruction**: "Give Dad a call"
  - **Output**: "Calling [Dad's Phone Number]..."

- **Instruction**: "Call my friend John"
  - **Output**: "Contact not found"

The application handles different phrasings and errors (e.g., contact not found) gracefully, providing a robust and user-friendly experience.


## Part 3: Additional Approaches

### Tool Calling (Function Calling)

In addition to the current implementation, another approach to consider is tool calling (function calling). This method involves the following steps:

1. **Action Identification**: Use the LLM to identify the specific function to call based on the user's instruction.
2. **Function Invocation**: Map the identified action to a pre-defined function and invoke it programmatically.
3. **Execution and Response**: Execute the function and provide a response based on the result.

#### Benefits of Tool Calling:

- **Flexibility**: Allows for the integration of a wide range of actions and functions, providing greater flexibility in handling user instructions.
- **Modularity**: Each function can be developed and tested independently, promoting modularity and maintainability.
- **Scalability**: Easy to add new functions as the application grows, enabling scalability.

#### Example Workflow:

1. **User Instruction**: "Send a message to Mom"
2. **Action Identification**: LLM identifies the action as "send_message".
3. **Function Invocation**: The application maps "send_message" to a pre-defined function `send_message_to_contact(contact_name, message)`.
4. **Execution and Response**: The function is executed, and the application responds with "Message sent to Mom".



---

For any further assistance, please contact with me : bahoangwork@gmail.com