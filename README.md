# Vector Search AI Assistant

A simple AI assistant that uses LangChain and Groq to answer questions from a PDF document using embeddings and a vector database.

---

## Features

- Load and process PDF documents.
- Split documents into chunks for better retrieval.
- Generate embeddings with HuggingFace.
- Store and retrieve chunks using Chroma vector database.
- Query the assistant interactively with context-aware answers.
- Answers limited to the content of the PDF.

---

## Requirements

- Python 3.10+
- langchain-core
- langchain-huggingface
- langchain-community
- langchain-groq
- langchain-text-splitters
- python-dotenv

Install dependencies:

pip install langchain-core langchain-huggingface langchain-community langchain-groq langchain-text-splitters python-dotenv

---


## Output Screenshot

![Output Screenshot](images/Screenshot%202026-02-27%20224402.png)

---

## Usage

1. Run the script with `python main.py`.
2. Ask questions in the terminal.
3. Type `exit` to quit.

---

## Notes

- The assistant only answers based on the content of the PDF.
- Ensure the vector database folder `chroma_db` exists for persistent storage.
- Adjust chunk size and overlap for better retrieval if needed.