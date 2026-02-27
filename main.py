from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import dotenv

dotenv.load_dotenv()

api_key = os.getenv("API_KEY")

loader = PyPDFLoader("data/My_CV.pdf")
document = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)
# Load
# set splitter
# chunk
# set the Embedding model
# set the vector database(chunk,embedding,directory)
# set retriever
# set generator(api-key,model)
# ask query
# retrieve(query)
# extract the retrived
# set context
# generate(retrived,context)

chunk = text_splitter.split_documents(document)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_dir = "./chroma_db"
if os.path.exists(persist_dir):
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )
else:
    vector_store = Chroma.from_documents(
        documents=chunk,
        embedding=embedding,
        persist_directory=persist_dir
    )
    vector_store.persist()
retriever = vector_store.as_retriever(search_kwargs={"k":3})

llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.1-8b-instant"
)

is_AI_on = True


while is_AI_on:
    query = input("What's your question? : ")
    if query.lower() == "exit":
        is_AI_on = False
        break
    retrieved = retriever.invoke(query)

    context = "\n\n".join([r.page_content for r in retrieved])

    prompt = f"""
    You are a helpful AI assistant.

    Answer ONLY from the provided context.
    If the answer is not in the context, say "I could not find it in the document".

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    print(response.content)