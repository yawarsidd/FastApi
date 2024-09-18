from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = FastAPI()

# Initialize embeddings, LLM, and vectorstore
embedding = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
llm_gemma2 = Ollama(model="gemma:2b")

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.
<context>
{context}
</context>
Question: {input}""")

document_chain_gemma2 = create_stuff_documents_chain(llm_gemma2, prompt)

# Placeholder for vector DB (will be initialized once PDF is uploaded)
vector_db = None

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application."}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_db
    try:
        # Load and process the PDF file
        loader = PyPDFLoader(file.file)
        docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)

        # Create FAISS vector store
        vector_db = FAISS.from_documents(documents=documents, embedding=embedding)

        return {"message": "PDF uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

class QuestionRequest(BaseModel):
    input: str

@app.post("/ask_question/")
async def ask_question(question_request: QuestionRequest):
    if vector_db is None:
        raise HTTPException(status_code=400, detail="No documents available. Please upload a PDF first.")

    try:
        # Create a retriever from vector DB and retrieval chain
        retriever = vector_db.as_retriever()
        retrieval_chain_gemma2 = create_retrieval_chain(retriever, document_chain_gemma2)

        # Run the question through the chain
        response = retrieval_chain_gemma2.invoke({"input": question_request.input})

        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)