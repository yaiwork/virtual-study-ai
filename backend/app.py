from fastapi import FastAPI, Request, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, glob, re
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from io import BytesIO
from docx import Document
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader

from fastapi import UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from typing import Optional

from langchain.chains.summarize import load_summarize_chain
# Load environment variables
load_dotenv(override=True)

MODEL = "gpt-4o"
#MODEL = "gpt-3.5-turbo"
DB_DIR = "vector_db"
KNOWLEDGE_FOLDER = "knowledge-base"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_chain = None
retriever = None

class HomeworkRequest(BaseModel):
    question: str

class LessonPlanRequest(BaseModel):
    subject: str
    grade: str
    topic: str

class ExamGeneratorRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    num_questions: int = range(5,20) # changed from = 5

class StudyGuideRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    
# ================= Document Loader - ORIGINAL ======================
def load_documents():
    documents = []
    folders = glob.glob(f"{KNOWLEDGE_FOLDER}/*")
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


# def initialize_vectorstore(documents):
#     text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
#     chunks = text_splitter.split_documents(documents)
#     embeddings = OpenAIEmbeddings()
#     if os.path.exists(DB_DIR):
#         Chroma(persist_directory=DB_DIR, embedding_function=embeddings).delete_collection()
#     return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)

def sanitize_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


#========== vectorstorer   =====================================

def initialize_vectorstore(documents):
    import os, shutil
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

    # === Step 1: Reset vector store ===
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    # === Step 2: Chunk documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # === Step 3: Sanitize all chunks
    def estimate_tokens(text): return len(text) // 4
    for chunk in chunks:
        chunk.page_content = sanitize_text(chunk.page_content)

    print(f"✅ Prepared {len(chunks)} chunks total")

    # === Step 4: Initialize vectorstore + embeddings
    embedding_fn = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_fn)

    # === Step 5: Add texts in token-safe batches
    batch, batch_tokens = [], 0
    BATCH_TOKEN_LIMIT = 100_000

    for chunk in chunks:
        tokens = estimate_tokens(chunk.page_content)
        if batch_tokens + tokens > BATCH_TOKEN_LIMIT:
            vectorstore.add_documents(batch)
            batch, batch_tokens = [], 0
        batch.append(chunk)
        batch_tokens += tokens

    # Final batch
    if batch:
        vectorstore.add_documents(batch)

    print("✅ Chroma vectorstore created successfully.")
    return vectorstore


def fix_latex_formatting(text: str) -> str:
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"\[\s*(\\pi.*?)\s*\]", r"$$\1$$", text)
    return text

def extract_text_from_pdf_bytes(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx_bytes(file_bytes):
    doc = Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image_bytes(file_bytes):
    img = Image.open(BytesIO(file_bytes))
    return pytesseract.image_to_string(img)

# === setup_chain with conversation_chain and  retriever  ===============
@app.on_event("startup")
def setup_chain():
    global conversation_chain, retriever
    documents = load_documents() # call the document loader function
    vectorstore = initialize_vectorstore(documents) 
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL) 
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ============ CREATE THE AI AGENTS  =====================
## ============  1. chat =================
@app.post("/chat")
async def chat(req: Request):
    global conversation_chain  # Added by me
    data = await req.json()
    question = data.get("question", "")
    result = conversation_chain.invoke({"question": question})
    return {"answer": fix_latex_formatting(result["answer"])}

## ============ 2. homework-assistant  ============
@app.post("/homework-assistant")
async def homework_assistant(
    request: Request,
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)):
    global conversation_chain # Added
    
    try:
        # Handle form-data (file + question)
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            if not question and not file:
                return {"answer": "❌ Please provide a question or upload a file."}

            extracted_text = ""
            if file:
                content = await file.read()
                filename = file.filename.lower()
                if filename.endswith(".pdf"):
                    extracted_text = extract_text_from_pdf_bytes(content)
                elif filename.endswith(".docx"):
                    extracted_text = extract_text_from_docx_bytes(content)
                elif filename.endswith((".png", ".jpg", ".jpeg")):
                    extracted_text = extract_text_from_image_bytes(content)
                else:
                    return {"answer": "❌ Unsupported file type."}

            full_prompt = (
                f"You are a homework assistant that helps students understand and solve problems using uploaded documents.\n"
                f"First, read the uploaded content below, then provide step-by-step explanations and final answers for the following question.\n"
                f"\nContent:\n{extracted_text}\n\nQuestion: {question or '(no specific question provided)'}"
            )

        # Handle JSON (text-only mode)
        else:
            json_data = await request.json()
            question = json_data.get("question", "").strip()
            if not question:
                return {"answer": "❌ Please provide a homework question."}
            full_prompt = (
                f"You are a helpful homework assistant. Solve the following question step-by-step with explanations:\n\n"
                f"{question}"
            )

        result = conversation_chain.invoke({"question": full_prompt})
        return {"answer": fix_latex_formatting(result["answer"])}

    except Exception as e:
        return JSONResponse(status_code=500, content={"answer": f"❌ Error: {str(e)}"})


## =====    3. lesson-plan   =================
@app.post("/lesson-plan")
async def lesson_plan(data: LessonPlanRequest):
    global conversation_chain
    try:
        prompt = (
        f"You are an educational expert assistant. Only respond if there are documents that match "
        f"**both** the subject '{data.subject}' and grade '{data.grade}'. If no such match is found, reply with "
        f"'🚨Please, try again for {data.subject} in Grade {data.grade}.'\n\n"
        f"Based on the provided documents, create a detailed lesson plan for Grade {data.grade} in {data.subject} "
        f"on the topic or a closely related topic to '{data.topic}'. Include:\n"
        f"- Learning objectives\n"
        f"- Materials needed\n"
        f"- Activities\n"
        f"- Assessment methods")

        result = conversation_chain.invoke({"question": prompt})
        return {"plan": fix_latex_formatting(result["answer"])}
    except Exception as e:
        return {"plan": f"❌ Lesson plan error: {str(e)}"}

## ==========  4. generate-exam       =============
@app.post("/generate-exam")
async def generate_exam(data: ExamGeneratorRequest):
    global conversation_chain
    try:
        prompt = (
    f"You are an expert exam generator and explainer.Please, mostly create multiple-choice questions with well-formatted outputs. Only respond if there are documents that match "
    f"**both** the subject '{data.subject}' and grade '{data.grade}'. If no such documents exist, reply with "
    f"'🚨Please, try again for {data.subject} in Grade {data.grade}.'\n\n"
    f"Based on the documents, generate {data.num_questions} exam questions for Grade {data.grade} in {data.subject} "
    f"on the topic or closely related topics to '{data.topic}'. For each question, provide answers with detailed explanations and/or calculations.")

        result = conversation_chain.invoke({"question": prompt})
        return {"questions": fix_latex_formatting(result["answer"])}
    except Exception as e:
        return {"questions": f"❌ Exam generation error: {str(e)}"}


## ====  5. study-guide   =====================
@app.post("/study-guide")
async def study_guide(data: StudyGuideRequest):
    global conversation_chain
    try:
        prompt = (
        f"You are an intelligent educational assistant. Only respond if there are documents that match "
        f"**both** the subject '{data.subject}' and grade '{data.grade}'. If no such match is found, reply with "
        f"'🚨 Please, try again for {data.subject} in Grade {data.grade}.'\n\n"
        f"Based on the documents provided, create a comprehensive study guide for Grade {data.grade} in {data.subject} "
        f"on the topic '{data.topic}' or any similar topic covered in the content. Include:\n"
        f"- Key concepts\n"
        f"- Definitions\n"
        f"- Notes\n"
        f"- Examples\n"
        f"- Formulas\n"
        f"- Tips")

        result = conversation_chain.invoke({"question": prompt})
        return {"study_guide": fix_latex_formatting(result["answer"])}
    except Exception as e:
        return {"study_guide": f"❌ Study guide generation error: {str(e)}"}

