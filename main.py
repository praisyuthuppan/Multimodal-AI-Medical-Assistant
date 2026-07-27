import os
import uuid
import sqlite3
from io import BytesIO
from pathlib import Path
from datetime import datetime
import logging
import cv2
import easyocr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gtts import gTTS
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from PIL import Image

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import (
    DOCUMENT_SYSTEM_PROMPT,
    GENERAL_MEDICAL_SYSTEM_PROMPT
)


# -----------------------------
# INITIAL SETUP
# -----------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Add it to the .env file.")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = str(DATA_DIR / "chroma")
DB_PATH = DATA_DIR / "medical_assistant.db"

DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Medical Assistant",description="Multimodal AI Assistant for understanding medical documents using OCR, Whisper and RAG.",
version="1.0"

)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

# First run downloads OCR models.
ocr_reader = easyocr.Reader(["en"], gpu=False)


# -----------------------------
# DATABASE
# -----------------------------
def db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    connection = db_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            source_type TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    

    connection.commit()
    connection.close()


create_tables()


# -----------------------------
# SAFETY
# -----------------------------
EMERGENCY_WORDS = [
    "chest pain",
    "cannot breathe",
    "trouble breathing",
    "severe bleeding",
    "stroke",
    "face drooping",
    "fainted",
    "unconscious",
    "suicide",
    "kill myself",
    "self harm"
]

EMERGENCY_RESPONSE = """
This may be a medical emergency.

Please contact your local emergency services or visit the nearest emergency department immediately.

This assistant can explain uploaded medical documents and answer general medical education questions.

It cannot diagnose emergencies or provide emergency treatment.
"""


def is_emergency(question: str) -> bool:
    question = question.lower()
    return any(word in question for word in EMERGENCY_WORDS)


# -----------------------------
# VECTOR DATABASE HELPERS
# -----------------------------
def safe_collection_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_").lower()


def patient_collection_name(document_id: str) -> str:
    return f"patient_{safe_collection_name(document_id)}"



def open_collection(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings
    )


def recreate_collection(collection_name: str, documents: list[Document]):
    import chromadb

    client = chromadb.PersistentClient(path=VECTOR_DIR)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=VECTOR_DIR
    )


# -----------------------------
# PDF / IMAGE / VOICE MODULES
# -----------------------------
def extract_pdf(file_bytes: bytes):
    reader = PdfReader(BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number
            })

    return pages


def extract_image(file_bytes: bytes):
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    image_array = np.array(image)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    processed = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    extracted_lines = ocr_reader.readtext(processed, detail=0)
    text = "\n".join(extracted_lines)

    return [{"text": text, "page": 1}]


def extract_voice(file_bytes: bytes, filename: str):
    result = groq_client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model="whisper-large-v3-turbo",
        response_format="json",
        temperature=0.0
    )

    return [{"text": result.text, "page": 1}]


def create_patient_documents(document_id: str, filename: str, pages: list):
    source_documents = []

    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])

        for chunk in chunks:
            source_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_data["page"],
                        "source_type": "patient_document"
                    }
                )
            )

    return source_documents



# -----------------------------
# RAG
# -----------------------------
def retrieve_documents(document_id: str | None, question: str):

    if not document_id:
        return []

    try:

        patient_store = open_collection(
            patient_collection_name(document_id)
        )

        return patient_store.similarity_search(
            question,
            k=5
        )

    except Exception:

        return []


def build_context_and_sources(documents: list[Document]) -> tuple[str, list]:

    context_parts = []

    sources = []

    for index, document in enumerate(documents, start=1):

        source_id = f"S{index}"

        label = f"{document.metadata['filename']} - Page {document.metadata['page']}"

        context_parts.append(

            f"[{source_id}] {label}\n{document.page_content}"

        )

        sources.append({

            "id": source_id,

            "label": label

        })

    return "\n\n".join(context_parts), sources




# -----------------------------
# REQUEST MODELS
# -----------------------------
class AskRequest(BaseModel):
    question: str
    document_id: str | None = None


class TTSRequest(BaseModel):
    text: str


# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/")
def home():
    return {
        "application": "AI Medical Assistant",
        "version": "1.0",
        "status": "Running"
    }



async def process_upload(file: UploadFile, source_type: str):
    raw_file = await file.read()

    if not raw_file:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    filename = file.filename or f"{source_type}_file"

    logger.info(
    f"Uploaded File: {filename} | "
    f"Type: {source_type}")
    document_id = str(uuid.uuid4())
    

    if source_type == "pdf":
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Upload a PDF file.")
        pages = extract_pdf(raw_file)

    elif source_type == "image":
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Upload JPG, JPEG, or PNG.")
        pages = extract_image(raw_file)

    elif source_type == "voice":
        allowed = (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mp4")
        if not filename.lower().endswith(allowed):
            raise HTTPException(status_code=400, detail="Unsupported audio format.")
        pages = extract_voice(raw_file, filename)

    else:
        raise HTTPException(status_code=400, detail="Unknown file type.")

    readable_text = "\n".join(page["text"] for page in pages).strip()

    if len(readable_text) < 20:
        raise HTTPException(
            status_code=400,
            detail="No readable text found. Try a clearer file."
        )

    documents = create_patient_documents(document_id, filename, pages)

    recreate_collection(
        patient_collection_name(document_id),
        documents
    )

    connection = db_connection()
    connection.execute(
        """
        INSERT INTO uploaded_documents
        (document_id, filename, source_type, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            document_id,
            filename,
            source_type,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    connection.commit()
    connection.close()

    return {
    "document_id": document_id,
    "filename": filename,
    "pages": len(pages),
    "chunks": len(documents),
    "preview": readable_text[:1200]
}


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    return await process_upload(file, "pdf")


@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    return await process_upload(file, "image")


@app.post("/upload/voice")
async def upload_voice(file: UploadFile = File(...)):
    return await process_upload(file, "voice")


@app.post("/ask")
def ask_question(request: AskRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    logger.info(f"Question Received: {request.question}")
    logger.info(f"Document ID: {request.document_id}")

    if is_emergency(request.question):
        return {
            "answer": EMERGENCY_RESPONSE,
            "sources": []
        }

    # Default values
    answer = ""
    sources = []

    # ==========================================================
    # MODE 1 : DOCUMENT RAG
    # ==========================================================
    if request.document_id:

        logger.info("Mode: Document RAG")

        documents = retrieve_documents(
            request.document_id,
            request.question
        )
        logger.info(f"Retrieved {len(documents)} document chunks.")
        if documents:

            context, sources = build_context_and_sources(documents)

            user_prompt = f"""
SOURCE CONTEXT

{context}

USER QUESTION

{request.question}
"""

            try:
                response = llm.invoke([
                    ("system", DOCUMENT_SYSTEM_PROMPT),
                    ("human", user_prompt)
                ])

                answer = response.content

            except Exception:
                logger.exception("LLM Error")

                raise HTTPException(
                    status_code=500,
                    detail="Unable to generate AI response."
                )

        else:
             logger.warning("No relevant document chunks found.")

             answer = (
                "I could not find relevant information in the uploaded document.\n\n"
                "Try asking another question or upload a different document."
            )

    # ==========================================================
    # MODE 2 : GENERAL MEDICAL QUESTIONS
    # ==========================================================
    else:

        logger.info("Mode: General Medical AI")

        try:

            response = llm.invoke([
                ("system", GENERAL_MEDICAL_SYSTEM_PROMPT),
                ("human", request.question)
            ])

            answer = response.content
            logger.info("LLM response generated successfully.")
        except Exception:

            logger.exception("LLM Error")

            raise HTTPException(
                status_code=500,
                detail="Unable to generate AI response."
            )

    connection = db_connection()

    connection.execute(
        """
        INSERT INTO chat_history
        (document_id, question, answer, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            request.document_id,
            request.question,
            answer,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    connection.commit()
    connection.close()

    logger.info("Response generated successfully.")

    return {
        "answer": answer,
        "sources": sources
    }


@app.post("/tts")
def text_to_speech(request: TTSRequest):
    # Demo-only TTS. Do not send real patient data to external TTS
    # services without privacy review and user consent.
    audio_buffer = BytesIO()

    speech = gTTS(
        text=request.text[:3500],
        lang="en"
    )

    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return StreamingResponse(
        audio_buffer,
        media_type="audio/mpeg"
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    logger.info(f"Deleting document: {document_id}")
    import chromadb

    client = chromadb.PersistentClient(path=VECTOR_DIR)

    try:
        client.delete_collection(patient_collection_name(document_id))
    except Exception:
        pass

    connection = db_connection()
    connection.execute(
        "DELETE FROM uploaded_documents WHERE document_id = ?",
        (document_id,)
    )
    connection.execute(
        "DELETE FROM chat_history WHERE document_id = ?",
        (document_id,)
    )
    connection.commit()
    connection.close()

    return {"message": "Document text and related history deleted."}