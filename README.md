# Multimodal-AI-Medical-Assistant
An end-to-end Multimodal AI Medical Assistant that combines Optical Character Recognition (OCR), Speech Recognition, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs) to help users understand medical documents and answer general medical education questions.

---

## ✨ Features

### 📄 Medical Document Analysis
- Upload PDF medical reports
- Upload prescription/laboratory images
- Upload voice recordings
- Automatic text extraction
- AI-powered report summarization
- Document-specific Question Answering using RAG
- Medical terminology explanation
- Text-to-Speech responses

### 🩺 General Medical Assistant
- Ask medical education questions without uploading documents
- AI-generated medical explanations
- Emergency keyword detection with safety response

---
## 🛠️ Tech Stack

#### Python • FastAPI • Streamlit • LangChain • ChromaDB • Groq Llama 3.3 • EasyOCR 
####  OpenCV • Whisper • SQLite • HuggingFace Embeddings • PyPDF • Docker
---

## 🏗 System Workflow

```text
User
 │
 ├── Upload PDF / Image / Voice
 │
 ▼
Text Extraction
(PDF | OCR | Whisper)
 │
 ▼
Text Chunking
 │
 ▼
Sentence Transformer Embeddings
 │
 ▼
ChromaDB Vector Store
 │
 ▼
Retrieval-Augmented Generation (RAG)
 │
 ▼
Groq Llama 3.3 70B
 │
 ▼
AI Response
 │
 ▼
Streamlit Interface
```

---

## 📂 Project Structure

```
AI-Medical-Assistant/
│
├── backend/
│   ├── main.py
│   └── config.py
│
├── frontend/
│   └── app.py
│
├── data/
│
├── requirements.txt
├── Dockerfile

```

## 📌 Key Highlights

- Multimodal AI application supporting PDF, Image, and Voice inputs
- Retrieval-Augmented Generation (RAG) for document-grounded responses
- OCR using EasyOCR with OpenCV preprocessing
- Speech-to-text using Whisper Large V3 Turbo
- Context-aware medical report summarization
- General medical education assistant
- Vector search using ChromaDB
- FastAPI REST APIs with Streamlit frontend
- Docker-ready architecture

---

## ⚠ Disclaimer

This application is designed for educational purposes only.
It does not diagnose diseases, prescribe medications, or replace professional medical advice.
Always consult a qualified healthcare professional before making medical decisions.

---

