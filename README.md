# 🩺 Multimodal AI Medical Assistant

An **end-to-end Multimodal AI Medical Assistant** that leverages **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLMs)**, **Optical Character Recognition (OCR)**, and **Speech Recognition** to help users understand medical documents and ask general medical education questions. The application supports intelligent analysis of PDF reports, prescription images, and voice recordings through a context-aware AI workflow.

---

## ✨ Features

### 📄 Medical Document Analysis
- Upload PDF medical reports
- Upload prescription and laboratory images
- Upload voice recordings
- Automatic text extraction using OCR and speech recognition
- AI-powered medical report summarization
- Context-aware document question answering using RAG
- Medical terminology explanation
- Text-to-Speech (TTS) responses
- Document lifecycle management

### 📄 General Medical Assistant
- Ask general medical education questions without uploading documents
- AI-generated educational responses powered by LLMs
- Emergency keyword detection with safety guidance

---

## 🛠 Tech Stack

**Programming Language**
- Python

**Backend**
- FastAPI
- SQLite

**Frontend**
- Streamlit

**Generative AI**
- Groq Llama 3.3 (70B)
- LangChain
- HuggingFace Sentence Transformers
- Retrieval-Augmented Generation (RAG)

**Vector Database**
- ChromaDB

**Document Processing**
- PyPDF
- EasyOCR
- OpenCV

**Speech Processing**
- Whisper Large V3 Turbo

**Speech Output**
- gTTS (Text-to-Speech)

---

## 🏗 System Workflow

```text
User
 │
 ├── Upload PDF / Image / Voice
 │
 ▼
Text Extraction
(PDF | EasyOCR | Whisper)
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
Semantic Retrieval (RAG)
 │
 ▼
Groq Llama 3.3
 │
 ▼
AI Response
 │
 ▼
Streamlit Interface
```

---

## 📂 Project Structure

```text
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
└── README.md
```

---

## 🚀 Key Highlights

- Built an **end-to-end Multimodal AI application** integrating OCR, Speech Recognition, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs).
- Developed a **context-aware medical document understanding system** capable of processing PDF reports, prescription images, and voice recordings.
- Implemented a **RAG pipeline** using LangChain, ChromaDB, and HuggingFace Sentence Transformers for semantic retrieval and document-grounded AI responses.
- Designed separate AI workflows for **medical document analysis** and **general medical education**.
- Developed a modular application using **FastAPI**, **Streamlit**, and **SQLite**, enabling REST APIs, report summarization, contextual question answering, and Text-to-Speech.

---

## 📚 Future Enhancements

- Multi-document support
- Conversational memory for follow-up questions
- Enhanced retrieval and reranking strategies
- Cloud deployment
- Authentication and user management
- Improved OCR support for handwritten prescriptions

---

## ⚠ Disclaimer

This application is developed for **educational and informational purposes only**. It is designed to help users understand medical documents and provide general medical education. It **does not diagnose diseases, prescribe medications, or replace professional medical advice**. Always consult a qualified healthcare professional before making medical decisions.

