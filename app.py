import os
import requests
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
)

st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --------------------------------------------------
# API HELPERS
# --------------------------------------------------

def api_get(path):

    return requests.get(
        f"{API_URL}{path}",
        timeout=60
    )


def api_post(path, **kwargs):

    return requests.post(
        f"{API_URL}{path}",
        timeout=180,
        **kwargs
    )


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

def upload_file(uploaded_file, endpoint):

    response = api_post(

        endpoint,

        files={
            "file":(
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

    )

    if response.status_code != 200:

        st.error(response.text)

        return

    data = response.json()

    st.session_state.document_id = data["document_id"]

    st.session_state.document_name = data["filename"]

    st.success("✅ Document processed successfully.")

    st.success(
        f"""
Document ID

{data["document_id"]}

Pages Indexed : {data["pages"]}

Chunks Created : {data["chunks"]}
"""
    )

    with st.expander("📄 Extracted Text Preview"):

        st.write(data["preview"])


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("🩺 AI Medical Assistant")


left, right = st.columns(2)

with left:
    st.markdown("""
### 📄 Document Analysis

Upload

- PDF
- Prescription Image
- Laboratory Report
- Voice Recording

Ask questions about your uploaded medical document.
""")

with right:
    st.markdown("""
### 📄 General Medical Questions

No upload required.

You can ask your medical related questions.
""")



st.divider()

# --------------------------------------------------
# TABS
# --------------------------------------------------

document_tab, general_tab = st.tabs(

    [

        "📄 Document Analysis",

        " 📄General Medical Questions"

    ]

)

# ==================================================
# DOCUMENT ANALYSIS TAB
# ==================================================

with document_tab:

    st.subheader("Upload Medical Document")

    uploaded_file = st.file_uploader(

        "Upload PDF, Image or Voice",

        type=[
            "pdf",
            "jpg",
            "jpeg",
            "png",
            "wav",
            "mp3",
            "m4a",
            "ogg",
            "webm",
            "flac"
        ]

    )

    if uploaded_file:

        extension = uploaded_file.name.split(".")[-1].lower()

        if extension == "pdf":

            endpoint = "/upload/pdf"

        elif extension in [

            "jpg",

            "jpeg",

            "png"

        ]:

            endpoint = "/upload/image"

        else:

            endpoint = "/upload/voice"

        if st.button(

            "Upload & Analyze",

            use_container_width=True

        ):

            with st.spinner(

                "Extracting medical information..."

            ):

                upload_file(

                    uploaded_file,

                    endpoint

                )

    if st.session_state.document_id:

        st.success(

            f"📄 Active Document : {st.session_state.document_name}"

        )
        # --------------------------------------------------
        # DOCUMENT QUESTION
        # --------------------------------------------------

        document_question = st.text_area(
            "Ask a question about your uploaded document",
            
            key="document_question"
        )

        col1, col2 = st.columns(2)

        with col1:

            ask_document = st.button(
                "Ask Document",
                type="primary",
                use_container_width=True
            )

        with col2:

            summarize = st.button(
                "Summarize Report",
                use_container_width=True
            )

        if summarize:

            document_question = (
                "Summarize this uploaded medical document in simple language. "
                "List important findings, abnormal values, medicines, "
                "follow-up instructions and questions I should ask my doctor."
            )

            ask_document = True

        if ask_document:

            if not document_question.strip():

                st.error("Please enter a question.")

            else:

                with st.spinner("Generating clinical insight..."):

                    response = api_post(
                        "/ask",
                        json={
                            "question": document_question,
                            "document_id": st.session_state.document_id
                        }
                    )

                    if response.status_code != 200:

                        st.error(response.text)

                    else:

                        data = response.json()

                        st.session_state.last_answer = data["answer"]

                        st.session_state.last_sources = data.get(
                            "sources",
                            []
                        )

                        st.subheader("Clinical Insight")

                        st.success(
                            "📄 Answer generated using your uploaded document."
                        )

                        st.write(data["answer"])

                        if data.get("sources"):

                            st.markdown("### Sources")

                            for source in data["sources"]:

                                st.caption(
                                    f"• [{source['id']}] {source['label']}"
                                )

        # --------------------------------------------------
        # TEXT TO SPEECH
        # --------------------------------------------------

        if st.session_state.last_answer:

            if st.button(
                "🔊 Read Answer Aloud",
                use_container_width=True
            ):

                with st.spinner("Generating speech..."):

                    response = api_post(
                        "/tts",
                        json={
                            "text": st.session_state.last_answer
                        }
                    )

                    if response.status_code == 200:

                        st.audio(
                            response.content,
                            format="audio/mp3"
                        )

                    else:

                        st.error("Unable to generate speech.")

        # --------------------------------------------------
        # DELETE DOCUMENT
        # --------------------------------------------------

        if st.button(
            "🗑 Delete Uploaded Document",
            use_container_width=True
        ):

            response = requests.delete(
                f"{API_URL}/documents/{st.session_state.document_id}"
            )

            if response.status_code == 200:

                st.session_state.document_id = None
                st.session_state.document_name = None
                st.session_state.last_answer = ""
                st.session_state.last_sources = []

                st.success("Document deleted successfully.")

                st.rerun()

            else:

                st.error("Unable to delete document.")

    else:

        st.info(
            """
No document uploaded.

Upload a PDF, Image or Voice recording to use Document Analysis.
"""
        )

# ==================================================
# GENERAL MEDICAL QUESTIONS TAB
# ==================================================

with general_tab:

    st.subheader("General Medical Questions")

    st.info(
        """
Ask educational medical questions.

Examples

• What is diabetes?

• Explain hypertension.

• What is cholesterol?

• What is HbA1c?

• Explain MRI.

• Explain CT Scan.

• What causes high blood pressure?

No document upload is required.
"""
    )

    general_question = st.text_area(
        "Ask your medical question",
        placeholder="Example: What is diabetes?",
        key="general_question"
    )

    ask_general = st.button(
        "Ask General Medical Question",
        type="primary",
        use_container_width=True
    )

    if ask_general:

        if not general_question.strip():

            st.error("Please enter a medical question.")

        else:

            with st.spinner("Generating answer..."):

                response = api_post(

                    "/ask",

                    json={
                        "question": general_question,
                        "document_id": None
                    }

                )

                if response.status_code != 200:

                    st.error(response.text)

                else:

                    data = response.json()

                    st.session_state.last_answer = data["answer"]

                    st.subheader("Medical Explanation")

                    st.info(
                        "🩺 Answer generated using General Medical AI."
                    )

                    st.write(data["answer"])

                    if data.get("sources"):

                        st.markdown("### Sources")

                        for source in data["sources"]:

                            st.caption(
                                f"• [{source['id']}] {source['label']}"
                            )

    if st.session_state.last_answer:

        if st.button(
            "🔊 Read Answer Aloud",
            key="general_tts",
            use_container_width=True
        ):

            response = api_post(

                "/tts",

                json={
                    "text": st.session_state.last_answer
                }

            )

            if response.status_code == 200:

                st.audio(
                    response.content,
                    format="audio/mp3"
                )

            else:

                st.error("Unable to generate speech.")


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
"""
🩺 AI Medical Assistant

Built With

• FastAPI
• Streamlit
• ChromaDB
• Sentence Transformers
• EasyOCR
• Whisper
• SQLite
• Groq LLM
• Docker

"""
)