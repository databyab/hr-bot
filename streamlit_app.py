import streamlit as st

from utils import (
    detect_intent,
    generate_answer,
    ingest_documents,
    retrieve_documents,
)


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="HR",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top, rgba(255, 255, 255, 0.05), transparent 28%),
                linear-gradient(180deg, #0b0b0b 0%, #050505 100%);
            color: #f5f5f5;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101010 0%, #070707 100%);
            border-right: 1px solid #232323;
        }

        [data-testid="stHeader"] {
            background: rgba(5, 5, 5, 0.75);
        }

        .hero-card,
        .panel-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
        }

        .hero-card {
            margin-bottom: 1rem;
        }

        .subtle-text {
            color: #b9b9b9;
        }

        .stChatMessage {
            background: transparent;
        }

        .stTextInput input,
        .stTextArea textarea {
            background: #101010 !important;
            color: #f5f5f5 !important;
            border: 1px solid #2a2a2a !important;
        }

        div[data-testid="stButton"] > button {
            background: linear-gradient(180deg, #1d1d1d, #0f0f0f);
            color: #ffffff;
            border: 1px solid #2d2d2d;
            border-radius: 999px;
            padding: 0.5rem 1rem;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #6f6f6f;
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask me anything about the HR policies, or upload new PDFs to refresh the knowledge base."
            ),
        }
    ]


st.sidebar.title("HR Policy Assistant")
st.sidebar.markdown(
    """
    <div class="panel-card">
        <strong>Actions</strong><br>
        <span class="subtle-text">Use the button below to ingest PDFs from <code>data/raw</code>.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("Run ingestion", use_container_width=True):
    with st.sidebar.spinner("Reading PDFs and building the vector store..."):
        result = ingest_documents()

    if result["pages"] == 0:
        st.sidebar.error("No PDF files found in data/raw.")
    else:
        st.sidebar.success(
            f"Ingestion complete: {result['pages']} pages, {result['chunks']} chunks."
        )

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask me anything about the HR policies, or upload new PDFs to refresh the knowledge base."
            ),
        }
    ]
    st.rerun()


st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin: 0.35rem 0 0.25rem 0; color: #ffffff;">HR Policy Assistant</h1>
        <div class="subtle-text">
            Query policy documents, inspect sources, and keep the whole app in one lightweight deployment.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


quick_prompts = [
    "What is the leave policy?",
    "How do I request work from home?",
    "What is the notice period?",
]

prompt_columns = st.columns(len(quick_prompts))
for column, prompt in zip(prompt_columns, quick_prompts):
    if column.button(prompt, use_container_width=True):
        st.session_state.pending_question = prompt


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


question = st.chat_input("Ask a question about the HR policy...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking policy context..."):
            intent = detect_intent(question)

            if intent == "GREETING":
                answer = "Hello. Ask me a policy question and I’ll look it up for you."
                sources = []
            elif intent == "OUT_OF_SCOPE":
                answer = (
                    "I can only help with questions related to the provided HR policies."
                )
                sources = []
            elif intent == "UNKNOWN":
                answer = (
                    "I’m not sure what you mean. Try rephrasing the question more clearly."
                )
                sources = []
            else:
                documents = retrieve_documents(question)

                if not documents:
                    answer = (
                        "I could not find this information in the provided HR policy."
                    )
                    sources = []
                else:
                    answer = generate_answer(question, documents)
                    sources = [
                        {
                            "file_name": document.metadata.get("file_name"),
                            "page_number": document.metadata.get("page_number"),
                        }
                        for document in documents
                    ]

        st.write(answer)

        if sources:
            with st.expander("Sources", expanded=True):
                for source in sources:
                    st.markdown(
                        f"- {source['file_name']} · page {source['page_number']}"
                    )

        st.caption(f"Detected intent: {intent}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
