import streamlit as st
import uuid
import shutil
import os
import time

from utils.pdf_loader import load_pdf
from utils.embeddings import chunk_documents
from utils.vectorstore import create_vectorstore

from agents.planner_agent import planner_agent
from agents.retrieval_agent import retrieval_agent
from agents.generator_agent import generator_agent
from agents.refinement_agent import refinement_agent
from agents.critic_agent import critic_agent
from agents.citation_agent import citation_agent
from agents.summarizer_agent import summarizer_agent

st.set_page_config(
    page_title="Agentic RAG Assistant",
    layout="wide"
)

if "sessions" not in st.session_state:

    st.session_state.sessions = {}

if "current_session" not in st.session_state:

    session_id = str(uuid.uuid4())

    st.session_state.current_session = session_id

    st.session_state.sessions[session_id] = {
        "title": "New Chat",
        "messages": [],
        "vectorstore": None,
        "documents_uploaded": False,
        "uploaded_files": []
    }

current_session = (
    st.session_state.sessions[
        st.session_state.current_session
    ]
)

st.sidebar.title("Agentic RAG")

if st.sidebar.button(
    "🗑 Clear Vector Database"
):

    try:

        session_id = (
            st.session_state.current_session
        )

        st.session_state.sessions[
            session_id
        ] = {
            "title": "New Chat",
            "messages": [],
            "vectorstore": None,
            "documents_uploaded": False,
            "uploaded_files": []
        }

        if os.path.exists("chroma_db"):

            time.sleep(2)

            shutil.rmtree(
                "chroma_db",
                ignore_errors=True
            )

        st.success(
            """
                Vector database cleared successfully.
                Upload fresh PDFs.
            """
        )

    except Exception as e:

        st.error(
            f"Error clearing DB: {e}"
        )

    st.rerun()

if st.sidebar.button(
    "➕ New Chat"
):

    session_id = str(
        uuid.uuid4()
    )

    st.session_state.current_session = (
        session_id
    )

    st.session_state.sessions[
        session_id
    ] = {
        "title": "New Chat",
        "messages": [],
        "vectorstore": None,
        "documents_uploaded": False,
        "uploaded_files": []
    }

    st.rerun()

st.sidebar.subheader(
    "Chat Sessions"
)

session_titles = {}

for session_id, session_data in (
    st.session_state.sessions.items()
):

    title = (
        session_data["title"]
        + f" ({session_id[:4]})"
    )

    session_titles[
        title
    ] = session_id

selected_title = st.sidebar.radio(
    "Select Chat",
    list(session_titles.keys())
)

st.session_state.current_session = (
    session_titles[selected_title]
)

current_session = (
    st.session_state.sessions[
        st.session_state.current_session
    ]
)

st.sidebar.subheader(
    "Agent Workflow"
)

st.sidebar.markdown("""
1. Planner Agent  
2. Retrieval Agent  
3. Generator Agent  
4. Critic Agent  
5. Refinement Agent  
6. Citation Agent  
7. Summarizer Agent  
""")

st.title(
    "Multi-Agent Hallucination-Aware RAG Assistant"
)

st.write(
    """
Upload research papers and interact with
them using a grounded multi-agent RAG system.
"""
)

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

selected_document = None

if uploaded_files:

    pdf_names = [
        file.name
        for file in uploaded_files
    ]

    selected_document = st.selectbox(
        "Select document for querying",
        pdf_names
    )

if (
    uploaded_files
    and not current_session[
        "documents_uploaded"
    ]
):

    all_pages = []

    current_session[
        "uploaded_files"
    ] = pdf_names

    with st.spinner(
        "Processing PDFs..."
    ):

        for uploaded_file in uploaded_files:

            with open(
                uploaded_file.name,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.read()
                )

            pages = load_pdf(
                uploaded_file.name
            )

            all_pages.extend(
                pages
            )

        chunks = chunk_documents(
            all_pages
        )

        vectorstore = create_vectorstore(
            chunks
        )

        current_session[
            "vectorstore"
        ] = vectorstore

        current_session[
            "documents_uploaded"
        ] = True

    st.success(
        "Documents processed successfully."
    )

if not current_session[
    "documents_uploaded"
]:

    st.warning(
        "Please upload PDFs for this chat session."
    )

    st.stop()

if st.button(
    "Generate Research Summary"
):

    with st.spinner(
        "Generating summary..."
    ):

        retriever = (
            current_session[
                "vectorstore"
            ]
            .as_retriever(
                search_kwargs={
                    "k": 10
                }
            )
        )

        docs = retriever.invoke(
            "summarize the paper"
        )

        context = "\n".join(
            [
                doc.page_content
                for doc in docs
            ]
        )

        summary = summarizer_agent(
            context
        )

        st.subheader(
            "Research Summary"
        )

        st.write(summary)

current_chat = current_session[
    "messages"
]


for chat in current_chat:

    with st.chat_message("user"):

        st.write(
            chat["question"]
        )

    with st.chat_message(
        "assistant"
    ):

        st.write(
            chat["answer"]
        )


        st.subheader(
            "Confidence Score"
        )

        st.write(
            f"{float(chat['confidence']):.2f}%"
        )

        st.progress(
            float(chat["confidence"]) / 100
        )


        with st.expander(
            "Critic Analysis"
        ):

            st.write(
                chat["critic_reason"]
            )

        if float(
            chat["confidence"]
        ) < 75:

            st.warning(
                """
Potential hallucination detected.
Retrieval grounding may be weak.
"""
            )

        else:

            st.success(
                """
Answer appears grounded
in retrieved context.
"""
            )

        st.subheader(
            "Retrieved Sources"
        )

        for citation in (
            chat["citations"]
        ):

            st.markdown(
                f"""
### Source:
{citation['source']}

### Page:
{citation['page']}

### Relevance Score:
{citation['score']}%

### Evidence:
{citation['text']}
"""
            )


query = st.chat_input(
    "Ask a question about the document..."
)

if query:


    if current_session[
        "title"
    ] == "New Chat":

        current_session[
            "title"
        ] = query[:40]


    with st.chat_message("user"):

        st.write(query)


    memory = ""

    for chat in current_chat:

        memory += f"""
User:
{chat['question']}

Assistant:
{chat['answer']}
"""


    with st.spinner(
        "Planner Agent Thinking..."
    ):

        improved_query = planner_agent(
            query,
            memory
        )

    st.subheader(
        "Planner Agent Output"
    )

    st.info(
        improved_query
    )


    docs = retrieval_agent(
        current_session[
            "vectorstore"
        ],
        improved_query,
        selected_document
    )

    context = "\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    with st.spinner(
        "Generating Answer..."
    ):

        answer = generator_agent(
            query,
            context,
            memory
        )


    critic_result = critic_agent(
        answer,
        context
    )

    confidence = (
        critic_result["score"]
    )

    critic_reason = (
        critic_result["reason"]
    )


    max_iterations = 3

    iteration = 0

    while (
        confidence < 75
        and iteration < max_iterations
    ):

        previous_confidence = confidence

        iteration += 1

        with st.spinner(
            f"Refinement Iteration {iteration}..."
        ):

            improved_answer = (
                refinement_agent(
                    query,
                    context,
                    answer
                )
            )

            improved_result = (
                critic_agent(
                    improved_answer,
                    context
                )
            )

            improved_confidence = (
                improved_result["score"]
            )

            improved_reason = (
                improved_result["reason"]
            )


        st.info(
            f"""
Iteration {iteration}

Previous Confidence:
{previous_confidence:.2f}%

New Confidence:
{improved_confidence:.2f}%
"""
        )

        answer = improved_answer

        confidence = improved_confidence

        critic_reason = improved_reason

        if confidence >= 75:

            st.success(
                f"""
                    Answer successfully improved
                    after {iteration} refinement iterations.
                """
            )

            break

        if improved_confidence <= previous_confidence:

            break


    citations = citation_agent(
        answer,
        docs
    )

    current_chat.append(
        {
            "question": query,
            "answer": answer,
            "confidence": float(
                confidence
            ),
            "critic_reason": critic_reason,
            "citations": citations
        }
    )

    st.rerun()