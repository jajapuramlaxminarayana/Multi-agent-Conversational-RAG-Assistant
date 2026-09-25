# Multi-Agent Conversational RAG Assistant

A hallucination-aware, multi-agent Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, **Llama 3.1**, and **SBERT**. Upload research papers (PDFs) and interact with them through a grounded, citation-backed conversational interface.

---

## Features

- **Multi-Agent Architecture** — 7 specialized agents collaborate in a pipeline for planning, retrieval, generation, critique, refinement, citation, and summarization
- **Hallucination Detection** — Critic agent scores every answer using cosine similarity (SBERT) + LLM evaluation; flags low-confidence responses
- **Critic-Refinement Loop** — Automatically re-generates and refines answers up to 3 times when confidence < 75%
- **Citation Grounding** — Citations ranked by cosine similarity between answer and retrieved chunks; top-3 sources displayed with page numbers and relevance scores
- **Conversational Memory** — Chat history is maintained across turns and injected into the planner for context-aware query reformulation
- **Multi-Document Support** — Upload multiple PDFs per session; select which document to query
- **Multiple Chat Sessions** — Create and switch between independent chat sessions with separate vector stores

---

## Architecture

```
User Query
    |
    v
[Planner Agent]         -- Reformulates query using chat history
    |
    v
[Retrieval Agent]       -- Semantic search via ChromaDB + SBERT embeddings
    |
    v
[Generator Agent]       -- Generates answer from context using Llama 3.1
    |
    v
[Critic Agent]          -- Scores answer: cosine similarity (40%) + LLM eval (60%)
    |
    v (if confidence < 75%)
[Refinement Agent]      -- Rewrites answer; loops up to 3 times
    |
    v
[Citation Agent]        -- Ranks retrieved chunks by similarity; returns top-3 citations
    |
    v
[Summarizer Agent]      -- (On-demand) Summarizes entire uploaded document
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.1 8B (via Groq API) |
| Embeddings | SBERT (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |
| Similarity Scoring | scikit-learn cosine similarity |
| PDF Parsing | LangChain PyPDFLoader |

---

## Project Structure

```
Multi-agent-Conversational-RAG-Assistant/
│
├── app.py                        # Main Streamlit app & agent orchestration
│
├── agents/
│   ├── planner_agent.py          # Query reformulation with memory
│   ├── retrieval_agent.py        # ChromaDB semantic retrieval
│   ├── generator_agent.py        # Answer generation (Llama 3.1)
│   ├── critic_agent.py           # Hallucination scoring & evaluation
│   ├── refinement_agent.py       # Answer refinement loop
│   ├── citation_agent.py         # Citation grounding & ranking
│   └── summarizer_agent.py       # Document summarization
│
├── utils/
│   ├── pdf_loader.py             # PDF ingestion with PyPDFLoader
│   ├── embeddings.py             # Document chunking with SBERT
│   └── vectorstore.py            # ChromaDB vector store creation
│
├── Basepaper.pdf                 # Reference research paper
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- A [Groq API Key](https://console.groq.com/) (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/jajapuramlaxminarayana/Multi-agent-Conversational-RAG-Assistant.git
cd Multi-agent-Conversational-RAG-Assistant
```

### 2. Install Dependencies

```bash
pip install streamlit langchain langchain-community chromadb \
            sentence-transformers scikit-learn groq pypdf \
            python-dotenv
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Usage

1. **Upload PDFs** — Drag and drop one or more research papers
2. **Select Document** — Choose which PDF to query from the dropdown
3. **Ask Questions** — Type your question in the chat input
4. **View Results** — See the answer, confidence score, critic analysis, and cited sources
5. **Generate Summary** — Click "Generate Research Summary" for an on-demand document overview
6. **New Chat** — Start a fresh session with a new vector store at any time

---

## Agent Details

| Agent | Role |
|-------|------|
| **Planner Agent** | Reformulates the user's query using conversation history for better retrieval |
| **Retrieval Agent** | Fetches top-k semantically relevant chunks from ChromaDB |
| **Generator Agent** | Generates a contextual answer using Llama 3.1 via Groq API |
| **Critic Agent** | Scores factual consistency using SBERT similarity (40%) + LLM evaluation (60%) |
| **Refinement Agent** | Refines low-confidence answers (score < 75) in up to 3 iterations |
| **Citation Agent** | Ranks retrieved chunks by cosine similarity to the answer; returns top-3 with page numbers |
| **Summarizer Agent** | Produces a structured summary of the entire uploaded document |

---

## Hallucination Mitigation

The system uses a **dual-scoring** approach in the Critic Agent:

```
Final Score = 0.4 * (SBERT cosine similarity) + 0.6 * (LLM factual evaluation score)
```

- **Score >= 75** → Answer is considered grounded ✅
- **Score < 75** → Refinement loop triggered (up to 3 iterations) ⚠️

---

## License

This project is for educational and research purposes.

---

## Acknowledgements

- [Groq](https://groq.com/) for ultra-fast Llama 3.1 inference
- [LangChain](https://langchain.com/) for the RAG pipeline framework
- [ChromaDB](https://www.trychroma.com/) for the vector store
- [Sentence Transformers](https://www.sbert.net/) for SBERT embeddings