from langchain_community.vectorstores import (
    Chroma
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

embedding_model = (
    HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
)

def create_vectorstore(
    documents
):

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    return vectorstore