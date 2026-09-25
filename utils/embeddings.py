from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import (
    Document
)

def chunk_documents(
    pages
):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250
        )
    )

    documents = []

    for page in pages:

        chunks = splitter.split_text(
            page["text"]
        )

        for chunk in chunks:

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "page": page["page"],
                        "source": page["source"]
                    }
                )
            )

    return documents