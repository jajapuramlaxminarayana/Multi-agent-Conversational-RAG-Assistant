def retrieval_agent(
    vectorstore,
    query,
    selected_document=None
):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20
        }
    )

    docs = retriever.invoke(query)

    filtered_docs = []

    for doc in docs:

        source = (
            doc.metadata.get(
                "source",
                ""
            )
        )

        lower_text = (
            doc.page_content.lower()
        )

        if (
            "references" in lower_text
            or "bibliography" in lower_text
            or "et al." in lower_text
            or "conference on" in lower_text
            or "proceedings of" in lower_text
        ):

            continue

        if selected_document:

            if selected_document not in source:

                continue

        filtered_docs.append(doc)

    unique_docs = []

    seen = set()

    for doc in filtered_docs:

        text = (
            doc.page_content[:300]
        )

        if text not in seen:

            seen.add(text)

            unique_docs.append(doc)

    return unique_docs[:5]