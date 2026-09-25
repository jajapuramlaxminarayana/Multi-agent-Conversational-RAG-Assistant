from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def citation_agent(
    answer,
    docs
):

    citations = []

    answer_embedding = (
        embedding_model.encode(
            [answer]
        )
    )

    for doc in docs:

        chunk_text = (
            doc.page_content
        )

        chunk_embedding = (
            embedding_model.encode(
                [chunk_text]
            )
        )

        similarity = (
            cosine_similarity(
                answer_embedding,
                chunk_embedding
            )[0][0]
        )

        citations.append(
            {
                "source": doc.metadata.get(
                    "source",
                    "Unknown"
                ),

                "page": doc.metadata.get(
                    "page",
                    "Unknown"
                ),

                "text": chunk_text[:500],

                "score": round(
                    similarity * 100,
                    2
                )
            }
        )

    citations = sorted(
        citations,
        key=lambda x: x["score"],
        reverse=True
    )

    final_citations = []

    seen = set()

    for citation in citations:

        unique_key = (
            f"{citation['source']}"
            f"{citation['page']}"
        )

        if unique_key not in seen:

            seen.add(unique_key)

            final_citations.append(
                citation
            )

        if len(final_citations) == 3:

            break

    return final_citations