import fitz

def load_pdf(
    pdf_path
):

    doc = fitz.open(pdf_path)

    pages = []

    for page_num in range(len(doc)):

        page = doc.load_page(page_num)

        text = page.get_text()

        pages.append(
            {
                "text": text,
                "page": page_num + 1,
                "source": pdf_path
            }
        )

    return pages