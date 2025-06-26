import os
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

def extract_text_from_file(filepath):
    """
    Extract plain text from a document.
    Supported formats: .txt, .pdf, .docx

    Args:
        filepath (str): Path to the input file

    Returns:
        str: Extracted text content
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext == ".txt":
        return _extract_text_from_txt(filepath)
    elif ext == ".pdf":
        return _extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return _extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def _extract_text_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _extract_text_from_pdf(filepath):
    return extract_pdf_text(filepath)


def _extract_text_from_docx(filepath):
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])

###  How to use  ###

# from extract_text import extract_text_from_file

# text = extract_text_from_file("data/fra.txt")
# print(text[:500])  # Preview first 500 characters


