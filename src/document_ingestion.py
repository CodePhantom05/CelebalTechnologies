from __future__ import annotations
import os
from typing import List, Dict

import fitz 


def load_pdf(path: str) -> Dict[str, str]:
    
    doc = fitz.open(path)
    pages_text = [page.get_text() for page in doc]
    doc.close()
    full_text = "\n".join(pages_text)
    return {"source": os.path.basename(path), "text": full_text}


def load_txt(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {"source": os.path.basename(path), "text": text}


def load_documents_from_folder(folder_path: str) -> List[Dict[str, str]]:
   
    documents: List[Dict[str, str]] = []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for filename in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, filename)
        lower = filename.lower()
        try:
            if lower.endswith(".pdf"):
                documents.append(load_pdf(full_path))
            elif lower.endswith(".txt") or lower.endswith(".md"):
                documents.append(load_txt(full_path))
        except Exception as e:
            print(f"[document_ingestion] Skipped '{filename}' due to error: {e}")

    if not documents:
        raise ValueError(f"No supported documents (.pdf/.txt/.md) found in {folder_path}")

    return documents


def load_huggingface_dataset(
    dataset_name: str = "vectara/open_ragbench",
    split: str = "train",
    text_column: str = "text",
    max_docs: int = 200,
) -> List[Dict[str, str]]:

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    documents: List[Dict[str, str]] = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        if text_column not in row:
            raise KeyError(
                f"Column '{text_column}' not found. Available columns: {list(row.keys())}"
            )
        documents.append({"source": f"{dataset_name}#{i}", "text": row[text_column]})
    return documents

