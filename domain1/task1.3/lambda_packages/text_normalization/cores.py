# cores.py

import os
import sys
import json
from typing import Any, Dict, List, Optional, Iterable 

import pymupdf

from .utils import normalize_text, now_iso, sha256_hex, sha256_path
from .readers import read_txt_text, read_pdf_text, read_docx_text, read_html_text, read_json_text, read_csv_text


# ------------------------------
# Document builder
# ------------------------------
def build_docs(path: str, id_strategy: str = "path") -> Optional[Dict[str, Any]]:
    """
    Build a normalized document object from a file for RAG ingestion.
    id_strategy: 'path' (stable per file) or 'content' (stable per content).
    """
    ext  = os.path.splitext(path)[1].lower()

    raw  = None
    mime = None

    # ---------------------------
    # Read content by extension
    # ---------------------------
    if ext == ".txt":
        raw = read_txt_text(path)
        mime = "text/plain"

    elif ext in (".docx", ".docm"):
        raw = read_docx_text(path, include_headers_footers=True)
        if ext == ".docx":
            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext == ".docm":
            mime = "application/vnd.ms-word.document.macroEnabled.12"

    elif ext == ".pdf":
        raw = read_pdf_text(path)
        mime = "application/pdf"

    elif ext == ".html":
        raw = read_html_text(path)
        mime = "text/html"

    elif ext in (".json", ".jsonl"):
        raw = read_json_text(path)
        mime = "application/json"

    elif ext == ".csv":
        raw = read_csv_text(path)
        mime = "text/csv"

    else:
        return None  # unsupported type

    if not raw:
        return None  # unreadable or empty input

    # ---------------------------
    # Normalize text
    # ---------------------------
    norm = normalize_text(raw)
    if not norm:
        return None  # empty after normalization

    # ---------------------------
    # Metadata
    # ---------------------------
    abs_path = os.path.abspath(path)
    file_name = os.path.basename(path)

    meta: Dict[str, Any] = {
        "ext": ext.lstrip("."),
        "size_bytes": os.path.getsize(path),
        "content_sha256": sha256_hex(norm),
        "path_sha256": sha256_path(path),
        "page_count": None
    }

    # PDF page count (best-effort)
    if ext == ".pdf" and pymupdf is not None:
        try:
            with pymupdf.open(path) as pdf:
                meta["page_count"] = len(pdf)
        except Exception:
            pass

    # ---------------------------
    # ID strategy
    # ---------------------------
    if id_strategy == "content":
        doc_id = meta["content_sha256"]
    elif id_strategy == "path":
        doc_id = meta["path_sha256"]
    else:
        raise ValueError("id_strategy must be 'path' or 'content'")

    # ---------------------------
    # Final document
    # ---------------------------
    return {
        "id": doc_id,
        "source_path": abs_path,
        "file_name": file_name,
        "title": file_name,
        "mime_type": mime,
        "created_at": now_iso(),
        "text": norm,
        "metadata": meta
    }



# ------------------------------
# Chunk builder
# ------------------------------
def build_chunks(doc: Dict[str, Any], *, max_chars: int = 1024, overlap: int = 64) -> List[Dict[str, Any]]:
    """
    Split a document into overlapping, paragraph-aware chunks.
    Preserves semantic boundaries and source metadata for RAG ingestion.
    """     
    paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]

    chunks = []
    buf = ""
    char_start = 0
    chunk_index = 0

    for p in paragraphs:
        if len(buf) + len(p) > max_chars:
            chunk_id = f"{doc['id']}:{chunk_index}"
            chunks.append({
                "id": chunk_id,
                "doc_id": doc["id"],
                "chunk_index": chunk_index,
                "text": buf.strip(),
                "char_start": char_start,
                "char_end": char_start + len(buf),
                "metadata": doc.get("metadata", {})
            })

            chunk_index += 1
            buf = buf[-overlap:] + "\n\n" + p
            char_start += len(buf) - overlap
        else:
            buf += "\n\n" + p if buf else p

    if buf.strip():
        chunks.append({
            "id": f"{doc['id']}:{chunk_index}",
            "doc_id": doc["id"],
            "chunk_index": chunk_index,
            "text": buf.strip(),
            "char_start": char_start,
            "char_end": char_start + len(buf),
            "metadata": doc.get("metadata", {})
        })

    return chunks



# ------------------------------
# Corpus builder
# ------------------------------
def find_files(
    root_dir: str, 
    allowed_extensions: Iterable[str] = (
        ".csv",".docx",".docm",".json",".jsonl",".html",".pdf",".txt"
        )
    ) -> List[str]:
    """Recursively scan a directory and return files matching allowed extensions"""

    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}"
               for ext in allowed_extensions}

    file_list = []
    for root, _, files in os.walk(root_dir, topdown=True):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in allowed:
                file_list.append(os.path.join(root, fname))

    if not file_list:
        raise FileNotFoundError(f"No files found in {root_dir} with extensions {allowed_extensions}")
    else:
        return sorted(file_list)



def build_corpus(root_dir: str, id_strategy: str = "path") -> List[Dict[str, Any]]:
    """
    Recursively scans root_dir for supported files and builds document dictionaries.
    
    Each document contains normalized text, metadata, and a deterministic ID.
    Returns a list of document dictionaries suitable for ingestion into a RAG pipeline.
    """
    
    allowed_extensions = [".csv",".docx",".docm",".json",".jsonl",".html",".pdf",".txt"]

    file_list = find_files(root_dir, allowed_extensions = allowed_extensions)
    
    doc_list      = []
    skipped_files = []
    for file_path in file_list:
        doc = build_docs(file_path, id_strategy=id_strategy)
        if doc:
            doc_list.append(doc)
        else:
            skipped_files.append(file_path)
            print(f"Skipped unreadable or empty file: {file_path}")
        
    return doc_list
