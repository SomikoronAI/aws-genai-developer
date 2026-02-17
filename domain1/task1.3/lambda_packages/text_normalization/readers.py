# readers.py

import os
import sys
import json
from typing import Any, Dict, List, Optional

import csv
import pymupdf
from docx import Document
from docx.document import Document as _Document
from docx.table import _Cell, _Column, _Row, Table
from docx.text.paragraph import Paragraph
from bs4 import BeautifulSoup


# ------------------------------
# Extractors Methods
# ------------------------------
def read_txt_text(path: str) -> Optional[str]:
    """Extract text from a .txt file"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] Failed to read TXT: {path} ({e})")
        return None
        

def read_pdf_text(path: str) -> Optional[str]:
    """Extract text from a .pdf file"""
    try:
        doc = pymupdf.open(path)
        if doc.is_encrypted:
            try:
                doc.authenticate("")  # try empty password
            except Exception:
                print(f"[WARN] Encrypted PDF (no text): {path}")
                return None

        texts = []
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            texts.append( page.get_text("text") )
        doc.close()
        text = "\n\n".join(texts)
        return text if text.strip() else None
    except Exception as e:
        print(f"[WARN] Failed to read PDF: {path} ({e})")
        return None


def read_html_text(path: str) -> Optional[str]:
    """Extract text from a .html file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read() 
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text()
            return text if text.strip() else None
    except Exception as e:
        print(f"[WARN] Failed to read HTML: {path} ({e})")
        return None    


# Microsoft document files
def _iter_block_items(parent):
    """
    Yield Paragraph and Table objects in the order they appear in the document.
    Works for the document body and for table cells.
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        return

    for child in parent_elm.iterchildren():
        # tags end with }p for paragraph, }tbl for table
        tag = child.tag.lower()
        if tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif tag.endswith("}tbl"):
            yield Table(child, parent)


def _extract_from_paragraph(par: Paragraph) -> str:
    # paragraph.text already concatenates runs
    return par.text or ""


def _extract_from_table(tbl: Table) -> str:
    """
    Extract table contents in a simple, embedding-friendly format.
    Each row on its own line; cells separated by a tab.
    """
    lines: List[str] = []
    for row in tbl.rows:
        cells = [cell.text.strip() for cell in row.cells]
        lines.append("\t".join(cells))
    return "\n".join(lines)


def _extract_section_header_footer(doc: Document) -> str:
    """
    Headers/footers often repeat; we include them **once per section**.
    This avoids bloating text while still capturing important info.
    """
    chunks: List[str] = []
    for i, sec in enumerate(doc.sections):
        # Header
        try:
            htxt = []
            for item in _iter_block_items(sec.header):
                if isinstance(item, Paragraph):
                    htxt.append(_extract_from_paragraph(item))
                elif isinstance(item, Table):
                    htxt.append(_extract_from_table(item))
            htxt = "\n".join([s for s in htxt if s.strip()])
            if htxt.strip():
                chunks.append(f"[Section {i+1} Header]\n{htxt}")
        except Exception:
            pass

        # Footer
        try:
            ftxt = []
            for item in _iter_block_items(sec.footer):
                if isinstance(item, Paragraph):
                    ftxt.append(_extract_from_paragraph(item))
                elif isinstance(item, Table):
                    ftxt.append(_extract_from_table(item))
            ftxt = "\n".join([s for s in ftxt if s.strip()])
            if ftxt.strip():
                chunks.append(f"[Section {i+1} Footer]\n{ftxt}")
        except Exception:
            pass
    return "\n\n".join(chunks).strip()


def read_docx_text(path: str, include_headers_footers: bool = True) -> Optional[str]:
    """
    Read a .docx/.docm files.
    Return text that preserves the reading order of paragraphs and tables. 
    Returns None if empty/unreadable.
    """
    try:
        doc = Document(path)
    except Exception as e:
        print(f"[WARN] Failed to open Word file: {path} ({e})")
        return None

    body_chunks: List[str] = []

    # Document body in natural order
    for item in _iter_block_items(doc):
        if isinstance(item, Paragraph):
            txt = _extract_from_paragraph(item)
            if txt.strip():
                body_chunks.append(txt)
        elif isinstance(item, Table):
            ttxt = _extract_from_table(item)
            if ttxt.strip():
                body_chunks.append(ttxt)

    # Optional: include section headers/footers once
    if include_headers_footers:
        try:
            hf = _extract_section_header_footer(doc)
            if hf:
                body_chunks.append(hf)
        except Exception:
            # Non-fatal; headers/footers are optional
            pass

    text = "\n\n".join(body_chunks).strip()
    return text if text else None


def extract_text_fields(obj: Any) -> list[str]:
    """ Recursively extract values from keys containing 'text' """
    texts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "text" in k.lower() and isinstance(v, (str, int, float)):
                texts.append(str(v))
            else:
                texts.extend( extract_text_fields(v) )

    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text_fields(item))
    return texts


def read_json_text(path: str) -> Optional[str]:
    """
    Reads text content from a JSON or JSONL file and returns a single text string.

    - JSON: expects either a dict or a list of dicts
    - JSONL: expects one JSON object per line
    """
    try:
        texts = []

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if path.lower().endswith(".jsonl"):    #jsonl
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    texts.extend( extract_text_fields(data) ) 
            else:                                  # .json
                data = json.load(f)
                texts.extend(extract_text_fields(data))
                                                
        return "\n".join(texts) if texts else None

    except Exception as e:
        print(f"[WARN] Failed to read JSON/JSONL: {path} ({e})")
        return None


def read_csv_text(path: str) -> Optional[str]:
    """
    Reads text content from a CSV file by extracting values from
    any column whose name contains 'text' (case-insensitive).
    """
    try:
        texts = []

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                return None

            text_columns = [
                col for col in reader.fieldnames
                if "text" in col.lower()
            ]

            if not text_columns:
                return None

            for row in reader:
                for col in text_columns:
                    value = row.get(col)
                    if value:
                        texts.append(value.strip())

        return "\n".join(texts) if texts else None

    except Exception as e:
        print(f"[WARN] Failed to read CSV: {path} ({e})")
        return None
