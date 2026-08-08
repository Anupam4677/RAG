"""PDF parsing and cleaning.

Extracts three kinds of content per page:
  - body text, reconstructed as markdown with '#'/'##'/'###' headers inferred
    from relative font size / boldness (gives us structural hierarchy without
    a layout-ML dependency)
  - tables, extracted with pdfplumber and rendered as markdown tables
  - images/charts, extracted with PyMuPDF and captioned with a vision LLM

Also strips repeated running headers/footers, bare page numbers, and
watermark-style boilerplate that would otherwise pollute every chunk.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import pymupdf as fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI, RateLimitError

from src import config

_PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?\s*$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"[ \t]+")


@dataclass
class TableBlock:
    page_num: int
    table_index: int
    markdown: str


@dataclass
class PageContent:
    page_num: int
    markdown_text: str
    tables: list[TableBlock] = field(default_factory=list)
    image_captions: list[str] = field(default_factory=list)


@dataclass
class DocumentContent:
    source_path: str
    filename: str
    title: str
    num_pages: int
    pages: list[PageContent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Boilerplate (headers/footers/watermarks) detection
# ---------------------------------------------------------------------------

def _normalize_line(line: str) -> str:
    line = _WHITESPACE_RE.sub(" ", line).strip()
    return re.sub(r"\d+", "#", line)  # collapse numbers so "Page 3" == "Page 4"


def _find_boilerplate_lines(pages_raw_lines: list[list[str]]) -> set[str]:
    """Lines repeated across many pages (headers/footers/watermarks)."""
    num_pages = max(len(pages_raw_lines), 1)
    counter: Counter[str] = Counter()
    for lines in pages_raw_lines:
        seen_this_page = set()
        for raw in lines:
            norm = _normalize_line(raw)
            if not norm or norm in seen_this_page:
                continue
            seen_this_page.add(norm)
            counter[norm] += 1
    threshold = max(3, int(num_pages * config.MIN_REPEAT_RATIO_FOR_BOILERPLATE))
    return {norm for norm, count in counter.items() if count >= threshold}


# ---------------------------------------------------------------------------
# Heading detection via font size
# ---------------------------------------------------------------------------

def _collect_span_sizes(doc: fitz.Document) -> list[float]:
    sizes = []
    for page in doc:
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes.append(round(span["size"], 1))
    return sizes


def _median(values: list[float]) -> float:
    if not values:
        return 10.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _heading_prefix(size: float, is_bold: bool, body_size: float) -> str:
    if size >= body_size * 1.4:
        return "# "
    if size >= body_size * 1.15:
        return "## "
    if is_bold and size >= body_size * 1.02:
        return "### "
    return ""


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def _table_to_markdown(table: list[list[str | None]]) -> str:
    if not table or not table[0]:
        return ""
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        # pad/truncate ragged rows to header width
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _extract_tables(pdf_path: str) -> dict[int, list[TableBlock]]:
    tables_by_page: dict[int, list[TableBlock]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                raw_tables = page.extract_tables()
            except Exception:
                raw_tables = []
            blocks = []
            for idx, t in enumerate(raw_tables):
                md = _table_to_markdown(t)
                if md and md.count("\n") >= 1:  # at least header+separator
                    blocks.append(TableBlock(page_num=i, table_index=idx, markdown=md))
            if blocks:
                tables_by_page[i] = blocks
    return tables_by_page


# ---------------------------------------------------------------------------
# Image extraction + vision captioning (with on-disk cache)
# ---------------------------------------------------------------------------

def _image_cache_path() -> str:
    return str(config.IMAGE_CACHE_DIR / "captions.json")


def _load_image_cache() -> dict:
    path = _image_cache_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_image_cache(cache: dict) -> None:
    with open(_image_cache_path(), "w", encoding="utf-8") as f:
        json.dump(cache, f)


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _caption_image(image_bytes: bytes, context_hint: str, cache: dict) -> str | None:
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest in cache:
        return cache[digest]

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    max_retries = 5
    caption = None
    for attempt in range(max_retries):
        try:
            resp = _get_client().chat.completions.create(
                model=config.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This image is from a bank annual report / investor "
                                    f"presentation ({context_hint}). Describe it factually in "
                                    "2-4 sentences: if it is a chart or table, state the metric, "
                                    "the time period, and the key numbers/trend shown. If it is a "
                                    "logo or decorative image, reply exactly with 'DECORATIVE'."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                max_tokens=220,
                temperature=0,
            )
            caption = resp.choices[0].message.content.strip()
            break
        except RateLimitError as e:
            wait = 2**attempt
            print(f"[pdf_parser] rate limited captioning image ({attempt + 1}/{max_retries}), waiting {wait}s: {e}")
            time.sleep(wait)
        except Exception as e:  # noqa: BLE001 - keep ingestion resilient to other API hiccups
            print(f"[pdf_parser] image captioning failed: {e}")
            break

    # Only cache confirmed outcomes (success or a real answer). A None here means
    # every retry failed - don't poison the cache, so a later ingestion run retries it.
    if caption is not None:
        cache[digest] = caption
    return caption


def _extract_and_caption_images(doc: fitz.Document, filename: str) -> dict[int, list[str]]:
    if not config.CAPTION_IMAGES:
        return {}
    cache = _load_image_cache()
    captions_by_page: dict[int, list[str]] = {}
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        images = page.get_images(full=True)[: config.MAX_IMAGES_PER_PAGE]
        captions = []
        for img in images:
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            width, height = base_image.get("width", 0), base_image.get("height", 0)
            if width < config.MIN_IMAGE_SIZE_PX or height < config.MIN_IMAGE_SIZE_PX:
                continue
            caption = _caption_image(
                base_image["image"], f"{filename}, page {page_num}", cache
            )
            if caption and caption.strip().upper() != "DECORATIVE":
                captions.append(caption)
        if captions:
            captions_by_page[page_num] = captions
    _save_image_cache(cache)
    return captions_by_page


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str) -> DocumentContent:
    doc = fitz.open(pdf_path)
    filename = pdf_path.split("\\")[-1].split("/")[-1]

    span_sizes = _collect_span_sizes(doc)
    body_size = _median(span_sizes)

    # First pass: raw lines per page, for boilerplate detection
    raw_lines_by_page: list[list[str]] = []
    for page in doc:
        text = page.get_text("text")
        raw_lines_by_page.append([ln for ln in text.split("\n") if ln.strip()])
    boilerplate = _find_boilerplate_lines(raw_lines_by_page)

    tables_by_page = _extract_tables(pdf_path)
    image_captions_by_page = _extract_and_caption_images(doc, filename)

    pages: list[PageContent] = []
    title_guess = filename.rsplit(".", 1)[0]
    title_found = False

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        page_dict = page.get_text("dict")
        md_lines: list[str] = []

        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                if _PAGE_NUM_RE.match(text):
                    continue
                if _normalize_line(text) in boilerplate:
                    continue

                max_span = max(spans, key=lambda s: s.get("size", 0))
                is_bold = bool(max_span.get("flags", 0) & 2**4)
                prefix = _heading_prefix(max_span.get("size", body_size), is_bold, body_size)

                if prefix and not title_found and page_num <= 2:
                    title_guess = text
                    title_found = True

                md_lines.append(f"{prefix}{text}" if prefix else text)

        markdown_text = "\n".join(md_lines)
        pages.append(
            PageContent(
                page_num=page_num,
                markdown_text=markdown_text,
                tables=tables_by_page.get(page_num, []),
                image_captions=image_captions_by_page.get(page_num, []),
            )
        )

    doc.close()
    return DocumentContent(
        source_path=pdf_path,
        filename=filename,
        title=title_guess,
        num_pages=len(pages),
        pages=pages,
    )
