import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass
class DocumentChunk:
    """A single text chunk ready for embedding and storage in ChromaDB."""

    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """
    Loads .pdf and .txt construction norm documents and splits them into chunks.

    PDF extraction strategy:
    - Tables are extracted separately and converted to readable text rows,
      preserving the numeric data that pdfplumber mangles when tables and
      prose are extracted together.
    - Regular prose is extracted from the remaining page text after tables
      are stripped out.

    This matters for МДС/СП documents where the most valuable data
    (durations, quantities) lives in tables.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def load_file(self, path: Path) -> list[DocumentChunk]:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw_text = self._extract_text_from_pdf(path)
        elif suffix == ".txt":
            raw_text = path.read_text(encoding="utf-8").strip()
        else:
            raise ValueError(f"Unsupported file type '{suffix}': {path}")

        if not raw_text:
            raise ValueError(f"Document is empty or could not be parsed: {path}")

        paragraphs = self._split_into_paragraphs(raw_text)
        merged = self._merge_short_chunks(paragraphs)

        return [
            DocumentChunk(
                text=text,
                source=path.name,
                chunk_index=idx,
                metadata={"source_path": str(path)},
            )
            for idx, text in enumerate(merged)
        ]

    def load_directory(self, directory: Path) -> list[DocumentChunk]:
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        all_chunks: list[DocumentChunk] = []
        supported_files = sorted(
            f for f in directory.rglob("*") if f.suffix.lower() in {".pdf", ".txt"}
        )

        for file_path in supported_files:
            try:
                chunks = self.load_file(file_path)
                all_chunks.extend(chunks)
                print(f"[DocumentLoader] Loaded {len(chunks)} chunks from {file_path.name}")
            except (FileNotFoundError, ValueError) as exc:
                print(f"[DocumentLoader] Skipping {file_path.name}: {exc}")

        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_text_from_pdf(self, path: Path) -> str:
        """
        Extract text from a PDF, handling tables and prose separately.

        For each page:
        - Tables are converted to clean pipe-separated rows so numeric
          data (durations, floor counts) is preserved and readable.
        - Prose text has table bounding boxes cropped out before extraction
          so table content isn't duplicated as garbled text.

        Pages are joined with double newlines to preserve paragraph structure.
        """
        pages: list[str] = []

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_parts: list[str] = []

                # Extract tables first
                tables = page.extract_tables()
                table_bboxes = [t.bbox for t in page.find_tables()] if tables else []

                for table in tables:
                    table_text = self._format_table(table)
                    if table_text:
                        page_parts.append(table_text)

                # Crop out table areas from the page before extracting prose
                # so we don't get the same data twice as mangled text
                cropped_page = page
                for bbox in table_bboxes:
                    try:
                        cropped_page = cropped_page.outside_bbox(bbox)
                    except Exception:
                        pass  # if crop fails, fall back to full page text

                prose = cropped_page.extract_text()
                if prose and prose.strip():
                    page_parts.append(prose.strip())

                if page_parts:
                    pages.append("\n\n".join(page_parts))

        return "\n\n".join(pages)

    @staticmethod
    def _format_table(table: list[list]) -> str:
        """
        Convert a pdfplumber table into semantically rich sentences.

        Plain pipe-separated rows embed poorly — ChromaDB sees
        "Кирпичное | 6,5 | 1 | 3" with no idea what the numbers mean.
        Instead we pair each data row with the detected header row and
        produce full descriptions:

            "Характеристика = Кирпичное и из мелких блоков; Общая = 6,5; Надземная часть = 3."

        This gives the embedding model enough context to match a query like
        "продолжительность строительства кирпичного 5-этажного здания".

        Falls back to pipe-separated rows if no header can be detected.
        """
        if not table:
            return ""

        # Normalise all cells
        cleaned: list[list[str]] = []
        for row in table:
            if row is None:
                continue
            cells = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
            if any(cells):
                cleaned.append(cells)

        if not cleaned:
            return ""

        # Detect header row — the first row where most cells are non-numeric
        def _is_header(row: list[str]) -> bool:
            non_empty = [c for c in row if c]
            if not non_empty:
                return False
            numeric = sum(1 for c in non_empty if re.match(r"^[\d,.\s]+$", c))
            return numeric / len(non_empty) < 0.5

        header: list[str] = []
        data_rows: list[list[str]] = []

        if cleaned and _is_header(cleaned[0]):
            header = cleaned[0]
            data_rows = cleaned[1:]
        else:
            data_rows = cleaned

        # If no header detected, fall back to pipe rows
        if not header:
            return "\n".join(" | ".join(row) for row in data_rows)

        # Expand each data row into "header_col = value" sentences
        sentences: list[str] = []
        for row in data_rows:
            pairs: list[str] = []
            for header_cell, value in zip(header, row):
                if value and header_cell:
                    pairs.append(f"{header_cell} = {value}")
                elif value:
                    pairs.append(value)
            if pairs:
                sentences.append("; ".join(pairs) + ".")

        return "\n".join(sentences)

    def _split_into_paragraphs(self, text: str) -> list[str]:
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
        result: list[str] = []

        for paragraph in raw_paragraphs:
            if not paragraph:
                continue
            if len(paragraph) <= self.max_chunk_size:
                result.append(paragraph)
            else:
                result.extend(self._split_on_sentences(paragraph))

        return result

    def _split_on_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.max_chunk_size:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks

    def _merge_short_chunks(self, chunks: list[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""

        for chunk in chunks:
            buffer = f"{buffer} {chunk}".strip() if buffer else chunk
            if len(buffer) >= self.min_chunk_size:
                merged.append(buffer)
                buffer = ""

        if buffer:
            if merged:
                merged[-1] = f"{merged[-1]} {buffer}".strip()
            else:
                merged.append(buffer)

        return merged