"""Token-aware clause chunking for standards and policy documents."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHUNK_SIZE = max(1, min(int(os.getenv("CHUNK_SIZE", "300")), 480))
CHUNK_OVERLAP = max(0, min(int(os.getenv("CHUNK_OVERLAP", "45")), CHUNK_SIZE - 1))
MIN_CLAUSE_TOKENS = int(os.getenv("MIN_CLAUSE_TOKENS", "80"))

_NUMERIC_ID = r"\d+(?:\.\d+)*"
_CLAUSE_LINE_RE = re.compile(
    rf"^\s*(?P<id>{_NUMERIC_ID})\s+(?P<title>[A-Z][^\n]*)\s*$"
    rf"|^\s*(?P<alpha>[A-Z]\.\d+(?:\.\d+)*)\s+(?P<alpha_title>[^\n]+)\s*$"
    r"|^\s*(?P<annex>Annex\s+[A-Z])(?:\s*[-:]?\s*(?P<annex_title>.*))$",
    re.MULTILINE,
)
_YEAR_OR_PAGE_RE = re.compile(r"^(?:19|20)\d{2}$|^\d{1,4}$")


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_num: int | None = None
    section_id: str | None = None
    heading_path: str | None = None
    page_nums: tuple[int, ...] = ()
    section_ids: tuple[str, ...] = ()


@dataclass
class Section:
    section_id: str | None
    heading_path: str | None
    text: str
    page_nums: tuple[int, ...] = ()
    section_ids: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(EMBED_MODEL, use_fast=True)


def _token_count(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _section_markers(text: str) -> list[tuple[int, int, str, str]]:
    lines = text.splitlines(keepends=True)
    markers = []
    offset = 0
    for index, line in enumerate(lines):
        content = line.strip()
        match = _CLAUSE_LINE_RE.match(content)
        if match:
            section_id = match.group("id") or match.group("alpha") or match.group("annex")
            title = match.group("title") or match.group("alpha_title") or match.group("annex_title")
            if section_id:
                markers.append((offset, offset + len(line), section_id, (title or "").strip()))
        offset += len(line)
    return markers


def parse_sections(
    text: str,
    page_num: int | None = None,
    heading_context: dict[str, str] | None = None,
) -> list[Section]:
    """Parse recognized clause headings and assign hierarchical heading paths."""
    markers = _section_markers(text)
    if not markers:
        return [Section(None, None, text.strip(), (page_num,) if page_num else ())] if text.strip() else []

    sections = []
    heading_by_id = heading_context if heading_context is not None else {}
    for idx, (start, heading_end, section_id, title) in enumerate(markers):
        end = markers[idx + 1][0] if idx + 1 < len(markers) else len(text)
        content_start = heading_end
        if section_id.isdigit() and title and text[heading_end:end].lstrip().startswith(title):
            title_end = text.find(title, heading_end, end) + len(title)
            content_start = title_end
        parent = section_id.rsplit(".", 1)[0] if "." in section_id else None
        ancestors = []
        while parent:
            if parent in heading_by_id:
                ancestors.insert(0, heading_by_id[parent])
            parent = parent.rsplit(".", 1)[0] if "." in parent else None
        heading = f"{section_id} {title}".strip()
        heading_by_id[section_id] = heading
        heading_path = " > ".join(ancestors + [heading])
        body = text[start:content_start] + text[content_start:end]
        body = body.strip()
        if body:
            sections.append(Section(
                section_id, heading_path, body, (page_num,) if page_num else (), (section_id,)
            ))
    prefix = text[:markers[0][0]].strip()
    if prefix and any(
        line.strip() and not _YEAR_OR_PAGE_RE.fullmatch(line.strip())
        for line in prefix.splitlines()
    ):
        sections.insert(0, Section(None, None, prefix, (page_num,) if page_num else (), ()))
    return sections


def _merge_short_sections(sections: list[Section]) -> list[Section]:
    def parents(section_ids):
        if not section_ids:
            return set()
        return {
            value.rsplit(".", 1)[0] if "." in value else ""
            for value in section_ids
        }

    def merge(left: Section, right: Section) -> Section:
        section_ids = tuple(dict.fromkeys(left.section_ids + right.section_ids))
        return Section(
            left.section_id or right.section_id,
            " / ".join(filter(None, [left.heading_path, right.heading_path])),
            f"{left.text}\n\n{right.text}".strip(),
            tuple(dict.fromkeys(left.page_nums + right.page_nums)),
            section_ids,
        )

    merged: list[Section] = []
    pending: Section | None = None
    for section in sections:
        if pending:
            if section.section_ids and parents(pending.section_ids) == parents(section.section_ids):
                section = merge(pending, section)
            else:
                merged.append(pending)
            pending = None

        is_short_clause = bool(
            section.section_id and _token_count(section.text) < MIN_CLAUSE_TOKENS
        )
        same_parent_as_previous = bool(
            merged and section.section_ids and merged[-1].section_ids
            and parents(section.section_ids) == parents(merged[-1].section_ids)
        )
        if is_short_clause and same_parent_as_previous:
            merged[-1] = merge(merged[-1], section)
        elif is_short_clause:
            pending = section
        else:
            merged.append(section)
    if pending:
        merged.append(pending)
    return merged


def _preferred_boundary(text: str, start: int, target_end: int) -> int:
    """Prefer paragraph/line boundaries while retaining all original newlines."""
    lower = max(start + 1, target_end - 240)
    candidates = [text.rfind("\n\n", lower, target_end), text.rfind("\n", lower, target_end)]
    boundary = max(candidates)
    return boundary + (2 if text[boundary:boundary + 2] == "\n\n" else 1) if boundary >= lower else target_end


def _token_windows(text: str) -> list[str]:
    tokenizer = _tokenizer()
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if not offsets:
        return []
    windows = []
    start_token = 0
    while start_token < len(offsets):
        end_token = min(start_token + CHUNK_SIZE, len(offsets))
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        if end_token < len(offsets):
            char_end = _preferred_boundary(text, char_start, char_end)
            if char_end <= char_start:
                char_end = offsets[end_token - 1][1]
        fragment = text[char_start:char_end].strip()
        if fragment:
            windows.append(fragment)
        if end_token == len(offsets):
            break
        actual_end_token = next(
            (index for index in range(start_token + 1, len(offsets)) if offsets[index][0] >= char_end),
            len(offsets),
        )
        next_start = max(start_token + 1, actual_end_token - CHUNK_OVERLAP)
        start_token = next_start
    return windows


def _chunks_from_sections(
    sections: list[Section], start_index: int = 0, default_page_num: int | None = None
) -> list[Chunk]:
    sections = _merge_short_sections(sections)
    chunks = []
    for section in sections:
        pieces = _token_windows(section.text)
        for piece in pieces:
            if _token_count(piece) > 480:
                raise ValueError("Tokenizer window exceeded the hard 480-token limit")
            chunks.append(Chunk(
                text=piece,
                chunk_index=start_index + len(chunks),
                page_num=section.page_nums[0] if section.page_nums else default_page_num,
                section_id=section.section_id,
                heading_path=section.heading_path,
                page_nums=section.page_nums,
                section_ids=section.section_ids,
            ))
    return chunks


def chunk_text(text: str, page_num: int | None = None, start_index: int = 0) -> list[Chunk]:
    """Split text to <=480 BGE tokens, targeting 300 tokens with 45-token overlap."""
    return _chunks_from_sections(parse_sections(text, page_num), start_index, page_num)


def chunk_pages(pages: list[dict], start_index: int = 0) -> list[Chunk]:
    """Chunk page-level text while allowing short sibling clauses across pages to merge."""
    sections = []
    heading_context: dict[str, str] = {}
    for page in pages:
        text = page.get("text", "").strip()
        if text:
            page_sections = parse_sections(text, page.get("page"), heading_context)
            if (
                sections
                and page_sections
                and page_sections[0].section_id is None
                and sections[-1].section_id is not None
            ):
                continuation = page_sections.pop(0)
                previous = sections[-1]
                sections[-1] = Section(
                    previous.section_id,
                    previous.heading_path,
                    f"{previous.text}\n\n{continuation.text}".strip(),
                    tuple(dict.fromkeys(previous.page_nums + continuation.page_nums)),
                    previous.section_ids,
                )
            sections.extend(page_sections)
    return _chunks_from_sections(sections, start_index)
