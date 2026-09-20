import asyncio
import json
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag

from utils.wiki_links import HEADERS

HEADING_TAGS = ("h2", "h3", "h4")


def split_url(url: str) -> tuple[str, str]:
    """Split a URL into (base_url, fragment); fragment is "" if absent."""
    base, _, fragment = url.partition("#")
    return base, fragment


def _clean_text(tag: Tag) -> str:
    """Get visible text from `tag`, dropping hidden microformat spans (e.g. duplicate ISO dates)."""
    for hidden in tag.select('span.bday, [style*="display:none"], [style*="display: none"]'):
        hidden.decompose()
    return tag.get_text(" ", strip=True)


def extract_match_info(html: str, fragment: str) -> str:
    """Extract prose, result/venue/referee, and squads/officials for the match identified by `fragment`."""
    if not fragment:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    anchor = soup.find(id=fragment)
    if anchor is None:
        return ""
    heading = anchor if anchor.name in HEADING_TAGS else anchor.find_parent(HEADING_TAGS)
    if heading is None:
        return ""
    container = heading.find_parent(class_="mw-heading") or heading

    paragraphs: list[str] = []
    footballbox: Tag | None = None
    detail_tables: list[Tag] = []
    for sibling in container.find_next_siblings():
        if sibling.name in HEADING_TAGS:
            break
        if sibling.name == "div" and "mw-heading" in (sibling.get("class") or []):
            break
        if sibling.name == "p":
            text = sibling.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
        elif sibling.name == "div" and "footballbox" in (sibling.get("class") or []):
            footballbox = sibling
        elif sibling.name == "table" and footballbox is not None:
            detail_tables.append(sibling)

    parts: list[str] = []
    if footballbox is not None:
        parts.append(_clean_text(footballbox))
    if paragraphs:
        parts.append("\n".join(paragraphs))
    if detail_tables:
        parts.append("\n".join(_clean_text(table) for table in detail_tables))
    return "\n\n".join(parts)


async def fetch_page(client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def fetch_match_descriptions(
    matches_path: Path = Path("data/matches"),
    out_path: Path = Path("data/match_descriptions.jsonl"),
    concurrency: int = 5,
) -> int:
    """Fetch each URL's page (deduped by base URL) and write {"url", "text"} rows to a JSONL file."""
    lines = [line.strip() for line in matches_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    split_lines = [split_url(url) for url in lines]
    unique_bases = {base for base, _ in split_lines}

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(headers=HEADERS, timeout=20, follow_redirects=True) as client:
        htmls = await asyncio.gather(*(fetch_page(client, base, semaphore) for base in unique_bases))
    html_by_base = dict(zip(unique_bases, htmls))

    rows_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for url, (base, fragment) in zip(lines, split_lines):
            text = extract_match_info(html_by_base[base], fragment)
            if not text:
                print(f"no text extracted for {url}")
            f.write(json.dumps({"url": url, "text": text}, ensure_ascii=False) + "\n")
            rows_written += 1
    return rows_written
