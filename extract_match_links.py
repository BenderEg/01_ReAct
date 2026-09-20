import asyncio
from pathlib import Path

from utils.wiki_links import extract_match_links, fetch_html


async def main() -> None:
    html = await fetch_html()
    links = extract_match_links(html)
    out_path = Path("data/matches")
    out_path.write_text("\n".join(links) + "\n", encoding="utf-8")
    print(f"wrote {len(links)} links to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
