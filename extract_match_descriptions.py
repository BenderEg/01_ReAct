import asyncio
from pathlib import Path

from utils.match_descriptions import fetch_match_descriptions


async def main() -> None:
    out_path = Path("data/match_descriptions.jsonl")
    count = await fetch_match_descriptions(out_path=out_path)
    print(f"wrote {count} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
