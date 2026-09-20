import argparse
import asyncio
from pathlib import Path

from utils.evaluate_quiz import IN_PATH, OUT_PATH, evaluate_quiz


async def main() -> None:
    parser = argparse.ArgumentParser(description="answer each quiz question with Sonnet alone and with the tool agent")
    parser.add_argument("--in-path", type=Path, default=IN_PATH)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    parser.add_argument("--concurrency", type=int, default=5, help="how many rows to answer at once")
    parser.add_argument("--no-resume", action="store_true", help="re-answer every row, overwriting the output")
    args = parser.parse_args()

    count = await evaluate_quiz(
        in_path=args.in_path,
        out_path=args.out_path,
        concurrency=args.concurrency,
        limit=args.limit,
        resume=not args.no_resume,
    )
    print(f"wrote {count} rows to {args.out_path}")


if __name__ == "__main__":
    asyncio.run(main())
