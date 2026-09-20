from pathlib import Path

from utils.match_quiz import generate_match_quiz


def main() -> None:
    out_path = Path("data/fresh.jsonl")
    count = generate_match_quiz(out_path=out_path)
    print(f"wrote {count} rows to {out_path}")


if __name__ == "__main__":
    main()
