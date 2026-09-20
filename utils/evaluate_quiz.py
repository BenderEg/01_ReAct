"""Answer every quiz question twice: with a strong model alone and with the tool-using agent.

Both answerers come from `lesson/fetch_fresh.py` and are synchronous, so they run in worker
threads via `asyncio.to_thread` while this module does the bounded-concurrency orchestration.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, TextIO

from lesson import fetch_fresh as ff

QuizRow = dict[str, Any]

IN_PATH = Path("data/fresh.jsonl")
OUT_PATH = Path("data/match_quiz_evaluated.jsonl")


def read_jsonl(path: Path) -> list[QuizRow]:
    """Parse a JSONL file into a list of dicts; a missing file yields an empty list."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def graded(gold: str, candidate: str) -> bool:
    """Grade like `fetch_fresh` does: normalized gold must appear inside the normalized answer."""
    gold_normalized, candidate_normalized = ff.normalize(gold), ff.normalize(candidate)
    return bool(gold_normalized) and gold_normalized in candidate_normalized


def _answer_or_empty(result: str | BaseException, row_id: str, label: str) -> str:
    """Unwrap a `gather(return_exceptions=True)` slot, reporting a failure as an empty answer."""
    if isinstance(result, BaseException):
        print(f"  {row_id}: {label} failed, {type(result).__name__}: {result}", flush=True)
        return ""
    return result


async def evaluate_row(
    row: QuizRow,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    out_file: TextIO,
    index: int,
    total: int,
) -> QuizRow:
    """Answer one question both ways and append the enriched row to the open output file."""
    question: str = row["question"]
    gold: str = row["answer"]
    async with semaphore:
        # `strong_answer` lets a RuntimeError out of `post()` escape; don't let it cancel the agent.
        sonnet_result, agent_result = await asyncio.gather(
            asyncio.to_thread(ff.strong_answer, question),
            asyncio.to_thread(ff.agent_answer, question),
            return_exceptions=True,
        )

    sonnet_answer = _answer_or_empty(sonnet_result, row["id"], "strong_answer")
    agent_answer = _answer_or_empty(agent_result, row["id"], "agent_answer")
    out_row: QuizRow = {
        **row,
        "sonnet_answer": sonnet_answer,
        "sonnet_ok": graded(gold, sonnet_answer),
        "agent_answer": agent_answer,
        "agent_ok": graded(gold, agent_answer),
    }

    async with write_lock:
        out_file.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        out_file.flush()
    print(
        f"[{index}/{total}] {out_row['id']} "
        f"{'sonnet+' if out_row['sonnet_ok'] else 'sonnet-'} "
        f"{'agent+' if out_row['agent_ok'] else 'agent-'} | "
        f"gold: {gold[:40]} | sonnet: {sonnet_answer[:40]} | agent: {agent_answer[:40]}",
        flush=True,
    )
    return out_row


async def evaluate_quiz(
    in_path: Path = IN_PATH,
    out_path: Path = OUT_PATH,
    concurrency: int = 5,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """Evaluate every row of `in_path` and write `in_path`'s rows plus four answer fields to `out_path`.

    With `resume`, rows whose `id` is already in `out_path` are kept as they are and not re-answered.
    """
    rows = read_jsonl(in_path)
    if limit is not None:
        rows = rows[:limit]
    existing = {row["id"]: row for row in read_jsonl(out_path)} if resume else {}
    todo = [row for row in rows if row["id"] not in existing]
    print(f"{len(rows)} rows, {len(existing)} already evaluated, {len(todo)} to do", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    start_cost = ff.COST
    # Append as we go, so a crash mid-run keeps the rows already paid for.
    mode = "a" if resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as out_file:
        evaluated = await asyncio.gather(
            *(
                evaluate_row(row, semaphore, write_lock, out_file, index, len(todo))
                for index, row in enumerate(todo, start=1)
            )
        )

    # Rewrite the file so the appended rows are deduped and ordered like the input.
    merged = {**existing, **{row["id"]: row for row in evaluated}}
    ordered = [merged[row["id"]] for row in rows if row["id"] in merged]
    with out_path.open("w", encoding="utf-8") as out_file:
        for row in ordered:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    sonnet_ok = sum(bool(row["sonnet_ok"]) for row in ordered)
    agent_ok = sum(bool(row["agent_ok"]) for row in ordered)
    print(f"sonnet_ok {sonnet_ok}/{len(ordered)}  agent_ok {agent_ok}/{len(ordered)}", flush=True)
    print(f"spent ${ff.COST - start_cost:.3f} this run, ${ff.COST:.3f} total", flush=True)
    return len(ordered)
