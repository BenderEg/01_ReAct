import json
import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        _key, _, _value = _line.partition("=")
        if _key and not os.getenv(_key):
            os.environ[_key] = _value.strip()

DEFAULT_MODEL = os.getenv("MATCH_QUIZ_MODEL", "haiku")

SYSTEM_PROMPT = """Each description comes from a Wikipedia about Football World Cup 2026 and represents info about one of 104 matches played.
For each description write one quiz question with a short unambiguous answer.
Rules: the answer is a single entity, at most five words: a person's name, a number, a title, a place or a date, nothing else;
the answer must appear verbatim in the fact; the question asks for a specific detail of a specific described event, never
"what happened" or "who is mentioned"; the question is self-contained: it must name both teams playing in the match and the
match round (for example, group stage or knock-out stage), so someone who has not read this description can still identify which match is being asked about; never put the
answer into the question;
prefer an interesting or unusual fact over a routine one: a record, a milestone, a historical first, an unusual refereeing
detail, a notable incident (red card, penalty, own goal, comeback), attendance, or context about the teams' history against
each other; avoid the Man of the Match and "who scored" as your default choice - only use them if the description offers
nothing more distinctive to ask about;
do not create questions from past events, use only current match description, if in description are mentioned events not from
world cup 2026 itself - skip this part, do not base questions on future or scheduled events and facts without a clear short checkable detail.
Return only a JSON object {"question": "...", "answer": "...", "evidence": "..."}."""


class GeneratedQuestion(BaseModel):
    question: str
    answer: str
    evidence: str


QUESTION_SCHEMA = GeneratedQuestion.model_json_schema()


def page_from_url(url: str) -> str:
    """Last segment of the URL path (before any #fragment), with underscores turned into spaces."""
    path = urlparse(url).path
    return path.rstrip("/").rsplit("/", 1)[-1].replace("_", " ")


def ask_claude(description: str, model: str = DEFAULT_MODEL) -> GeneratedQuestion | None:
    """Run the `claude` CLI once to turn a match description into a question/answer/evidence triple."""
    cmd = [
        "claude", "-p",
        "--system-prompt", SYSTEM_PROMPT,
        "--output-format", "json",
        "--json-schema", json.dumps(QUESTION_SCHEMA),
        "--no-session-persistence",
        "--tools", "",
        "--model", model,
    ]

    try:
        result = subprocess.run(cmd, input=description, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("claude CLI failed: %s", e.stderr)
        return None

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("could not parse claude CLI output as JSON: %s", result.stdout[:200])
        return None

    structured = payload.get("structured_output")
    if not structured:
        logger.warning("claude CLI response has no structured_output: %s", payload.get("result"))
        return None

    try:
        return GeneratedQuestion.model_validate(structured)
    except ValidationError as e:
        logger.warning("claude CLI structured_output failed validation: %s", e)
        return None


def generate_match_quiz(
    descriptions_path: Path = Path("data/match_descriptions.jsonl"),
    out_path: Path = Path("data/fresh.jsonl"),
    model: str = DEFAULT_MODEL,
) -> int:
    """Turn each row of `descriptions_path` into a quiz question via the claude CLI, writing jsonl rows to `out_path`."""
    lines = [line for line in descriptions_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    counter = 0
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            row = json.loads(line)
            url, text = row["url"], row["text"]
            structured = ask_claude(text, model=model)
            if structured is None:
                continue
            out_row = {
                "id": f"fresh-{counter}",
                "source": "fresh",
                "question": structured.question,
                "answer": structured.answer,
                "evidence": structured.evidence,
                "page": page_from_url(url),
                "url": url,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()
            counter += 1
    return counter
