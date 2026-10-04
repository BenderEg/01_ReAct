"""Тесты проверки ответа: utils.answer_check, main.is_correct и переоценка main.rescore.

Случаи взяты из настоящих трейсов: верный по сути ответ в другой форме должен
засчитываться, неверный — нет, даже если цифры похожи.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import main
from utils.answer_check import answer_tokens, is_correct


@pytest.mark.parametrize(("gold", "answer"), [
    ("two", "2"),
    ("sixth", "6"),
    ("sixth", "6th"),
    ("fourteen", "14 matches"),
    ("44 years", "44"),
    ("84th minute", "84"),
    ("64,478", "64478 spectators"),
    ("Tomas Soucek", "Tomáš Souček"),
    ("United States", "the United States."),
    ("90+7'", "90+7"),
    ("1,000th World Cup match", "The 1,000th World Cup match"),
])
def test_accepts_same_answer_in_another_form(gold: str, answer: str) -> None:
    assert is_correct(gold, answer)


@pytest.mark.parametrize(("gold", "answer"), [
    ("6", "16"),
    ("two", "12"),
    ("1958", "1954"),
    ("44 years", "4 years"),
    ("44 years", "144"),
    ("Cristiano Ronaldo", "Криштиану Роналду"),
    ("Toronto", "unknown"),
    ("", "anything"),
])
def test_rejects_wrong_answer(gold: str, answer: str) -> None:
    assert not is_correct(gold, answer)


def test_answer_tokens_normalize_form() -> None:
    assert answer_tokens("The Sixth goal, 64,478 fans — Souček") == ["6", "goal", "64478", "fans", "soucek"]


def test_main_is_correct_never_accepts_failed_run() -> None:
    answer = 'ошибка: HTTP 429 : {"error": {"code": 10}}'

    assert not main.is_correct({"answer": "10"}, answer)


def test_rescore_regrades_csv_and_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results, traces = tmp_path / "results", tmp_path / "traces"
    folder = traces / "поиск" / "gpt-4o-mini"
    folder.mkdir(parents=True)
    results.mkdir()
    monkeypatch.setattr(main, "RESULTS", results)
    monkeypatch.setattr(main, "TRACES", traces)

    trace = {"task": {"id": "q1", "answer": "two"}, "answer": "Thinking...\nFINAL: 2", "correct": False}
    (folder / "q1.json").write_text(json.dumps(trace), encoding="utf-8")
    pd.DataFrame([{"config": "поиск", "model": "gpt-4o-mini", "id": "q1", "source": "fresh", "correct": False,
                   "steps": 1, "tool_calls": 0, "cost": 0.0, "seconds": 0.1, "answer": "2", "gold": "two"}]
                 ).to_csv(results / "поиск__cheap.csv", index=False)

    rescored = main.rescore()

    assert rescored["correct"].tolist() == [True]
    assert pd.read_csv(results / "all_runs.csv")["correct"].tolist() == [True]
    assert json.loads((folder / "q1.json").read_text(encoding="utf-8"))["correct"] is True
