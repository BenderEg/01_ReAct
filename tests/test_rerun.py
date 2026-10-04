"""Тесты повторного запуска неверных задач: main.rerun_failed и main.rerun_comparison.

Модель не вызывается: main.agent подменяется заглушкой, results/ и traces/ — во временной папке.
"""

from pathlib import Path

import pandas as pd
import pytest

import main

CONFIG, LEVEL = "поиск и чтение страницы", "cheap"
TASKS = [{"id": f"fresh-{i}", "source": "fresh", "question": f"q{i}", "answer": f"gold{i}"} for i in range(4)]


@pytest.fixture
def asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Старый прогон: fresh-0 и fresh-2 верные, fresh-1 и fresh-3 нет; агент теперь отвечает верно.

    Возвращает список вопросов, которые получил агент.
    """
    monkeypatch.setattr(main, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(main, "TRACES", tmp_path / "traces")
    monkeypatch.setattr(main, "load_tasks", lambda: TASKS)
    questions: list[str] = []

    def fake_agent(question: str, model: str, tool_names: list[str]) -> main.Run:
        questions.append(question)
        return main.Run(question, f"FINAL: gold{question[1:]}", 3, [{"role": "user", "content": question}], 0.01, 1.0)

    monkeypatch.setattr(main, "agent", fake_agent)
    main.RESULTS.mkdir()
    old = pd.DataFrame([{"config": CONFIG, "model": "gpt-4o-mini", "id": t["id"], "source": "fresh",
                         "correct": i % 2 == 0, "steps": 5, "tool_calls": 4, "cost": 0.002, "seconds": 10.0,
                         "answer": "x", "gold": t["answer"]} for i, t in enumerate(TASKS)])
    old.to_csv(main.RESULTS / f"{CONFIG}__{LEVEL}.csv", index=False, encoding="utf-8")
    return questions


def test_rerun_failed_runs_only_wrong_tasks_into_new_folders(asked: list[str]) -> None:
    old_csv = main.RESULTS / f"{CONFIG}__{LEVEL}.csv"
    old_text = old_csv.read_text(encoding="utf-8")

    redo = main.rerun_failed([LEVEL], [CONFIG])

    assert asked == ["q1", "q3"]
    assert redo["id"].tolist() == ["fresh-1", "fresh-3"]
    assert redo["correct"].all()
    assert (main.RESULTS / main.RERUN / f"{CONFIG}__{LEVEL}.csv").exists()
    traces = sorted(p.name for p in (main.TRACES / main.RERUN / CONFIG / "gpt-4o-mini").iterdir())
    assert traces == ["fresh-1.json", "fresh-3.json"]
    assert not (main.TRACES / CONFIG).exists()
    assert old_csv.read_text(encoding="utf-8") == old_text


def test_rerun_comparison_before_after_and_merged(asked: list[str]) -> None:
    main.rerun_failed([LEVEL], [CONFIG])

    table, merged = main.rerun_comparison([LEVEL], [CONFIG])

    assert table["config"].tolist() == [f"{CONFIG}, до правок", f"{CONFIG}, после правок"]
    assert table["n"].tolist() == [2, 2]
    assert table["accuracy"].tolist() == [0.0, 1.0]
    row = merged.iloc[0]
    assert (row["n"], row["old_correct"], row["fixed"]) == (4, 2, 2)
    assert (row["accuracy_before"], row["accuracy_merged"]) == (0.5, 1.0)
    assert row["cost_per_task"] == pytest.approx((2 * 0.002 + 2 * 0.01) / 4)
    assert row["cost_per_correct"] == pytest.approx((2 * 0.002 + 2 * 0.01) / 4)
