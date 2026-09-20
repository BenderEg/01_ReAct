"""Тесты инструментов агента: web_search, page_find и обёртка run_tool.

Сеть не трогаем: в main подменяется wiki() — единственная точка выхода наружу.
Для каждого инструмента проверяются нормальный вход, пустой результат и ошибка.
"""

import json

import pytest

import main

from conftest import InstallWiki

HITS = [{"title": "A"}, {"title": "B"}, {"title": "C"}]


def tool_call(name: str, arguments: dict[str, object], call_id: str = "c1") -> dict[str, object]:
    """Вызов инструмента в том виде, в каком его присылает модель."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


# --- web_search: нормальный вход -------------------------------------------------

def test_web_search_returns_intro_and_other_titles(install_wiki: InstallWiki) -> None:
    fake = install_wiki(search=HITS, extract="Intro\n  text")

    result = main.web_search("mexico")

    assert result == "[A] Intro text | other articles: B, C"
    assert len(fake.calls) == 2
    assert fake.calls[0]["srsearch"] == "mexico"
    assert fake.calls[1]["titles"] == "A", "интро берётся у лучшего хита"


def test_web_search_single_hit_has_no_other_articles(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], extract="Intro text")

    assert main.web_search("mexico") == "[A] Intro text"


def test_web_search_truncates_long_intro(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], extract="x" * 2000)

    result = main.web_search("mexico")

    assert result == "[A] " + "x" * 1500


# --- web_search: пустой результат ------------------------------------------------

def test_web_search_without_hits_skips_second_request(install_wiki: InstallWiki) -> None:
    fake = install_wiki(search=[])

    assert main.web_search("несуществующий запрос") == "nothing found"
    assert len(fake.calls) == 1, "за текстом страницы ходить незачем"


def test_web_search_survives_page_without_extract(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], extract=None)

    assert main.web_search("mexico") == "[A] "


# --- web_search: ошибка ----------------------------------------------------------

def test_web_search_propagates_wiki_error(install_wiki: InstallWiki) -> None:
    install_wiki(error=RuntimeError("Википедия ответила 503"))

    with pytest.raises(RuntimeError, match="503"):
        main.web_search("mexico")


# --- page_find: нормальный вход --------------------------------------------------

PAGE = "\n".join([
    "Mexico scored three goals in the final.",
    "Attendance was 80,000 people.",
    "The stadium hosted the final match.",
    "",
    "   ",
    "Only goals matter here.",
])


def test_page_find_keeps_lines_with_half_of_the_keywords(install_wiki: InstallWiki) -> None:
    fake = install_wiki(extract=PAGE)

    result = main.page_find("2026 FIFA World Cup", "goals scored stadium final")

    assert result.splitlines() == [
        "Mexico scored three goals in the final.",
        "The stadium hosted the final match.",
    ], "строка с одним совпадением из четырёх слов не проходит порог"
    assert fake.calls[0]["titles"] == "2026 FIFA World Cup"
    assert fake.calls[0]["redirects"] == 1, "запрос должен идти по редиректам"


def test_page_find_ignores_short_keywords(install_wiki: InstallWiki) -> None:
    install_wiki(extract="Mexico scored twice.\nNothing relevant.")

    # из "a of goals scored" остаются два слова, порог опускается до одного совпадения
    assert main.page_find("T", "a of goals scored") == "Mexico scored twice."


def test_page_find_returns_at_most_six_lines(install_wiki: InstallWiki) -> None:
    install_wiki(extract="\n".join(f"goals line {i}" for i in range(10)))

    assert len(main.page_find("T", "goals").splitlines()) == 6


def test_page_find_truncates_long_line(install_wiki: InstallWiki) -> None:
    install_wiki(extract="goals " + "y" * 600)

    assert len(main.page_find("T", "goals")) == 400


# --- page_find: пустой результат -------------------------------------------------

@pytest.mark.parametrize("extract", [None, ""], ids=["нет поля extract", "пустой extract"])
def test_page_find_without_text_reports_missing_page(install_wiki: InstallWiki,
                                                     extract: str | None) -> None:
    install_wiki(extract=extract)

    assert main.page_find("Нет такой статьи", "goals scored") == "no such page"


def test_page_find_without_matches(install_wiki: InstallWiki) -> None:
    install_wiki(extract=PAGE)

    assert main.page_find("T", "weather forecast") == "keywords not found on the page"


def test_page_find_with_only_short_keywords(install_wiki: InstallWiki) -> None:
    install_wiki(extract=PAGE)

    assert main.page_find("T", "a of in") == "keywords not found on the page"


# --- page_find: ошибка -----------------------------------------------------------

def test_page_find_propagates_wiki_error(install_wiki: InstallWiki) -> None:
    install_wiki(error=RuntimeError("Википедия ответила 503"))

    with pytest.raises(RuntimeError, match="503"):
        main.page_find("T", "goals scored")


# --- run_tool: как ошибку инструмента видит агент --------------------------------

def test_run_tool_returns_tool_message(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], extract="Intro text")

    message = main.run_tool(tool_call("web_search", {"query": "mexico"}))

    assert message == {"role": "tool", "tool_call_id": "c1", "content": "[A] Intro text"}


def test_run_tool_wraps_wiki_error(install_wiki: InstallWiki) -> None:
    install_wiki(error=RuntimeError("Википедия ответила 503"))

    message = main.run_tool(tool_call("web_search", {"query": "mexico"}))

    assert message["content"].startswith("НЕ удалось вызвать инструмент web_search")
    assert "503" in message["content"]


def test_run_tool_wraps_invalid_arguments(install_wiki: InstallWiki) -> None:
    fake = install_wiki(extract=PAGE)

    message = main.run_tool(tool_call("page_find", {"wrong": 1}))

    assert message["content"].startswith("НЕ удалось вызвать инструмент page_find")
    assert fake.calls == [], "до Википедии дело не доходит"


def test_run_tool_wraps_unknown_tool(install_wiki: InstallWiki) -> None:
    install_wiki(search=HITS)

    message = main.run_tool(tool_call("nope", {"query": "mexico"}))

    assert message["content"].startswith("НЕ удалось вызвать инструмент nope")
