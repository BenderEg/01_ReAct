"""Тесты инструментов агента: web_search, page_find, обёртка run_tool и разбор HTML.

Сеть не трогаем: в main подменяется wiki() — единственная точка выхода наружу.
Для каждого инструмента проверяются нормальный вход, пустой результат и ошибка.
"""

import json
from typing import Any

import pytest

import main
from utils.wiki_text import html_lines

from conftest import InstallWiki

HITS = [{"title": "A"}, {"title": "B"}, {"title": "C"}]


def tool_call(name: str, arguments: dict[str, object], call_id: str = "c1") -> dict[str, object]:
    """Вызов инструмента в том виде, в каком его присылает модель."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


# --- web_search: нормальный вход -------------------------------------------------

def test_web_search_returns_intro_and_other_titles(install_wiki: InstallWiki) -> None:
    fake = install_wiki(search=HITS, html="<p>Intro</p>\n<p>  text</p>")

    result = main.web_search("mexico")

    assert result == "[A] Intro text | other articles: B, C"
    assert len(fake.calls) == 2
    assert fake.calls[0]["srsearch"] == "mexico"
    assert fake.calls[1]["action"] == "parse"
    assert fake.calls[1]["page"] == "A", "интро берётся у лучшего хита"
    assert fake.calls[1]["section"] == 0, "только вводная секция"


def test_web_search_single_hit_has_no_other_articles(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], html="<p>Intro text</p>")

    assert main.web_search("mexico") == "[A] Intro text"


def test_web_search_truncates_long_intro(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], html="<p>" + "x" * 2000 + "</p>")

    result = main.web_search("mexico")

    assert result == "[A] " + "x" * 1500


# --- web_search: пустой результат ------------------------------------------------

def test_web_search_without_hits_skips_second_request(install_wiki: InstallWiki) -> None:
    fake = install_wiki(search=[])

    assert main.web_search("несуществующий запрос") == "nothing found"
    assert len(fake.calls) == 1, "за текстом страницы ходить незачем"


def test_web_search_survives_missing_page(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], html=None)

    assert main.web_search("mexico") == "[A] "


# --- web_search: ошибка ----------------------------------------------------------

def test_web_search_propagates_wiki_error(install_wiki: InstallWiki) -> None:
    install_wiki(error=RuntimeError("Википедия ответила 503"))

    with pytest.raises(RuntimeError, match="503"):
        main.web_search("mexico")


# --- page_find: нормальный вход --------------------------------------------------

PAGE = """
<h2>Final</h2>
<p>Mexico scored three goals in the final. Attendance was 80,000 people.</p>
<ul><li>The stadium hosted the final match.</li></ul>
<p>   </p>
<p>Only goals matter here.</p>
"""

FOOTBALLBOX = (
    '<div class="footballbox"><div class="fleft"><time><div class="fdate">June 24, 2026'
    '<span style="display: none;"> (<span class="bday">2026-06-24</span>)</span></div></time></div>'
    '<table class="fevent"><tbody><tr><th class="fhome">Scotland</th><th class="fscore">0–3</th>'
    '<th class="faway">Brazil</th></tr></tbody></table>'
    '<div class="fright"><div><a href="#">Hard Rock Stadium</a>, Miami Gardens</div>'
    '<div>Attendance: 64,478</div><div>Referee: César Ramos (<a href="#">Mexico</a>)</div></div></div>'
)
FOOTBALLBOX_LINE = ("June 24, 2026 Scotland 0–3 Brazil Hard Rock Stadium, Miami Gardens "
                    "Attendance: 64,478 Referee: César Ramos (Mexico)")


def test_page_find_keeps_lines_with_half_of_the_keywords(install_wiki: InstallWiki) -> None:
    fake = install_wiki(html=PAGE)

    result = main.page_find("2026 FIFA World Cup", "goals scored stadium final")

    assert result.splitlines() == [
        "Mexico scored three goals in the final.",
        "The stadium hosted the final match.",
    ], "строка с одним совпадением из четырёх слов не проходит порог"
    assert fake.calls[0]["page"] == "2026 FIFA World Cup"
    assert fake.calls[0]["redirects"] == 1, "запрос должен идти по редиректам"
    assert "section" not in fake.calls[0], "нужна вся статья, а не только интро"


def test_page_find_reads_footballbox(install_wiki: InstallWiki) -> None:
    install_wiki(html=FOOTBALLBOX)

    assert main.page_find("T", "Scotland Brazil attendance") == FOOTBALLBOX_LINE


def test_page_find_ranks_lines_by_matched_keywords(install_wiki: InstallWiki) -> None:
    install_wiki(html="<p>Scotland and Brazil met in 1998.</p>" + FOOTBALLBOX)

    result = main.page_find("T", "Scotland Brazil attendance")

    assert result.splitlines() == [FOOTBALLBOX_LINE, "Scotland and Brazil met in 1998."], \
        "строка со всеми тремя словами идёт раньше строки с двумя"


def test_page_find_ignores_short_keywords(install_wiki: InstallWiki) -> None:
    install_wiki(html="<p>Mexico scored twice.</p><p>Nothing relevant.</p>")

    # из "a of goals scored" остаются два слова, порог опускается до одного совпадения
    assert main.page_find("T", "a of goals scored") == "Mexico scored twice."


def test_page_find_returns_at_most_six_lines(install_wiki: InstallWiki) -> None:
    install_wiki(html="<ul>" + "".join(f"<li>goals line {i}</li>" for i in range(10)) + "</ul>")

    assert len(main.page_find("T", "goals").splitlines()) == 6


def test_page_find_truncates_long_line(install_wiki: InstallWiki) -> None:
    install_wiki(html="<p>goals " + "y" * 600 + "</p>")

    assert len(main.page_find("T", "goals")) == 400


# --- page_find: пустой результат -------------------------------------------------

@pytest.mark.parametrize("html", [None, ""], ids=["нет страницы", "пустой html"])
def test_page_find_without_text_reports_missing_page(install_wiki: InstallWiki,
                                                     html: str | None) -> None:
    install_wiki(html=html)

    assert main.page_find("Нет такой статьи", "goals scored") == "no such page"


def test_page_find_without_matches(install_wiki: InstallWiki) -> None:
    install_wiki(html=PAGE)

    assert main.page_find("T", "weather forecast") == "keywords not found on the page"


def test_page_find_with_only_short_keywords(install_wiki: InstallWiki) -> None:
    install_wiki(html=PAGE)

    assert main.page_find("T", "a of in") == "keywords not found on the page"


# --- page_find: ошибка -----------------------------------------------------------

def test_page_find_propagates_wiki_error(install_wiki: InstallWiki) -> None:
    install_wiki(error=RuntimeError("Википедия ответила 503"))

    with pytest.raises(RuntimeError, match="503"):
        main.page_find("T", "goals scored")


# --- run_tool: как ошибку инструмента видит агент --------------------------------

def test_run_tool_returns_tool_message(install_wiki: InstallWiki) -> None:
    install_wiki(search=[{"title": "A"}], html="<p>Intro text</p>")

    message = main.run_tool(tool_call("web_search", {"query": "mexico"}))

    assert message == {"role": "tool", "tool_call_id": "c1", "content": "[A] Intro text"}


def test_run_tool_wraps_wiki_error(install_wiki: InstallWiki) -> None:
    install_wiki(error=RuntimeError("Википедия ответила 503"))

    message = main.run_tool(tool_call("web_search", {"query": "mexico"}))

    assert message["content"].startswith("НЕ удалось вызвать инструмент web_search")
    assert "503" in message["content"]


def test_run_tool_wraps_invalid_arguments(install_wiki: InstallWiki) -> None:
    fake = install_wiki(html=PAGE)

    message = main.run_tool(tool_call("page_find", {"wrong": 1}))

    assert message["content"].startswith("НЕ удалось вызвать инструмент page_find")
    assert fake.calls == [], "до Википедии дело не доходит"


def test_run_tool_wraps_unknown_tool(install_wiki: InstallWiki) -> None:
    install_wiki(search=HITS)

    message = main.run_tool(tool_call("nope", {"query": "mexico"}))

    assert message["content"].startswith("НЕ удалось вызвать инструмент nope")


# --- html_lines: разбор HTML парсера Википедии ------------------------------------

def test_html_lines_keeps_footballbox_on_one_line() -> None:
    assert html_lines(FOOTBALLBOX) == [FOOTBALLBOX_LINE], "скрытая ISO-дата выброшена"


def test_html_lines_splits_paragraph_into_sentences() -> None:
    html = '<p>Mexico won.<sup class="reference">[1]</sup> The U.S. President attended. Fans sang.</p>'

    assert html_lines(html) == ["Mexico won.", "The U.S. President attended.", "Fans sang."]


# --- wiki: ошибки API ---------------------------------------------------------------

class FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json; charset=utf-8"}

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def json(self) -> dict[str, Any]:
        return self.payload


def test_wiki_returns_empty_dict_for_missing_page(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"error": {"code": "missingtitle", "info": "The page you specified doesn't exist."}}
    monkeypatch.setattr(main.requests, "get", lambda *a, **kw: FakeResponse(payload))

    assert main.wiki({"action": "parse", "page": "Нет такой статьи"}) == {}


def test_wiki_raises_on_other_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"error": {"code": "badvalue", "info": "Unrecognized value"}}
    monkeypatch.setattr(main.requests, "get", lambda *a, **kw: FakeResponse(payload))

    with pytest.raises(RuntimeError, match="Unrecognized value"):
        main.wiki({"action": "parse", "page": "T"})
