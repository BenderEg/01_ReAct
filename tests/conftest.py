"""Общие фикстуры для тестов инструментов агента.

main.py на импорте требует OPENROUTER_API_KEY, поэтому ключ-заглушку выставляем
здесь: conftest выполняется раньше, чем pytest соберёт тестовые модули.
"""

import os
from typing import Any, Callable

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from utils import wiki_tools  # noqa: E402

Params = dict[str, Any]


class FakeWiki:
    """Подмена wiki_tools.wiki: отдаёт заготовленные ответы и помнит, о чём её спросили.

    search — хиты для action=query&list=search;
    html — HTML страницы для action=parse (None = страницы нет: настоящая wiki()
    в этом случае возвращает {});
    error — если задан, любой вызов падает с этим исключением.
    """

    def __init__(self, search: list[dict[str, str]] | None = None,
                 html: str | None = None, error: Exception | None = None) -> None:
        self.search = search if search is not None else []
        self.html = html
        self.error = error
        self.calls: list[Params] = []

    def __call__(self, params: Params, attempts: int = 3) -> dict[str, Any]:
        self.calls.append(params)
        if self.error is not None:
            raise self.error
        if params.get("list") == "search":
            return {"search": self.search}
        return {} if self.html is None else {"title": params.get("page", ""), "text": self.html}


InstallWiki = Callable[..., FakeWiki]


@pytest.fixture
def install_wiki(monkeypatch: pytest.MonkeyPatch) -> InstallWiki:
    """Ставит FakeWiki вместо wiki_tools.wiki и возвращает его, чтобы смотреть вызовы."""

    def install(**kwargs: Any) -> FakeWiki:
        fake = FakeWiki(**kwargs)
        monkeypatch.setattr(wiki_tools, "wiki", fake)
        return fake

    return install
