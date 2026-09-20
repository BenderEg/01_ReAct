"""Общие фикстуры для тестов инструментов агента.

main.py на импорте требует OPENROUTER_API_KEY, поэтому ключ-заглушку выставляем
здесь: conftest выполняется раньше, чем pytest соберёт тестовые модули.
"""

import os
from typing import Any, Callable

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

import main  # noqa: E402  (импорт только после подмены ключа)

Params = dict[str, Any]


class FakeWiki:
    """Подмена main.wiki: отдаёт заготовленные ответы и помнит, о чём её спросили.

    search — хиты для action=query&list=search;
    extract — полный текст страницы (None = страницы без поля extract, как у Википедии
    для несуществующего заголовка);
    error — если задан, любой вызов падает с этим исключением.
    """

    def __init__(self, search: list[dict[str, str]] | None = None,
                 extract: str | None = None, error: Exception | None = None) -> None:
        self.search = search if search is not None else []
        self.extract = extract
        self.error = error
        self.calls: list[Params] = []

    def __call__(self, params: Params, attempts: int = 3) -> dict[str, Any]:
        self.calls.append(params)
        if self.error is not None:
            raise self.error
        if params.get("list") == "search":
            return {"search": self.search}
        page: dict[str, Any] = {"pageid": 1, "title": params.get("titles", "")}
        if self.extract is not None:
            page["extract"] = self.extract
        return {"pages": {"1": page}}


InstallWiki = Callable[..., FakeWiki]


@pytest.fixture
def install_wiki(monkeypatch: pytest.MonkeyPatch) -> InstallWiki:
    """Ставит FakeWiki вместо main.wiki и возвращает его, чтобы смотреть вызовы."""

    def install(**kwargs: Any) -> FakeWiki:
        fake = FakeWiki(**kwargs)
        monkeypatch.setattr(main, "wiki", fake)
        return fake

    return install
