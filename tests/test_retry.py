"""Тесты повторов запроса к OpenRouter: main.retry_delay, main.post_with_retry, main.Throttle.

Тело 429 взято из настоящего трейса haiku: лимит нового аккаунта, 20 запросов в минуту,
момент снятия лимита — error.metadata.headers.X-RateLimit-Reset в миллисекундах.
"""

import json
from typing import Any

import pytest
import requests

import main

NOW = 1_789_904_000.0


def response(status: int, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body or {}).encode()
    r.headers.update(headers or {})
    return r


def rate_limited(reset_ms: int) -> requests.Response:
    return response(429, {"error": {"code": 429, "message": "Rate limit exceeded: new-account-rpm",
                                    "metadata": {"headers": {"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0",
                                                             "X-RateLimit-Reset": str(reset_ms)}}}})


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Замороженное время: time.time() отдаёт NOW, time.sleep только записывает паузы."""
    sleeps: list[float] = []
    monkeypatch.setattr(main.time, "time", lambda: NOW)
    monkeypatch.setattr(main.time, "sleep", sleeps.append)
    return sleeps


def test_delay_from_reset_in_body(clock: list[float]) -> None:
    assert main.retry_delay(rate_limited(int((NOW + 37) * 1000)), 0) == pytest.approx(38)


def test_delay_from_reset_header(clock: list[float]) -> None:
    r = response(429, headers={"X-RateLimit-Reset": str(int((NOW + 10) * 1000))})
    assert main.retry_delay(r, 0) == pytest.approx(11)


def test_retry_after_wins(clock: list[float]) -> None:
    r = rate_limited(int((NOW + 37) * 1000))
    r.headers["Retry-After"] = "5"
    assert main.retry_delay(r, 0) == 5


def test_delay_is_clamped(clock: list[float]) -> None:
    assert main.retry_delay(rate_limited(int((NOW + 600) * 1000)), 0) == main.MAX_WAIT
    assert main.retry_delay(rate_limited(int((NOW - 5) * 1000)), 0) == 1


@pytest.mark.parametrize(("attempt", "delay"), [(0, 3), (2, 12), (10, 60)])
def test_fallback_backoff(clock: list[float], attempt: int, delay: float) -> None:
    assert main.retry_delay(response(503, {"error": "busy"}), attempt) == delay
    assert main.retry_delay(None, attempt) == delay


def test_post_waits_for_reset_then_succeeds(monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
    replies = iter([rate_limited(int((NOW + 20) * 1000)), rate_limited(int((NOW + 40) * 1000)),
                    response(200, {"choices": [], "usage": {"cost": 0.001}})])
    monkeypatch.setattr(main.requests, "post", lambda *a, **kw: next(replies))
    assert main.post_with_retry({"model": "m"}) == {"choices": [], "usage": {"cost": 0.001}}
    assert clock == [pytest.approx(21), pytest.approx(41)]


def test_post_retries_network_errors(monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
    def post(*args: Any, **kwargs: Any) -> requests.Response:
        if not clock:
            raise requests.Timeout("read timeout")
        return response(200, {"ok": True})

    monkeypatch.setattr(main.requests, "post", post)
    assert main.post_with_retry({"model": "m"}) == {"ok": True}
    assert clock == [3]


def test_post_gives_up_after_attempts(monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
    monkeypatch.setattr(main.requests, "post", lambda *a, **kw: rate_limited(int((NOW + 30) * 1000)))
    with pytest.raises(RuntimeError, match=r"^HTTP 429 : "):
        main.post_with_retry({"model": "m"}, attempts=3)
    assert len(clock) == 2  # после последней попытки не ждём


def test_post_does_not_retry_client_error(monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
    monkeypatch.setattr(main.requests, "post", lambda *a, **kw: response(400, {"error": "bad request"}))
    with pytest.raises(RuntimeError, match=r"^HTTP 400 : "):
        main.post_with_retry({"model": "m"})
    assert clock == []


def test_throttle_spaces_requests_per_model(monkeypatch: pytest.MonkeyPatch, clock: list[float]) -> None:
    monkeypatch.setattr(main.time, "monotonic", lambda: 100.0)
    throttle = main.Throttle(rpm=20)
    throttle.wait("a")
    throttle.wait("b")
    throttle.wait("a")
    assert clock == [pytest.approx(3)]


def test_throttle_off_by_default(clock: list[float]) -> None:
    throttle = main.Throttle()
    throttle.wait("a")
    throttle.wait("a")
    assert clock == []
