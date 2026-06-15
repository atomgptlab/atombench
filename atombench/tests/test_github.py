"""Offline tests for atombench._github (requests fully mocked, no network)."""
from __future__ import annotations

import pytest

from atombench import _github as G


# ── fake requests ────────────────────────────────────────────────────────────
class FakeResp:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeRequests:
    """Routes (method, url-substring) -> FakeResp; matched routes are one-shot."""

    def __init__(self, routes):
        self.routes = list(routes)
        self.calls = []

    def _resolve(self, method, url):
        self.calls.append((method, url))
        for i, (m, substr, resp) in enumerate(self.routes):
            if m == method and substr in url:
                self.routes.pop(i)
                return resp
        raise AssertionError(f"no route for {method} {url}")

    def get(self, url, **kw):
        return self._resolve("GET", url)

    def post(self, url, **kw):
        return self._resolve("POST", url)


def _use(monkeypatch, *routes):
    fake = FakeRequests(routes)
    monkeypatch.setattr(G, "_requests", lambda: fake)
    return fake


# ── token resolution ─────────────────────────────────────────────────────────
def test_resolve_token_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(G.GitHubError):
        G.resolve_token()


def test_resolve_token_precedence(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "from_gh")
    assert G.resolve_token() == "from_gh"
    monkeypatch.setenv("GITHUB_TOKEN", "from_github")
    assert G.resolve_token() == "from_github"  # GITHUB_TOKEN wins
    assert G.resolve_token("explicit") == "explicit"  # explicit beats env


# ── REST helpers ─────────────────────────────────────────────────────────────
def test_whoami(monkeypatch):
    _use(monkeypatch, ("GET", "/user", FakeResp(200, {"login": "alice"})))
    assert G.whoami("t") == "alice"


def test_whoami_auth_failure(monkeypatch):
    _use(monkeypatch, ("GET", "/user", FakeResp(401, {"message": "Bad credentials"})))
    with pytest.raises(G.GitHubError):
        G.whoami("t")


def test_ensure_fork_existing(monkeypatch):
    monkeypatch.setattr(G, "whoami", lambda token: "alice")
    _use(
        monkeypatch,
        ("GET", "/repos/alice/jarvis_leaderboard", FakeResp(200, {"default_branch": "develop"})),
        ("POST", "/merge-upstream", FakeResp(409, {})),  # best-effort sync, ignored
    )
    full, branch = G.ensure_fork("t", "usnistgov", "jarvis_leaderboard")
    assert full == "alice/jarvis_leaderboard"
    assert branch == "develop"


def test_ensure_fork_creates(monkeypatch):
    monkeypatch.setattr(G, "whoami", lambda token: "bob")
    monkeypatch.setattr(G.time, "sleep", lambda *_: None)  # don't actually wait
    _use(
        monkeypatch,
        ("GET", "/repos/bob/jarvis_leaderboard", FakeResp(404)),       # not forked yet
        ("POST", "/forks", FakeResp(202, {})),                          # fork accepted
        ("GET", "/repos/bob/jarvis_leaderboard", FakeResp(200, {"default_branch": "main"})),
    )
    full, branch = G.ensure_fork("t", "atomgptlab", "jarvis_leaderboard")
    assert full == "bob/jarvis_leaderboard"
    assert branch == "main"


def test_open_pr_success(monkeypatch):
    _use(monkeypatch, ("POST", "/pulls", FakeResp(201, {"html_url": "https://github.com/o/r/pull/7"})))
    url = G.open_pr("t", "o", "r", head="bob:branch", base="main", title="T", body="B")
    assert url.endswith("/pull/7")


def test_open_pr_failure(monkeypatch):
    _use(monkeypatch, ("POST", "/pulls", FakeResp(422, {"message": "Validation failed"})))
    with pytest.raises(G.GitHubError):
        G.open_pr("t", "o", "r", head="bob:branch", base="nope", title="T", body="B")


# ── helpers ──────────────────────────────────────────────────────────────────
def test_redact():
    assert G._redact("token=SECRET in url", "SECRET") == "token=*** in url"
    assert G._redact("no token here", None) == "no token here"
