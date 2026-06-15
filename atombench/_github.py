"""
atombench._github — minimal, clean GitHub fork/branch/commit/push/PR helpers.

A deliberately small reimplementation of what jarvis_leaderboard's jarvis_upload.py
does, but: token-based (never reads passwords from ``git config``), uses
``subprocess`` instead of ``os.system``, never echoes the token, and targets a
configurable upstream (default ``atomgptlab/jarvis_leaderboard``).

Authentication uses a GitHub Personal Access Token from ``--token`` or the
``GITHUB_TOKEN`` / ``GH_TOKEN`` environment variables.  ``git`` must be on PATH
and ``requests`` must be installed (``pip install 'atombench[submit]'``).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

API = "https://api.github.com"


class GitHubError(Exception):
    """Raised for any GitHub/git failure during a push."""


# ── token / requests ────────────────────────────────────────────────────────
def resolve_token(explicit: Optional[str] = None) -> str:
    """Return a GitHub token from *explicit*, ``GITHUB_TOKEN`` or ``GH_TOKEN``."""
    token = explicit or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token or not token.strip():
        raise GitHubError(
            "No GitHub token found. Set GITHUB_TOKEN (or GH_TOKEN), or pass --token.\n"
            "Create one at https://github.com/settings/tokens — classic token with "
            "the 'repo' scope, or a fine-grained token with Contents + Pull requests "
            "set to 'Read and write'."
        )
    return token.strip()


def _requests():
    try:
        import requests

        return requests
    except Exception as exc:  # pragma: no cover - import guard
        raise GitHubError(
            "The 'requests' package is required to push. Install it with: "
            "pip install 'atombench[submit]'"
        ) from exc


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _msg(resp) -> str:
    try:
        j = resp.json()
        m = j.get("message", "")
        errs = j.get("errors")
        return f"{m}{(' ' + str(errs)) if errs else ''}".strip()
    except Exception:
        return (resp.text or "")[:200]


# ── REST helpers ────────────────────────────────────────────────────────────
def whoami(token: str) -> str:
    """Return the authenticated user's login."""
    resp = _requests().get(f"{API}/user", headers=_headers(token), timeout=30)
    if resp.status_code != 200:
        raise GitHubError(
            f"GitHub authentication failed ({resp.status_code}). "
            f"Check your token. {_msg(resp)}"
        )
    return resp.json()["login"]


def get_repo(token: str, owner: str, repo: str) -> Optional[dict]:
    """Return repo metadata, or ``None`` if it doesn't exist / isn't visible."""
    resp = _requests().get(
        f"{API}/repos/{owner}/{repo}", headers=_headers(token), timeout=30
    )
    return resp.json() if resp.status_code == 200 else None


def _sync_fork(token: str, fork_full: str, branch: str) -> None:
    """Best-effort: fast-forward the fork's *branch* to upstream."""
    try:
        _requests().post(
            f"{API}/repos/{fork_full}/merge-upstream",
            headers=_headers(token),
            json={"branch": branch},
            timeout=30,
        )
    except Exception:
        pass


def ensure_fork(
    token: str,
    owner: str,
    repo: str,
    *,
    login: Optional[str] = None,
    sync: bool = True,
    timeout: int = 120,
) -> Tuple[str, str]:
    """Ensure the authenticated user has a fork of ``owner/repo``.

    Returns ``(fork_full_name, default_branch)``.  Creates the fork if needed and
    waits for it to become available; syncs an existing fork best-effort.
    """
    requests = _requests()
    login = login or whoami(token)

    existing = get_repo(token, login, repo)
    if existing is None:
        resp = requests.post(
            f"{API}/repos/{owner}/{repo}/forks", headers=_headers(token), timeout=30
        )
        if resp.status_code not in (200, 202):
            raise GitHubError(
                f"Failed to fork {owner}/{repo} ({resp.status_code}). {_msg(resp)}"
            )
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(3)
            existing = get_repo(token, login, repo)
            if existing is not None:
                break
        if existing is None:
            raise GitHubError(
                "Timed out waiting for the fork to be created. "
                "Try again in a minute."
            )
    elif sync:
        _sync_fork(token, f"{login}/{repo}", existing.get("default_branch", "main"))

    return f"{login}/{repo}", existing.get("default_branch", "main")


def open_pr(
    token: str,
    owner: str,
    repo: str,
    *,
    head: str,
    base: str,
    title: str,
    body: str,
) -> str:
    """Open a pull request and return its HTML URL."""
    resp = _requests().post(
        f"{API}/repos/{owner}/{repo}/pulls",
        headers=_headers(token),
        json={"title": title, "head": head, "base": base, "body": body},
        timeout=30,
    )
    if resp.status_code == 201:
        return resp.json()["html_url"]
    raise GitHubError(
        f"Failed to open pull request against {owner}/{repo} ({resp.status_code}). "
        f"Base branch {base!r}; head {head!r}. {_msg(resp)}"
    )


# ── git helpers ─────────────────────────────────────────────────────────────
def _redact(text: str, token: Optional[str]) -> str:
    return text.replace(token, "***") if token else text


def _git(args, cwd: Path, token: Optional[str] = None) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip())
        raise GitHubError(
            f"git {args[0]} failed:\n{_redact(detail, token)}"
        )
    return proc.stdout


def clone_branch_commit_push(
    fork_full: str,
    token: str,
    branch: str,
    mutate_fn: Callable[[Path], None],
    *,
    base_branch: str = "main",
    commit_message: str,
    work_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Clone the fork, create *branch*, apply *mutate_fn*, commit and push.

    *mutate_fn* receives the repository root and should create/modify files
    (e.g. drop in a contribution directory).  Returns ``(repo_dir, work_root)``;
    the caller is responsible for removing *work_root* unless it wants to keep it.
    """
    root = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="atombench_submit_")
    )
    root.mkdir(parents=True, exist_ok=True)
    repo_dir = root / fork_full.split("/")[-1]
    authed = f"https://x-access-token:{token}@github.com/{fork_full}.git"

    try:
        _git(
            ["clone", "--depth", "1", "--branch", base_branch, authed, str(repo_dir)],
            cwd=root,
            token=token,
        )
    except GitHubError:
        # base_branch may not exist on the fork — fall back to its default branch.
        _git(["clone", "--depth", "1", authed, str(repo_dir)], cwd=root, token=token)

    # Ensure a committer identity exists even if git isn't configured globally.
    _git(["config", "user.name", "atombench"], cwd=repo_dir, token=token)
    _git(
        ["config", "user.email", "atombench@users.noreply.github.com"],
        cwd=repo_dir,
        token=token,
    )
    _git(["checkout", "-b", branch], cwd=repo_dir, token=token)

    mutate_fn(repo_dir)

    _git(["add", "-A"], cwd=repo_dir, token=token)
    if not _git(["status", "--porcelain"], cwd=repo_dir, token=token).strip():
        raise GitHubError(
            "No changes to submit — the contribution may already exist upstream."
        )
    _git(["commit", "-m", commit_message], cwd=repo_dir, token=token)
    _git(["push", "-u", "origin", branch], cwd=repo_dir, token=token)
    return repo_dir, root
