#!/usr/bin/env python3
"""
atombench.submit — submit reconstruction benchmarks to the JARVIS-Leaderboard.

This wraps the JARVIS-Leaderboard contribution workflow without modifying the
``jarvis_leaderboard`` package: it validates a benchmark CSV against the
leaderboard's real requirements, builds a valid *AI / AtomGen* contribution
(and, with ``--new-benchmark``, a brand-new benchmark), and opens a pull request
via a clean, token-based GitHub flow.

Usage (CLI)::

    atombench-submit predictions.csv \\
        --dataset dft_3d --prop Tc_supercon \\
        --model-name MyModel --author-email me@example.com \\
        --project-url https://example.com --git-url https://github.com/me/mymodel

    # Build the contribution locally without pushing (inspect, then PR yourself):
    atombench-submit predictions.csv --dataset dft_3d --prop Tc_supercon \\
        --model-name MyModel --author-email me@example.com \\
        --project-url ... --git-url ... --no-push --out ./submission

Usage (Python)::

    from atombench import submit
    submit("predictions.csv", dataset="dft_3d", prop="Tc_supercon",
           model_name="MyModel", author_email="me@example.com",
           project_url="...", git_url="...", push=False, out_dir="./submission")
"""
from __future__ import annotations

import csv as csv_mod
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import click
import pandas as pd

from atombench._common import discover_benchmark_csvs
from atombench._leaderboard import (
    CATEGORY,
    TASK,
    SubmissionError,
    append_description_row,
    benchmark_exists,
    benchmark_name,
    build_benchmark_zip,
    build_contribution_dir,
    default_contribution_name,
    descriptions_row,
    read_benchmark_ids_with_source,
    skeleton_md,
    validate_submission,
)
from atombench._github import (
    GitHubError,
    clone_branch_commit_push,
    ensure_fork,
    get_repo,
    open_pr,
    resolve_token,
    whoami,
)

DEFAULT_REPO = "atomgptlab/jarvis_leaderboard"


# ── Helpers ─────────────────────────────────────────────────────────────────
def _resolve_single_csv(path: Path) -> Path:
    pairs = discover_benchmark_csvs(path)
    if len(pairs) != 1:
        names = ", ".join(name for name, _ in pairs)
        raise SubmissionError(
            f"Expected exactly one benchmark CSV but found {len(pairs)} ({names}). "
            "Submit one benchmark at a time: pass a single CSV with --dataset/--prop."
        )
    return pairs[0][1]


def _load_splits(splits_json: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not splits_json:
        return {}
    data = json.loads(Path(splits_json).read_text())
    out: Dict[str, Dict[str, str]] = {}
    for key in ("train", "val"):
        if isinstance(data.get(key), dict):
            out[key] = {str(k): str(v) for k, v in data[key].items()}
    return out


def _doc_title(dataset: str, prop: str) -> str:
    return f"Generative reconstruction benchmark — {dataset} / {prop}"


def _materialize(
    *,
    contrib_parent: Path,
    benchmarks_dir: Path,
    descriptions_csv: Path,
    docs_dir: Path,
    name: str,
    df: pd.DataFrame,
    dataset: str,
    prop: str,
    meta: dict,
    new_benchmark: bool,
    description: Optional[str],
    splits: Dict[str, Dict[str, str]],
    append_descriptions: bool,
) -> Dict[str, List[str]]:
    """Write all submission artifacts under the given directories.

    Returns a dict of created paths grouped by kind.
    """
    created: Dict[str, List[str]] = {"contribution": [], "benchmark": [], "docs": [], "descriptions": []}

    contrib_dir = build_contribution_dir(contrib_parent, name, [(df, dataset, prop)], meta)
    created["contribution"].append(str(contrib_dir))

    if new_benchmark:
        bench_zip = benchmarks_dir / f"{dataset}_{prop}.json.zip"
        build_benchmark_zip(
            bench_zip, dataset, prop, df,
            train=splits.get("train"), val=splits.get("val"),
        )
        created["benchmark"].append(str(bench_zip))

        md = docs_dir / f"{dataset}_{prop}.md"
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(skeleton_md(_doc_title(dataset, prop)))
        created["docs"].append(str(md))

        if append_descriptions and descriptions_csv.exists():
            append_description_row(descriptions_csv, dataset, prop, description or "")
            created["descriptions"].append(str(descriptions_csv))
        else:
            snippet = contrib_parent / "descriptions_row.csv"
            with snippet.open("w", newline="", encoding="utf-8") as fh:
                csv_mod.writer(fh).writerow(descriptions_row(dataset, prop, description or ""))
            created["descriptions"].append(str(snippet))

    return created


def _run_rebuild(repo_root: Path, echo: Callable[[str], None]) -> None:
    """Best-effort run of the clone's own rebuild.py (opt-in via --rebuild)."""
    inner = repo_root / "jarvis_leaderboard"
    script = inner / "rebuild.py"
    if not script.exists():
        echo("  ! rebuild.py not found in clone; skipping --rebuild")
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    echo("  • running rebuild.py in the clone (this can take a while)…")
    proc = subprocess.run(
        [sys.executable, "rebuild.py"], cwd=str(inner), env=env,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        echo("  ! rebuild failed (non-fatal) — maintainers' CI will rebuild on merge.")


def _pr_body(name: str, bn: str, dataset: str, prop: str, meta: dict, new_benchmark: bool) -> str:
    lines = [
        f"Adds AtomGen contribution **{name}** generated with "
        "[`atombench`](https://github.com/atomgptlab/atombench).",
        "",
        f"- **Model:** {meta.get('model_name')}",
        f"- **Team:** {meta.get('team_name', meta.get('model_name'))}",
        f"- **Benchmark:** `{bn}`",
        f"- **Dataset / property:** `{dataset}` / `{prop}`",
    ]
    if new_benchmark:
        lines.append(
            f"- **New benchmark:** adds `benchmarks/{CATEGORY}/{TASK}/{dataset}_{prop}.json.zip`, "
            f"a `descriptions.csv` row, and `docs/{CATEGORY}/{TASK}/{dataset}_{prop}.md`."
        )
    lines += [
        "",
        "Reproduce the reconstruction metrics:",
        "```bash",
        "pip install atombench",
        "atombench <your_benchmark.csv> out/",
        "```",
    ]
    return "\n".join(lines)


# ── Public API ──────────────────────────────────────────────────────────────
def submit(
    csv: str,
    *,
    dataset: str,
    prop: str,
    model_name: str,
    author_email: str,
    project_url: str,
    git_url: str,
    team_name: Optional[str] = None,
    description: Optional[str] = None,
    new_benchmark: bool = False,
    splits_json: Optional[str] = None,
    repo: str = DEFAULT_REPO,
    base: Optional[str] = None,
    contribution_name: Optional[str] = None,
    push: bool = True,
    out_dir: Optional[str] = None,
    token: Optional[str] = None,
    rebuild: bool = False,
    keep: bool = False,
    check_structures: bool = True,
    extra_metadata: Optional[dict] = None,
    echo: Callable[[str], None] = print,
) -> dict:
    """Validate, build, and (optionally) push a JARVIS-Leaderboard contribution.

    Returns a dict with ``contribution_dir``, ``created`` (artifact paths),
    ``pr_url`` (or ``None``), and ``validation`` (the :class:`ValidationReport`).
    See the module docstring for usage. Raises :class:`SubmissionError` /
    :class:`GitHubError` on failure.
    """
    csv_path = Path(csv)
    resolved = _resolve_single_csv(csv_path)
    df = pd.read_csv(resolved)
    echo(f"Read {len(df)} rows from {resolved.name}")

    name = contribution_name or default_contribution_name(model_name)
    meta = {
        "model_name": model_name,
        "author_email": author_email,
        "project_url": project_url,
        "git_url": git_url,
    }
    if team_name:
        meta["team_name"] = team_name
    if extra_metadata:
        meta.update({k: v for k, v in extra_metadata.items() if v is not None})

    splits = _load_splits(splits_json)

    # ── Validation ────────────────────────────────────────────────────────────
    if new_benchmark:
        if not description:
            raise SubmissionError("--new-benchmark requires --description.")
        if benchmark_exists(dataset, prop, repo=repo, branch=base):
            raise SubmissionError(
                f"A benchmark '{dataset}_{prop}' already exists upstream — drop "
                "--new-benchmark to submit against it."
            )
        report = validate_submission(
            df, dataset, prop, benchmark_ids=None, check_structures=check_structures
        )
    else:
        bench_ids, source = read_benchmark_ids_with_source(
            dataset, prop, repo=repo, branch=base
        )
        if bench_ids is None:
            echo(
                f"  ! could not locate benchmark '{dataset}_{prop}'. If it does not "
                "exist yet, re-run with --new-benchmark --description '...'."
            )
        else:
            echo(f"  • validating ids against {source} ({len(bench_ids)} test ids)")
        report = validate_submission(
            df, dataset, prop, benchmark_ids=bench_ids, check_structures=check_structures
        )

    for w in report.warnings:
        echo(f"  ! {w}")
    report.raise_if_failed()
    echo("  ✓ validation passed")

    bn = benchmark_name(prop, dataset)

    # ── No-push: build a local preview ────────────────────────────────────────
    if not push:
        out_root = Path(out_dir or "atombench_submission").resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        created = _materialize(
            contrib_parent=out_root,
            benchmarks_dir=out_root / "benchmarks" / CATEGORY / TASK,
            descriptions_csv=out_root / "benchmarks" / "descriptions.csv",
            docs_dir=out_root / "docs" / CATEGORY / TASK,
            name=name, df=df, dataset=dataset, prop=prop, meta=meta,
            new_benchmark=new_benchmark, description=description, splits=splits,
            append_descriptions=False,
        )
        contrib_dir = created["contribution"][0]
        echo("")
        echo(f"Built contribution at: {contrib_dir}")
        if new_benchmark:
            echo("New-benchmark artifacts (copy into a jarvis_leaderboard checkout):")
            echo(f"  • benchmark : {created['benchmark'][0]}  -> jarvis_leaderboard/benchmarks/{CATEGORY}/{TASK}/")
            echo(f"  • docs page : {created['docs'][0]}  -> docs/{CATEGORY}/{TASK}/")
            echo(f"  • add row from {created['descriptions'][0]} to jarvis_leaderboard/benchmarks/descriptions.csv")
        echo("")
        echo("To submit, re-run without --no-push (set GITHUB_TOKEN first).")
        return {
            "contribution_dir": contrib_dir,
            "created": created,
            "pr_url": None,
            "validation": report,
        }

    # ── Push: fork → branch → commit → PR ─────────────────────────────────────
    token = resolve_token(token)
    if "/" not in repo:
        raise SubmissionError(f"--repo must be 'owner/name', got {repo!r}")
    owner, repo_name = repo.split("/", 1)

    repo_info = get_repo(token, owner, repo_name)
    if repo_info is None:
        raise GitHubError(
            f"Cannot access {repo}. Check --repo and that your token can see it."
        )
    base = base or repo_info.get("default_branch", "main")

    login = whoami(token)
    echo(f"Authenticated as {login}; ensuring fork of {repo}…")
    fork_full, _ = ensure_fork(token, owner, repo_name, login=login)

    branch = f"atombench-{name}-{int(time.time())}"

    def mutate(repo_root: Path) -> None:
        inner = repo_root / "jarvis_leaderboard"
        _materialize(
            contrib_parent=inner / "contributions",
            benchmarks_dir=inner / "benchmarks" / CATEGORY / TASK,
            descriptions_csv=inner / "benchmarks" / "descriptions.csv",
            docs_dir=repo_root / "docs" / CATEGORY / TASK,
            name=name, df=df, dataset=dataset, prop=prop, meta=meta,
            new_benchmark=new_benchmark, description=description, splits=splits,
            append_descriptions=True,
        )
        if rebuild:
            _run_rebuild(repo_root, echo)

    echo(f"Cloning fork and committing to branch {branch}…")
    repo_dir, work_root = clone_branch_commit_push(
        fork_full, token, branch, mutate,
        base_branch=base,
        commit_message=f"Add AtomGen contribution '{name}' ({bn})",
    )

    head = f"{login}:{branch}"
    title = f"Add AtomGen contribution: {name}"
    body = _pr_body(name, bn, dataset, prop, meta, new_benchmark)
    try:
        try:
            pr_url = open_pr(token, owner, repo_name, head=head, base=base, title=title, body=body)
        except GitHubError:
            alt = (get_repo(token, owner, repo_name) or {}).get("default_branch")
            if alt and alt != base:
                echo(f"  ! base {base!r} rejected; retrying against {alt!r}")
                pr_url = open_pr(token, owner, repo_name, head=head, base=alt, title=title, body=body)
            else:
                raise
    finally:
        if keep:
            echo(f"  kept working clone at {repo_dir}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)

    echo("")
    echo(f"✓ Pull request opened: {pr_url}")
    return {
        "contribution_dir": str(repo_dir),
        "created": {},
        "pr_url": pr_url,
        "validation": report,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("csv", type=click.Path(exists=True, path_type=Path))
@click.option("--dataset", required=True, help="Dataset id, e.g. dft_3d (no '-').")
@click.option("--prop", required=True, help="Property id, e.g. Tc_supercon (no '-').")
@click.option("--model-name", required=True, help="Model name (also the default team/contribution name).")
@click.option("--author-email", required=True, help="Contact email (must contain '@').")
@click.option("--project-url", required=True, help="Project/paper URL.")
@click.option("--git-url", required=True, help="Source code URL.")
@click.option("--team-name", default=None, help="Team name (defaults to --model-name).")
@click.option("--software-used", default=None, help="Software stack used.")
@click.option("--hardware-used", default=None, help="Hardware used.")
@click.option("--date-submitted", default=None, help="MM-DD-YYYY (defaults to today).")
@click.option("--new-benchmark", is_flag=True, default=False,
              help="Also create a new benchmark from this CSV's target column.")
@click.option("--description", default=None, help="Benchmark description (required with --new-benchmark).")
@click.option("--splits-json", type=click.Path(exists=True, path_type=Path), default=None,
              help="Optional JSON with 'train'/'val' {id: structure} for a new benchmark.")
@click.option("--repo", default=DEFAULT_REPO, show_default=True, help="Upstream owner/name to submit to.")
@click.option("--base", default=None, help="Base branch for the PR (defaults to the repo's default branch).")
@click.option("--contribution-name", default=None, help="Contribution directory name (defaults to a slug of --model-name).")
@click.option("--push/--no-push", default=True, show_default=True,
              help="Open a PR (default) or just build the contribution locally.")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=None,
              help="Output directory for --no-push (default: ./atombench_submission).")
@click.option("--token", default=None, help="GitHub token (else $GITHUB_TOKEN/$GH_TOKEN).")
@click.option("--rebuild", is_flag=True, default=False, help="Run rebuild.py in the clone before pushing (slow).")
@click.option("--keep", is_flag=True, default=False, help="Keep the temporary clone after pushing.")
@click.option("--check-structures/--no-check-structures", default=True, show_default=True,
              help="Parse every prediction during validation (disable for speed).")
def main(
    csv: Path,
    dataset: str,
    prop: str,
    model_name: str,
    author_email: str,
    project_url: str,
    git_url: str,
    team_name: Optional[str],
    software_used: Optional[str],
    hardware_used: Optional[str],
    date_submitted: Optional[str],
    new_benchmark: bool,
    description: Optional[str],
    splits_json: Optional[Path],
    repo: str,
    base: Optional[str],
    contribution_name: Optional[str],
    push: bool,
    out_dir: Optional[Path],
    token: Optional[str],
    rebuild: bool,
    keep: bool,
    check_structures: bool,
) -> None:
    """Submit a reconstruction benchmark CSV to the JARVIS-Leaderboard.

    \b
    CSV must have columns: id, target, prediction (the same file atombench scores).
    The leaderboard scores 'prediction' against its own ground truth; structures
    are normalized to POSCAR before submission.
    """
    extra_metadata = {
        "software_used": software_used,
        "hardware_used": hardware_used,
        "date_submitted": date_submitted,
    }
    try:
        submit(
            str(csv),
            dataset=dataset, prop=prop,
            model_name=model_name, author_email=author_email,
            project_url=project_url, git_url=git_url,
            team_name=team_name, description=description,
            new_benchmark=new_benchmark,
            splits_json=str(splits_json) if splits_json else None,
            repo=repo, base=base, contribution_name=contribution_name,
            push=push, out_dir=str(out_dir) if out_dir else None,
            token=token, rebuild=rebuild, keep=keep,
            check_structures=check_structures,
            extra_metadata=extra_metadata,
            echo=click.echo,
        )
    except (SubmissionError, GitHubError) as exc:
        raise click.ClickException(str(exc))


if __name__ == "__main__":
    main()
