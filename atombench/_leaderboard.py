"""
atombench._leaderboard — offline building blocks for JARVIS-Leaderboard submissions.

This module contains everything needed to turn an atombench benchmark CSV
(columns: id, target, prediction) into a valid JARVIS-Leaderboard *AI / AtomGen*
contribution — and, optionally, into a brand-new benchmark — without any network
access and without modifying the jarvis_leaderboard package.

Nothing here pushes anything; see atombench._github for the GitHub side and
atombench.submit for the orchestration / CLI.

Key facts about the leaderboard that shape this code (verified against
jarvis_leaderboard/jarvis_leaderboard/rebuild.py):

* A contribution lives in ``contributions/<name>/`` and holds one or more
  ``AI-AtomGen-<prop>-<dataset>-test-rmse.csv.zip`` files (inner file named
  ``<benchmark>.csv``) plus a ``metadata.json``.
* The AtomGen scorer ignores the CSV ``target`` column.  It loads ground truth
  from ``benchmarks/AI/AtomGen/<dataset>_<prop>.json.zip`` (``test`` split),
  merges on ``id`` (requiring equal row counts and matching ids), and parses the
  ``prediction`` column with jarvis ``Poscar.from_string(s.replace("\\n","\\n"))``
  — so predictions must be **POSCAR-escaped**, not CIF.
* A new benchmark additionally needs the ground-truth ``.json.zip``, a row in
  ``benchmarks/descriptions.csv`` and a skeleton ``docs/AI/AtomGen/<name>.md``.
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from atombench._structure_io import parse_structure


# ── Constants ───────────────────────────────────────────────────────────────
CATEGORY = "AI"
TASK = "AtomGen"
DEFAULT_METRIC = "rmse"
DEFAULT_SPLIT = "test"

# Fields rebuild.py:check_metadata_info_exists requires for a contribution to be
# accepted.  We guarantee all of these are present in metadata.json.
REQUIRED_METADATA_FIELDS = (
    "model_name",
    "team_name",
    "author_email",
    "project_url",
    "git_url",
    "time_taken_seconds",
    "software_used",
    "hardware_used",
)

# Path of the benchmarks dir relative to a jarvis_leaderboard checkout root.
_BENCH_SUBPATH = ("jarvis_leaderboard", "benchmarks", CATEGORY, TASK)


class SubmissionError(Exception):
    """Raised for any user-facing problem while preparing a submission."""


# ── Naming ──────────────────────────────────────────────────────────────────
def benchmark_name(
    prop: str,
    dataset: str,
    *,
    metric: str = DEFAULT_METRIC,
    split: str = DEFAULT_SPLIT,
    category: str = CATEGORY,
    task: str = TASK,
) -> str:
    """Return the canonical benchmark name, e.g. ``AI-AtomGen-Tc_supercon-dft_3d-test-rmse``.

    The leaderboard parses this name by splitting on ``-`` (rebuild.py:
    ``get_metric_value``), so the individual fields must not contain a dash.
    """
    for label, value in (
        ("category", category),
        ("task", task),
        ("prop", prop),
        ("dataset", dataset),
        ("split", split),
        ("metric", metric),
    ):
        if not value or not str(value).strip():
            raise SubmissionError(f"benchmark {label} must be non-empty")
        if "-" in str(value):
            raise SubmissionError(
                f"benchmark {label} {value!r} must not contain '-' "
                "(the leaderboard splits the benchmark name on '-'); use '_' instead"
            )
    return f"{category}-{task}-{prop}-{dataset}-{split}-{metric}"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def default_contribution_name(model_name: str) -> str:
    """Slugify *model_name* into a default contribution directory name."""
    slug = _SLUG_RE.sub("_", str(model_name).strip().lower()).strip("_")
    return slug or "atombench_contribution"


# ── Structure normalisation ─────────────────────────────────────────────────
def _to_poscar_text(cell_str: str) -> str:
    """Parse any supported structure representation → plain-newline POSCAR text.

    Prefers jarvis-tools (byte-compatible with the leaderboard scorer, which uses
    jarvis ``Poscar``) and falls back to pymatgen if jarvis-tools is unavailable.
    """
    structure = parse_structure(cell_str)
    try:  # best compatibility with the AtomGen scorer
        from jarvis.core.atoms import pmg_to_atoms
        from jarvis.io.vasp.inputs import Poscar as JarvisPoscar

        return JarvisPoscar(pmg_to_atoms(structure)).to_string()
    except Exception:
        from pymatgen.io.vasp.inputs import Poscar as PmgPoscar

        return str(PmgPoscar(structure))


def normalize_to_poscar_escaped(cell_str: str) -> str:
    """Return *cell_str* as a single-line, ``\\n``-escaped POSCAR string.

    This is the exact representation the AtomGen scorer expects for both the
    ``prediction`` column and the benchmark ground truth.
    """
    text = _to_poscar_text(cell_str)
    return text.replace("\r\n", "\n").replace("\n", "\\n")


# ── Benchmark ground-truth access ───────────────────────────────────────────
def _read_json_zip(path: Path, inner: Optional[str] = None) -> dict:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        name = inner if inner and inner in names else names[0]
        return json.loads(z.read(name))


def _local_benchmark_zip(dataset: str, prop: str) -> Optional[Path]:
    """Locate ``<dataset>_<prop>.json.zip`` in an installed jarvis_leaderboard, if any."""
    try:
        import jarvis_leaderboard
    except Exception:
        return None
    root = Path(jarvis_leaderboard.__path__[0])
    candidate = root / "benchmarks" / CATEGORY / TASK / f"{dataset}_{prop}.json.zip"
    return candidate if candidate.exists() else None


def _raw_benchmark_url(repo: str, branch: str, dataset: str, prop: str) -> str:
    return (
        f"https://raw.githubusercontent.com/{repo}/{branch}/"
        f"jarvis_leaderboard/benchmarks/{CATEGORY}/{TASK}/{dataset}_{prop}.json.zip"
    )


def _fetch_benchmark_json(
    repo: str, branch: str, dataset: str, prop: str
) -> Optional[dict]:
    try:
        import requests
    except Exception:
        return None
    for br in (branch, "main", "develop", "master"):
        if not br:
            continue
        try:
            resp = requests.get(_raw_benchmark_url(repo, br, dataset, prop), timeout=30)
        except Exception:
            continue
        if resp.status_code == 200:
            try:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    return json.loads(z.read(z.namelist()[0]))
            except Exception:
                return None
    return None


def read_benchmark_split_with_source(
    dataset: str,
    prop: str,
    *,
    split: str = DEFAULT_SPLIT,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
) -> Tuple[Optional[Dict[str, str]], str]:
    """Return ``({id: structure_str}, source)`` for *split*.

    Tries an installed jarvis_leaderboard first, then a raw GitHub fetch from
    *repo* (e.g. ``atomgptlab/jarvis_leaderboard``).  *source* is a human-readable
    description of where the ground truth came from, so callers can report it.
    """
    local = _local_benchmark_zip(dataset, prop)
    if local is not None:
        data = _read_json_zip(local, f"{dataset}_{prop}.json")
        return data.get(split), f"local jarvis_leaderboard install ({local})"
    if repo:
        data = _fetch_benchmark_json(repo, branch or "main", dataset, prop)
        if data is not None:
            return data.get(split), f"{repo} on GitHub"
    return None, "unavailable"


def read_benchmark_split(
    dataset: str,
    prop: str,
    *,
    split: str = DEFAULT_SPLIT,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Return ``{id: structure_str}`` for *split*, or ``None`` if unavailable."""
    return read_benchmark_split_with_source(
        dataset, prop, split=split, repo=repo, branch=branch
    )[0]


def read_benchmark_ids_with_source(
    dataset: str,
    prop: str,
    *,
    split: str = DEFAULT_SPLIT,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
) -> Tuple[Optional[Set[str]], str]:
    """Return ``(ids, source)`` for the benchmark *split*."""
    sp, source = read_benchmark_split_with_source(
        dataset, prop, split=split, repo=repo, branch=branch
    )
    return ({str(k) for k in sp} if sp is not None else None), source


def read_benchmark_ids(
    dataset: str,
    prop: str,
    *,
    split: str = DEFAULT_SPLIT,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
) -> Optional[Set[str]]:
    """Return the set of ids in the benchmark *split*, or ``None`` if unavailable."""
    return read_benchmark_ids_with_source(
        dataset, prop, split=split, repo=repo, branch=branch
    )[0]


def benchmark_exists(
    dataset: str,
    prop: str,
    *,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
) -> bool:
    """True iff the benchmark json.zip can be located locally or fetched."""
    if _local_benchmark_zip(dataset, prop) is not None:
        return True
    if repo:
        return _fetch_benchmark_json(repo, branch or "main", dataset, prop) is not None
    return False


# ── Validation ──────────────────────────────────────────────────────────────
@dataclass
class ValidationReport:
    """Outcome of :func:`validate_submission`."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise SubmissionError(
                "Submission validation failed:\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).strip().lower() for c in df.columns})


def validate_submission(
    df: pd.DataFrame,
    dataset: str,
    prop: str,
    *,
    benchmark_ids: Optional[Set[str]] = None,
    check_structures: bool = True,
    max_struct_checks: Optional[int] = None,
) -> ValidationReport:
    """Validate a benchmark DataFrame against the leaderboard's real requirements.

    Catches the failure modes that otherwise produce a silently-zeroed score or a
    rejected PR: missing columns, empty/duplicate ids, id sets that don't match
    the benchmark test split, and predictions that can't be parsed.
    """
    errors: List[str] = []
    warnings: List[str] = []

    df = _norm_cols(df)
    cols = set(df.columns)
    for required in ("id", "prediction"):
        if required not in cols:
            errors.append(f"missing required column {required!r}")
    if "target" not in cols:
        warnings.append(
            "no 'target' column — fine for submission (the leaderboard scores "
            "against its own ground truth), but atombench metrics need it"
        )
    if errors:
        return ValidationReport(False, errors, warnings)

    ids = [str(x) for x in df["id"].tolist()]
    seen: Set[str] = set()
    dups: Set[str] = set()
    empties = 0
    for i in ids:
        if not i or not i.strip() or i.lower() == "nan":
            empties += 1
            continue
        if i in seen:
            dups.add(i)
        seen.add(i)
    if empties:
        errors.append(f"{empties} row(s) have an empty id")
    if dups:
        errors.append(
            f"{len(dups)} duplicate id(s), e.g. {sorted(dups)[:5]}"
        )

    if benchmark_ids is not None:
        missing = benchmark_ids - seen
        extra = seen - benchmark_ids
        if missing:
            errors.append(
                f"{len(missing)} benchmark id(s) are missing from your CSV "
                f"(scoring requires every test id), e.g. {sorted(missing)[:5]}"
            )
        if extra:
            errors.append(
                f"{len(extra)} id(s) in your CSV are not in the benchmark "
                f"test set, e.g. {sorted(extra)[:5]}"
            )
    else:
        warnings.append(
            "could not load the benchmark id list; skipped the id-match check "
            "(the leaderboard will still enforce it at scoring time)"
        )

    if check_structures and "prediction" in cols:
        preds = [str(x) for x in df["prediction"].tolist()]
        rows = list(enumerate(preds))
        if max_struct_checks:
            rows = rows[:max_struct_checks]
        bad: List[Tuple[str, str]] = []
        for idx, cell in rows:
            try:
                parse_structure(cell)
            except Exception as exc:
                rid = ids[idx] if idx < len(ids) else str(idx)
                bad.append((rid, str(exc).splitlines()[0][:80]))
                if len(bad) >= 10:
                    break
        if bad:
            sample = "; ".join(f"{rid} ({msg})" for rid, msg in bad[:3])
            errors.append(
                f"{len(bad)}{'+' if len(bad) >= 10 else ''} prediction(s) failed "
                f"to parse, e.g. {sample}"
            )

    return ValidationReport(len(errors) == 0, errors, warnings)


# ── Metadata ────────────────────────────────────────────────────────────────
def complete_metadata(meta: Optional[dict], *, csv_zip_names: Sequence[str]) -> dict:
    """Fill defaults and assert the field set required by rebuild.py.

    Raises :class:`SubmissionError` if a field that cannot be defaulted is missing.
    """
    m = dict(meta or {})

    if not m.get("model_name"):
        raise SubmissionError("metadata: 'model_name' is required")
    if "@" not in str(m.get("author_email", "")):
        raise SubmissionError(
            "metadata: 'author_email' must be a valid email address (contain '@')"
        )
    for key in ("project_url", "git_url"):
        if not m.get(key):
            raise SubmissionError(f"metadata: {key!r} is required")

    # git_url is a list in the canonical examples; accept a bare string too.
    if isinstance(m["git_url"], str):
        m["git_url"] = [m["git_url"]]

    m.setdefault("team_name", m["model_name"])
    m.setdefault("date_submitted", date.today().strftime("%m-%d-%Y"))
    m.setdefault("software_used", "atombench")
    m.setdefault("hardware_used", "not specified")
    m.setdefault("language", "python")
    m.setdefault("os", sys.platform)

    tts = m.get("time_taken_seconds")
    if not isinstance(tts, dict):
        fill = "" if tts in (None, "") else tts
        m["time_taken_seconds"] = {name: fill for name in csv_zip_names}
    else:
        for name in csv_zip_names:
            tts.setdefault(name, "")

    missing = [f for f in REQUIRED_METADATA_FIELDS if not m.get(f)]
    if missing:
        raise SubmissionError(f"metadata is missing required fields: {missing}")
    return m


def _run_sh() -> str:
    return (
        "#!/bin/bash\n"
        "# Generated by atombench-submit.\n"
        "# Reproduce this contribution's reconstruction metrics:\n"
        "#   pip install atombench\n"
        "#   atombench <your_benchmark.csv> out/\n"
    )


# ── Contribution / benchmark / docs builders ────────────────────────────────
def _write_csv_zip(df: pd.DataFrame, out_zip: Path, inner_name: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(inner_name, buf.getvalue())


def build_contribution_dir(
    out_dir: Path,
    name: str,
    items: Sequence[Tuple[pd.DataFrame, str, str]],
    metadata: dict,
) -> Path:
    """Write a complete contribution directory and return its path.

    *items* is a sequence of ``(df, dataset, prop)``; one ``.csv.zip`` is written
    per item, with the ``prediction`` column normalised to POSCAR-escaped.
    """
    contrib = Path(out_dir) / name
    contrib.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for df, dataset, prop in items:
        bn = benchmark_name(prop, dataset)
        out = _norm_cols(df)
        if "id" not in out.columns or "prediction" not in out.columns:
            raise SubmissionError(
                f"{bn}: CSV must have 'id' and 'prediction' columns"
            )
        out = out[[c for c in ("id", "target", "prediction") if c in out.columns]].copy()
        out["prediction"] = out["prediction"].astype(str).map(normalize_to_poscar_escaped)
        zip_name = f"{bn}.csv.zip"
        _write_csv_zip(out, contrib / zip_name, f"{bn}.csv")
        written.append(zip_name)

    meta = complete_metadata(metadata, csv_zip_names=written)
    (contrib / "metadata.json").write_text(json.dumps(meta, indent=4) + "\n")
    (contrib / "run.sh").write_text(_run_sh())
    return contrib


def build_benchmark_zip(
    out_path: Path,
    dataset: str,
    prop: str,
    df: pd.DataFrame,
    *,
    train: Optional[Dict[str, str]] = None,
    val: Optional[Dict[str, str]] = None,
    normalize: bool = True,
) -> Path:
    """Build a new benchmark ground-truth ``<dataset>_<prop>.json.zip`` from a CSV.

    The ``test`` split is taken from the CSV (``id`` → ``target``); ``train`` /
    ``val`` are optional and default to empty.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    d = _norm_cols(df)
    if "id" not in d.columns or "target" not in d.columns:
        raise SubmissionError(
            "building a new benchmark requires 'id' and 'target' columns "
            "(the 'target' column becomes the ground truth)"
        )

    def conv(value: object) -> str:
        return normalize_to_poscar_escaped(str(value)) if normalize else str(value)

    test = {str(i): conv(t) for i, t in zip(d["id"], d["target"])}
    data: Dict[str, dict] = {"train": dict(train or {}), "test": test}
    if val:
        data["val"] = dict(val)

    inner = f"{dataset}_{prop}.json"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(inner, json.dumps(data))
    return out


def descriptions_row(dataset: str, prop: str, description: str) -> List[str]:
    """Return a ``benchmarks/descriptions.csv`` row for a new benchmark.

    Columns: Category, Sub-category, Benchmark, Description, Experimental DOI.
    """
    if not description or not str(description).strip():
        raise SubmissionError(
            "a new benchmark requires a non-empty --description "
            "(it appears on the leaderboard and in descriptions.csv)"
        )
    return [CATEGORY, TASK, f"{dataset}_{prop}", str(description).strip(), ""]


def append_description_row(
    descriptions_csv: Path, dataset: str, prop: str, description: str
) -> bool:
    """Append a description row if one for ``<dataset>_<prop>`` isn't already present.

    Returns True if a row was added, False if it already existed.
    """
    import csv as csv_mod

    bench = f"{dataset}_{prop}"
    path = Path(descriptions_csv)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv_mod.reader(fh):
                if len(row) >= 3 and row[0] == CATEGORY and row[1] == TASK and row[2] == bench:
                    return False
    needs_newline = path.exists() and path.read_bytes()[-1:] not in (b"\n", b"")
    with path.open("a", newline="", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        csv_mod.writer(fh).writerow(descriptions_row(dataset, prop, description))
    return True


def skeleton_md(title: str) -> str:
    """Return a skeleton ``docs/AI/AtomGen/<name>.md`` with the markers rebuild.py fills."""
    return (
        f"# {title}\n"
        "<!--benchmark_description-->\n\n\n"
        "<h2>Model benchmarks</h2>\n\n"
        '<table style="width:100%" id="j_table">\n'
        " <thead>\n"
        "  <tr>\n"
        "<th>Model name</th><th>Dataset</th>\n"
        "   <!-- <th>Method</th>-->\n"
        "    <th>RMSE</th>\n"
        "    <th>Team name</th>\n"
        "    <th>Dataset size</th>\n"
        "    <th>Date submitted</th>\n"
        "    <th>Notes</th>\n"
        "  </tr>\n"
        " </thead>\n"
        "<!--table_content--><!--table_content-->\n"
        "</table>\n"
    )
