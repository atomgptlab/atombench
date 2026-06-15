"""Offline tests for atombench._leaderboard (no network, no GitHub)."""
from __future__ import annotations

import json
import zipfile

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Poscar as PmgPoscar

from atombench import _leaderboard as L


# ── fixtures / helpers ───────────────────────────────────────────────────────
def _struct(a: float = 3.0) -> Structure:
    return Structure(Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _poscar_escaped(s: Structure) -> str:
    return str(PmgPoscar(s)).replace("\r\n", "\n").replace("\n", "\\n")


def _df(ids) -> pd.DataFrame:
    cell = _poscar_escaped(_struct())
    return pd.DataFrame({"id": list(ids), "target": [cell] * len(ids), "prediction": [cell] * len(ids)})


# ── naming ───────────────────────────────────────────────────────────────────
def test_benchmark_name_ok():
    assert L.benchmark_name("Tc_supercon", "dft_3d") == "AI-AtomGen-Tc_supercon-dft_3d-test-rmse"


def test_benchmark_name_rejects_dash():
    with pytest.raises(L.SubmissionError):
        L.benchmark_name("Tc-bad", "dft_3d")
    with pytest.raises(L.SubmissionError):
        L.benchmark_name("Tc", "dft-3d")


def test_default_contribution_name():
    assert L.default_contribution_name("My Cool Model!!") == "my_cool_model"
    assert L.default_contribution_name("   ") == "atombench_contribution"


# ── normalisation ────────────────────────────────────────────────────────────
def test_normalize_is_single_line_escaped():
    norm = L.normalize_to_poscar_escaped(_poscar_escaped(_struct()))
    assert "\\n" in norm and "\n" not in norm


def test_normalize_from_cif_roundtrips_via_jarvis():
    """A CIF prediction must normalize to something the AtomGen scorer can parse."""
    norm = L.normalize_to_poscar_escaped(str(CifWriter(_struct())))
    assert "\n" not in norm and "\\n" in norm
    from jarvis.io.vasp.inputs import Poscar  # the scorer's parser

    atoms = Poscar.from_string(norm.replace("\\n", "\n")).atoms
    assert atoms.num_atoms == 2


# ── benchmark ground truth ───────────────────────────────────────────────────
def test_build_benchmark_zip_roundtrip(tmp_path):
    df = _df(["A-1", "A-2", "A-3"])
    out = L.build_benchmark_zip(tmp_path / "alex_Tc.json.zip", "alex", "Tc", df)
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ["alex_Tc.json"]
        data = json.loads(z.read("alex_Tc.json"))
    assert "train" in data  # rebuild.py does len(json_data["train"])
    assert set(data["test"]) == {"A-1", "A-2", "A-3"}
    from jarvis.io.vasp.inputs import Poscar

    for cell in data["test"].values():
        Poscar.from_string(cell.replace("\\n", "\n"))  # must not raise


def test_build_benchmark_zip_requires_target():
    df = pd.DataFrame({"id": ["a"], "prediction": ["x"]})
    with pytest.raises(L.SubmissionError):
        L.build_benchmark_zip("/tmp/x.json.zip", "d", "p", df)


# ── contribution dir ─────────────────────────────────────────────────────────
def test_build_contribution_dir(tmp_path):
    df = _df(["X-1", "X-2"])
    meta = {"model_name": "M", "author_email": "a@b.com", "project_url": "u", "git_url": "g"}
    contrib = L.build_contribution_dir(tmp_path, "mymodel", [(df, "dft_3d", "Tc_supercon")], meta)

    bn = "AI-AtomGen-Tc_supercon-dft_3d-test-rmse"
    zp = contrib / f"{bn}.csv.zip"
    assert zp.exists()
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == [f"{bn}.csv"]  # inner file named after the benchmark
        text = z.read(f"{bn}.csv").decode()
    lines = text.strip().splitlines()
    assert lines[0] == "id,target,prediction"
    assert len(lines) == 3  # header + 2 rows; predictions stayed single-line

    meta_out = json.loads((contrib / "metadata.json").read_text())
    for field in L.REQUIRED_METADATA_FIELDS:
        assert meta_out.get(field), f"missing {field}"
    assert meta_out["team_name"] == "M"  # defaulted from model_name
    assert meta_out["time_taken_seconds"] == {f"{bn}.csv.zip": ""}
    assert (contrib / "run.sh").exists()


# ── metadata ─────────────────────────────────────────────────────────────────
def test_complete_metadata_defaults():
    m = L.complete_metadata(
        {"model_name": "M", "author_email": "a@b.com", "project_url": "u", "git_url": "g"},
        csv_zip_names=["z.csv.zip"],
    )
    assert m["team_name"] == "M"
    assert isinstance(m["git_url"], list)
    assert m["time_taken_seconds"] == {"z.csv.zip": ""}
    for field in L.REQUIRED_METADATA_FIELDS:
        assert m.get(field)


def test_complete_metadata_rejects_bad_email():
    with pytest.raises(L.SubmissionError):
        L.complete_metadata(
            {"model_name": "M", "author_email": "bad", "project_url": "u", "git_url": "g"},
            csv_zip_names=["z"],
        )


def test_complete_metadata_requires_model_name():
    with pytest.raises(L.SubmissionError):
        L.complete_metadata({"author_email": "a@b.com", "project_url": "u", "git_url": "g"}, csv_zip_names=["z"])


# ── validation ───────────────────────────────────────────────────────────────
def test_validate_happy_path():
    df = _df(["a", "b"])
    report = L.validate_submission(df, "d", "p", benchmark_ids={"a", "b"}, check_structures=True)
    assert report.ok, report.errors


def test_validate_missing_prediction_column():
    df = pd.DataFrame({"id": ["a"], "target": ["x"]})
    report = L.validate_submission(df, "d", "p", check_structures=False)
    assert not report.ok
    assert any("prediction" in e for e in report.errors)


def test_validate_missing_target_is_warning_only():
    df = pd.DataFrame({"id": ["a"], "prediction": [_poscar_escaped(_struct())]})
    report = L.validate_submission(df, "d", "p", benchmark_ids={"a"}, check_structures=True)
    assert report.ok
    assert any("target" in w for w in report.warnings)


def test_validate_catches_duplicate_and_id_mismatch():
    df = _df(["a", "a", "b"])
    report = L.validate_submission(df, "d", "p", benchmark_ids={"a", "b", "c"}, check_structures=False)
    assert not report.ok
    assert any("duplicate" in e for e in report.errors)
    assert any("missing" in e for e in report.errors)  # 'c' not in CSV


def test_validate_catches_unparseable_prediction():
    df = pd.DataFrame({"id": ["a"], "target": ["x"], "prediction": ["definitely not a crystal"]})
    report = L.validate_submission(df, "d", "p", check_structures=True)
    assert not report.ok
    assert any("parse" in e for e in report.errors)


# ── descriptions / docs ──────────────────────────────────────────────────────
def test_descriptions_row():
    assert L.descriptions_row("alex", "Tc", "hi") == ["AI", "AtomGen", "alex_Tc", "hi", ""]


def test_descriptions_row_requires_text():
    with pytest.raises(L.SubmissionError):
        L.descriptions_row("alex", "Tc", "")


def test_append_description_row_dedupes(tmp_path):
    csv = tmp_path / "descriptions.csv"
    csv.write_text("Category,Sub-category,Benchmark,Description,Experimental DOI (If Applicable)\n")
    assert L.append_description_row(csv, "alex", "Tc", "first") is True
    assert L.append_description_row(csv, "alex", "Tc", "again") is False  # already present
    assert csv.read_text().count("alex_Tc") == 1


def test_skeleton_md_has_markers():
    md = L.skeleton_md("Title")
    assert md.startswith("# Title")
    assert "<!--benchmark_description-->" in md
    assert md.count("<!--table_content-->") == 2
