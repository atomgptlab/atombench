#!/usr/bin/env python3
"""
Convert mattergen-generate output (extxyz) + test CSV targets into the
standard atombench benchmark CSV: id,target,prediction (POSCAR strings).

Generated structures are paired 1-to-1 with test structures in order.
If more structures were generated than needed, the extras are ignored.
"""

import argparse
from pathlib import Path

import pandas as pd
from ase.io import read as ase_read
from jarvis.core.atoms import Atoms as JAtoms
from jarvis.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


def ase_to_poscar_string(ase_atoms) -> str:
    pmg = AseAtomsAdaptor.get_structure(ase_atoms)
    jarvis_atoms = JAtoms.from_dict(
        {
            "lattice_mat": pmg.lattice.matrix.tolist(),
            "elements": [str(s) for s in pmg.species],
            "coords": pmg.frac_coords.tolist(),
            "cartesian": False,
        }
    )
    return Poscar(jarvis_atoms).to_string().replace("\n", r"\n")


def cif_to_poscar_string(cif_str: str) -> str:
    pmg = Structure.from_str(cif_str, fmt="cif")
    jarvis_atoms = JAtoms.from_dict(
        {
            "lattice_mat": pmg.lattice.matrix.tolist(),
            "elements": [str(s) for s in pmg.species],
            "coords": pmg.frac_coords.tolist(),
            "cartesian": False,
        }
    )
    return Poscar(jarvis_atoms).to_string().replace("\n", r"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True,
                        help="Directory containing generated_crystals.extxyz")
    parser.add_argument("--test_csv", required=True,
                        help="Test split CSV with material_id and cif columns")
    parser.add_argument("--output_csv", required=True,
                        help="Output benchmark CSV path")
    args = parser.parse_args()

    results_path = Path(args.results_path)
    extxyz_path = results_path / "generated_crystals.extxyz"
    if not extxyz_path.exists():
        raise FileNotFoundError(f"No generated_crystals.extxyz found at {extxyz_path}")

    test_df = pd.read_csv(args.test_csv)
    n_test = len(test_df)

    print(f"Reading generated structures from {extxyz_path}")
    generated = ase_read(str(extxyz_path), index=":", format="extxyz")
    n_gen = len(generated)
    print(f"Generated {n_gen} structures, test set size {n_test}")

    if n_gen < n_test:
        print(f"WARNING: only {n_gen} generated structures for {n_test} test samples; "
              "some predictions will be missing")

    with open(args.output_csv, "w") as f:
        f.write("id,target,prediction\n")
        for i, row in tqdm(test_df.iterrows(), total=n_test, desc="Writing CSV"):
            jid = row["material_id"]
            target_poscar = cif_to_poscar_string(row["cif"])
            if i < n_gen:
                pred_poscar = ase_to_poscar_string(generated[i])
            else:
                pred_poscar = target_poscar  # fallback: repeat target
            f.write(f"{jid},{target_poscar},{pred_poscar}\n")

    print(f"[✓] Benchmark CSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
