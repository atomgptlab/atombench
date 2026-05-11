#!/usr/bin/env python3
"""
Per-crystal Tc-conditioned inference for a finetuned mattergen model.

Loads the model once, then for each crystal in the test CSV generates one
structure conditioned on that crystal's individual Tc value (classifier-free
guidance).  Output is written as generated_crystals.extxyz to --output_path
in the same row order as the test CSV, making it compatible with
write_benchmark.py.
"""

import argparse
import tempfile
from pathlib import Path

import ase.io
import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to finetuned model outputs directory")
    parser.add_argument("--test_csv", required=True,
                        help="Test CSV with material_id, cif, and <tc_column> columns")
    parser.add_argument("--tc_column", required=True,
                        help="Column name for Tc values (e.g. Tc or Tc_supercon)")
    parser.add_argument("--output_path", required=True,
                        help="Output directory; generated_crystals.extxyz is written here")
    parser.add_argument("--guidance_factor", type=float, default=2.0,
                        help="Diffusion guidance scale for CFG (default: 2.0)")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_info = MatterGenCheckpointInfo(
        model_path=Path(args.model_path).resolve(),
        load_epoch="best",
    )

    gen = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=1,
        num_batches=1,
        properties_to_condition_on={},
        diffusion_guidance_factor=args.guidance_factor,
        record_trajectories=False,
    )
    # Trigger model load once; subsequent generate() calls reuse the loaded weights.
    gen.prepare()

    all_structures = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                           desc="Generating (per-crystal Tc)"):
            tc_value = float(row[args.tc_column])
            gen.properties_to_condition_on = {"tc_supercon": tc_value}
            structures = gen.generate(
                batch_size=1,
                num_batches=1,
                output_dir=tmp_dir,
            )
            all_structures.extend(structures)

    ase_atoms = [AseAtomsAdaptor.get_atoms(s) for s in all_structures]
    out_file = out_dir / "generated_crystals.extxyz"
    ase.io.write(str(out_file), ase_atoms, format="extxyz")
    print(f"[✓] Written {len(ase_atoms)} structures to {out_file}")


if __name__ == "__main__":
    main()
