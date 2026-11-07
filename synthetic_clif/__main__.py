"""CLI entry point for synthetic CLIF dataset generation."""

import argparse
from pathlib import Path
import sys

from synthetic_clif.generators.dataset import SyntheticCLIFDataset
from synthetic_clif.config.mcide import MCIDELoader


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic CLIF (Common Longitudinal ICU data Format) datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate small test dataset (10 patients, 12 hospitalizations)
  python -m synthetic_clif --patients 10 --hospitalizations 12 --output data/test/

  # Generate larger dataset with specific seed
  python -m synthetic_clif --hospitalizations 1000 --output data/full/ --seed 12345

  # Generate only beta tables (exclude concept tables)
  python -m synthetic_clif --hospitalizations 100 --output data/ --no-concept-tables

  # Use custom mCIDE directory
  python -m synthetic_clif --mcide-dir /path/to/mcide --output data/

  # Output as CSV instead of parquet
  python -m synthetic_clif --output data/ --format csv

  # Generate CLIF 2.1.0 tables instead of 3.0.0
  python -m synthetic_clif --hospitalizations 100 --output data/v21/ --clif-version 2.1

  # Generate with 4 parallel workers (speeds up large datasets)
  python -m synthetic_clif --hospitalizations 10000 --output data/ -w 4
        """,
    )

    parser.add_argument(
        "--patients",
        "-p",
        type=int,
        default=None,
        help="Number of patients to generate (default: derived from hospitalizations)",
    )
    parser.add_argument(
        "--hospitalizations",
        "-n",
        type=int,
        default=12,
        help="Number of hospitalizations to generate (default: 12)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--mcide-dir",
        type=str,
        default=None,
        help="Path to custom mCIDE CSV directory",
    )
    parser.add_argument(
        "--fetch-mcide",
        action="store_true",
        help="Fetch latest mCIDE from GitHub before generating",
    )
    parser.add_argument(
        "--no-concept-tables",
        action="store_true",
        help="Skip generating concept tables (draft status)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers for generation (default: 1, sequential)",
    )
    parser.add_argument(
        "--clif-version",
        default="3.0",
        help="CLIF version of the output: 3.0 (default) or 2.1",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Derive number of patients if not specified
    n_patients = args.patients
    if n_patients is None:
        # Roughly 80% of hospitalizations have unique patients
        n_patients = max(1, int(args.hospitalizations * 0.8))

    # Handle mCIDE directory
    mcide_dir = Path(args.mcide_dir) if args.mcide_dir else None

    # Fetch mCIDE from GitHub if requested
    if args.fetch_mcide:
        print("Fetching mCIDE from GitHub...")
        loader = MCIDELoader()
        mcide_dir = loader.fetch_from_github()
        print(f"Downloaded mCIDE to {mcide_dir}")

    # Create dataset generator
    dataset = SyntheticCLIFDataset(
        n_patients=n_patients,
        n_hospitalizations=args.hospitalizations,
        seed=args.seed,
        mcide_dir=mcide_dir,
        include_concept_tables=not args.no_concept_tables,
        workers=args.workers,
        clif_version=args.clif_version,
    )

    # Generate dataset
    tables = dataset.generate()

    # Write output
    output_dir = Path(args.output)
    if args.format == "parquet":
        dataset.to_parquet(output_dir)
    else:
        dataset.to_csv(output_dir)

    # Print summary
    if args.verbose:
        print("\nSummary:")
        print(dataset.summary().to_string(index=False))

    print(f"\nGenerated {len(tables)} tables to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
