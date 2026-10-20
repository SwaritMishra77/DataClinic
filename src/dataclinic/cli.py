"""Command-line interface for experiment and report generation."""

import argparse
from pathlib import Path

from .audit import audit_outputs
from .experiment import Config, run
from .report import write_report


def main():
    parser = argparse.ArgumentParser(description="DataClinic reproducible benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    reproduce = sub.add_parser("reproduce", help="Run the benchmark and build report/demo")
    reproduce.add_argument("--config", default="configs/reproduce.json")
    reproduce.add_argument("--output", default="results/main")
    reproduce.add_argument("--docs", default="docs")
    audit = sub.add_parser("audit", help="Check saved output integrity and data provenance")
    audit.add_argument("--output", default="results/main")
    audit.add_argument("--docs", default="docs")
    args = parser.parse_args()

    if args.command == "reproduce":
        config = Config.from_file(args.config)
        ranking, reviews = run(config, args.output)
        write_report(ranking, reviews, Path(args.output), Path(args.docs))
        print(f"Results: {Path(args.output).resolve()}")
        print(f"Demo: {(Path(args.docs) / 'index.html').resolve()}")
    elif args.command == "audit":
        results = audit_outputs(args.output, args.docs)
        print("Audit passed:", results)
