"""Entrada histórica da rodada AP; o fluxo compartilhado está em classification_search."""

import argparse
import logging
from pathlib import Path

from fraud_detection.classification_search import run, search_threshold, select_parameters
from fraud_detection.data import ROOT, load_config

__all__ = ["run", "search_threshold", "select_parameters"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(
        load_config(),
        load_config(ROOT / "configs/ablation.yaml"),
        load_config(ROOT / "configs/xgboost_ap_threshold.yaml"),
        args.output,
    )


if __name__ == "__main__":
    main()
