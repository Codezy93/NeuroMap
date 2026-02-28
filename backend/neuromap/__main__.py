from __future__ import annotations

import argparse

from . import api, predict, train


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neuromap",
        description="NeuroMap: 3D volumetric segmentation with uncertainty quantification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a 3D U-Net with MONAI.")
    train.add_args(train_parser)

    predict_parser = subparsers.add_parser("predict", help="Run MC-dropout inference for one case.")
    predict.add_args(predict_parser)

    serve_parser = subparsers.add_parser("serve", help="Start FastAPI for slice-serving.")
    api.add_args(serve_parser)

    args = parser.parse_args()
    if args.command == "train":
        train.run(args)
    elif args.command == "predict":
        predict.run(args)
    elif args.command == "serve":
        api.run(args)


if __name__ == "__main__":
    main()
