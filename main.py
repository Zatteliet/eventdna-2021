import json
from pathlib import Path
from typing import Iterable

import joblib
import typer
from loguru import logger

from experiments.corpus import (
    DATA_EXTRACTED,
    DATA_ZIPPED,
    check_extract,
    get_examples,
)
from experiments.training import (
    Fold,
    average_scores,
    make_folds,
    train_crossval,
)


def main(
    out_dir: str,
    n_folds: int = 10,
    max_iter: int = 500,
    verbose: bool = False,
    test: bool = False,
):
    """Run experiments, write out results to `out_dir`."""

    cfg = {
        "n_folds": n_folds,
        "max_iter": max_iter,
        "verbose": verbose,
        "test": test,
    }

    if test:
        cfg["n_folds"] = 2
        cfg["max_iter"] = 10
        logger.warning(f"Using test config: {cfg}")
    else:
        logger.info(f"Starting training with config: {cfg}")

    # Setup directories.

    out_dir = Path(out_dir)
    iob_scores_dir = out_dir / "scores_iob"
    event_scores_dir = out_dir / "scores_event_spans"
    model_dir = out_dir / "models"
    for p in [out_dir, iob_scores_dir, event_scores_dir, model_dir]:
        setup(p)

    write(cfg, out_dir / "config.json")

    # Prepare the X and y examples.
    check_extract(DATA_ZIPPED, DATA_EXTRACTED)
    examples = list(get_examples(DATA_EXTRACTED))
    logger.info(f"Training with {len(examples)} training examples.")

    _n_no_events = len([ex for ex in examples if set(ex.y) == {"O"}])
    logger.info(
        f"{_n_no_events} examples have no events. {len(examples) -_n_no_events} do."
    )

    # Initialize training folds.
    folds: Iterable[Fold] = list(make_folds(examples, cfg["n_folds"]))

    # Perform cross-validation training.
    train_crossval(folds, max_iter=cfg["max_iter"], verbose=cfg["verbose"])

    # Dump the models.
    for fold in folds:
        joblib.dump(fold.crf, model_dir / f"model_{fold.id}.pkl")

    # Write out scores per fold and averaged.

    for fold in folds:
        write(fold.iob_scores, iob_scores_dir / f"scores_{fold.id}.json")
        write(fold.event_scores, event_scores_dir / f"scores_{fold.id}.json")

    write(
        average_scores([fold.iob_scores for fold in folds]),
        iob_scores_dir / "averaged.json",
    )

    write(
        average_scores([fold.event_scores for fold in folds]),
        event_scores_dir / "averaged.json",
    )

    logger.success(f"Done training, wrote models and scores to {out_dir}")


def setup(dir: Path):
    """Create the given directory if it does not exist.
    If it does, clean out its contents.
    """

    def clean(dir: Path):
        for item in dir.iterdir():
            if item.is_dir():
                clean(item)
                item.rmdir()
            else:
                item.unlink()

    if not dir.exists():
        dir.mkdir()
    else:
        logger.warning(f"Found existing data in {dir}. Erasing...")
        clean(dir)


def write(json_dict, path):
    with open(path, "w") as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    typer.run(main)
