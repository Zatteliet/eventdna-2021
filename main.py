import json
from pathlib import Path

import joblib
import typer
from loguru import logger

from experiments.corpus import (
    DATA_EXTRACTED,
    DATA_ZIPPED,
    check_extract,
    get_examples,
)
from experiments.training import score_crossval, train_crossval


def main(
    out_dir: str,
    n_folds: int = None,
    max_iter: int = None,
    verbose: bool = False,
    test: bool = False,
):
    """Run experiments, write out results to `out_dir`."""

    cfg = {
        "n_folds": n_folds if n_folds else 10,
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
    scores_dir = out_dir / "scores"
    model_dir = out_dir / "models"
    for p in [out_dir, scores_dir, model_dir]:
        setup(p)

    # Prepare the X and y examples.
    check_extract(DATA_ZIPPED, DATA_EXTRACTED)
    examples = list(get_examples(DATA_EXTRACTED))
    logger.info(f"Training with {len(examples)} training examples.")

    # Perform cross-validation training and score the results.
    cv = train_crossval(examples, cfg)
    fold_scores = []
    for id, model, pretty_report, dict_report in cv:
        fold_scores.append(dict_report)
        with open(scores_dir / f"scores_{id}.txt", "w") as f:
            f.write(pretty_report)
        joblib.dump(model, model_dir / f"model_{id}.pkl")

    logger.info("Computing overall scores...")
    overall_score = score_crossval(fold_scores)
    with open(scores_dir / f"overall_scores.json", "w") as f:
        json.dump(overall_score, f, sort_keys=True, indent=4)

    logger.success(f"Done training, wrote models and scores to {out_dir}")


def setup(dir: Path):
    """Create the given directory if it does not exist.
    If it does, clean out its contents.
    """

    def clean(dir: Path):
        for item in dir.iterdir():
            if item.is_dir():
                clean(item)
            else:
                item.unlink()

    if not dir.exists():
        dir.mkdir()
    else:
        logger.warning(f"Found existing data in {dir}. Erasing...")
        clean(dir)


if __name__ == "__main__":
    typer.run(main)
