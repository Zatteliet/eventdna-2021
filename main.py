import json
from pathlib import Path

import joblib
import typer
from loguru import logger

from experiments.training.training import score_crossval, train_crossval
from experiments.corpus import get_examples, check_extract, DATA_EXTRACTED, DATA_ZIPPED

def main(
    out_dir: str,
    n_folds: int = None,
    max_iter: int = None,
    verbose: bool = False,
    test: bool = False,
):
    """Run experiments, write out results to `out_dir`."""
    # The CRF implementation has a sensible default `max_iter`

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
    if not out_dir.exists():
        out_dir.mkdir()

    scores_dir = out_dir / "scores"
    if not scores_dir.exists():
        scores_dir.mkdir()

    model_dir = out_dir / "models"
    if not model_dir.exists():
        model_dir.mkdir()


    # Prepare the X and y examples.
    check_extract(DATA_ZIPPED, DATA_EXTRACTED)
    examples = list(get_examples(DATA_EXTRACTED))

    # Perform cross-validation training and score the results.
    cv = train_crossval(examples, cfg)
    fold_scores = []
    for id, model, score in cv:
        fold_scores.append(score)
        with open(scores_dir / f"scores_{id}.json", "w") as f:
            json.dump(score, f)
        joblib.dump(model, model_dir / f"model_{id}.pkl")

    logger.info("Computing overall scores...")
    overall_score = score_crossval(fold_scores)
    with open(scores_dir / f"overall_scores.json", "w") as f:
        json.dump(overall_score, f)

    logger.success(f"Done training, wrote models and scores to {out_dir}")


if __name__ == "__main__":
    typer.run(main)
