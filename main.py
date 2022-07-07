import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import typer

from experiments import corpus, training

logger = logging.getLogger(__name__)


def main(
    out_dir: str,
    n_folds: int = 10,
    max_iter: int = 500,
    verbose: bool = False,
    main_events_only: bool = False,
    test: bool = False,
):
    """Run experiments, write out results to a time-stamped dir under `out_dir`."""

    cfg = {
        "n_folds": n_folds,
        "max_iter": max_iter,
        "verbose": verbose,
        "main_events_only": main_events_only,
        "test": test,
    }

    if test:
        cfg["n_folds"] = 2
        cfg["max_iter"] = 10
        logger.warning(f"Using test config: {cfg}")
    else:
        logger.info(f"Starting training with config: {cfg}")

    # Setup directories.

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir) / f"output-{timestamp}"
    out_dir.mkdir()

    micro_iob_scores_dir = out_dir / "scores_iob_micro"
    macro_iob_scores_dir = out_dir / "scores_iob_macro"
    micro_event_scores_dir = out_dir / "scores_event_spans_micro"
    macro_event_scores_dir = out_dir / "scores_event_spans_macro"
    model_dir = out_dir / "models"
    for p in [
        out_dir,
        micro_iob_scores_dir,
        macro_iob_scores_dir,
        micro_event_scores_dir,
        macro_event_scores_dir,
        model_dir,
    ]:
        p.mkdir()

    write(cfg, out_dir / "config.json")

    # Prepare the X and y examples.
    examples = corpus.get_examples(main_events_only=cfg["main_events_only"])
    logger.info(f"Training with {len(examples)} training examples.")

    # # Info.
    # _n_no_events = len([ex for ex in examples if set(ex.y) == {"O"}])
    # logger.info(
    #     f"{_n_no_events} examples have no events. {len(examples) -_n_no_events} do."
    # )

    # Initialize training folds.
    folds = list(training.make_folds(examples, cfg["n_folds"]))

    # Perform cross-validation training.
    training.train_crossval(
        folds, max_iter=cfg["max_iter"], verbose=cfg["verbose"]
    )

    # Dump the models.
    for fold in folds:
        joblib.dump(fold.crf, model_dir / f"model_{fold.id}.pkl")

    # Write out scores per fold and averaged.

    for fold in folds:
        write(
            fold.micro_iob_scores,
            micro_iob_scores_dir / f"scores_{fold.id}.json",
        )
        write(
            fold.macro_iob_scores,
            macro_iob_scores_dir / f"scores_{fold.id}.json",
        )
        write(
            fold.macro_event_scores,
            macro_event_scores_dir / f"scores_{fold.id}.json",
        )
        write(
            fold.micro_event_scores,
            micro_event_scores_dir / f"scores_{fold.id}.json",
        )

    write(
        training.average_scores([fold.micro_iob_scores for fold in folds]),
        micro_iob_scores_dir / "averaged.json",
    )

    write(
        training.average_scores([fold.macro_iob_scores for fold in folds]),
        macro_iob_scores_dir / "averaged.json",
    )

    write(
        training.average_scores([fold.micro_event_scores for fold in folds]),
        micro_event_scores_dir / "averaged.json",
    )
    write(
        training.average_scores([fold.macro_event_scores for fold in folds]),
        macro_event_scores_dir / "averaged.json",
    )

    logger.info(f"Done training, wrote models and scores to {out_dir}")


def write(json_dict, path):
    with open(path, "w") as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    typer.run(main)
