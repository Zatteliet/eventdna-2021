import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import typer

from experiments import corpus, training
from experiments.evaluation import event_level
from sklearn.metrics import classification_report


logger = logging.getLogger(__name__)


def main(
    n_folds: int = 10,
    max_iter: int = 500,
    verbose: bool = False,
    main_events_only: bool = False,
    test: bool = False,
):
    """Run experiments, write out results to a time-stamped dir under `./output/`."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if test:
        out_dir = Path("output") / f"output-{timestamp}-test"
    else:
        out_dir = Path("output") / f"output-{timestamp}"
    out_dir.mkdir(parents=True)

    logging.basicConfig(filename=out_dir / "log.log", level=logging.DEBUG)

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

    micro_iob_scores_dir = out_dir / "scores_iob_micro"
    # macro_iob_scores_dir = out_dir / "scores_iob_macro"
    micro_event_scores_dir = out_dir / "scores_event_spans_micro"
    # macro_event_scores_dir = out_dir / "scores_event_spans_macro"
    folds_defs_dir = out_dir / "fold_checks"
    model_dir = out_dir / "models"
    for p in [
        micro_iob_scores_dir,
        # macro_iob_scores_dir,
        micro_event_scores_dir,
        # macro_event_scores_dir,
        folds_defs_dir,
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

    # Write out the example texts in each fold, to check consistency over training runs.
    for fold in folds:
        ids = [ex.id for ex in fold.train]
        with open(folds_defs_dir / f"fold_{fold.id}.txt", mode="w") as f:
            f.write("\n".join(ids))

    # Perform cross-validation training.
    training.train_crossval(
        folds, max_iter=cfg["max_iter"], verbose=cfg["verbose"]
    )

    # Dump the models.
    for fold in folds:
        joblib.dump(fold.crf, model_dir / f"model_{fold.id}.pkl")

    # # Run micro->micro evaluation for event-level.
    # logger.info("RUNNING EVENT LEVEL EVAL")

    # # 1. Concatenate found/not-found vectors over all examples.
    # gold_vector = []
    # pred_vector = []
    # for fold in folds:
    #     dev_xs = [ex.x for ex in fold.dev]
    #     predictions = fold.crf.predict(dev_xs)
    #     for example, prediction in zip(fold.dev, predictions):
    #         gold_events = list(
    #             event_level.get_events(example.y, example.alpino_tree)
    #         )
    #         pred_events = list(
    #             event_level.get_events(prediction, example.alpino_tree)
    #         )
    #         gv, pv = event_level.match_between(gold_events, pred_events)
    #         gold_vector.extend(gv)
    #         pred_vector.extend(pv)

    # report = classification_report(gold_vector, pred_vector, output_dict=False)
    # logger.info(report)

    # Write out scores per fold and averaged.

    for fold in folds:
        write(
            fold.micro_iob_scores,
            micro_iob_scores_dir / f"scores_{fold.id}.json",
        )
        # write(
        #     fold.macro_iob_scores,
        #     macro_iob_scores_dir / f"scores_{fold.id}.json",
        # )
        # write(
        #     fold.macro_event_scores,
        #     macro_event_scores_dir / f"scores_{fold.id}.json",
        # )
        write(
            fold.micro_event_scores,
            micro_event_scores_dir / f"scores_{fold.id}.json",
        )

    write(
        training.average_scores([fold.micro_iob_scores for fold in folds]),
        micro_iob_scores_dir / "averaged.json",
    )

    # write(
    #     training.average_scores([fold.macro_iob_scores for fold in folds]),
    #     macro_iob_scores_dir / "averaged.json",
    # )

    write(
        training.average_scores([fold.micro_event_scores for fold in folds]),
        micro_event_scores_dir / "averaged.json",
    )
    # write(
    #     training.average_scores([fold.macro_event_scores for fold in folds]),
    #     macro_event_scores_dir / "averaged.json",
    # )

    logger.info(f"Done training, wrote models and scores to {out_dir}")


def write(json_dict, path):
    with open(path, "w") as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    typer.run(main)
