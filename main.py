import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import joblib

from experiments import corpus, training
from experiments.evaluation import alpino

logger = logging.getLogger(__name__)


config = {
    "n_folds": 10,
    "max_iter": 500,
    "main_events_only": False,
}


def train(args):

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.test:
        out_dir = Path("output") / f"output-{timestamp}-test"
    else:
        out_dir = Path("output") / f"output-{timestamp}"
    out_dir.mkdir(parents=True)

    logging.basicConfig(filename=out_dir / "train.log", level=logging.DEBUG)

    if args.test:
        args.n_folds = 2
        args.max_iter = 10
        logger.warning(f"Using test config: {args}")
    else:
        logger.info(f"Starting training with config: {args}")

    # Prepare the X and y examples.
    examples = corpus.get_examples(main_events_only=args.main_events_only)
    logger.info(f"Training with {len(examples)} training examples.")

    # Initialize training folds.
    folds = list(training.make_folds(examples, args.n_folds))

    # Perform cross-validation training.
    training.train_crossval(folds, max_iter=args.max_iter)

    def save_examples(examples, dir):
        for ex in examples:
            ex_dir = dir / ex.id
            ex_dir.mkdir(parents=True)
            with open(ex_dir / "x.json", "w") as f:
                json.dump(ex.x, f)
            with open(ex_dir / "y.json", "w") as f:
                json.dump(ex.y, f)
            shutil.copy(ex.alpino_tree.file, ex_dir / "alpino.xml")

    # Dump the fold info.
    for fold in folds:

        fold_dir = out_dir / "folds" / f"fold_{fold.id}"
        fold_dir.mkdir(parents=True)
        # joblib.dump(fold, fold_dir / "fold.pkl") # TEST
        joblib.dump(fold.crf, fold_dir / "crf.pkl")

        # Save each example in the fold.
        save_examples(fold.dev, fold_dir / "data" / "dev")

    logger.info(f"Finished training -> {out_dir}")


def eval(args):

    eval_dir = args.dir / "eval"

    if eval_dir.exists():
        raise ValueError(
            "Eval dir already exists, hinting that evaluation has already been run on this dir. Remove the existing eval dir first."
        )
    else:
        eval_dir.mkdir(parents=True)

    logging.basicConfig(filename=eval_dir / "eval.log", level=logging.DEBUG)

    def load_examples(dir):
        examples = []
        for ex_dir in dir.iterdir():
            with open(ex_dir / "x.json", "r") as f:
                x = json.load(f)
            with open(ex_dir / "y.json", "r") as f:
                y = json.load(f)
            alpino_tree = alpino.AlpinoTree(
                ex_dir / "alpino.xml", restricted_mode=True
            )
            example = corpus.Example(
                id=ex_dir.stem, x=x, y=y, alpino_tree=alpino_tree
            )
            examples.append(example)
        return examples

    # Load fold data.
    fold_dirs = ((args.dir) / "folds").iterdir()

    folds = []
    for fold_dir in fold_dirs:
        id = fold_dir.stem.split("_")[-1]
        crf = joblib.load(fold_dir / "crf.pkl")
        dev = load_examples(fold_dir / "data" / "dev")

        # For evaluation, no need to search for training data.
        fold = training.Fold(id=id, train=[], dev=dev)
        fold.crf = crf
        folds.append(fold)

    # Setup directories.
    micro_iob_scores_dir = eval_dir / "scores_iob_micro"
    micro_event_scores_dir = eval_dir / "scores_event_spans_micro"

    # Write out scores per fold and averaged.
    for fold in folds:
        training.score_fold(fold)

        write(
            fold.micro_iob_scores,
            micro_iob_scores_dir / f"scores_{fold.id}.json",
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
        training.average_scores([fold.micro_event_scores for fold in folds]),
        micro_event_scores_dir / "averaged.json",
    )

    logger.info(f"Finished evaluation -> {eval_dir}")


def write(json_dict, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(json_dict, f, sort_keys=True, indent=4)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser("train")
parser_train.add_argument(
    "--n-folds",
    help="Number of folds to use in crossval training.",
    type=int,
    default=10,
)
parser_train.add_argument(
    "--max-iter",
    help="Number of iterations to run training in each fold.",
    type=int,
    default=500,
)
parser_train.add_argument(
    "--main-events-only",
    help="If True, only use main events for training and evaluation.",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser_train.add_argument(
    "--test",
    help="Use a test configuration that trains quickly.",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser_train.set_defaults(func=train)

parser_eval = subparsers.add_parser("eval")
parser_eval.add_argument(
    "dir",
    help="Directory containing folds saved after training.",
    type=Path,
)
parser_eval.set_defaults(func=eval)

args = parser.parse_args()


args.func(args)
