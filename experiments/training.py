from collections import defaultdict
from typing import Iterable, List

import sklearn_crfsuite
from experiments.corpus import Example
from loguru import logger
from sklearn.model_selection import KFold
from experiments.scoring import score
from experiments.util import merge_list, map_over_leaves
from statistics import mean


def make_folds(
    examples: Iterable[Example], n
) -> Iterable[tuple[list[Example], list[Example]]]:
    """Separate `examples` in `n` sets of `(train, dev)` splits.
    Docs: https://scikit-learn.org/stable/modules/cross_validation.html#k-fold
    """
    kf = KFold(n_splits=n, shuffle=True)

    # `folds` is a list of tuples, where a tuple = 2 numpy arrays of indices representing train-test sets.
    for train, test in kf.split(examples):
        # Convert np arrays to lists for ease of use.
        train = [examples[i] for i in train.tolist()]
        test = [examples[i] for i in test.tolist()]
        yield train, test


def train_crossval(examples: List[Example], config):
    """Run crossvalidation training. Yield trained CRF models and their scores."""

    # Split the data into folds.
    folds = make_folds(examples, config["n_folds"])

    # Train and score each fold.
    for fold_id, (tr, dv) in enumerate(folds):
        logger.info(f"Training fold {fold_id}")
        crf = train(tr, dv, config["max_iter"], config["verbose"])
        scores_pretty, scores_dict = score("iob", crf, dv)
        yield fold_id, crf, scores_pretty, scores_dict


def score_crossval(fold_score_dicts: Iterable[dict]):
    """Flatten and process the scores from crossval training."""

    merged = merge_list(fold_score_dicts)
    averaged = map_over_leaves(merged, mean)
    return averaged

    # def flatten(score_dict):
    #     new = {}
    #     for label_name, label_scores in score_dict.items():
    #         for score_name, value in label_scores.items():
    #             new_key = label_name + "_" + score_name
    #             new[new_key] = value
    #     return new

    # avg_scores = defaultdict(list)
    # for sc in score_dicts:
    #     sc = flatten(sc)
    #     for k, v in sc.items():
    #         avg_scores[k].append(v)

    # return avg_scores


def train(
    train_set: List[Example],
    dev_set: List[Example],
    max_iter: int = None,
    verbose: bool = False,
) -> dict:
    """Return a trained crf."""

    # Initialize CRf.
    # Docs & options = https://sklearn-crfsuite.readthedocs.io/en/latest/api.html#module-sklearn_crfsuite
    crf = sklearn_crfsuite.CRF(verbose=verbose, max_iterations=max_iter)

    # Fit a model.
    X_train, y_train = [ex.x for ex in train_set], [ex.y for ex in train_set]
    X_dev, y_dev = [ex.x for ex in dev_set], [ex.y for ex in dev_set]
    crf.fit(X=X_train, y=y_train, X_dev=X_dev, y_dev=y_dev)

    return crf
