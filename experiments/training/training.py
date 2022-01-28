from collections import defaultdict
from typing import Iterable, List

import sklearn.metrics as skl_metrics
import sklearn_crfsuite
from experiments.corpus import Example
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer


def make_folds(examples: Iterable[Example], n):
    """Separate `examples` in `n` sets of `(train, dev)` splits.
    Docs: https://scikit-learn.org/stable/modules/cross_validation.html#k-fold
    """
    kf = KFold(n_splits=n, shuffle=True)

    # `folds` is a list of tuples, where a tuple = 2 numpy arrays of indices representing train-test sets.
    for train, test in kf.split(examples):
        # Convert np arrays to lists for ease of use.
        yield train.tolist(), test.tolist()


def train_crossval(examples: List[Example], config):
    """Run crossvalidation training. Yield trained CRF models and their scores."""

    # Split the data into folds.
    folds: List[List[Example], List[Example]] = make_folds(
        examples, config["n_folds"]
    )

    # Train and score each fold.
    for fold_id, (fold_train_is, fold_dev_is) in enumerate(folds):
        fold_train = [examples[i] for i in fold_train_is]
        fold_dev = [examples[i] for i in fold_dev_is]

        logger.info(f"Training fold {fold_id}")
        crf = train(
            fold_train, fold_dev, config["max_iter"], config["verbose"]
        )

        # logger.info(f"Scoring fold {fold_id}")
        scores = score(crf, fold_dev)

        yield fold_id, crf, scores


def score_crossval(scores: Iterable[dict]):
    """Flatten and process the scores from crossval training."""

    def flatten(score_dict):
        new = {}
        for label_name, label_scores in score_dict.items():
            for score_name, value in label_scores.items():
                new_key = label_name + "_" + score_name
                new[new_key] = value
        return new

    avg_scores = defaultdict(list)
    for sc in scores:
        sc = flatten(sc)
        for k, v in sc.items():
            avg_scores[k].append(v)

    return avg_scores


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


def score(crf, dev_set) -> dict:
    """Score a `crf` model over `X_dev` against `y_dev`.

    We use classification reports as provided by `sklearn.metrics`.

    Return a scores dict.
    """
    # CC: I have not found a way to get score directly during model training.
    # So this solution calculates scores by using the model to predict after it's been trained.

    X_dev, y_dev = (ex.x for ex in dev_set), (ex.y for ex in dev_set)
    y_pred = crf.predict(X_dev)

    # Transform the data to SKLearn's preference.
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
    targets = ["I", "O", "B"]
    mlb = MultiLabelBinarizer(classes=targets)
    mlb_y_gold = mlb.fit_transform(y_dev)
    mlb_y_pred = mlb.fit_transform(y_pred)

    # Collect various scores and return them.
    return skl_metrics.classification_report(
        mlb_y_gold,
        mlb_y_pred,
        output_dict=True,
        target_names=targets,
    )
