from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List

from loguru import logger
from sklearn.model_selection import KFold
from sklearn_crfsuite import CRF

from experiments.corpus import Example
from experiments.scoring import ScoreReport
from experiments.util import map_over_leaves, merge_list


@dataclass
class Fold:
    id: int
    train: Iterable[Example]
    dev: Iterable[Example]
    scores: ScoreReport = None
    crf: CRF = None


def make_folds(
    examples: Iterable[Example], n
) -> Iterable[tuple[list[Example], list[Example]]]:
    """Separate `examples` in `n` sets of `(train, dev)` splits.
    Docs: https://scikit-learn.org/stable/modules/cross_validation.html#k-fold
    """
    kf = KFold(n_splits=n, shuffle=True)

    # `folds` is a list of tuples, where a tuple = 2 numpy arrays of indices representing train-test sets.
    for i, (train, test) in enumerate(kf.split(examples)):
        # Convert np arrays to lists for ease of use.
        train = [examples[i] for i in train.tolist()]
        test = [examples[i] for i in test.tolist()]
        yield Fold(i, train, test)


def train_crossval(folds: Iterable[Fold], max_iter, verbose) -> None:
    """Run crossvalidation training. Yield trained CRF models and their scores."""
    for fold in folds:
        logger.info(f"Training fold {fold.id}")
        fold.crf = train(fold.train, fold.dev, max_iter, verbose)
        fold.scores = ScoreReport(fold.dev, fold.crf)


def average_scores(fold_score_dicts: Iterable[dict]):
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
    crf = CRF(verbose=verbose, max_iterations=max_iter)

    # Fit a model.
    X_train, y_train = [ex.x for ex in train_set], [ex.y for ex in train_set]
    X_dev, y_dev = [ex.x for ex in dev_set], [ex.y for ex in dev_set]
    crf.fit(X=X_train, y=y_train, X_dev=X_dev, y_dev=y_dev)

    return crf
