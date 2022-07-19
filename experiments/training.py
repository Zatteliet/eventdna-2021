import logging
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List

from sklearn.model_selection import KFold
from sklearn_crfsuite import CRF

from experiments.corpus import Example
from experiments.evaluation import event_level, iob_level

# from experiments.evaluation.scoring import ScoreReport
from experiments.util import map_over_leaves, merge_list

logger = logging.getLogger(__name__)


@dataclass
class Fold:
    id: int
    train: Iterable[Example]
    dev: Iterable[Example]
    # micro_iob_scores: dict = None
    # event_scores: dict = None
    # crf: CRF = None


def make_folds(
    examples: Iterable[Example], n
) -> Iterable[tuple[list[Example], list[Example]]]:
    """Separate `examples` in `n` sets of `(train, dev)` splits.
    Docs: https://scikit-learn.org/stable/modules/cross_validation.html#k-fold
    """
    kf = KFold(n_splits=n, shuffle=True, random_state=42)

    # `folds` is a list of tuples, where a tuple = 2 numpy arrays of indices representing train-test sets.
    for i, (train, test) in enumerate(kf.split(examples)):
        # Convert np arrays to lists for ease of use.
        train = [examples[i] for i in train.tolist()]
        test = [examples[i] for i in test.tolist()]
        yield Fold(i, train, test)


def train_crossval(folds: Iterable[Fold], max_iter) -> None:
    """Run crossvalidation training."""
    for fold in folds:

        # Train fold.
        logger.info(f"Training fold {fold.id}")
        fold.crf = train(fold.train, fold.dev, max_iter)


def score_fold(fold):
    # Compute micro and macro IOB scores.
    fold.micro_iob_scores = iob_level.score_micro_average(fold.dev, fold.crf)
    # fold.macro_iob_scores = iob_level.score_macro_average(
    #     fold.dev, fold.crf
    # )

    # Compute micro and macro event scores.
    fold.micro_event_scores = event_level.score_micro_average(
        fold.dev, fold.crf
    )
    # fold.macro_event_scores = event_level.score_macro_average(
    #     fold.dev, fold.crf
    # )


def average_scores(fold_score_dicts: Iterable[dict]):
    """Flatten and process the scores from crossval training."""

    merged = merge_list(fold_score_dicts)
    averaged = map_over_leaves(merged, mean)
    return averaged


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
