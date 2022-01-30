from typing import Iterable

from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

from experiments.corpus import Example
from experiments.util import merge_mean


def score_macro_average(examples: Iterable[Example], crf: CRF):

    predictions = crf.predict([ex.x for ex in examples])

    reports = []
    for example, prediction in zip(examples, predictions):
        reports.append(score(example.y, prediction))

    averaged = merge_mean(reports)
    return averaged


def score(gold, pred):
    """
    We use `zero_division = 1` so that cases with missing labels e.g.
        gold = ["O", "O", "O", "O"]
        pred = ["O", "O", "O", "O"]
    the failure to find any I or B labels in pred is counted as f-score == 1 on those classes. Without `zero_division = 1`, prec-rec-f scores for the I and B classes unfairly default to 0.

    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    """
    report = classification_report(
        gold,
        pred,
        output_dict=True,
        zero_division=1,
        labels=["B", "I", "O"],
    )

    return {"B": report["B"], "I": report["I"], "O": report["O"]}
