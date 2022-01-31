from typing import Iterable

from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

from experiments.corpus import Example
from experiments.util import merge_mean, map_over_leaves
from itertools import chain
from statistics import mean


def score_micro_average(examples: Iterable[Example], crf: CRF):

    golds = [ex.y for ex in examples]
    predictions = crf.predict([ex.x for ex in examples])

    flat_golds = list(chain.from_iterable(golds))
    flat_preds = list(chain.from_iterable(predictions))
    report = classification_report(flat_golds, flat_preds, output_dict=True)
    print(report)
    return report


"""
Macro-averaging the results leads me into technical difficulties.
There are many instances where recall or precision are ill-defined because those labels are missing.
So macro-averaging doesn't really work, because some of the class scores getting passed through are missing, so we are working with a different # of B, I, O class examples.
"""


def score_macro_average(examples: Iterable[Example], crf: CRF):
    def f1_score(prec, rec):
        return (2 * (prec * rec)) / (prec + rec)

    predictions = crf.predict([ex.x for ex in examples])

    scores = {"B": [], "I": [], "O": []}
    for example, prediction in zip(examples, predictions):
        report = score(example.y, prediction)

        if report.get("B"):
            scores["B"].append(report["B"])
        if report.get("I"):
            scores["I"].append(report["I"])
        if report.get("O"):
            scores["O"].append(report["O"])

    for label in ["I", "O", "B"]:
        scores[label] = merge_mean(scores[label])
        p = scores[label]["precision"]
        r = scores[label]["recall"]
        scores[label]["f1-score"] = f1_score(p, r)

    return scores


def score(gold, pred):
    """Score a single pairs of IOB sequences, representing a sentence.
    We use `zero_division = 1` so that cases with missing labels e.g.
        gold = ["O", "O", "O", "O"]
        pred = ["O", "O", "O", "O"]
    the failure to find any I or B labels in pred is counted as f-score == 1 on those classes. Without `zero_division = 1`, prec-rec-f scores for the I and B classes unfairly default to 0.

    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    """
    # print(gold, pred)

    report = classification_report(
        gold,
        pred,
        output_dict=True,
        zero_division=0,
        # labels=["B", "I", "O"],
    )

    # return {"B": report["B"], "I": report["I"], "O": report["O"]}
    return report
