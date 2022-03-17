from itertools import chain
from typing import Iterable

from experiments.corpus import Example
from experiments.util import merge_mean
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF


def score_macro_average(examples: Iterable[Example], crf: CRF):
    """Score each prediction made by `crf` on an `example` against its gold. Return the macro-averaged precision, recall and f1 scores."""

    def f1_score(prec, rec):
        return (2 * (prec * rec)) / (prec + rec)

    predictions = crf.predict([ex.x for ex in examples])

    # Gather precision and recall scores for each gold, pred pair.
    scores = {"B": [], "I": [], "O": []}
    for example, prediction in zip(examples, predictions):
        report = score(example.y, prediction)

        if report.get("B"):
            scores["B"].append(report["B"])
        if report.get("I"):
            scores["I"].append(report["I"])
        if report.get("O"):
            scores["O"].append(report["O"])

    # Average out the prec/rec scores and compute f1.
    # We compute f1 after the facts, because macro-averaging that score over examples that don't consistently return scores for each class label creates unexpected behaviour.
    for label in ["I", "O", "B"]:
        scores[label] = merge_mean(scores[label])
        p = scores[label]["precision"]
        r = scores[label]["recall"]
        scores[label]["f1-score"] = f1_score(p, r)

    return scores


def score(gold, pred):
    """Score a single pairs of IOB sequences, representing a sentence.

    Note that the output dict does not carry scores for labels that weren't found in the sequences.

    Docs: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
    """
    report = classification_report(
        gold,
        pred,
        output_dict=True,
        zero_division=0,
    )
    return report


def score_micro_average(examples: Iterable[Example], crf: CRF):
    """Score each prediction made by `crf` on an `example` against its gold. Return a micro-averaged score report."""

    golds = [ex.y for ex in examples]
    predictions = crf.predict([ex.x for ex in examples])

    flat_golds = list(chain.from_iterable(golds))
    flat_preds = list(chain.from_iterable(predictions))
    report = classification_report(flat_golds, flat_preds, output_dict=True)
    return report
