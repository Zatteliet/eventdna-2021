from itertools import chain
from typing import Iterable

from experiments.corpus import Example
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF


def score_micro_average(examples: Iterable[Example], crf: CRF):
    """Score each prediction made by `crf` on an `example` against its gold. Return a micro-averaged score report."""

    golds = [ex.y for ex in examples]
    predictions = crf.predict([ex.x for ex in examples])

    flat_golds = list(chain.from_iterable(golds))
    flat_preds = list(chain.from_iterable(predictions))
    report = classification_report(flat_golds, flat_preds, output_dict=True)
    return report
