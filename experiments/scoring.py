from itertools import chain
from typing import Iterable

from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

from experiments.corpus import Example

IOBSequence = list[str]


class ScoreReport:
    def __init__(self, dev_set: Iterable[Example], crf: CRF) -> None:
        y_gold, y_pred = predict_on(dev_set, crf)

        iob_vs = iob_vectors(y_gold, y_pred)
        self.iob = classification_report(*iob_vs, output_dict=True)

        event_vs = event_vectors(y_gold, y_pred)
        self.events = classification_report(*event_vs, output_dict=True)


def iob_vectors(
    y_gold: Iterable[IOBSequence], y_pred: Iterable[IOBSequence]
) -> dict:
    gold_iob_tags: IOBSequence = list(chain.from_iterable(y_gold))
    pred_iob_tags: IOBSequence = list(chain.from_iterable(y_pred))
    return gold_iob_tags, pred_iob_tags


def event_vectors(
    y_gold: Iterable[IOBSequence], y_pred: Iterable[IOBSequence]
):
    def collect_outcomes():
        for gold_sent, pred_sent in zip(y_gold, y_pred):
            gold_events = list(get_events(gold_sent))
            pred_events = list(get_events(pred_sent))
            for outcome in match_between(gold_events, pred_events):
                yield outcome

    outcomes = list(collect_outcomes())

    # Sanity check. We expect all types of outcomes to occur.
    assert set(outcomes) == {"TP", "TN", "FP", "FN"}, collect_outcomes

    gold_vector, pred_vector = [], []
    for outcome in outcomes:
        if outcome == "TP":
            gold_vector.append(1)
            pred_vector.append(1)
        if outcome == "TN":
            gold_vector.append(0)
            pred_vector.append(0)
        if outcome == "FP":
            gold_vector.append(0)
            pred_vector.append(1)
        if outcome == "FN":
            gold_vector.append(1)
            pred_vector.append(0)

    return gold_vector, pred_vector


def dice_fuzzy_match(set1, set2):
    def dice_coef(items1, items2) -> float:
        if len(items1) + len(items2) == 0:
            return 0
        intersect = set(items1).intersection(set(items2))
        return 2.0 * len(intersect) / (len(items1) + len(items2))

    return dice_coef(set1, set2) > 0.8


def match_between(gold_events: Iterable[set], pred_events: Iterable[set]):
    """Yield strings indication a True Positive, etc. score.

    Note that this function support matching multiple events in a sentence, though current experimental setting only counts on one event per sentence.
    """
    if not pred_events:
        if gold_events:
            yield "FN"
        elif not gold_events:
            yield "TN"
    elif pred_events:
        for p in pred_events:
            if not gold_events:
                yield "FP"
            elif gold_events:
                if any(dice_fuzzy_match(p, g) for g in gold_events):
                    yield "TP"

    # if gold_events and pred_events:
    #     for g in gold_events:
    #         for p in pred_events:
    #             if dice_fuzzy_match(g, p):
    #                 yield "TP"
    #             else:
    #                 yield "FN"
    # elif gold_events and not pred_events:
    #     yield "FN"
    # elif not gold_events and pred_events:
    #     yield "FP"
    # elif not gold_events and not pred_events:
    #     yield "TN"


def get_events(sent: IOBSequence):
    assert len(sent) > 0, sent
    assert all(tag in {"I", "O", "B"} for tag in sent), sent
    current_event = set()
    for i, iob_tag in enumerate(sent):
        if iob_tag in {"I", "B"}:
            current_event.update({i})
        else:
            if len(current_event) > 0:
                yield current_event
                current_event = set()


def predict_on(
    examples: Iterable[Example], crf: CRF
) -> tuple[Iterable[IOBSequence], Iterable[IOBSequence]]:
    x, y_gold = [ex.x for ex in examples], [ex.y for ex in examples]
    y_pred = crf.predict(x)
    return y_gold, y_pred
