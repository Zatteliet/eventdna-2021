from typing import Iterable

from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

from experiments.corpus import Example
from experiments.util import merge_mean
from experiments.evaluation.alpino import AlpinoTree
from dataclasses import dataclass


FOUND = "Found"
NOT_FOUND = "Not found"


@dataclass
class Event:
    tokens: set[int]
    heads: set[int]


def score_macro_average(examples: Iterable[Example], crf: CRF):

    predictions = crf.predict([ex.x for ex in examples])

    reports = []
    for example, prediction in zip(examples, predictions):

        gold_events = list(get_events(example.y, example.alpino_tree))
        pred_events = list(get_events(prediction, example.alpino_tree))

        s = score(gold_events, pred_events)
        if s is not None:
            reports.append(s)

    averaged = merge_mean(reports)
    return averaged


def score_macro_average(examples: Iterable[Example], crf: CRF):
    def f1_score(prec, rec):
        return (2 * (prec * rec)) / (prec + rec)

    predictions = crf.predict([ex.x for ex in examples])

    scores = {FOUND: [], NOT_FOUND: []}
    for example, prediction in zip(examples, predictions):

        gold_events = list(get_events(example.y, example.alpino_tree))
        pred_events = list(get_events(prediction, example.alpino_tree))

        report = score(gold_events, pred_events)

        if report.get(FOUND):
            scores[FOUND].append(report[FOUND])
        if report.get(NOT_FOUND):
            scores[NOT_FOUND].append(report[NOT_FOUND])

    for label in [FOUND, NOT_FOUND]:
        scores[label] = merge_mean(scores[label])
        p = scores[label]["precision"]
        r = scores[label]["recall"]
        scores[label]["f1-score"] = f1_score(p, r)

    return scores


def score(gold_events, pred_events):
    gold_vector, pred_vector = match_between(gold_events, pred_events)
    report = classification_report(
        gold_vector,
        pred_vector,
        output_dict=True,
        zero_division=0,
        # labels=[FOUND, NOT_FOUND],
    )
    return report
    # return {FOUND: report[FOUND], NOT_FOUND: report[NOT_FOUND]}


def match_between(gold_events: list[Event], pred_events: list[Event]) -> list:
    """Yield strings indicating a True Positive, etc. score.

    Note that this function support matching multiple events in a sentence, though current experimental setting only counts on one event per sentence.
    """
    assert isinstance(gold_events, list)
    assert isinstance(pred_events, list)

    def has(l):
        return len(l) > 0

    gold_vector = []
    pred_vector = []

    if not has(pred_events) and not has(gold_events):
        # True negative.
        gold_vector.append(NOT_FOUND)
        pred_vector.append(NOT_FOUND)

    elif not has(pred_events) and has(gold_events):
        # False negative.
        gold_vector.append(FOUND)
        pred_vector.append(NOT_FOUND)

    elif has(pred_events) and not has(gold_events):
        # One false positive for each predicted event.
        for p in pred_events:
            gold_vector.append(NOT_FOUND)
            pred_vector.append(FOUND)

    elif has(pred_events) and has(gold_events):
        for p in pred_events:
            # One true positive for each predicted event matching any gold event.
            if any(fallback_match(p, g) for g in gold_events):
                gold_vector.append(FOUND)
                pred_vector.append(FOUND)
            # One false positive for each predicted event not matching any gold events.
            else:
                gold_vector.append(NOT_FOUND)
                pred_vector.append(FOUND)

    return gold_vector, pred_vector


def get_events(sent: list[str], tree: AlpinoTree):

    # Sanity checks.
    assert len(sent) > 0, sent
    assert all(tag in {"I", "O", "B"} for tag in sent), sent

    current_event = []
    for i, iob_tag in enumerate(sent):
        if iob_tag in {"I", "B"}:
            current_event.append(i)
        else:
            if len(current_event) > 0:
                heads = list(get_head_set(current_event, tree))
                yield Event(tokens=current_event, heads=heads)
                current_event = []


def get_head_set(event_tokens: list[int], alpino_tree: AlpinoTree):
    heads = alpino_tree.head_indices
    for token in event_tokens:
        if token in heads:
            yield token


def fallback_match(gold: Event, pred: Event):
    if fuzzy_match(gold.heads, pred.heads):
        return True
    return fuzzy_match(gold.tokens, pred.tokens)


def fuzzy_match(set1, set2):
    def dice_coef(items1, items2) -> float:
        if len(items1) + len(items2) == 0:
            return 0
        intersect = set(items1).intersection(set(items2))
        return 2.0 * len(intersect) / (len(items1) + len(items2))

    return dice_coef(set1, set2) > 0.8
