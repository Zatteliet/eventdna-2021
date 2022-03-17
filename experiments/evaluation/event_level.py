from dataclasses import dataclass
from typing import Iterable

from experiments.corpus import Example
from experiments.evaluation.alpino import AlpinoTree
from experiments.util import merge_mean
from loguru import logger
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

logger.add("log.log", level="TRACE", format="{message}", mode="w")


FOUND = "Found"
NOT_FOUND = "Not found"


@dataclass
class Event:
    tokens: set[int]
    heads: set[int]


def score_micro_average(examples: Iterable[Example], crf: CRF):
    """Score the performance of `crf` against the gold in each `example`. Return a report of micro-averaged scores."""
    predictions = crf.predict([ex.x for ex in examples])

    gold_vector = []
    pred_vector = []
    for example, prediction in zip(examples, predictions):
        gold_events = list(get_events(example.y, example.alpino_tree))
        pred_events = list(get_events(prediction, example.alpino_tree))

        gv, pv = match_between(gold_events, pred_events)
        gold_vector.extend(gv)
        pred_vector.extend(pv)

    report = classification_report(gold_vector, pred_vector, output_dict=True)
    return report


def score_macro_average(examples: Iterable[Example], crf: CRF):
    """Score the performance of `crf` against the gold in each `example`. Return a report of macro-averaged scores."""

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


def score(gold_events: list[Event], pred_events: list[Event]):
    """Use SKLearn to score lis list of predicted events against their gold equivalent."""
    gold_vector, pred_vector = match_between(gold_events, pred_events)
    report = classification_report(
        gold_vector,
        pred_vector,
        output_dict=True,
        zero_division=0,
    )
    return report


def match_between(gold_events: list[Event], pred_events: list[Event]):
    """Yield gold and pred event vectors for the given sets of events. These are binary vectors, where the positive class (`FOUND`) represents that the event is present in the event list.

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
        for _ in gold_events:
            # False negative.
            gold_vector.append(FOUND)
            pred_vector.append(NOT_FOUND)

    elif has(pred_events) and not has(gold_events):
        # One false positive for each predicted event.
        for _ in pred_events:
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
    """Find and yield `Event` objects from `sent`. These encode the tokens and head tokens of the event, encoded as integer indices over the sentence tokens.
    `sent` is a list of IOB tags, without label information.
    """

    # Sanity checks.
    assert len(sent) > 0, sent
    assert all(tag in {"I", "O", "B"} for tag in sent), sent

    current_event = []
    for i, iob_tag in enumerate(sent):
        if iob_tag in {"I", "B"}:
            current_event.append(i)
        else:
            if len(current_event) > 0:
                heads = get_head_set(current_event, tree)
                yield Event(tokens=set(current_event), heads=set(heads))
                current_event = []


def get_head_set(event_tokens: list[int], alpino_tree: AlpinoTree):
    """Given a list of event tokens (as indices over sentence tokens), yield those indices that also mark head tokens."""
    heads = alpino_tree.head_indices
    for token in event_tokens:
        if token in heads:
            yield token


def fallback_match(gold: Event, pred: Event):
    """Perform fuzzy matching to compare `gold` and `pred` events."""
    if gold.tokens == pred.tokens:
        return True
    if len(gold.tokens.intersection(pred.tokens)) == 0:
        return False
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
