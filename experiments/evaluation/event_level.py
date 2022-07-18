import logging
from dataclasses import dataclass
from typing import Iterable

from experiments.corpus import Example
from experiments.evaluation.alpino import AlpinoTree
from experiments.util import merge_mean
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

logger = logging.getLogger(__name__)


FOUND = "Found"
NOT_FOUND = "Not found"


@dataclass
class Event:
    tokens: set[int]
    heads: set[int]


def score_micro_average(examples: Iterable[Example], crf: CRF):
    """Score the performance of `crf` against the gold in each `example`. Return a report of micro-averaged scores.

    This is used to score a single fold in micro-average fashion.
    """
    predictions = crf.predict([ex.x for ex in examples])

    gold_vector = []
    pred_vector = []
    for example, prediction in zip(examples, predictions):

        # Get sets of tokens indices and head token indices for each gold and pred event.
        gold_events = list(get_events(example.y, example.alpino_tree))
        pred_events = list(get_events(prediction, example.alpino_tree))

        # Compute a y_actual and y_pred vector by tallying TP, FN, FP, TN event matches for this example.
        gv, pv = match_between(gold_events, pred_events)

        gold_vector.extend(gv)
        pred_vector.extend(pv)

    # Report a count of CM categories.
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for g, p in zip(gold_vector, pred_vector):
        if g == p == FOUND:
            counts["tp"] += 1
        elif g == FOUND and p == NOT_FOUND:
            counts["fn"] += 1
        elif g == NOT_FOUND and p == FOUND:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    logger.info(
        f"In event level scoring, collected the following CM counts: {counts}"
    )

    report = classification_report(gold_vector, pred_vector, output_dict=True)
    return report


def score_macro_average(examples: Iterable[Example], crf: CRF):
    """Score the performance of `crf` against the gold in each `example`. Return a report of macro-averaged scores."""

    def f1_score(prec, rec):
        return (2 * (prec * rec)) / (prec + rec)

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


def match_between(gold_events: list[Event], pred_events: list[Event]):
    """Yield gold and pred event vectors for the given sets of events. These are binary vectors, where the positive class (`FOUND`) represents that the event is present in the event list.

    Note that this function support matching multiple events in a sentence, though current experimental setting only counts on one event per sentence.
    """

    assert isinstance(gold_events, list)
    assert isinstance(pred_events, list)

    def has(l):
        return len(l) > 0

    gv = []
    pv = []

    # There are pred events, but no gold events.
    # -> Count one FP for each predicted event.
    if has(pred_events) and not has(gold_events):
        for _ in pred_events:
            gv.append(NOT_FOUND)
            pv.append(FOUND)

    # There are gold events, but no pred events.
    # -> Count one FN for each gold event.
    elif has(gold_events) and not has(pred_events):
        for _ in gold_events:
            gv.append(FOUND)
            pv.append(NOT_FOUND)

    # There are gold events AND there are pred events.
    # For each predicted event, attempt to match it to a gold event.
    # If there is a match, count a TP. If there is no match, count a FP.
    # For every gold event that was NOT matched to a pred event, also count a FN.

    # TODO 1. count FN for each leftover gold event
    # TODO 2. reverse: count TP for each gold event predicted, and FNs for leftover pred events.
    elif has(pred_events) and has(gold_events):

        matched_gold_events = []

        for p in pred_events:

            p_was_matched = False
            for g in gold_events:
                if fallback_match(p, g):
                    matched_gold_events.append(g)
                    gv.append(FOUND)
                    pv.append(FOUND)
                    p_was_matched = True
                    break
            if not p_was_matched:
                gv.append(NOT_FOUND)
                pv.append(FOUND)

        unmatched_gold_events = [
            g for g in gold_events if g not in matched_gold_events
        ]
        for g in unmatched_gold_events:
            gv.append(FOUND)
            pv.append(NOT_FOUND)

    # There are no gold or pred events in the example.
    # -> Count 1x TN.
    else:
        gv.append(NOT_FOUND)
        pv.append(NOT_FOUND)

    return gv, pv


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
    """Perform fuzzy matching to compare `gold` and `pred` events.

    The match is always True if the gold tokens match the pred tokens exactly, and always False if there is no overlap between the tokens of pred and gold.

    If neither these conditions pass, perform a fuzzy match on the heads of the events and return True if that check passes.

    Else, perform fuzzy match on the tokens of the events and return that conclusion.
    """

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
