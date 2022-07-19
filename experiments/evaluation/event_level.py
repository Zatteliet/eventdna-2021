import logging
from dataclasses import dataclass
from typing import Iterable

from experiments.corpus import Example
from experiments.evaluation.alpino import AlpinoTree
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF

logger = logging.getLogger(__name__)


FOUND = "Found"
NOT_FOUND = "Not found"


@dataclass
class Event:
    """Defines an event in terms of token indices over the host sentence. Additionally encodes the head tokens, derived from alpino trees."""

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

        # Assume there is only 1 gold event. Assert this.
        gold_events = list(get_events(example.y, example.alpino_tree))
        assert len(gold_events) <= 1
        gold_event = gold_events[0] if gold_events else None

        # The CRF was trained with example each containing 1 event, so we expect mostly output with 1 event, but this is not guaranteed.
        # In this evaluation, to conform with the original evaluation, we ignore the significance of having multiple pred events.

        pred_events = list(get_events(prediction, example.alpino_tree))

        # Determine wether there the gold and pred event match as TP, FP, TN, FN. Add to the vectors accordingly.

        # No gold or pred events -> TN
        if not gold_event and not pred_events:
            gold_vector.append(NOT_FOUND)
            pred_vector.append(NOT_FOUND)
        # Pred events BUT no gold event -> FP
        elif not gold_event and pred_events:
            gold_vector.append(NOT_FOUND)
            pred_vector.append(FOUND)
        # Gold event BUT no pred event -> FN
        elif gold_event and not pred_events:
            gold_vector.append(FOUND)
            pred_vector.append(NOT_FOUND)
        # Gold event AND pred event -> TP is there is a fuzzy match, otherwise FN
        else:
            if any(fallback_match(gold_event, p) for p in pred_events):
                gold_vector.append(FOUND)
                pred_vector.append(FOUND)
            else:
                gold_vector.append(FOUND)
                pred_vector.append(NOT_FOUND)

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


def get_events(sent: list[str], tree: AlpinoTree):
    """Find and yield `Event` objects from `sent`. These encode the tokens and head tokens of the event, encoded as integer indices over the sentence tokens.

    `sent` is a list of IOB tags, without label information.
    """

    def get_head_set(event_tokens: list[int], alpino_tree: AlpinoTree):
        """Given a list of event tokens (as indices over sentence tokens), yield those indices that also mark head tokens."""
        heads = alpino_tree.head_indices
        for token in event_tokens:
            if token in heads:
                yield token

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
                event = Event(tokens=set(current_event), heads=set(heads))
                yield event
                current_event = []


def fallback_match(gold: Event, pred: Event):
    """Perform fuzzy matching to compare `gold` and `pred` events.

    The match is always True if the gold tokens match the pred tokens exactly, and always False if there is no overlap between the tokens of pred and gold.

    If neither these conditions pass, perform a fuzzy match on the heads of the events and return True if that check passes.

    Else, perform fuzzy match on the tokens of the events and return that conclusion.
    """

    def dice_coef(s1: set, s2: set) -> float:
        if not isinstance(s1, set) or not isinstance(s2, set):
            raise TypeError("Arguments must be sets.")

        if len(s1) + len(s2) == 0:
            return 0
        num = 2.0 * len(s1.intersection(s2))
        den = len(s1) + len(s2)
        return num / den

    def fuzzy_match(set1, set2):
        return dice_coef(set1, set2) > 0.8

    if gold.tokens == pred.tokens:
        return True
    if len(gold.tokens.intersection(pred.tokens)) == 0:
        return False
    if fuzzy_match(gold.heads, pred.heads):
        return True
    return fuzzy_match(gold.tokens, pred.tokens)
