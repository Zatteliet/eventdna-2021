import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


def get_iob(dnaf_path: Path, main_events_only: bool):
    """Get the IOB representation of the events of a document.
    Yield (sent_id, iob_list) tuples.
    """
    with open(dnaf_path) as j:
        dnaf = json.load(j)

    # Get events from the document.
    events = dnaf["doc"]["annotations"]["events"].values()

    # Filter out non-main-events.
    if main_events_only:
        events = [
            ann
            for ann in events
            if ann.get("features").get("prominence") == "Main"
        ]

    # For each sentence, get the list of main events in that sentence.
    sent_to_events = defaultdict(list)
    for event in events:
        sent_to_events[event["home_sentence"]].append(event)

    # For each sentence, pick the longest main event and get the IOB sequence for it.
    # If there is no main event in the sentence, return all O's.
    for sent_id, sent in dnaf["doc"]["sentences"].items():
        events = sent_to_events[sent_id]

        # NOTE ! Select longest main event, discard the rest.
        if len(events) == 0:
            iob = get_iob_sequence(sent, None)
        else:
            target_ev = max(events, key=lambda me: len(me["features"]["span"]))
            iob = get_iob_sequence(sent, target_ev)

        yield sent_id, iob


def get_iob_sequence(sent, event_ann: Optional[dict]):
    """Return the IOB sequence over a sentence, given an event annotation in that sentence."""

    sent_tokens: list[str] = sent["token_ids"]

    if event_ann is None:
        return ["O" for _ in sent_tokens]

    iob = []
    event_tokens = event_ann["features"]["span"]
    for sent_token in sent_tokens:
        if sent_token == event_tokens[0]:
            iob.append("B")
        elif sent_token in event_tokens:
            iob.append("I")
        else:
            iob.append("O")

    return iob
