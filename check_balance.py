import itertools
from experiments.corpus import get_examples
from itertools import chain
from pathlib import Path
from collections import Counter
from experiments.evaluation.event_level import get_events

examples = list(get_examples(Path("data/extracted"), main_events_only=False))

print(len(examples), "examples.")

n_events = 0
for ex in examples:
    for event in get_events(ex.y, ex.alpino_tree):
        n_events += 1
print(n_events, "events.")


# def has_events(example):
#     return set(example.y) == {"I", "O", "B"}


# print(Counter([has_events(example) for example in examples]))

iob_labels = chain.from_iterable([ex.y for ex in examples])

counts = Counter(iob_labels)

print(counts)
