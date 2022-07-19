import csv
import json
from pathlib import Path

from experiments.errors import FeaturizationError


def featurize(dnaf_path: Path, lets_path: Path):
    """Featurize a DNAF document, using only LETS features.
    The DNAF file is used to fetch sentence id codes, for later sanity checking against y labels.

    Yield (`sent_id`, `sent_feats`) tuples, where each sentence in `sent_feats` is a list of token feature dictionaries.
    """

    with open(dnaf_path) as j:
        dnaf = json.load(j)

    # Get the sentence ids used in this doc.
    sent_ids = [s_id for s_id in dnaf["doc"]["sentences"]]

    # Get the LETS features in this doc, sentence by sentence.
    for sent_id, lets_sent in zip(sent_ids, get_sentences(lets_path)):
        yield sent_id, list(featurize_lets_sentence(lets_sent))


def featurize_lets_sentence(rows):
    for i, row in enumerate(rows):
        previous = rows[i - 1] if i > 0 else None
        next = rows[i + 1] if i < (len(rows) - 1) else None
        yield featurize_lets_token(row, previous, next)


def get_sentences(lets_path):
    """Each row of a LETS csv file is a 5-tuple:
        (token, lemma, POS, chunk_iob, named_ent_iob)
    One file contains multiple sentences, separated by rows of five empty elements.

    Read in `lets_path` and yield `(sentence_id, [row])` tuples.
    """

    def is_separator(row):
        return len(row[0].strip()) == 0

    with open(lets_path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    current_sent = []
    for row in rows:
        # Check wellformedness.
        if not len(row) == 5:
            m = f"{lets_path.stem}: LETS file badly formed. -> {row}"
            raise FeaturizationError(m)

        if is_separator(row):
            yield current_sent
            current_sent = []
        else:
            current_sent.append(row)


def featurize_lets_token(current, previous=None, next=None):
    """Return a dict of features for the given token."""

    def get_features(row, prefix=None):
        """Return a dict with the token features.
        If `prefix` is True, all feature names are given this prefix.
        """
        token, lemma, pos, lets_chunk, lets_named_entity = row
        f = {
            "token": token,
            "lemma": lemma,
            "pos": pos,
            "lets_chunk": lets_chunk,
            "lets_named_entity": lets_named_entity,  # ! check if not the same as NE type, can be binary.
            "token_all_lower": token.islower(),
            "token[-3:]": token[-3:],
            "token[-2:]": token[-2:],
            "token_all_upper": token.isupper(),
            "token_contains_upper": any(c.isupper() for c in token),
            "token_isDigit": token.isdigit(),
            "token_containsOnlyAlpha": all(c.isalpha() for c in token),
            "token_capitalized": token.istitle(),
            "postag_major_cat": pos.split("(")[0],
            "chunk_major_cat": lets_chunk.split("-")[0],
            "ne_type": lets_named_entity.split("-")[-1],
        }
        if prefix:
            return {f"{prefix}{name}": val for name, val in f.items()}
        return f

    features = get_features(current)

    # Features of the preceding token.
    if previous:
        features.update(get_features(previous, "prev_"))
    # Add a beginning-of-sentence feature otherwise.
    else:
        features["BOS"] = True

    # Features of the following token.
    if next:
        features.update(get_features(next, "next_"))
    # Add an end-of-sentence feature otherwise.
    else:
        features["EOS"] = True

    return features
