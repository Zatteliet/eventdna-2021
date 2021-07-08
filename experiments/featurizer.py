from pathlib import Path
from collections import defaultdict
import json
import csv
import xml.etree.ElementTree as ET


def featurize(dnaf: Path, lets: Path, alpino: Path):
    """Featurize a DNAF document.
    Returns a json-like with the form:
    ```
    [  # document
        [  # sentence
            tokens: [{feature: str}]
        [
    ]
    ```
    """

    with open(dnaf) as j:
        dnaf = json.load(j)

    # Get the sentence ids used in this doc. This is useful for many variable definitions down the road.
    sent_ids = [s_id for s_id in dnaf["doc"]["sentences"]]
    # Little helper function to simply an id e.g. "sentence_1" to "1".
    as_int = lambda sent_id: sent_id.split("_")[-1]

    # Find the right alpino file for each sentence id.
    sent_to_alpino = {}
    for i in sent_ids:
        filename = as_int(i) + ".xml"
        sent_to_alpino[i] = alpino / filename

    # Read in the LETS rows.
    with open(lets, newline="") as lets_in:
        lets_reader = csv.reader(lets_in, delimiter="\t")
        lets_rows = list(lets_reader)

    # Get the LETS rows pertaining to each sentence.
    sent_to_lets = defaultdict(list)
    current_sent = 0
    for row in lets_rows:
        if len(row[0].strip()) == 0:
            current_sent += 1
        else:
            sent_to_lets[sent_ids[current_sent]].append(row)

    # Process each sentence.
    featurized_doc = []
    for sent_id, sent_dict in dnaf["doc"]["sentences"].items():
        # read Alpino data for this sentence
        alpino_tree = ET.parse(sent_to_alpino[sent_id])

        sentence_feats = []
        s_tokens = sent_dict["token_ids"]
        for token_i_in_sent, _ in enumerate(s_tokens):
            token_feats = {"sentence_index": as_int(sent_id)}

            # get LETS features
            token_feats.update(lets_features(lets_rows, token_i_in_sent))

            # get Alpino features
            token_feats.update(alpino_features(alpino_tree, token_i_in_sent))

            sentence_feats.append(token_feats)
        featurized_doc.append(sentence_feats)

    return featurized_doc


def alpino_features(alpino_tree, token_idx_in_sentence):
    """Return a dict of features for the given token."""
    # for testing
    return {"dummyFeature": "dummyValue"}


def lets_features(sentence_rows, token_idx_in_sentence):
    """Return a dict of features for the given token."""
    # logger.debug("Processing token {}", token_idx_in_sentence)

    token, lemma, pos, lets_chunk, lets_named_entity = [
        el.strip() for el in sentence_rows[token_idx_in_sentence]
    ]

    features = {
        "token": token,
        "lemma": lemma,
        "pos": pos,
        "lets_chunk": lets_chunk,
        "lets_named_entity": lets_named_entity,  # ! check if not the same as NE type, can be binary.
        "token_all_lower()": token.islower(),
        "token[-3:]": token[-3:],
        "token[-2:]": token[-2:],
        "token_all_upper": token.isupper(),
        "token_contains_upper": any(c.isupper() for c in token),
        "token_isDigit": token.isdigit(),
        "token_containsOnlyAlpha": all(c.isalpha() for c in token),
        "token_capitalized": token.istitle(),
        "postag_majorCategory": pos.split("(")[0],
        "chunk_majorCategory": lets_chunk.split("-")[0],
        "ne_type": lets_named_entity.split("-")[-1],
    }

    # in sentence with word before
    if token_idx_in_sentence > 0:
        prev_token, prev_lemma, prev_pos, prev_chunk, prev_ne = sentence_rows[
            token_idx_in_sentence - 1
        ]

        features.update(
            {
                "-1:token": prev_token,
                "-1:lemma": prev_lemma,
                "-1:postag": prev_pos,
                "-1:chunk": prev_chunk,
                "-1:ne": prev_ne,
                "-1:token_all_lower()": prev_token.islower(),
                "-1:token[-3:]": prev_token[-3:],
                "-1:token[-2:]": prev_token[-2:],
                "-1:token_all_upper": prev_token.isupper(),
                "-1:token_contains_upper": any(c.isupper() for c in prev_token),
                "-1:token_isdigit": prev_token.isdigit(),
                "-1:token_containsonlyalpha": all(c.isalpha() for c in prev_token),
                "-1:token_capitalized": prev_token.istitle(),
                "-1:postag_majorcategory": prev_pos.split("(")[0],
                "-1:chunk_majorcategory": prev_chunk.split("-")[0],
                "-1:ne_type": prev_ne.split("-")[-1],
            }
        )
    else:  # beginning of sentence
        features["BOS"] = True

    if token_idx_in_sentence < len(sentence_rows) - 1:  # in sentence with word after
        next_token, next_lemma, next_pos, next_chunk, next_ne = sentence_rows[
            token_idx_in_sentence + 1
        ]

        features.update(
            {
                "+1:token": next_token,
                "+1:lemma": next_lemma,
                "+1:postag": next_pos,
                "+1:chunk": next_chunk,
                "+1:ne": next_ne,
                "+1:token_all_lower()": next_token.islower(),
                "+1:token[-3:]": next_token[-3:],
                "+1:token[-2:]": next_token[-2:],
                "+1:token_all_upper": next_token.isupper(),
                "+1:token_contains_upper": any(c.isupper() for c in next_token),
                "+1:token_isdigit": next_token.isdigit(),
                "+1:token_containsonlyalpha": all(c.isalpha() for c in next_token),
                "+1:token_capitalized": next_token.istitle(),
                "+1:postag_majorcategory": next_token.split("(")[0],
                "+1:chunk_majorcategory": next_token.split("-")[0],
                "+1:ne_type": next_token.split("-")[-1],
            }
        )
    else:
        features["EOS"] = True

    return features
