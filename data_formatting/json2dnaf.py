"""Transforming Webanno output formats to the EventDNA Data Format (DNAf).
"""

import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from wasabi import msg

from data_formatting.iptc_tree import iptc_handler

# def format_over_corpus(corpus_json_dirp, out_dirp, test=False):
#     """Expects an input dir of json files. For each json, writes a corresponding dnaf file to `our_dirp`.
#     If `test` is true, only a small number of files will be processed.
#     """
#     source_jsons = list(corpus_json_dirp.iterdir())

#     assert len(source_jsons) == 1745
#     assert all(f.suffix == ".json" for f in source_jsons)

#     if test == True:
#         source_jsons = source_jsons[:100]

#     for source_f in source_jsons:
#         try:
#             with open(source_f, "r") as json_fo:
#                 dnaf = json2dnaf(json_fo, source_f.stem)
#             outfile = (out_dirp / source_f.stem).with_suffix(".json")
#             with open(outfile, "w") as o:
#                 json.dump(dnaf, o)
#         except AssertionError as ae:
#             msg.warn("Failed to parse {}. Skipping document.", source_f.stem)
#             print(ae)


def json2dnaf(inpath: Path, doc_id: str, strict: bool = True):
    """Return a dnaf-formatted json.

    If strict is True, articles that contain no IPTC topic annotations, entities or events are discarded.
    """

    ## AUX

    def is_in_spanObject(query, target):
        """Both `query` and `target` are spanObjects.
        A `spanObject` is a dict object that has `"begin"` and `"end"` as keys.
        The `"begin"` attritube of each object is missing if it is 0.
        Returns true if the query object's span in inside the target object's.
        """
        qb = query["begin"] if "begin" in query else 0
        qe = query["end"]
        tb = target["begin"] if "begin" in target else 0
        te = target["end"]
        return qb >= tb and qe <= te

    def annotation_tokens(annotation, source_tokens):
        """`source_tokens` is a list of e.g. `{"sofa" : 12,  "begin" : 26,  "end" : 32 }`."""
        return [
            i
            for i, tok in enumerate(source_tokens)
            if is_in_spanObject(tok, annotation)
        ]

    def tokensIds_to_string(token_ids, token_string):
        return " ".join([token_string.split(" ")[i] for i in token_ids])

    # id generators
    sentIds = (f"sentence_{n}" for n in range(1, 100))
    entity_annIds = (f"entity_{n}" for n in range(1, 100))
    event_annIds = (f"event_{n}" for n in range(1, 100))

    ## BODY

    with open(inpath, newline="") as json_in:
        source = json.load(json_in)
    source_initialView = source["_views"]["_InitialView"]

    if strict:
        # check: doc must contain at least one entity and at least one event annotation to appear in the corpus
        assert "Entities" in source_initialView, f"{doc_id} contains no entities."
        assert "Eventclauses" in source_initialView, f"{doc_id} contains no events."

    # initialize dnaf skeleton with initial values
    dnaf = {
        "meta": {"id": None, "author": None},
        "doc": {
            "token_string": None,
            "token_ids": [],
            "sentences": {},
            "annotations": {
                "entities": {},
                "events": {},
                "coreference": {},
                "iptc_codes": {},
            },
        },
    }

    ## METADATA

    dnaf["meta"]["id"] = doc_id

    author = source_initialView["DocumentMetaData"][0]["documentId"]
    dnaf["meta"]["author"] = author

    ## TEXT AND TOKENS

    # the sofa string is always in the node mysteriously named "12"
    tokens = source["_referenced_fss"]["12"]["sofaString"].split(" ")
    tokens = [tok.strip() for tok in tokens]
    dnaf["doc"]["token_string"] = " ".join(tokens)
    dnaf["doc"]["token_ids"] = [i for i, _ in enumerate(tokens)]

    # check
    # get tokens defined in source and verify they match with the dnaf tokens
    # e.g. [{"sofa" : 12,  "end" : 9 }, {"sofa" : 12,  "begin" : 10,  "end" : 19 }]
    source_tokens = source_initialView["Token"]
    assert len(dnaf["doc"]["token_ids"]) == len(
        source_tokens
    ), f"{doc_id}: tokens badly parsed."

    ## SENTENCES

    # e.g. [{"sofa" : 12,  "end" : 32 }, {"sofa" : 12,  "begin" : 34,  "end" : 106 }]
    source_sentences = source_initialView["Sentence"]
    sort_by_end = lambda l: sorted(l, key=lambda el: el["end"])

    # for each sentence, find which tokens belong to it
    # then add sentences to dnaf
    for source_sentence in sort_by_end(source_sentences):
        ss_tokenIndices = [
            i
            for i, source_tok in enumerate(source_tokens)
            if is_in_spanObject(source_tok, source_sentence)
        ]
        sentence = {"token_ids": ss_tokenIndices}
        dnaf["doc"]["sentences"][next(sentIds)] = sentence

    ## ANNOTATIONS
    # annotations in `source_initialView["Entities"]` or the corresponding `events` entries come in two flavours
    # either directly as a dict e.g. `{"sofa" : 12,  "begin" : 253,  "end" : 283,  "Entitytype" : "PER",  "Head" : [{"_type" : "EntitiesHeadLink",  "role" : "Head",  "target" : 714 } ],  "Individuality" : "GROUP",  "Mentionleveltype" : "NOM" }`
    # either as an int reference to an object listed in `source["_referenced_fss]"`

    ## COREF BUCKETS
    # if the relevant keys are not found, it indicates no coreference was annotated on that doc.
    source_coref_anns = {}
    try:
        source_coref_anns["events"] = source_initialView["Eventscoreference"]
    except KeyError:
        source_coref_anns["events"] = []
        # logger.warning("No event coreference found.")
    try:
        source_coref_anns["entities"] = source_initialView["Entitiescoreference"]
    except KeyError:
        source_coref_anns["entities"] = []
        # logger.warning("No entity coreference found.")

    # each of these annotations is a link between 2 ids
    # link e.g. {"sofa": 12, "end": 25, "Dependent": 603, "Governor": 633}
    # --> transform them into buckets
    def resolve_buckets(link_annotations):
        links = [(l["Dependent"], l["Governor"]) for l in link_annotations]
        buckets = []
        for a, b in links:
            bucket = [a, b]
            for l in links:
                if a in l or b in l:
                    bucket.extend(l)
            bucket = set(bucket)
            if bucket not in buckets:
                buckets.append(bucket)
        return buckets

    coref_buckets = {
        "events": resolve_buckets(source_coref_anns["events"]),
        "entities": resolve_buckets(source_coref_anns["entities"]),
    }

    sourceId_to_dnafId = {}

    ## COLLECT ENTITIES ##

    for source_entity in source_initialView["Entities"]:

        ann_id = next(entity_annIds)

        if isinstance(source_entity, int):
            sourceId_to_dnafId[source_entity] = ann_id
            source_entity = source["_referenced_fss"][str(source_entity)]

        # get basic features
        toks = annotation_tokens(source_entity, source_tokens)  # will be reused
        new_entity = {
            "type": "entity",
            "string": tokensIds_to_string(toks, dnaf["doc"]["token_string"]),
            "features": {
                "type": source_entity["Entitytype"],
                "span": toks,
                "individuality": source_entity["Individuality"],
                "mention_level_type": source_entity["Mentionleveltype"],
            },
        }

        # get entity heads & add it to the new annotation element
        # heads come in a list with 0 or 1 element. If there are more this is an annotator error
        # if there are 0 elements, the head is taken to be the same span as the entity mention itself
        # e.g. "Head" : [{"_type" : "EntitiesHeadLink",  "role" : "Head",  "target" : 714 } ]
        # `"target"` refers to objects in `source["_referenced_fss"]`
        head_list = source_entity["Head"]
        assert len(head_list) < 2, f"{doc_id}: an entity has more than 1 head."
        if len(head_list) == 1:
            hd = source["_referenced_fss"][
                str(head_list[0]["target"])
            ]  # e.g. {"_type" : "Entitieshead",  "sofa" : 12,  "begin" : 122,  "end" : 132 }
            toks = annotation_tokens(hd, source_tokens)
            hd_dict = {
                "type": "entityHead",
                "string": tokensIds_to_string(toks, dnaf["doc"]["token_string"]),
                "features": {"span": toks},
            }
        elif len(head_list) == 0:
            hd_dict = {
                "type": "entityHead",
                "string": tokensIds_to_string(toks, dnaf["doc"]["token_string"]),
                "features": {"span": toks},
            }
        new_entity["features"]["head"] = hd_dict

        dnaf["doc"]["annotations"]["entities"][ann_id] = new_entity

    ## COLLECT EVENTS ##

    events = source_initialView.get("Eventclauses")
    if events == None:
        events = []
    for source_ann in events:

        ann_id = next(event_annIds)

        if isinstance(source_ann, int):
            sourceId_to_dnafId[source_ann] = ann_id
            source_ann = source["_referenced_fss"][str(source_ann)]

        toks = annotation_tokens(source_ann, source_tokens)
        new_event = {
            "annotation_type": "event",
            "string": tokensIds_to_string(toks, dnaf["doc"]["token_string"]),
            "features": {
                "span": toks,
                "arguments": [],  # will be filled up
                "type": source_ann["Eventtype"],
                "subtype": source_ann["Eventsubtype"],
                "modality": source_ann["Modality"],
                "pos_neg": source_ann["Positivenegative"],
                "prominence": source_ann["Prominence"],
                "tense": source_ann["Tense"],
            },
        }

        ## GET ARGUMENTS ##

        # an arglink is e.g. `{"_type": "EventclausesArgumentsLink","role": "Victim","target": 767}`
        argLink_list = source_ann["Arguments"]
        for source_argLink in argLink_list:
            # a source arg is e.g. `{"_type": "Eventsarguments","sofa": 12,"end": 25,"Argumententitytype": "PER"}`
            source_arg = source["_referenced_fss"][str(source_argLink["target"])]
            arg_toks = annotation_tokens(source_arg, source_tokens)
            new_arg = {
                "string": tokensIds_to_string(arg_toks, dnaf["doc"]["token_string"]),
                "features": {
                    "span": arg_toks,
                    "argument_entity_type": source_arg["Argumententitytype"],
                    "argument_role": source_argLink["role"],
                },
            }
            new_event["features"]["arguments"].append(new_arg)

        dnaf["doc"]["annotations"]["events"][ann_id] = new_event

    # switch the source ids now used in the coreference buckets with the DNAF ids

    coref_bucket_ids = (f"coref_bucket_{n}" for n in range(1, 100))
    new_ids = lambda bucket: [sourceId_to_dnafId[old_id] for old_id in bucket]
    entity_coref_buckets = [new_ids(b) for b in coref_buckets["entities"]]
    event_coref_buckets = [new_ids(b) for b in coref_buckets["events"]]
    dnaf["doc"]["annotations"]["coreference"] = {
        "entities": {next(coref_bucket_ids): bucket for bucket in entity_coref_buckets},
        "events": {next(coref_bucket_ids): bucket for bucket in event_coref_buckets},
    }

    # add info to the annotations: in which sentence does it occur?

    def which_sentence(annotation):
        for s_id, s_tokens in dnaf["doc"]["sentences"].items():
            if annotation["features"]["span"][0] in s_tokens["token_ids"]:
                return s_id
        assert (
            False
        ), f"{doc_id}: no home sentence found for event. Should be impossible."

    for entity in dnaf["doc"]["annotations"]["entities"].values():
        entity["home_sentence"] = which_sentence(entity)
    for event in dnaf["doc"]["annotations"]["events"].values():
        event["home_sentence"] = which_sentence(event)

    ## IPTC CODES

    mtt = iptc_handler.iptc_tree
    if strict:
        assert source_initialView.get(
            "IPTCMediaTopic"
        ), f"{doc_id}: no IPTC topics annotated."
    topic_annotation = source_initialView["IPTCMediaTopic"][0]
    cert_to_topicChains = iptc_handler.get_topics(topic_annotation, mtt)

    topicDict_to_name_tuple = lambda topicDict: (
        topicDict["qcode"],
        topicDict["prefLabel"]["en-GB"],
    )
    cert_to_topicChainNames = {
        cert: [[topicDict_to_name_tuple(el) for el in chain] for chain in tn_list]
        for cert, tn_list in cert_to_topicChains.items()
    }

    dnaf["doc"]["annotations"]["iptc_codes"] = cert_to_topicChainNames
    if not dnaf["doc"]["annotations"]["iptc_codes"].get("uncertain"):
        # uncertain list needs to exist to guarantee no reading errors later
        dnaf["doc"]["annotations"]["iptc_codes"]["uncertain"] = []

    # with open(outpath, "w", newline="") as out:
    #     json.dump(dnaf, out, indent=4, sort_keys=True)
    return dnaf
