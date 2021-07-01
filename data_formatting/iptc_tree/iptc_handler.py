"""Read and define methods for interacting with the IPTC media topic tree.
"""

import json
from collections import defaultdict


class MediaTopicTree:
    """ Topics in `self.topic_objects` are stored as a flat list of dicts pulled directly from the json.
    eg.
    {
        "uri" : "http://cv.iptc.org/newscodes/mediatopic/20000003", 
        "qcode" : "medtop:20000003", 
        "type": ["http://www.w3.org/2004/02/skos/core#Concept"],
        "inScheme" : [ "http://cv.iptc.org/newscodes/mediatopic/" ], 
        "modified" : "2010-12-14T21:53:19+00:00", 
        "prefLabel" : {
        "en-GB" : "animation"},
        "definition" : {
        "en-GB" : "Stories told through animated drawings in either full-length or short format"},
        "broader" : [
        "http://cv.iptc.org/newscodes/mediatopic/20000002"
        ],
        "exactMatch" : [
        "http://cv.iptc.org/newscodes/subjectcode/01025000", "https://www.wikidata.org/entity/Q11425"],
        "created" : "2009-10-22T02:00:00+00:00"
    }
    """

    def __init__(self, json_fp):

        with open(json_fp) as json_in:
            self.raw_tree = json.load(json_in)

        self.topic_objects = list(self.raw_tree["conceptSet"])

    def _name(self, topic_dict):
        return topic_dict["prefLabel"]["en-GB"]

    def _code(self, topic_dict):
        return topic_dict["qcode"]

    def find_topic(self, query):
        """Find and return a topicDict from the tree.
        `query` can be the topic's code or its name.
        If no topic is found, an exception is raised.
        """
        matches = []
        for match_candidate in self.topic_objects:
            if (
                self._name(match_candidate) == query
                or self._code(match_candidate) == query
            ):
                matches.append(match_candidate)

        if len(matches) == 0:
            raise Exception(f"Topic '{query}' could not be found.")

        # CC 16/05: ['online media', 'mountaineering'] both appear twice as topic names.
        # Since they are extremely specific and exist under the same top node anyway, I
        # ignore the ambiguity and pick an arbitrary topic as the 'correct' one.
        return matches[0]

    def get_parents(self, start_topic):
        """Return a list of topic dicts, such that the given topic is the first element
        and the last topic is the top-level topic.
        """
        chain = [start_topic]
        while True:
            try:
                broader = chain[-1]["broader"]
                broader_medtopCode = f"medtop:{broader[0].split('/')[-1]}"
                broader_topic = self.find_topic(broader_medtopCode)
                chain.append(broader_topic)
            except KeyError:
                break
        return chain


iptc_tree = MediaTopicTree("data_formatting/iptc_tree/iptc_tree.json")


def get_topics(annotation_object, mediaTopicTree):
    """Given an annotation object, gather the certain and uncertain media topics. The output topic are the full topic dict objects.

    `annotation_object`, e.g.
    ```
    {
        "sofa": 12,
        "end": 9,
        "Certainoptics": "heads of state; homicide"
    }
    ```
    """

    certain_topics_string, uncertain_topics_string = (
        annotation_object["Certainoptics"]
        if annotation_object.get("Certainoptics")
        else "",
        annotation_object["Uncertaintopics"]
        if annotation_object.get("Uncertaintopics")
        else "",
    )

    # if an annotation string ends with ';', it is not parsed properly.
    if certain_topics_string[-1] == ";":
        certain_topics_string = certain_topics_string[:-1]
    if len(uncertain_topics_string) > 0 and uncertain_topics_string[-1] == ";":
        uncertain_topics_string = uncertain_topics_string[:-1]

    to_list = lambda annotation_string: [
        topic.strip()
        for topic in annotation_string.split(";")
        if len(topic.strip()) > 0
    ]
    certain_topics, uncertain_topics = (
        to_list(certain_topics_string),
        to_list(uncertain_topics_string),
    )

    all_topics = [(topic_name, "certain") for topic_name in certain_topics] + [
        (topic_name, "uncertain") for topic_name in uncertain_topics
    ]

    # dict of substitutions for wrong annotations.
    correction_dict = {
        "elections": "election",
        "automotive equipment": "automotive",
        "sexual assault": "sexual misconduct",
        "cultural policy": "cultural policies",
        "health organisation": "health organisations",
        "telecommunication": "telecommunication service",
        "gays and lesbians": "LGBT",
        "punishment": "punishment (criminal)",
        "national politics": "politics",
        "referendum": "referenda",
        "regulatory of industry": "regulation of industry",
        "international realtions": "international relations",
        "political parties": "political parties and movements",
        "Christianism": "Christianity",
        "non-governmental organization": "non-governmental organisation",
        "assault, court": "assault",
        "trail (court)": "trial (court)",
        "hospital and clinic": "hospital",
        "printing/promotional serivice": "printing/promotional service",
        "computer crime": "cyber crime",
        "national goverment": "national government",
        "acts of terror": "act of terror",
        "nationa elections": "national elections",
        "summit": "summit meetings",
        "constitution": "constitution (law)",
        "justice and rights": "justice",
        "security measures": "security measures (defence)",
        "forgein aid": "foreign aid",
        "high society": "high-society",
        "regional government": "regional government and authority",
        "international relationship": "international relations",
        "rstructuring and recapitalisation": "restructuring and recapitalisation",
        "military equiment": "military equipment",
        "bioterrorism": "act of bioterrorism",
        "newsagency": "news agency",
        "illegal immmigrants": "illegal immigrants",
        "trade unions": "unions",
        "Parliament": "parliament",
        "fine": "fine (penalty)",
        "international economic organisation": "international economic institution",
        "interantional organisation": "international organisation",
        "international organization": "international organisation",
        "ministers government": "ministers (government)",
        "bribe": "bribery",
        "c periodical": "periodical",
        "islam": "Islam",
        "nation government": "national government",
        "upper house (Parliament)": "upper house (parliament)",
        "royalty": "heads of state",
        "homeless": "homelessness",
        "awards and prizes": "award and prize",
        "investigation criminal": "investigation (criminal)",
        "local government": "local government and authority",
        "political parties and governments": "political parties and movements",
        "religion": "religious belief",
        "accident and emergency": "accident and emergency incident",
    }

    all_topicDicts = []
    for topic_name, certainty in all_topics:
        try:
            td = mediaTopicTree.find_topic(topic_name)
        except:
            try:
                td = mediaTopicTree.find_topic(
                    topic_name[:-1]
                )  # take care of -s at the end
            except:
                try:
                    td = mediaTopicTree.find_topic(topic_name + "s")
                except:
                    # an exception for "coup d'état" that is misread as "coup d'Ã©ta", not easy to fix
                    if "coup" in topic_name:
                        continue
                    else:
                        td = mediaTopicTree.find_topic(correction_dict[topic_name])

        all_topicDicts.append((td, certainty))

    certainty_to_topicChain = defaultdict(list)
    for t, cert in all_topicDicts:
        chain = mediaTopicTree.get_parents(t)
        certainty_to_topicChain[cert].append(chain)

    return certainty_to_topicChain
