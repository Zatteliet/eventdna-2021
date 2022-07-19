"""Classes and functions to read Alpino parse trees and determine what the heads are in EventDNA annotations."""

import xml.etree.ElementTree as ET


class AlpinoTree:
    """Read in and do operations on an alpino .xml file.
    Provides methods to find which tokens are heads.

    An easy way to check whether the token at index `i` in a sentence is part of a syntactic head:
    `bool(self.head_vector[i])`
    """

    def __init__(self, alpino_file, restricted_mode):
        self.file = alpino_file
        self.tree = ET.parse(alpino_file)
        self.nodes_to_parents = {c: p for p in self.tree.iter() for c in p}
        self.head_vector = self._head_vector(restricted_mode)
        self.head_indices = list(self._head_indices())

    def _head_indices(self):
        for i, flag in enumerate(self.head_vector):
            if flag == 1:
                yield i

    def get_parents(self, node):
        """Yield, in order, the queried node and then every ancestor up to and including the root."""
        current = node
        while True:
            yield current
            current = self.nodes_to_parents.get(current)
            if not current:
                break

    def find_head_leaves(self, restricted_mode: bool):
        """Depth-search through the tree. Only leaves are returned.
        If `restricted_mode` is True, only return heads that are not part of modifiers.
        """

        def check_stop(node, restricted_mode: bool):
            """If a node gets a True check here, search stops at that node and doesn't travel deeper in the tree."""
            if restricted_mode:
                if node.get("cat") in ["ap", "advp", "pp"]:
                    return True
                if node.get("rel") == "mod":
                    return True
            return False

        ## Negative restriction
        # Travel through all nodes in the tree in depth-first fashion, starting at the root.
        # Eliminate all those nodes that don't conform to the requirements.

        to_visit = [self.tree.getroot()]
        found = []
        while True:
            # If all nodes in the tree have been travelled, stop the loop.
            if len(to_visit) == 0:
                break

            # Pick the next node in the to_visit list and remove it from that list.
            current_node = to_visit.pop(0)

            # Determine whether the search for heads must stop here or continue.
            if check_stop(current_node, restricted_mode):
                continue

            is_leaf = lambda node: len(list(node)) == 0
            if is_leaf(current_node):
                found.append(current_node)
            else:
                kids = [n for n in current_node]
                to_visit.extend(kids)
                continue

        ## Positive restriction
        # Go over the `found` list and filter out all nodes that DON'T have a head somewhere in their ancestry.

        def has_hd_ancestor(node):
            """True if any of the query node's ancestors is a head node."""
            parents = self.get_parents(node)
            for p in parents:
                if p.get("rel") == "hd":
                    return True
            return False

        found = [n for n in found if has_hd_ancestor(n)]

        return found

    def _head_vector(self, restricted_mode):
        """Given an alpino tree, give a binary vector mapping over the tokens of the sentence described by the tree,
        such that 1 indicates that a token is part of a head node.
        e.g. [0, 1, 0, 1, 0, 0] --> tokens at index 1 and 3 are part of head nodes over the sentence.

        If `restricted_mode` is True, only return heads that are not part of modifiers.
        """

        def is_leaf(node):
            return len(list(node)) == 0

        # Get leaf nodes that are heads, as nodes.
        leaf_heads = self.find_head_leaves(restricted_mode)

        # Get nodes that are tokens. These are always leaves.
        sentence_tokens = [
            node
            for node in self.tree.iter("node")
            if is_leaf(node) and node.get("word") is not None
        ]
        sentence_tokens = sorted(
            sentence_tokens, key=lambda node: int(node.get("begin"))
        )

        # Sanity check: the sentence found by ordering the nodes is equal to the sentence given as a Sentence element in the xml.
        # For unknown reasons a None node is added to the list. This naively removes it (CC 13/05/2019).
        tokens_from_sentence_nodes = [
            node.get("word") for node in sentence_tokens
        ]

        x_tokens_by_sentence = self.tree.findall("./sentence")[0].text.split()
        assert (
            tokens_from_sentence_nodes == x_tokens_by_sentence
        ), "{} != {}".format(tokens_from_sentence_nodes, x_tokens_by_sentence)

        # Collect information: go over token nodes and show 1 if the token is part of the list of head tokens and 0 otherwise.
        head_flags = [
            (1 if node in leaf_heads else 0) for node in sentence_tokens
        ]
        assert len(tokens_from_sentence_nodes) == len(
            head_flags
        )  # Sanity check.

        return head_flags


def add_heads(dnaf, alpino_dir, restricted_mode) -> None:
    """Add head set info to the event annotations found in the given DNAF.

    If `restricted_mode` is True, only return heads that are not part of modifiers.
    """

    # Get a dict of sentence numbers to the correct head vector.
    head_vector_map = {
        int(file.stem): AlpinoTree(file)._head_vector(restricted_mode)
        for file in alpino_dir.iterdir()
    }

    # Go over all events in the dnaf document.
    for _, event in dnaf["doc"]["annotations"]["events"].items():

        home_sentence_id = event["home_sentence"]

        # Build vector for this event annotation over the sentence.
        # EG. "[President Trump addressed Congress] ." --> [1, 1, 1, 1, 0]
        sentence_tokens = sorted(
            dnaf["doc"]["sentences"][home_sentence_id]["token_ids"]
        )  # Tokens are represented as indices.
        event_tokens = sorted(event["features"]["span"])
        event_over_sentence_vector = [
            (1 if st in event_tokens else 0) for st in sentence_tokens
        ]

        # Fetch vector of all heads over the sentence from the head_vector_map defined previously.
        # eg. "President Trump addressed Congress ." --> [1, 0, 0, 1, 0]
        sentence_number = int(
            home_sentence_id.split("_")[1]
        )  # from e.g. "sentence_2" to 2
        head_over_sentence_vector = head_vector_map[sentence_number]

        # Get the overlap between head vector and sentence vector to get a set of tokens that are heads in an annotation.
        head_set = {
            i
            for i, (val1, val2) in enumerate(
                zip(event_over_sentence_vector, head_over_sentence_vector)
            )
            if val1 == val2 == 1
        }  # the `== 1` is there so as not to count the 0 values also.

        # Write the resulting head set as additional info to the DNAF.
        event["head_set"] = head_set
