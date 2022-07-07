from experiments.evaluation.alpino import AlpinoTree

def test_alpino_tree_handler():
    case = "tests/data/example_alpino.xml"
    
    tree = AlpinoTree(case, restricted_mode=False)

    head_leaves = tree.find_head_leaves(restricted_mode=False)
    print([h.get("word") for h in head_leaves])

    print(tree._head_vector)