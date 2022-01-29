from experiments.scoring import match_between, iob_vectors, get_events
from sklearn.metrics import classification_report

def test_get_events():
    case = ["O", "B", "I", "I", "O", "O", "B", "I", "O", "O"]
    assert list(get_events(case)) == [{1, 2, 3}, {6, 7}]


def test_score_iob():
    case = {
        "gold": [
            ["B", "I", "O"],
            ["O", "O", "O", "O"],
            ["B", "I", "I", "O", "O"],
        ],
        "pred": [
            ["B", "I", "I"],
            ["O", "O", "B", "I"],
            ["B", "I", "I", "O", "O"],
        ],
    }

    g, p = iob_vectors(case["gold"], case["pred"])
    print(g, p)
    report = classification_report(g, p, output_dict=True)

    assert report["O"]["precision"] == 1.0
    print(report)


def test_match_between():
    cases = [([{1}, {3, 4}], [{1}, {3, 4, 5}])]
    for g, p in cases:
        print(list(match_between(g, p)))
