from experiments.scoring import score_iob, get_events


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

    pretty_report, dict_report = score_iob(case["gold"], case["pred"])
    assert dict_report["O"]["precision"] == 1.0
    print(pretty_report)
