from experiments.util import merge_list, map_over_leaves, merge_mean


def test_merge():
    cases = [
        {"a": 1, "b": 2, "c": {"ca": 30, "cb": 50}},
        {"a": 17, "b": 18, "c": {"ca": 130, "cb": 150}},
    ]
    expected = {
        "a": [1, 17],
        "b": [2, 18],
        "c": {"ca": [30, 130], "cb": [50, 150]},
    }

    assert merge_list(cases) == expected


def test_map_over_leaves():
    case = {
        "a": [1, 17],
        "b": [2, 18],
        "c": {"ca": [30, 130], "cb": [50, 150]},
    }

    expected = {
        "a": 18,
        "b": 20,
        "c": {"ca": 160, "cb": 200},
    }

    assert map_over_leaves(case, sum) == expected


def test_merge_mean():
    case = [
        {"a": 1, "b": 2, "c": {"ca": 30, "cb": 50}},
        {"a": 17, "b": 18, "c": {"ca": 130, "cb": 150}},
    ]
    expected = {"a": 9, "b": 10, "c": {"ca": 80, "cb": 100}}
    assert merge_mean(case) == expected
