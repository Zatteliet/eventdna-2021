from typing import Callable, Iterable
from copy import deepcopy
from statistics import mean


def merge_mean(ds: Iterable[dict]) -> dict:
    d = merge_list(ds)
    return map_over_leaves(d, mean)


def merge_list(ds: Iterable[dict]) -> dict:
    """Recursively aggregate dictionaries with the same keys.

    Given two dictionaries with the same structure
    """

    result = {}
    first = ds[0]

    # # Check: all dictionaries should have the same keys.
    # for other in ds[1:]:
    #     assert (a:= set(first.keys())) == (b:= set(other.keys())), f"{a} != {b}"

    for k, v in first.items():
        if isinstance(v, dict):
            result[k] = merge_list([each[k] for each in ds])
        else:
            result[k] = [each[k] for each in ds]
    return result


def map_over_leaves(d: dict, c: Callable) -> dict:
    """Call `c` on each "leaf" of `d`, i.e. a value in `d` or a sub-dict of `d` that is not a dict itself.
    Return a new dict, leaving `d` intact.
    """
    r = deepcopy(d)
    for k, v in r.items():
        if isinstance(v, dict):
            r[k] = map_over_leaves(v, c)
        else:
            r[k] = c(v)
    return r
