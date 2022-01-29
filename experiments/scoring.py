from typing import Iterable
import sklearn.metrics as skl_metrics
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
from experiments.corpus import Example
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import dice

IOBSequence = list[str]


def score_iob(
    y_gold: Iterable[IOBSequence], y_pred: Iterable[IOBSequence]
) -> dict:
    """Score a `crf` model over `X_dev` against `y_dev`.

    We use classification reports as provided by `sklearn.metrics`.

    Return a scores dict.
    """

    def flatten(lists):
        for l in lists:
            for item in l:
                yield item

    gold_iob_tags: IOBSequence = list(flatten(y_gold))
    pred_iob_tags: IOBSequence = list(flatten(y_pred))

    # # Transform the data to SKLearn's preference.
    # # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
    # targets = ["I", "O", "B"]
    # mlb = MultiLabelBinarizer(classes=targets)
    # y_gold = mlb.fit_transform(y_gold)
    # y_pred = mlb.fit_transform(y_pred)

    # Collect various scores and return them.
    dict_report = skl_metrics.classification_report(
        gold_iob_tags, pred_iob_tags, output_dict=True
    )

    pretty_report = skl_metrics.classification_report(
        gold_iob_tags, pred_iob_tags, output_dict=False
    )

    return pretty_report, dict_report


def score_event_matches(
    y_gold: Iterable[IOBSequence], y_pred: Iterable[IOBSequence]
):
    """Score events according to a Dice-based metric."""

    def match(set1, set2):
        return dice(set1, set2) > 0.8

    def score_sentence(gold: IOBSequence, pred: IOBSequence):
        pass

    pass


def get_events(sent: IOBSequence):
    assert set(tag for tag in sent) == {"I", "O", "B"}
    current_event = set()
    for i, iob_tag in enumerate(sent):
        if iob_tag in {"I", "B"}:
            current_event.update({i})
        else:
            if len(current_event) > 0:
                yield current_event
                current_event = set()


METHODS = {"iob": score_iob, "event_spans": score_event_matches}


def score(method: str, crf, dev_set: Iterable[Example]):
    method = METHODS[method]

    X, y_gold = [ex.x for ex in dev_set], [ex.y for ex in dev_set]
    y_pred = crf.predict(X)

    # Sanity check.
    for g, p in zip(y_gold, y_pred):
        assert len(g) == len(p)

    return method(y_gold, y_pred)
