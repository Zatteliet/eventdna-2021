from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
from loguru import logger

from experiments.errors import FeaturizationError
from experiments.featurizer import featurize
from experiments.iob_fmt import get_iob

DATA_ZIPPED = Path("data/eventdna_corpus_2020-13-02.zip")
DATA_EXTRACTED = DATA_ZIPPED.parent / "extracted"


@dataclass
class Example:
    id: str
    x: dict
    y: dict


def get_examples(data_dir):
    """Read and yield features and labels from a data dir.
    Every sentence in the corpus will become a training example.
    """
    logger.info("Featurizing data...")
    for doc_id, dnaf_p, lets_p, _ in read_files(data_dir):
        try:
            examples = list(get_featurized_sents(doc_id, dnaf_p, lets_p))
        except FeaturizationError as e:
            logger.error(e)
        for example in examples:
            yield example


def get_featurized_sents(doc_id: str, dnaf: Path, lets: Path):
    """Stream examples from a single document directory."""

    # Extract X and y features.
    x_sents = list(featurize(dnaf, lets))
    y_sents = list(get_iob(dnaf, main_events_only=True))

    for (x_id, x), (y_id, y) in zip(x_sents, y_sents):

        # Check the correct sentences are matched.
        if x_id != y_id:
            raise FeaturizationError("Sentence ids do not match.")

        # Check the n of tokens in each sentence is the same.
        if not len(x) == len(y):
            t = [d["token"] for d in x]
            m = f"{doc_id}: number of tokens in x and y don't match.\n{t} != {y}"
            raise FeaturizationError(m)

        ex_id = f"{doc_id}_{x_id}"
        yield Example(id=ex_id, x=x, y=y)


def check_extract(zip: Path, target: Path):
    """Extract `zip_p` to `target` if this has not been done already."""
    if not target.exists():
        target.mkdir()
        logger.info(f"Extracting corpus to {target}")
        with ZipFile(zip) as z:
            z.extractall(target)
    else:
        logger.info(f"Using existing data dir: {target}")


def read_files(data_dir: Path):

    all_paths = data_dir.rglob("*")

    files = defaultdict(dict)
    for p in all_paths:
        if p.parent.stem in {"dnaf", "lets", "alpino"}:
            files[str(p.stem)][str(p.parent.stem)] = p
    for id, paths in files.items():
        yield id, paths["dnaf"], paths["lets"], paths["alpino"]
