import logging
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from experiments.errors import FeaturizationError
from experiments.evaluation.alpino import AlpinoTree
from experiments.featurizer import featurize
from experiments.iob_fmt import get_iob

logger = logging.getLogger(__name__)

ZIP = Path("data/EventDNA_dnaf_corpus.zip")
DATA_DIR = Path("extracted")


@dataclass
class Example:
    id: str
    x: dict
    y: dict
    alpino_tree: AlpinoTree


def get_examples(main_events_only: bool):
    """Read and yield features and labels from a data dir.
    Every sentence in the corpus will become a training example.
    """

    examples = []

    # Extract the zipped data if needed.
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()
        logger.info(f"Extracting corpus to {DATA_DIR}")
        with ZipFile(ZIP) as z:
            z.extractall(DATA_DIR)
    else:
        if len(list(DATA_DIR.iterdir())) == 0:
            raise ValueError(
                f"No data files found in {DATA_DIR.resolve()}. Delete this dir to allow unzipping."
            )
        logger.info(f"Using existing data dir: {DATA_DIR}")

    # Read in the files and extract features.
    for doc_dir in DATA_DIR.iterdir():

        try:
            sentence_examples = _get_featurized_sents(
                doc_id=doc_dir.stem,
                dnaf=doc_dir / "dnaf.json",
                lets=doc_dir / "lets.csv",
                alpino_dir=doc_dir / "alpino",
                main_events_only=main_events_only,
            )
        except FeaturizationError as e:
            logger.error(e)
            continue

        examples.extend(sentence_examples)

    logger.info(f"Loaded {len(examples)} examples.")

    return examples


def _get_featurized_sents(
    doc_id: str, dnaf: Path, lets: Path, alpino_dir: Path, main_events_only
):
    """Return examples from a single document directory."""

    examples = []

    # Extract X and y features.
    x_sents = list(featurize(dnaf, lets))
    y_sents = list(get_iob(dnaf, main_events_only))

    for (x_sent_id, x_sent), (y_sent_id, y_sent) in zip(x_sents, y_sents):

        # Check the correct sentences are matched.
        if x_sent_id != y_sent_id:
            raise FeaturizationError("Sentence ids do not match.")

        # Check the n of tokens in each sentence is the same.
        if not len(x_sent) == len(y_sent):
            t = [d["token"] for d in x_sent]
            m = f"{doc_id}: number of tokens in x and y don't match.\n\t-> {t} != {y_sent}"
            raise FeaturizationError(m)

        # Parse and attach the alpino tree.
        sentence_number = x_sent_id.split("_")[-1]
        alp = alpino_dir / f"{sentence_number}.xml"
        tree = AlpinoTree(alpino_file=alp, restricted_mode=True)

        ex_id = f"{doc_id}_{x_sent_id}"
        example = Example(id=ex_id, x=x_sent, y=y_sent, alpino_tree=tree)
        examples.append(example)

    return examples
