from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from loguru import logger

from experiments.errors import FeaturizationError
from experiments.featurizer import featurize
from experiments.iob_fmt import get_iob


DATA_ZIP = Path("data/eventdna_corpus_2020-13-02.zip")


@dataclass
class Example:
    id: str
    x: dict
    y: dict


class Corpus:

    def __init__(self, data_zip) -> None:

        # Extract the corpus data to `data_dir` if it does not already exist.
        self.data_dir = data_zip.parent / f"{data_zip}_extracted"
        if not self.data_dir.exists():
            self.data_dir.mkdir()
            logger.info(f"Extracting corpus to {self.data_dir}")
            with ZipFile(self.data_zip) as z:
                z.extractall(self.data_dir)
        else:
            logger.info(
                f"Found data dir: {self.data_dir}. Assuming it's ok to use it."
            )

    def examples(self) -> list:
        """Read and yield features and labels from a data dir."""

        # Every sentence in the corpus will become a training example.

        # The data in data_dir is given in the form of documents.
        # Transform these to the sentence level.
        # TODO adapt logic to work on sentence level.

        logger.info("Featurizing data...")
        for doc_dir in Path(self.data_dir).iterdir():
            try:
                examples = list(self._examples_from_dir(doc_dir))
            except FeaturizationError as e:
                logger.error(e)
            for example in examples:
                yield example

    @staticmethod
    def _examples_from_dir(doc_dir):
        """Stream examples from a single document directory."""

        dnaf = doc_dir / "dnaf.json"
        lets = doc_dir / "lets.csv"

        # Extract X and y features.
        x_sents = list(featurize(dnaf, lets))
        y_sents = list(get_iob(dnaf, main_events_only=True))

        for (x_id, x), (y_id, y) in zip(x_sents, y_sents):

            # Check the correct sentences are matched.
            if x_id != y_id:
                raise FeaturizationError("Sentence ids do not match.")

            # Check the n of tokens in each sentence is the same.
            if not len(x) == len(y):
                m = "{}: number of tokens in x and y don't match.\n{} != {}".format(
                    doc_dir.stem,
                    [d["token"] for d in x],
                    y,
                )
                raise FeaturizationError(m)

            ex_id = f"{doc_dir.stem}_{x_id}"
            yield Example(id=ex_id, x=x, y=y)


def 