import json
from pathlib import Path
from zipfile import ZipFile

from wasabi import msg

from experiments.featurizer import featurize
from experiments.iob_fmt import get_iob
from experiments.util import clean_dir


class DataPrepper:
    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path: Path = corpus_path

        self.corpus = Path("./temp")
        if not self.corpus.exists():
            self.corpus.mkdir()
        else:
            msg.warn(
                f"Corpus path at <{self.corpus}> already exists. Removing its contents."
            )

        # Read in and validate the corpus.
        # Note that the data is not read in memory, but stored in a temporary directory.
        msg.info(f"Extracting corpus to <{self.corpus}>...")
        with ZipFile(corpus_path) as z:
            z.extractall(self.corpus)

    def write_Xy(self, out_path: Path, info_out_path: Path):

        # Every sentence in the corpus will become a training example.
        X = []
        Y = []

        msg.info("Featurizing data...")
        for doc_dir in self.corpus.iterdir():
            doc_id = doc_dir.stem

            # Extract X features.
            x: dict = featurize(
                dnaf=doc_dir / "dnaf.json",
                lets=doc_dir / "lets.csv",
                alpino=doc_dir / "alpino",
            )
            x["id"] = doc_id
            X.append(x)

            # Extract y labels (IOB sequences).
            y: dict = get_iob(dnaf=doc_dir / "dnaf.json")
            y["id"] = doc_id
            Y.append(y)
