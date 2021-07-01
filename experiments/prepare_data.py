import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from wasabi import msg

from experiments.featurizer import featurize
from experiments.iob_fmt import get_iob


class Corpus:
    def __init__(self, corpus_path: Path) -> None:
        self.zip: Path = corpus_path

        msg.info(f"Extracting corpus...")
        with TemporaryDirectory() as d:
            with ZipFile(self.zip) as z:
                z.extractall(d)
                self.Xy = self._get_Xy(d)

    def write_Xy(self, out_path: Path) -> None:
        with open(out_path, "w") as f:
            json.dump(self.Xy, f, indent=4, sort_keys=True)
        msg.good(f"Written Xy data at: {out_path}")

    def _get_Xy(self, data_dir) -> list:
        """Get data features and labels. Return these as an (X, y) tuple."""

        # Every sentence in the corpus will become a training example.
        # An example is a dict with three keys: X, y and doc_id.
        Xy = []

        msg.info("Featurizing data...")
        for doc_dir in Path(data_dir).iterdir():
            doc_id = doc_dir.stem

            # Extract X features.
            X_instance: dict = featurize(
                dnaf=doc_dir / "dnaf.json",
                lets=doc_dir / "lets.csv",
                alpino=doc_dir / "alpino",
            )

            # Extract y labels (IOB sequences).
            y_instance: dict = get_iob(dnaf=doc_dir / "dnaf.json")

            Xy.append({"X": X_instance, "y": y_instance, "doc_id": doc_id})

        return Xy
