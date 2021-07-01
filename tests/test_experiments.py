from pathlib import Path

from experiments.prepare_data import Corpus


def test_dataprepper():
    case = Path("assets/eventdna_data/EventDNA_dnaf_corpus.zip")
    out_dir = Path("tests/data/out")

    prepper = Corpus(case)
    prepper.write_Xy(out_dir/"Xy.json")
