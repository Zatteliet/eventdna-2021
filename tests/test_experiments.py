from pathlib import Path

from experiments.data_handler import Corpus


def test_dataprepper():
    case = Path("/mnt/c/Users/camie/OneDrive/LRE paper/data/EventDNA_dnaf_corpus.zip")
    out_dir = Path("tests/data/out")

    prepper = Corpus(case)
    prepper.write_Xy(out_dir/"Xy.json")
