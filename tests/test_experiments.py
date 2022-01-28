from pathlib import Path

from experiments.corpus import Corpus


def test_dataprepper():
    corpus_path = Path(
        "/mnt/c/Users/camie/OneDrive/LRE paper/data/EventDNA_dnaf_corpus.zip"
    )
    out_path = Path("tests/data/out") / "Xy.json"

    corpus = Corpus(corpus_path)
    corpus.write_Xy(out_path)
