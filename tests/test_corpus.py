from experiments.corpus import Corpus
from pathlib import Path

def test_dataprepper():
    corpus_path = Path(
        "/mnt/c/Users/camie/OneDrive/LRE paper/data/EventDNA_dnaf_corpus.zip"
    )
    # out_path = Path("tests/data/out") / "Xy.json"

    corpus = Corpus(corpus_path)
    examples = list(corpus.examples())
    assert len(examples) > 0
    print(len(examples))

    # corpus.write_Xy(out_path)