from experiments import corpus
from pathlib import Path


def test_read_files():
    corpus.check_extract(corpus.DATA_ZIPPED, corpus.DATA_EXTRACTED)
    file_sets = corpus.read_files(corpus.DATA_EXTRACTED)
    assert len(list(file_sets)) == 1773


def test_featurized_examples():
    examples = list(corpus.get_examples(corpus.DATA_EXTRACTED))
    assert len(examples) > 0
    print(len(examples))
    print(examples[0])


def test_get_featurized_sents():
    case_dir = Path("tests/data/in/doc_dir")
    examples = corpus.get_featurized_sents(
        "id",
        case_dir / "dnaf.json",
        case_dir / "lets.tsv",
        case_dir / "alpino",
        main_events_only=True,
    )
    for ex in examples:
        print(ex.id)
        print(ex.alpino_tree._head_indices)
