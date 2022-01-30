from experiments.corpus import (
    read_files,
    get_examples,
    check_extract,
    DATA_EXTRACTED,
    DATA_ZIPPED,
    get_featurized_sents,
)
from pathlib import Path


def test_read_files():
    check_extract(DATA_ZIPPED, DATA_EXTRACTED)
    file_sets = read_files(DATA_EXTRACTED)
    assert len(list(file_sets)) == 1773


def test_featurized_examples():
    examples = list(get_examples(DATA_EXTRACTED))
    assert len(examples) > 0
    print(len(examples))
    print(examples[0])


def test_get_featurized_sents():
    case_dir = Path("tests/data/in/doc_dir")
    examples = get_featurized_sents(
        "id",
        case_dir / "dnaf.json",
        case_dir / "lets.tsv",
        case_dir / "alpino",
    )
    for ex in examples:
        print(ex.id)
        print(ex.alpino_tree._head_indices)
