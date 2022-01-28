from experiments.corpus import (
    read_files,
    get_examples,
    check_extract,
    DATA_EXTRACTED,
    DATA_ZIPPED,
)


def test_read_files():
    check_extract(DATA_ZIPPED, DATA_EXTRACTED)
    file_sets = read_files(DATA_EXTRACTED)
    assert len(list(file_sets)) == 1773


def test_featurized_examples():
    examples = list(get_examples(DATA_EXTRACTED))
    assert len(examples) > 0
    print(len(examples))
    print(examples[0])
