from pathlib import Path

from data_formatting.json2dnaf import json2dnaf


def test_json2dnaf():
    test_data_dir = Path("tests/data")

    example_json = test_data_dir / "in/bankdirecteur.json"
    assert example_json.exists()

    outpath = test_data_dir / "out/bankdirecteur_dnaf.json"

    json2dnaf(example_json, outpath, "example_id")
