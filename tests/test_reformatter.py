from pathlib import Path

from data_formatting.json2dnaf import json2dnaf


def test_json2dnaf():
    test_data_dir = Path("tests/data")
    example_json = test_data_dir/ "in/bankdirecteur.json"
    outpath = test_data_dir/ "out/bankdirecteur_dnaf.json"
    assert example_json.exists()
    json2dnaf(example_json, outpath, "example_id")
        
