# README

This repo bundles experimental code for the EventDNA LRE paper.

## Usage

### Data

_Info on obtaining the dataset here._

The dataset should be extracted to `./extracted`. This should result in a dir with one dir per EventDNA document, such that each document dir has the structure:

```text
eventdna_1/
    dnaf.json
    lets.csv
    alpino/
        1.xml
        2.xml
        3.xml
        4.xml
```

`dnaf.json` is annotated data in a custom format. `lets.csv` contains syntactic information derived from the LETS analysis tool. The `alpino` folder contains parse trees for each sentence in the document, produced by the Alpino parser for dutch.

### Training and evaluation

`main.py` is a CLI file. The command has two main options: `train` and `eval`.

`python main.py train` will train CRF models in crossvalidation fashion. The resulting models as well as the dev datasets are saved to a timestamped dir under `./output`. Training parameters can be changed on the CLI, and there is a `--test` option to run a limited training routine quickly. Use `--help` to check the options.

To evaluate the resulting models, use `python main.py eval <timestamped_output_dir>`. A directory called `eval` will be created in the same dir, containing PRF evaluations per fold and averaged.
