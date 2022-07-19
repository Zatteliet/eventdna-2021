# README

This repo bundles experimental code for the EventDNA LRE paper.

## Usage

`main.py` is a CLI file. The command has two main options: `train` and `eval`.

`python main.py train` will train CRF models in crossvalidation fashion. The resulting models as well as the dev datasets are saved to a timestamped dir under `./output`. Training parameters can be changed on the CLI, and there is a `--test` option to run a limited training routine quickly. Use `--help` to check the options.

To evaluate the resulting models, use `python main.py eval <timestamped_output_dir>`. A directory called `eval` will be created in the same dir, containing PRF evaluations per fold and averaged.
