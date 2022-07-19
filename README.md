# README

This repo bundles experimental code for the EventDNA LRE paper.

## Usage

`main.py` is a CLI file. Simply run it with `python main.py` to start training and evaluation. `python main.py --help` details some options; in particular, `--test` can be used to run a limited training routine taking very little time to complete.

The results will be output to a timestamped directory under `./output`. An `info.log` file tracks parameters and intermediary results.
