# DePLOI Experiment Materials
This repository contains all code and scripts required to run the experiments used to test DePLOI, a system to synthesize and audit access control. This repo is organized as follows:

1. init_exp: this folder contains the script for running an initial experiment, showing that C3+DDL outperforms Naive and Rolled prompting.
2. deploi_eval: this folder contains the scripts for running evaluations of DePLOI's synthesis and auditing functions.
3. benchgen: this folder contains the scripts to generate IBACBench, a benchmark for testing access control synthesis and auditing when the input abstractions are role hierarchy lists (RHLs), natural language access control matrices (NLACMs), and Temporal Access Control Matrices (TACMs).

## Steps to Run
1. Download the BIRD, Spider, and Dr. Spider benchmarks.
2. Download the Amazon Access dataset.
3. Set your OpenAI API key as an environment variable.
4. Run each script in benchgen/ to generate RHLs, NLACMs, and TACMs.
5. Run each script in deploi_eval to evaluate DePLOI's synthesis and auditing performance on the generated RHLs, NLACMs, and TACMs.
6. (Optional) To run the initial experiment, simply run init_exp/initial_exp.py.
