#!/bin/sh
# This script starts a batch job on Alvis (https://www.c3se.chalmers.se/about/Alvis/)
module load GCCcore/11.2.0 Python/3.9.6
poetry run python ./hyperparameters.py
