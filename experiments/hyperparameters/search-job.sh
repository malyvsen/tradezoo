#!/bin/sh
# This script is to be run by run-search.sh
module load Python/3.9.6-GCCcore-11.2.0
poetry run python ./search.py
