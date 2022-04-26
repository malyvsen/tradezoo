#!/bin/sh
# This script starts a batch job on Alvis (https://www.c3se.chalmers.se/about/Alvis/)
sbatch --gpus=A40:1 --job-name=hyperparameters --time=150:00:00 hyperparameters.sh
