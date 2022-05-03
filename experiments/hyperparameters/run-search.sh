#!/bin/sh
# This script starts a batch job on Alvis (https://www.c3se.chalmers.se/about/Alvis/)
sbatch --gpus=V100:1 --job-name=search --time=150:00:00 search-job.sh
