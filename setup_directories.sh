#!/bin/bash

# Create the main data folder
mkdir -p data


mkdir -p data/filtering_framework/data

# Create the subfolders inside the data folder
mkdir -p data/wav_data
mkdir -p data/manifest_data

mkdir -p data/manifest_data/finetuning

mkdir -p data/manifest_data/finetuning/raw
mkdir -p data/manifest_data/finetuning/preprocessed