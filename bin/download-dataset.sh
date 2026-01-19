#!/bin/bash

set -eu

cd $(dirname $0)/.. || exit 1

echo "Downloading dataset 'Garbage Classification' from Kaggle..."

temp_directory=$(mktemp -d)

curl -L -o "${temp_directory}/garbage-classification.zip" \
  https://www.kaggle.com/api/v1/datasets/download/hassnainzaidi/garbage-classification

unzip "${temp_directory}/garbage-classification.zip" -d "${temp_directory}"

# Dataset has been downloaded, let's create and ensure that no data is there.
mkdir -p data
rm -rf data/*

mv "${temp_directory}/Garbage classification/test" data/test
mv "${temp_directory}/Garbage classification/train" data/train
mv "${temp_directory}/Garbage classification/val" data/val

rm -rf "${temp_directory}"