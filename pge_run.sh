#!/bin/bash

source activate python38

pge_dir=$(cd "$(dirname "$0")" ; pwd -P)

echo "Creating runconfig"
python ${pge_dir}/generate_runconfig.py inputs.json

echo "Running L2B Vegetation Biochemistry PGE"
python ${pge_dir}/trait_estimate.py runconfig.json
