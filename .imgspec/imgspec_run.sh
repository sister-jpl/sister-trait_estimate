#!/bin/bash

imgspec_dir=$(cd "$(dirname "$0")" ; pwd -P)
pge_dir=$(dirname ${imgspec_dir})

source activate sister

mkdir output

for a in `ls -1 input/*.tar.gz`; do tar -xzvf $a -C input; done

rfl_file=$(ls input/*/*rfl)
file_base=$(basename $rfl_file)

if [[ $file_base == f* ]]; then
    output_prefix=$(echo $file_base | cut -c1-16)
elif [[ $file_base == ang* ]]; then
    output_prefix=$(echo $file_base | cut -c1-18)
elif [[ $file_base == PRS* ]]; then
    output_prefix=$(echo $file_base | cut -c1-38)
elif [[ $file_base == DESIS* ]]; then
    output_prefix=$(echo $file_base | cut -c1-44)
fi

out_dir=${output_prefix}_traits
mkdir output/${out_dir}

python ${pge_dir}/trait_estimate.py $rfl_file output/${out_dir} --models ${pge_dir}/models/*.json

cd output
tar -czvf ${out_dir}.tar.gz $out_dir
rm -r $out_dir
