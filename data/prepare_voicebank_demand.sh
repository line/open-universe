#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

output_dir="./data/voicebank_demand"
mkdir -p "${output_dir}"

# URLs for the dataset
# Refer to https://datashare.ed.ac.uk/handle/10283/2791
LINKS=(
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/trainset_28spk_txt.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"
)

#################################
# Download data
#################################
for link in "${LINKS[@]}"; do
    name=`basename "${link}"`
    stem="${name%.*}"
    if [ ! -d "${output_dir}/${stem}" ]; then
        wget --continue "${link}" -P "${output_dir}"
        unzip ${output_dir}/${name} -d ${output_dir}
    else
        echo "Already unzipped ${name}"
    fi
done

rm -rf ${output_dir}/__MACOSX

#################################
# Prepare 48 kHz data
#################################

dir_48k="${output_dir}/48k"

if [ ! -d "${dir_48k}" ]; then

    mkdir -p ${dir_48k}

    # train set
    cp -r "${output_dir}/clean_trainset_28spk_wav" "${dir_48k}/train/clean"
    mkdir -p ${dir_48k}/val/clean
    mv ${dir_48k}/train/clean/{p226,p287}_*.wav ${dir_48k}/val/clean

    # val set
    cp -r "${output_dir}/noisy_trainset_28spk_wav" "${dir_48k}/train/noisy"
    mkdir -p ${dir_48k}/val/noisy
    mv ${dir_48k}/train/noisy/{p226,p287}_*.wav ${dir_48k}/val/noisy

    # test set
    cp -r "${output_dir}/clean_testset_wav" "${dir_48k}/test/clean"
    cp -r "${output_dir}/noisy_testset_wav" "${dir_48k}/test/noisy"

else
    echo "Already prepared 48k data"
fi

#################################
# Prepare 16 kHz and 24 kHz data
#################################
if [ ! -d "${output_dir}/16k" ]; then
    python -m open_universe.prepare.resample ${output_dir}/48k ${output_dir}/16k --fs 16000
else
    echo "Already prepared 16k data"
fi
if [ ! -d "${output_dir}/24k" ]; then
    python -m open_universe.prepare.resample ${output_dir}/48k ${output_dir}/24k --fs 24000
else
    echo "Already prepared 24k data"
fi

for sr in 16k 24k 48k; do
    # test train set
    for sub in clean noisy; do
        path_train="${output_dir}/${sr}/train/${sub}"
        n_files_train=`find ${path_train} -name '*.wav' | wc -l`
        if [ ${n_files_train} -ne 10802 ]; then
            echo "Error: expected 10802 files in ${path_train}, found ${n_files_train}"
        fi

        path_val="${output_dir}/${sr}/val/${sub}"
        n_files_val=`find ${path_val} -name "*.wav" | wc -l`
        if [ ${n_files_val} -ne 770 ]; then
            echo "Error: expected 770 files in ${path_val}, found ${n_files_val}"
        fi

        path_test="${output_dir}/${sr}/test/${sub}"
        n_files_test=`find ${path_test} -name "*.wav" | wc -l`
        if [ ${n_files_test} -ne 824 ]; then
            echo "Error: expected 824 files in ${path_test}, found ${n_files_test}"
        fi
    done
done
