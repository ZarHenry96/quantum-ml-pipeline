#!/usr/bin/bash

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Illegal number of parameters!"
    echo "Correct usage: ./run_experiments.sh experiments_dir datasets_dir [res_subdir_name]"
    echo "    experiments_dir = directory containing the .template configuration files for the experiments"
    echo "    datasets_dir = directory containing the .csv datasets and the .json class mapping files"
    echo "    res_subdir_name = name of the sub-directory of 'results' where the results will be stored (optional)"
    exit 0
fi

# Parameters
CONFIG_TEMPLATE_DIR="$(cd "$1" && pwd -P)"
DATASETS_DIR="$(cd "$2" && pwd -P)"

SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
if [ "$#" -eq 3 ]; then
    RES_SUBDIR_NAME="/$3"
else
    RES_SUBDIR_NAME=""
fi
RES_DIR="${SCRIPT_DIR}/results${RES_SUBDIR_NAME}"

K_FOLDS=5
K_FOLD_RANDOM_SEED=7

K_NN=5

NUM_PROCESSES=5

# Create a temporary directory to store experiment configuration files
TMP_CONFIG_DIR="${SCRIPT_DIR}/tmp"
mkdir -p ${TMP_CONFIG_DIR}

# Iterate over template files
for template_file in "${CONFIG_TEMPLATE_DIR}"/*.template; do
    # Get the experiment type from the template filename
    exp_type="${template_file##*/}"
    exp_type="${exp_type%.*}"

    # Iterate over datasets
    for dataset_file in "${DATASETS_DIR}"/*.csv; do
        # Set variables for dynamic binding
        class_mapping_file="${dataset_file%.*}_class_mapping.json"
        dataset_name="${dataset_file##*/}"
        exp_res_dir="${RES_DIR}/${dataset_name%.*}_${exp_type}"
        exp_config_file="${TMP_CONFIG_DIR}/${dataset_name%.*}_${exp_type}.json"

        # Create the experiment configuration file starting from the template
        sed -e "s@\${dataset}@${dataset_file}@" -e "s@\${class_mapping}@${class_mapping_file}@" \
            -e "s@\${res_dir}@${exp_res_dir}@" -e "s@\${k_folds}@${K_FOLDS}@" \
            -e "s@\${k_fold_seed}@${K_FOLD_RANDOM_SEED}@" -e "s@\${k_nn}@${K_NN}@" \
            -e "s@\${num_processes}@${NUM_PROCESSES}@" "${template_file}" > "${exp_config_file}"

        # Run the experiment
        python main.py "${exp_config_file}"
        printf "\n\n\n"
    done
done

# Delete the temporary directory
rm -rf "${TMP_CONFIG_DIR}"