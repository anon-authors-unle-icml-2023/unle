#!/usr/bin/env bash
mkdir -p .env

# Setup a conda-based package manager (mambaforge in that case) locally
curl -L  -o "./.env/Mambaforge-$(uname)-$(uname -m).sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash ".env/Mambaforge-$(uname)-$(uname -m).sh" -b -p ./.env/mambaforge

# Enabble conda within this shell
eval "$('.env/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# Install all dependencies to run the experiments within a conda virtual environment named `unle`
# Look for a --gpu flag. If it's there, use the GPU environment (environment.yml), otherwise use the CPU-only environment (environment-cpu.yml)
if [[ " $* " =~ " --gpu " ]]; then
    echo "Using GPU environment"
    .env/mambaforge/bin/mamba env create -f environment.yml
else
    echo "Using CPU environment"
    .env/mambaforge/bin/mamba env create -f environment-cpu.yml
fi

# Pass some julia environment variables to the environment
mkdir -p .env/mambaforge/envs/unle/etc/conda/activate.d
mkdir -p .env/mambaforge/envs/unle/etc/conda/deactivate.d
printf "export JULIA_SYSIMAGE_DIFFEQTORCH=$(pwd)/.env/.julia_sysimage_diffeqtorch.so" > .env/mambaforge/envs/unle/etc/conda/activate.d/env_vars.sh
printf "unset JULIA_SYSIMAGE_DIFFEQTORCH" > .env/mambaforge/envs/unle/etc/conda/deactivate.d/env_vars.sh

# Activate the `unle` virtual environment
conda activate unle

# Install the diffeqtorch package (used to simulate from the `lotka_volterra` model)
# This takes approximately 15 minutes to run
python -c "from diffeqtorch.install import install_and_test; install_and_test()"
