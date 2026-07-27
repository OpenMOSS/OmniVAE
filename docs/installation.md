# Installation

Install the CUDA/PyTorch stack that matches your machine first, then install the
three local packages:

```bash
cd /path/to/OmniVAE

pip install -e vae
pip install -e generation
pip install -e sync
```

For generation and evaluation, the original experiments used separate inference
and metric environments. The release launchers keep this convention:

```bash
export INFER_CONDA_ENV=dit
export METRIC_CONDA_ENV=aveval
```

If the current environment is already correct, set either variable to an empty
string before running the launcher.

## Release Assets

Download the HuggingFace asset bundle:

```bash
pip install -U huggingface_hub
huggingface-cli download OpenMOSS-Team/OmniVAE \
  --repo-type model \
  --local-dir /path/to/omnivae_release \
  --local-dir-use-symlinks False
```

Point the code to the downloaded directory:

```bash
export OMNIVAE_RELEASE_ROOT=/path/to/omnivae_release
export OPEN_SOURCE_ROOT="${OMNIVAE_RELEASE_ROOT}"
```

or:

```bash
source scripts/setup_release_root.sh /path/to/omnivae_release
```

The scripts also auto-detect nearby `open_source` directories, but setting
`OMNIVAE_RELEASE_ROOT` is the recommended reproducible setup.

## Distributed Jobs

The T2AV release launcher supports both local multi-GPU and already allocated
distributed jobs. In PET-style jobs, these variables are detected:

```bash
PET_NNODES
PET_NPROC_PER_NODE
PET_NODE_RANK
PET_MASTER_ADDR
PET_MASTER_PORT
```

The public scripts do not submit jobs through an internal scheduler; run them
inside the allocated job.
