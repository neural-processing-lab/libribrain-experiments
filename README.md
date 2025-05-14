# LibriBrain Experiments

This repository provides experimental setups and code accompanying the figures presented in the publication:

> **LibriBrain: Over 50 Hours of Within-Subject MEG to Improve Speech Decoding Methods at Scale**

---
## Installation

You can install the package in editable/development mode so that any local changes are immediately reflected:

```bash
pip install -e .
```

## Experiment Configuration

Configuration files for the phoneme decoding experiments detailed in our paper can be found in:

```
libribrain_experiments/phoneme/configs
```

**Important Configuration Notes:**

Before running the project, make sure to update the configuration files with the correct local paths:

- **`data_path`**: Specify the paths for your training, validation, and testing datasets.
- **`output_path`**: Set this to the directory where output results (e.g., logs, predictions) will be saved.
- **`checkpoint_path`**: Define the location where model checkpoints should be stored.

---

## Running an Experiment

Use the following command format to execute an experiment:

```bash
python libribrain_experiments/hpo.py \
    --config=libribrain_experiments/configs/phoneme/<config-name>/base-config.yaml \
    --search-space=libribrain_experiments/configs/phoneme/<config-name>/search-space.yaml \
    --run-name=<run-name> \
    --run-index=<run-id>
```

Replace `<config-name>`, `<run-name>`, and `<run-id>` with your own values—`<config-name>` selects which experiment folder under `libribrain_experiments/phoneme/configs`, `<run-name>` is the Weights & Biases run name, and `<run-id>` is the hyperparameter/seed configuration index.

---