# D3M: Reliably detecting model failures in deployment without labels

Implementation of the D3M algorithm for post-deployment deterioration monitoring. D3M provably monitors model deterioration at deployment time. D3M:

- Flags deteriorating shifts in the unsupervised deployment data distribution
- Resists flagging non-deteriorating shifts, unlike classical OOD detection leveraging distances and/or metrics between data distributions. 

## Environment configuration and installing dependencies

First, ``cd`` into the working directory:

```
cd d3m/
```

In this example, we use ``anaconda3``. Create a new d3m environment:
``` 
conda create -n d3m python=3.11

conda activate d3m

pip install -r requirements.txt
```

## Dataset installation and configurations

This repository uses Hydra (``hydra-core`` on PyPI) in order to streamline hyperparameter loading. Best hyperparameter presets are provided in ``d3m/experiments/configs/``. In the paper, the test size $m$ corresponds to the parameter ``d3m.data_sample_size`` in our configuration, available to all experiments. 

The Camelyon17 dataset and CIFAR-10 dataset will be automatically installed the first time ``experiments/run.py`` is run using their corresponding configs. To install the UCI Heart Disease preprocessed dataset as well as the CIFAR-10.1 dataset, run:
```
chmod +x install_datasets.sh
./install_datasets.sh
```
Warning: the Camelyon17 dataset takes a long time to download and process. 

## Reproduction of results in the manuscript

All experiments are run as follows. For example, to run the UCI Heart Disease with $m=100$,  activate the ``d3m`` environment and run:

```
python experiments/run.py --config-name=uci_best d3m.data_sample_size=100
```

Optional arguments:
- ``wandb_enabled``: enabled by default, setting it to False disables live-logging with ``wandb``. Will require logging in with a ``wandb`` account.
- ``self_log``: enabled by default, setting it to False, ``run.py`` will not write results into ``results/<name_of_dataset>_<d3m.data_sample_size>.csv``. 
- ``seed``: set to $57$ by default. 

For a given dataset, run ``experiments/run.py`` with the same configuration and the same ``d3m.data_sample_size``, incrementing the seed by $1$ for each independent run and enable ``self_log``. We recommend doing 20-50 runs at low $m$ since at these test sizes, dataset sampling variance results in ID FPRs beyond our tolerance. In our paper, we discard runs with ID FPR below $0.10$ as in a real-world scenario, the ML practitioner would be sweeping D3M over random hyperparameters until random independent initializations of D3M yield ID FPRs below their tolerance level anyway. 

## How D3M works and tutorials

In short, training a ``D3MMonitor`` consists of a three steps.

1. Train the base model of the monitor on I.D. training data
2. With a held-out set of I.D. validation data, train the distribution of I.D. disagreement rates (Phi) of the monitor
3. Deploy the base model and monitor by periodically running ``d3m_test`` on batches of unsupervised deployment data 

When ``d3m_test`` returns ``True``, the monitor recognizes that the base model may severely underperform on the unsupervised deployment data. This is the cue for ML practitioners to inspect the problem further and consider further measures such as adapting and retraining. 

For a full tutorial on how to deploy ``d3m`` to monitor a downstream task, consider running the guidebook ``tutorials/classification.ipynb`` where we train a ``D3MBayesianMonitor`` to monitor an induced deteriorating shift on the UCI Heart Disease dataset. 



