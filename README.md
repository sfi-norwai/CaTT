# DynaCL: Dynamic Contrastive Learning for Time-Series Representation

This repository contains the Pytorch implementation of [**DynaCL**](https://arxiv.com), a method for unsupervised representation learning of time series data. The DynaCL method demonstrates the ability to learn semantically meaningful representations off the shelf and outperforms previous time series representation learning methods in downstream linear evaluation.

![DynaCL Framework](./images/dynacl.png?raw=true "Title")


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sfi-norwai/contrastive-learning.git
    cd contrastive-learning
    ```

2. Install the required dependencies:
    ```bash
    conda create -n dynacl-env python=3.11
    conda activate dynacl-env
    pip install -r requirements.txt
    ```

## Data


To use **DynaCL** and the baseline models, you will need access to relevant time-series datasets. The following datasets are commonly used in this repository:

- [**HARTH**](https://archive.ics.uci.edu/dataset/779/harth): This is a human activity recognition (HAR) dataset that contains recordings from 22 participants, each wearing two 3-axial Axivity AX3 accelerometers for approximately 2 hours in a free-living setting at a sampling rate of 50Hz.

- [**SleepEEG**](https://www.physionet.org/content/sleep-edfx/1.0.0/): This dataset contains 153 whole-night electroencephalography (EEG) sleep recordings from 82 healthy subjects, sampled at 100 Hz.

- [**ECG**](https://physionet.org/content/afdb/1.0.0/): We use the MIT-BIH Atrial Fibrillation dataset, which includes 25 long-term electrocardiogram (ECG) recordings of human subjects with atrial fibrillation, each with a duration of 10 hours.

Make sure to place the dataset in the appropriate directory (e.g., `data/dataset_name`) as specified in the configuration files.


## Usage

### Running DynaCL

To run the DynaCL model, use the following command:

```bash
python models/dynacl.py -p configs/dynacl_config.yml -d data/dataset_name -s seed
```

- `-p` specifies the configuration file.
- `-d` specifies the dataset directory.
- `-s` sets the random seed for reproducibility.

### Example
```bash
python models/dynacl.py -p configs/sleepconfig.yml -d data/sleepeeg -s 42
```

### Running Baseline Models
To compare DynaCL against the baseline models, you can use similar commands. For example, to run the CoST baseline:

```bash
python models/cost.py -p configs/sleepconfig.yml -d data/sleepeeg -s 42
```
## Visualizations

The figure below shows a t-SNE plot of the learned representation from all baselines on all three datasets.

![t-SNE Visualization](./images/tsne.png?raw=true "Title")



## Acknowledgements

This repository provides reimplementations of several baselines for time-series representation learning using some parts of the codes provided by the following  works:

- [**TS2Vec**](https://github.com/zhihanyue/ts2vec): Towards Universal Representation of Time Series.
- [**CoST**](https://github.com/salesforce/CoST): Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting.
- [**InfoTS**](https://github.com/chengw07/InfoTS): Time Series Contrastive Learning with Information-Aware Augmentations.
- [**CPC**](https://github.com/davidtellez/contrastive-predictive-coding): Contrastive Predictive coding for unsupervised learning of representations.
- [**TNC**](https://github.com/sanatonek/TNC_representation_learning): Time-series representation through neighborhood contrasting.
- [**SelfPAB**](https://github.com/ntnu-ai-lab/SelfPAB): Self-supervised learning with randomized cross-sensor masked reconstruction for human activity recognition.

Please check out the original repositories for more details.


## Citations

If you use **DynaCL** in your research, please consider citing it as follows:

```bibtex
@article{dynacl2024,
  title={DynaCL: Dynamic Contrastive Learning for Time-Series Representation},
  author={Your Name, et al.},
  journal={Conference/Journal Name},
  year={2024}
}
