<h1 align="center">CaTT</h1>
<h2 align="center">Contrast All The Time: Learning Time Series Representation from Temporal Consistency</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2410.15416">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2410.15416-b31b1b.svg">
  </a>
  <img alt="License" src="https://img.shields.io/github/license/sfi-norwai/CaTT">
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/sfi-norwai/CaTT">
  <img alt="Stars" src="https://img.shields.io/github/stars/sfi-norwai/CaTT?style=social">
</p>

<p align="center">
  <a href="https://ecai2025.eu/">
    <img alt="ECAI 2025 Accepted" src="https://img.shields.io/badge/Accepted%20at-ECAI%202025-blueviolet">
  </a>
</p>

This repository contains the official Pytorch implementation of the "[**Contrast All The Time (CaTT)**](https://arxiv.org/abs/2410.15416)" paper (ECAI 2025), a new approach to unsupervised contrastive learning for time series, which takes advantage of dynamics between temporally similar moments more efficiently and effectively than existing methods..


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sfi-norwai/CaTT.git
    cd contrastive-learning
    ```

2. Install the required dependencies:
    ```bash
    conda create -n catt-env python=3.11
    conda activate catt-env
    pip install -r requirements.txt
    ```

## Data


To use **CaTT** and the baseline models, you will need access to relevant time-series datasets. The following datasets are used in this repository:

- [**HARTH**](https://archive.ics.uci.edu/dataset/779/harth): This is a human activity recognition (HAR) dataset that contains recordings from 22 participants, each wearing two 3-axial Axivity AX3 accelerometers for approximately 2 hours in a free-living setting at a sampling rate of 50Hz.

- [**SleepEEG**](https://www.physionet.org/content/sleep-edfx/1.0.0/): This dataset contains 153 whole-night electroencephalography (EEG) sleep recordings from 82 healthy subjects, sampled at 100 Hz.

- [**ECG**](https://physionet.org/content/afdb/1.0.0/): We use the MIT-BIH Atrial Fibrillation dataset, which includes 25 long-term electrocardiogram (ECG) recordings of human subjects with atrial fibrillation, each with a duration of 10 hours.

Make sure to place the dataset in the appropriate directory (e.g., `datasets/sleepeeg`) as specified in the configuration files.

For the forecasting tasks use the following datasets.

- [**3 ETT**](https://github.com/zhouhaoyi/ETDataset): This datasets should be placed at datasets/ETTh1.csv, datasets/ETTh2.csv and datasets/ETTm1.csv.

- [**Weather**](https://github.com/zhouhaoyi/Informer2020): This dataset link is from the Informer repository and should be places in datasets/weather.csv.

Anomaly detection code and datasets are adapted from the [TSB-AD repository](https://github.com/TheDatumOrg/TSB-AD).



## Usage

### Unsupervised pretraining of CaTT

To pretrain the CaTT model for classification, use the following command:

```bash
python pretrain.py <model> <dataset> -p <configs/harthconfig.yml> -s < > --evaluate < >

```
- `model` specifies the model to train.
- `dataset` specifies the dataset directory.
- `-p` specifies the configuration file.
- `-s` sets the seed for reproducibility.
- `--evaluate` define the task to perform.

### Example
For example, to pretrain CaTT model on the harth dataset with a seed of 1 and evaluate of supervised classification, run:
```bash
python pretrain.py CaTT harth -p configs/harthconfig.yml -s 1 --evaluate supervised
```
Check the scripts/ directory for complete list of training scripts for all tasks in the paper as well as the different seeds used for reproducibility.

### Running Baseline Models
To compare CaTT against the baseline models, you can use similar commands to pretrain the baselines. For example, to pretrain the CoST baseline on SleepEEG:

```bash
python pretrain.py CoST sleepeeg -p configs/sleepconfig.yml -s 1 --evaluate supervised
```

## Visualizations

The figure below shows a t-SNE plot of the learned representation from all baselines on all three datasets.

![t-SNE Visualization](./images/CaTT_embeddings.png?raw=true "Title")


## Acknowledgements

This repository provides reimplementations of several baselines for time-series representation learning using some parts of the codes provided by the following  works:

- [**MF-CLR**](https://github.com/duanjufang/MF-CLR): Multi-Frequency Contrastive Learning Representation for Time Series.

- [**TS2Vec**](https://github.com/zhihanyue/ts2vec): Towards Universal Representation of Time Series.

- [**Soft**](https://github.com/seunghan96/softclt?tab=readme-ov-file): Soft Contrastive Learning for Time Series.
- [**TimeDRL**](https://github.com/blacksnail789521/TimeDRL): Disentangled Representation Learning for Multivariate Time-Series.

- [**SimMTM**](https://github.com/thuml/SimMTM): A Simple Pre-Training Framework for Masked Time-Series Modeling.

- [**CoST**](https://github.com/salesforce/CoST): Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting.
- [**InfoTS**](https://github.com/chengw07/InfoTS): Time Series Contrastive Learning with Information-Aware Augmentations.
- [**TNC**](https://github.com/sanatonek/TNC_representation_learning): Unsupervised Representation Learning for TimeSeries with Temporal Neighborhood Coding.
- [**SelfPAB**](https://github.com/ntnu-ai-lab/SelfPAB): Self-supervised learning with randomized cross-sensor masked reconstruction for human activity recognition.
- [**TSB-AD**](https://github.com/TheDatumOrg/TSB-AD): A Benchmark for Time Series Anomaly Detection.

Please check out the original repositories for more details.

## Citations

If you use **CaTT** in your research, please consider citing it as follows:

```bibtex
@misc{catt2025,
      title={Contrast All The Time: Learning Time Series Representation from Temporal Consistency}, 
      author={Abdul-Kazeem Shamba and Kerstin Bach and Gavin Taylor},
      year={2025},
      eprint={2410.15416},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.15416}, 
}
