# CaTT: Learning Time Series Representation from Temporal Consistency

This repository contains the Pytorch implementation of CaTT, a method for unsupervised representation learning of time series data. The CaTT method demonstrates the ability to learn semantically meaningful representations off the shelf and outperforms previous time series representation learning methods in downstream linear evaluation.

## Data


To use **CaTT** and the baseline models, you will need access to relevant time-series datasets. The following datasets are used in this repository:

- [**HARTH**](https://archive.ics.uci.edu/dataset/779/harth): This is a human activity recognition (HAR) dataset that contains recordings from 22 participants, each wearing two 3-axial Axivity AX3 accelerometers for approximately 2 hours in a free-living setting at a sampling rate of 50Hz.

- [**SleepEEG**](https://www.physionet.org/content/sleep-edfx/1.0.0/): This dataset contains 153 whole-night electroencephalography (EEG) sleep recordings from 82 healthy subjects, sampled at 100 Hz.

- [**ECG**](https://physionet.org/content/afdb/1.0.0/): We use the MIT-BIH Atrial Fibrillation dataset, which includes 25 long-term electrocardiogram (ECG) recordings of human subjects with atrial fibrillation, each with a duration of 10 hours.

Make sure to place the dataset in the appropriate directory (e.g., `datasets/sleepeeg`) as specified in the configuration files.

For the forecasting tasks use the following datasets.

- [**3 ETT**](https://github.com/zhouhaoyi/ETDataset): This datasets should be placed at datasets/ETTh1.csv, datasets/ETTh2.csv and datasets/ETTm1.csv.

- [**Weather**](https://github.com/zhouhaoyi/Informer2020): This dataset link is from the Informer repository and should be places in datasets/weather.csv.


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

![t-SNE Visualization](./images/tsne.png?raw=true "Title")


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

Please check out the original repositories for more details.