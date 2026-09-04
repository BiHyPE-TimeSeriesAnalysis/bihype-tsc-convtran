# BiHyPE: Binary-based Hybrid Positional Encoding for Time-series Analysis - [Baseline: ConvTran]

[![Python 3.6 (Legacy)](https://img.shields.io/badge/Python-3.6-yellow?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch >=1.4.0](https://img.shields.io/badge/PyTorch-%3E%3D1.4.0%2C_%3C%3D1.10.2-red?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## Description
This repository provides an implementation/reproduction of the baseline **[ConvTran]**, based on the original paper "Improving Position Encoding of Transformers for Multivariate Time Series Classification" (Foumani, Navid Mohammadi, et al). It serves as a benchmark Transformer-based baseline for evaluating our proposed method **BiHyPE**, on the Time-series Classification (TSC) task.
* **Paper:** [https://arxiv.org/abs/2305.16642]
* **Original Source code:** [https://github.com/navidfoumani/convtran]
* **Role in Research:** Benchmark baseline compared against the BiHyPE-integrated version of the same model, without modifying core architecture of original model.

## Dataset Information
 
Benchmarks from the UEA time-series classification archive are used, pre-processed by the original ConvTran authors.
 
- **Download:** [Google Drive - TSC Datasets](https://drive.google.com/drive/folders/1kQXlX2C_Bp9Zy9r5UI1M37umWT4VIpLp?usp=sharing)
- **Directory layout:** after downloading, place the data in the root folder, e.g. `../Dataset/UEA/`.
---

## Code Information
 
- **Baseline model code:** implementation of the ConvTran architecture, configurable with different fixed and relative positional encoding modules.
- **Positional encoding comparison:** the codebase supports six positional encoding scenarios for direct comparison against BiHyPE — **tAPE**, **eRPE**, **LAPE** (Learnable), **LRPE** (Vector), **ConvTran (tAPE + eRPE)**, and **BiHyPE (Proposed)**.
- **Checkpoints:** checkpoints for all six positional encoding scenarios above are provided at [Checkpoints_PE_Comparison](https://drive.google.com/drive/folders/1FFTjWRNXupvVPqg_KtATKlo1stivTdF4?usp=sharing).
---

## Usage Instructions

1. **Install dependencies:**
```bash
   pip install -r requirements.txt
```
2. **Download the dataset** from the link in [Dataset Information](#dataset-information) and place it under `../Dataset/UEA/` as described.
3. **(Optional) Download checkpoints** from the link in [Code Information](#code-information) if you want to evaluate without retraining.
4. **Training/evaluation** for each positional encoding scenario on the UEA dataset using the corresponding command:

| PE module | `Fix_pos_encode` | `Rel_pos_encode` | Epochs | Command |
| :--- | :--- | :--- | :---: | :--- |
| **tAPE** | `tAPE` | `None` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode tAPE --Rel_pos_encode None` |
| **eRPE** | `None` | `eRPE` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode None --Rel_pos_encode eRPE` |
| **Learnable** | `Learn` | `None` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode Learn --Rel_pos_encode None` |
| **Vector** | `None` | `Vector` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode None --Rel_pos_encode Vector` |
| **ConvTran(tAPE + eRPE)** | `tAPE` | `eRPE` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode tAPE --Rel_pos_encode eRPE` |
| **Proposed Method** | `Proposed` | `None` | 1000 | `python main.py --epochs 1000 --data_path Dataset/UEA/ --Fix_pos_encode Proposed --Rel_pos_encode None` |

## Requirements
 
- Python 3.6
- PyTorch >= 1.4.0, <= 1.10.2
- NumPy, Pandas
- scikit-learn
Install dependencies with:
 
```bash
pip install -r requirements.txt
```
