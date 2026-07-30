# BiHyPE: Binary-based Hybrid Positional Encoding for Time-series Analysis - [Baseline: ConvTran]

[![Python 3.6 (Legacy)](https://img.shields.io/badge/Python-3.6-yellow?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch >=1.4.0](https://img.shields.io/badge/PyTorch-%3E%3D1.4.0%2C_%3C%3D1.10.2-red?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## Baseline Overview
This repository provides an implementation/reproduction of the baseline **[MEMTO]** based on the original paper:
* **Paper Title:** *[Improving Position Encoding of Transformers for Multivariate Time Series Classification]*
* **Authors:** *[Foumani, Navid Mohammadi, et al]*
* **Source / Links:** [https://arxiv.org/abs/2305.16642] | [https://github.com/navidfoumani/convtran]
* **Role in Research:** Serves as a benchmark Transformer-based baseline to evaluate against our proposed method [**BiHyPE**].

## Implementation Details

1. Install Python, install requirements: `pip install -r requirements.txt`. 
2. You can obtain five benchmarks from [Google Drive](https://drive.google.com/drive/folders/1kQXlX2C_Bp9Zy9r5UI1M37umWT4VIpLp?usp=sharing). **All the datasets are well pre-processed by the authors of ConvTran**. You can download this data, into the root folder (valid dataset directory example: ../Dataset/UEA/).
3. Checkpoint: We implement six scenarios to compare BiHyPE with existing Positional Encodings, including tAPE, eRPE, LAPE, LRPE, ConvTran(tAPE + eRPE) and **BiHyPE**, as summarized in [Checkpoints_PE_Comparison](https://drive.google.com/drive/folders/1FFTjWRNXupvVPqg_KtATKlo1stivTdF4?usp=sharing).
4. Train and evaluate. To reproduce the positional encoding experiments on the UEA dataset (a set of various datasets):

| PE module | `Fix_pos_encode` | `Rel_pos_encode` | Epochs | Command |
| :--- | :--- | :--- | :---: | :--- |
| **tAPE** | `tAPE` | `None` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode tAPE --Rel_pos_encode None` |
| **eRPE** | `None` | `eRPE` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode None --Rel_pos_encode eRPE` |
| **Learnable** | `Learn` | `None` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode Learn --Rel_pos_encode None` |
| **Vector** | `None` | `Vector` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode None --Rel_pos_encode Vector` |
| **ConvTran(tAPE + eRPE)** | `tAPE` | `eRPE` | 1500 | `python main.py --epochs 1500 --data_path Dataset/UEA/ --Fix_pos_encode tAPE --Rel_pos_encode eRPE` |
| **Proposed Method** | `Proposed` | `None` | 1000 | `python main.py --epochs 1000 --data_path Dataset/UEA/ --Fix_pos_encode Proposed --Rel_pos_encode None` |
