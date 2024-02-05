# ml4graphs-vu
This project implements Graph Attention Networks (GAT) for machine learning from graphs, specifically on the Cora dataset.

## Introduction

Graph Attention Networks (GAT) is a powerful deep learning model that can effectively learn from graph-structured data. It leverages attention mechanisms to capture the importance of neighboring nodes when making predictions on a target node. The Cora dataset is a popular benchmark dataset for graph-based machine learning tasks.

## Features

- Implementation of Graph Attention Networks (GAT) using [insert library/framework here]
- Training and evaluation on the Cora dataset
- Support for various GAT configurations, such as different number of attention heads and hidden units
- Preprocessing utilities for handling graph data

# Installation

1. Clone the repository:

```bash
git clone https://github.com/cheremyp/ml4graphs-vu.git
cd ml4graphs-vu
```
2. Install the necesary packagesL 
You can install the necessary packages using pip. We recommend creating a virtual environment before installing the packages.

Option 1: Using conda
First, create a new conda environment

```
conda create -n ml4graphs python=3.8
conda activate ml4graphs
```

Then, install the packages:

```
pip install -r requirements.txt
```

Option 2: Using Poetry

```
curl -sSL https://install.python-poetry.org | python -
```

Then, install the packages:

```
poetry install 
```

# Usage


## Train

## Predict 



# Acknowledgement 

The original GAT paper: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
The GATv2 paper: [Graph Attention Networks](https://arxiv.org/abs/2106.05238)
The Cora dataset: [A Relational Database of Predicted Regulatory Interactions for Saccharomyces cerevisiae](https://academic.oup.com/bioinformatics/article/24/21/2498/196825)

