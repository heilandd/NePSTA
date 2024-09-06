# NePSTA
NeuroPathology Spatial Transcriptomic Analysis

<img width="1385" alt="image" src="https://github.com/user-attachments/assets/2feeeba8-34da-4b16-9dc4-88a24f8022f2">


## Spatially resolved transcriptomics and graph-based deep-learning improve accuracy of routine CNS tumor diagnostics

Abstract:
The diagnostic landscape of brain tumors has recently evolved to integrate comprehensive molecular markers alongside traditional histopathological evaluation. Foremost, genome-wide DNA methylation profiling and next generation sequencing (NGS) has become a cornerstone in classifying Central Nervous System (CNS) tumors, as recognized by its inclusion into the 2021 WHO classification. Despite its diagnostic precision, a limiting requirement for NGS and methylation profiling is sufficient DNA quality and quantity which restricts its feasibility, especially in cases with small biopsy samples or low tumor cell content, both frequent challenges in specimen of diffusely growing CNS lesions. Addressing these challenges, we demonstrate a novel application, namely NePSTA (NeuroPathology Spatial Transcriptomic Analysis), which is capable of generating comprehensive morphological and molecular neuropathological diagnostics from single 5 µm tissue sections. Our framework employs 10x Visium spatial transcriptomics with graph neural networks for automated histological and molecular evaluations. Trained and evaluated across 130 patients with CNS malignancies and healthy donors across four medical centers, NePSTA integrates spatial gene expression data and inferred CNAs to predict tissue histology and methylation-based subclasses with high accuracy. Further, we demonstrate the ability to reconstruct immunohistochemistry and genotype profiling on single thin slides of minute tissue biopsies. Our approach has minimal tissue requirements, often inadequate for conventional molecular diagnostics, demonstrating the potential to transform neuropathological diagnostics and enhance tumor subtype identification with implications for fast and precise diagnostic work-up.

Manuscript will be avaiable soon!

# Graph Neural Networks: NePSTA

## Data Split

The graph neural network framework is part of the NePSTA project, which aims to implement deep learning strategies to explore spatially resolved multi-omics. To assess the performance of NePSTA ([GitHub – heilandd/NePSTA](https://github.com/heilandd/NePSTA)) and comparative methods, we conducted evaluations on our Visium dataset. The datasets were partitioned into training and evaluation subsets using the following procedure:

### Data Split

From 107 patients characterized by EPIC, each dataset was split into training and validation segments. In samples with multiple biopsies, the dataset was split by individual biopsy cores. For single-specimen datasets, we segmented the spots manually using the `createSegmentation` function from the `SPATA2` package.

### Training Dataset Construction

We created the training dataset using the PyTorch Geometric library, selecting up to 500 subgraphs from the training split. Clinical attributes, such as tumor type (RTKI, RTKII, MES, etc.) and histological region, were included. This yielded a comprehensive training set of 97,000 subgraphs, including 12,000 subgraphs from healthy controls.

### Evaluation Dataset Construction

For evaluation, we used the validation datasets covering a spectrum of epigenetic classes. From each, we extracted up to 500 subgraphs using the 3-hop method.

## Evaluation Metrics

In the evaluation of our graph-neural network (GNN), we employed a set of metrics to assess classification performance in predicting clinical and histological parameters.

- **Accuracy**
- **Precision** 
- **Recall** 
- **F1 Score** 

Additionally, the confusion matrix was presented, offering a detailed breakdown of the model's predictions across different classes, showing true positives, true negatives, false positives, and false negatives.

## NePSTA Graph-Neural Network Architecture

The NePSTA prediction network consists of a **Graph Isomorphism Network (GIN)** backbone and multiple **Multilayer Perceptron (MLP)** prediction heads, selected based on defined prediction tasks. NePSTA processes the local spatial graphical structures of gene expression values derived from the 3-hop neighborhood of Visium spots.

### Node Features

- **Expression Data**: Encapsulated in an $\(N \times G\)$ matrix (where $\(N\)$ is the number of nodes and $\(G\)$ represents the set of genes). These features are derived from the log-scaled, normalized expression values of the top 5000 most variably expressed genes in the cohort.
  
- **Copy Number Alterations**: Encapsulated in an $\(N \times C\)$ matrix (where $\(C\)$ contains chromosomal alterations).

- **Histological Annotations**: Encapsulated in an $\(N \times A\)$ matrix with one-hot encoded histological classifications.

- **H&E Images**: Encoded using a **Convolutional Neural Network (CNN)** designed to process a $\(256 \times 256\)$ image into an $\(N\)$-length vector.

### Edge Features

Each node has up to six neighbors, reflecting spatial arrangement. Self-loops (each node connected to itself) are incorporated, allowing each node to retain its original feature information during message passing.

### GIN Layers

GIN convolution updates node features by aggregating information from neighboring nodes:

$$
x_v^\prime = \text{ReLU}\left(\text{BN}\left((1 + \epsilon) \cdot x_v + \sum_{u \in \mathcal{N}(v)} \text{ReLU}(x_u)\right)\right)
$$

where $\epsilon$ is a learnable parameter and $mathcal{N}(v)$ represents the neighbors of node $v$.

## MLP Modules

Each MLP consists of a linear layer, ReLU activation, batch normalization, dropout, and a final linear layer that outputs predictions:

$$
h(x) = W_2 \cdot D \cdot B \cdot \phi(W_1 \cdot x + b_1) + b_2
$$

where:
- $x$ is the input vector,
- $W_1$ and $W_2$ are the weight matrices,
- $b_1$ and $b_2$ are bias vectors,
- $phi$ is the ReLU activation function,
- $B$ is batch normalization,
- $D$ is the dropout operation.

## Loss Functions

We applied different loss strategies for individual prediction heads, which were integrated based on the task.

- **Cross-Entropy Loss** for categorical variables:

  $$
  \text{CrossEntropyLoss} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})
  $$

- **L1 Norm Loss** for continuous variables:

  $$
  L1 = \frac{1}{N} \sum_{i=1}^{N} | y_i - \hat{y}_i |
  $$

- **Mean Squared Error (MSE)**:

  $$
  MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
  $$

The losses from multiple MLPs are integrated using a weighted sum:

$$
\text{loss} = \frac{1}{N} \sum_{i=1}^{N} l_i \times \omega_i
$$

where \( \omega_i \) is the weight for each loss \( l_i \).

## Model Training and Inference

The model was trained over multiple epochs, iterating through data batches. The optimizer’s gradients were reset before each forward pass. The model then generated predictions, calculated losses, and adjusted weights using the **Adam optimizer** to minimize losses. Various tasks, including survival predictions and neuron score predictions, were performed with corresponding losses optimized accordingly.


## Code use
We recommend to use the NEePSTA algorithm as presented in the Jupiter notebook. The notbook is optimized for Colab use including instalation of the required dependencies. No further installations are required. The source code .py files (required in the notebook) can be accesed here. 

## Dataset
The training dataset can be [Download here](https://drive.google.com/uc?export=download&id=1HQ1-QfBCkRSmBtuu8Zq8YqZu1tatDu01) (~25Gb)

## Licences Information
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
