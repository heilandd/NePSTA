# NePSTA
NeuroPathology Spatial Transcriptomic Analysis

<img width="1385" alt="image" src="https://github.com/user-attachments/assets/2feeeba8-34da-4b16-9dc4-88a24f8022f2">


## Spatially resolved transcriptomics and graph-based deep-learning improve accuracy of routine CNS tumor diagnostics

Abstract:
The diagnostic landscape of brain tumors has recently evolved to integrate comprehensive molecular markers alongside traditional histopathological evaluation. Foremost, genome-wide DNA methylation profiling and next generation sequencing (NGS) has become a cornerstone in classifying Central Nervous System (CNS) tumors, as recognized by its inclusion into the 2021 WHO classification. Despite its diagnostic precision, a limiting requirement for NGS and methylation profiling is sufficient DNA quality and quantity which restricts its feasibility, especially in cases with small biopsy samples or low tumor cell content, both frequent challenges in specimen of diffusely growing CNS lesions. Addressing these challenges, we demonstrate a novel application, namely NePSTA (NeuroPathology Spatial Transcriptomic Analysis), which is capable of generating comprehensive morphological and molecular neuropathological diagnostics from single 5 µm tissue sections. Our framework employs 10x Visium spatial transcriptomics with graph neural networks for automated histological and molecular evaluations. Trained and evaluated across 130 patients with CNS malignancies and healthy donors across four medical centers, NePSTA integrates spatial gene expression data and inferred CNAs to predict tissue histology and methylation-based subclasses with high accuracy. Further, we demonstrate the ability to reconstruct immunohistochemistry and genotype profiling on single thin slides of minute tissue biopsies. Our approach has minimal tissue requirements, often inadequate for conventional molecular diagnostics, demonstrating the potential to transform neuropathological diagnostics and enhance tumor subtype identification with implications for fast and precise diagnostic work-up.

Manuscript will be avaiable soon!

# Graph Neural Networks: NePSTA

## Overview

The graph neural network framework is part of the NePSTA project, which aims to implement deep learning strategies to explore spatially resolved multi-omics data. NePSTA is evaluated on the Visium dataset, and the framework integrates various clinical, histological, and gene expression data to enhance predictive accuracy.

## Data Split

To assess the performance of NePSTA and comparative methods, the following data partitioning approach was adopted:

### Data Split:

- From the 107 patients characterized by EPIC, each dataset was divided into training and validation subsets.
- Samples with multiple biopsies were split by individual biopsy cores.
- For datasets with a single specimen, the spots were manually segmented using the `createSegmentation` function of the `SPATA2` package.

### Training Dataset Construction:

- The training dataset, implemented using the PyTorch Geometric library, included up to 500 subgraphs per training split. 
- Clinical attributes such as tumor type (RTKI, RTKII, MES, etc.) and histological region were incorporated.
- The resulting training set comprised 97,000 subgraphs, including 12,000 subgraphs from healthy controls.

### Evaluation Dataset Construction:

- For evaluation, validation datasets covered a spectrum of epigenetic classes. 
- From each dataset, up to 500 subgraphs were extracted using the 3-hop method.

## Evaluation Metrics

In evaluating the performance of NePSTA, we employed several metrics:

- **Accuracy**: The proportion of correct predictions among the total predictions:
  \[
  \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
  \]

- **Precision** (macro-average): The ratio of true positive predictions to the total number of positive predictions:
  \[
  \text{Precision}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FP_i}
  \]

- **Recall** (macro-average): The ratio of true positive predictions to the total number of actual positives:
  \[
  \text{Recall}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FN_i}
  \]

- **F1 Score** (macro-average): The harmonic mean of precision and recall:
  \[
  \text{F1 Score}_{\text{macro}} = 2 \times \frac{\text{Precision}_{\text{macro}} \times \text{Recall}_{\text{macro}}}{\text{Precision}_{\text{macro}} + \text{Recall}_{\text{macro}}}
  \]

- **Confusion Matrix**: A detailed breakdown of true positives, true negatives, false positives, and false negatives across different classes.

## NePSTA Graph-Neural Network Architecture

The NePSTA prediction network consists of a **Graph Isomorphism Network (GIN)** backbone, with multiple **multilayer perceptron (MLP)** heads selected based on the prediction tasks. Below are the input features and their respective structures.

### Node Features

- **Expression Data**: Encapsulated in an \( N \times G \) matrix, where \( N \) is the number of nodes and \( G \) represents genes. Non-expressed genes (zero counts) are masked to avoid skewing.
- **Copy Number Alterations**: Represented by an \( N \times C \) matrix, where \( C \) contains chromosomal alterations.
- **Histological Annotations**: Represented by an \( N \times A \) matrix with one-hot encoded histological classifications.
- **H&E Images**: Encoded into a vector through a **Convolutional Neural Network (CNN)** from a \( 256 \times 256 \) pixel image.

### Edge Features

- Each node connects to up to six neighbors, reflecting the spatial arrangement. Subgraphs with fewer than 15 nodes are excluded for complexity control. Self-loops are incorporated for each node to retain its original feature information during message passing.

### GIN Layers

The GIN convolution operation updates node features by aggregating features from neighboring nodes:
\[
x_v^\prime = \text{ReLU}\left(\text{BN}\left((1 + \epsilon) \cdot x_v + \sum_{u \in \mathcal{N}(v)} \text{ReLU}(x_u)\right)\right)
\]
where \( \epsilon \) is a learnable parameter and \( \mathcal{N}(v) \) is the set of neighbors of node \( v \).

### MLP Modules

Each MLP module consists of a linear layer, ReLU activation, batch normalization, dropout, and a final linear layer that outputs predictions:
\[
h(x) = W_2 \cdot D \cdot B \cdot \phi(W_1 \cdot x + b_1) + b_2
\]
where:
- \( x \) is the input vector,
- \( W_1 \) and \( W_2 \) are weight matrices,
- \( b_1 \) and \( b_2 \) are bias vectors,
- \( \phi \) is the ReLU activation function,
- \( B \) represents batch normalization, and
- \( D \) represents dropout.

## Loss Functions

NePSTA uses different loss strategies depending on the prediction task:

- **Cross-Entropy Loss** for categorical variables:
  \[
  \text{CrossEntropyLoss} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})
  \]

- **L1 Norm Loss** for continuous variables:
  \[
  L1 = \frac{1}{N} \sum_{i=1}^{N} \left| y_i - \widehat{y_i} \right|
  \]

- **Mean Squared Error (MSE)**:
  \[
  MSE = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \widehat{y_i} \right)^2
  \]

The losses from multiple MLPs are combined as a weighted sum:
\[
\text{loss} = \frac{1}{N} \sum_{i=1}^{N} l_i \times \omega_i
\]
where \( \omega_i \) is the weight assigned to each loss \( l_i \).

## Model Training and Inference

The model is trained over several epochs, iterating through batches of data. The training process involves:
1. Resetting the optimizer's gradients.
2. Performing a forward pass to generate predictions.
3. Calculating loss (using the L1 norm for neuron scores).
4. Backpropagating the loss to compute gradients.
5. Adjusting model weights to minimize loss using the Adam optimizer.

## Code use
We recommend to use the NEePSTA algorithm as presented in the Jupiter notebook. The notbook is optimized for Colab use including instalation of the required dependencies. No further installations are required. The source code .py files (required in the notebook) can be accesed here. 


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
