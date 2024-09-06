# NePSTA
NeuroPathology Spatial Transcriptomic Analysis

<img width="1385" alt="image" src="https://github.com/user-attachments/assets/2feeeba8-34da-4b16-9dc4-88a24f8022f2">


## Spatially resolved transcriptomics and graph-based deep-learning improve accuracy of routine CNS tumor diagnostics

Abstract:
The diagnostic landscape of brain tumors has recently evolved to integrate comprehensive molecular markers alongside traditional histopathological evaluation. Foremost, genome-wide DNA methylation profiling and next generation sequencing (NGS) has become a cornerstone in classifying Central Nervous System (CNS) tumors, as recognized by its inclusion into the 2021 WHO classification. Despite its diagnostic precision, a limiting requirement for NGS and methylation profiling is sufficient DNA quality and quantity which restricts its feasibility, especially in cases with small biopsy samples or low tumor cell content, both frequent challenges in specimen of diffusely growing CNS lesions. Addressing these challenges, we demonstrate a novel application, namely NePSTA (NeuroPathology Spatial Transcriptomic Analysis), which is capable of generating comprehensive morphological and molecular neuropathological diagnostics from single 5 µm tissue sections. Our framework employs 10x Visium spatial transcriptomics with graph neural networks for automated histological and molecular evaluations. Trained and evaluated across 130 patients with CNS malignancies and healthy donors across four medical centers, NePSTA integrates spatial gene expression data and inferred CNAs to predict tissue histology and methylation-based subclasses with high accuracy. Further, we demonstrate the ability to reconstruct immunohistochemistry and genotype profiling on single thin slides of minute tissue biopsies. Our approach has minimal tissue requirements, often inadequate for conventional molecular diagnostics, demonstrating the potential to transform neuropathological diagnostics and enhance tumor subtype identification with implications for fast and precise diagnostic work-up.

Manuscript will be avaiable soon!

# Graph Neural Networks: NePSTA
## Data Split

The graph neural network framework is part of the NePSTA project, which aims to implement deep learning strategies to explore spatially resolved multi-omics. To assess the performance of NePSTA ([GitHub - heilandd/NePSTA](https://github.com/heilandd/NePSTA)) and comparative methods, we conducted evaluations on our Visium dataset. The datasets were partitioned into training and evaluation subsets using the following stratified procedure:

### Data Split:
From the 107 patients characterized by EPIC, each dataset was split into a training and a validation segment. For samples containing multiple biopsies, the dataset was split by individual biopsy cores. In datasets with a single specimen, spots were segmented manually using the `createSegmentation` function of the `SPATA2` package.

### Training Dataset Construction:
We created the training dataset using the PyTorch Geometric library, selecting up to 500 subgraphs from the training split. Clinical attributes such as tumor type (RTKI, RTKII, MES, etc.) and histological region were incorporated. This resulted in a comprehensive training set of 97,000 subgraphs, including 12,000 from healthy controls.

### Evaluation Dataset Construction:
For evaluation, we used the validation datasets to cover a spectrum of epigenetic classes. Up to 500 subgraphs were extracted from each dataset using the 3-hop method.

## Evaluation Metrics

In evaluating our graph-neural network (GNN), we employed several metrics to assess classification performance in predicting clinical and histological parameters.

- **Accuracy**: Measures the overall correctness of the model:
  \[
  \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
  \]
  
- **Precision** (macro-average): Calculated as the ratio of true positives to total positive predictions.
  \[
  \text{Precision}_{\text{macro}} = \frac{1}{N}\sum_{i=1}^{N}\frac{TP_i}{TP_i + FP_i}
  \]
  
- **Recall** (macro-average): The ratio of true positives to actual positives.
  \[
  \text{Recall}_{\text{macro}} = \frac{1}{N}\sum_{i=1}^{N}\frac{TP_i}{TP_i + FN_i}
  \]
  
- **F1 Score** (macro-average): The harmonic mean of precision and recall.
  \[
  \text{F1 Score}_{\text{macro}} = 2 \times \frac{\text{Precision}_{\text{macro}} \times \text{Recall}_{\text{macro}}}{\text{Precision}_{\text{macro}} + \text{Recall}_{\text{macro}}}
  \]

Additionally, we presented the confusion matrix, providing a detailed breakdown of the model's predictions across different classes.

## NePSTA Graph-Neural Network Architecture

The NePSTA prediction network is based on a **Graph Isomorphism Network (GIN)** backbone, with multiple **multilayer perceptron (MLP)** prediction heads selected according to the defined prediction tasks. The input consists of local spatial graphical structures derived from gene expression values from Visium spots.

### Node Features:
- **Expression Data**: Encapsulated in an \( N \times G \) matrix (where \( N \) is the number of nodes and \( G \) represents genes). Non-expressed genes are masked.
- **Copy Number Alterations**: Extracted in an \( N \times C \) matrix where \( C \) represents chromosomal alterations.
- **Histological Annotations**: Hot-one encoded classification in an \( N \times A \) matrix.
- **H&E Images**: Encoded into a vector through a **Convolutional Neural Network (CNN)** from a \( 256 \times 256 \) pixel image.

### Edge Features:
Edges represent connections between nodes, with each node having up to six neighbors to reflect spatial arrangement. Subgraphs with fewer than 15 nodes are excluded to maintain complexity.

### GIN Layers:
The GIN convolution updates node features by aggregating features from neighboring nodes, as follows:
\[
x_v^\prime = \text{ReLU}\left(\text{BN}\left((1 + \epsilon) \cdot x_v + \sum_{u \in \mathcal{N}(v)} \text{ReLU}(x_u)\right)\right)
\]
where \( \epsilon \) is a learnable parameter and \( \mathcal{N}(v) \) represents neighbors of node \( v \).

### MLP Modules:
Each MLP comprises a linear layer, ReLU activation, batch normalization, dropout, and a final linear layer that outputs predictions. The MLP formula:
\[
h(x) = W_2 \cdot D \cdot B \cdot \phi(W_1 \cdot x + b_1) + b_2
\]

### Loss Functions:
- **Cross-Entropy Loss** for categorical variables:
  \[
  \text{CrossEntropyLoss} = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c})
  \]
- **L1 Norm Loss** for continuous variables:
  \[
  L1 = \frac{1}{N} \sum_{i=1}^{N} \left| y_i - \widehat{y_i} \right|
  \]
- **MSE** (Mean Square Error):
  \[
  MSE = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \widehat{y_i} \right)^2
  \]

The losses from different MLPs are combined through a weighted sum:
\[
\text{loss} = \frac{1}{N} \sum_{i=1}^{N} l_i \times \omega_i
\]

### Model Training and Inference:
The model was trained over several epochs using the Adam optimizer. Gradients were reset before each forward pass, predictions were generated, and the loss was minimized through backpropagation.

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
