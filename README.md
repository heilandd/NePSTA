# NePSTA
NeuroPathology Spatial Transcriptomic Analysis

<img width="1385" alt="image" src="https://github.com/user-attachments/assets/2feeeba8-34da-4b16-9dc4-88a24f8022f2">


Spatially resolved transcriptomics and graph-based deep-learning improve accuracy of routine CNS tumor diagnostics

Abstract:
The diagnostic landscape of brain tumors has recently evolved to integrate comprehensive molecular markers alongside traditional histopathological evaluation. Foremost, genome-wide DNA methylation profiling and next generation sequencing (NGS) has become a cornerstone in classifying Central Nervous System (CNS) tumors, as recognized by its inclusion into the 2021 WHO classification. Despite its diagnostic precision, a limiting requirement for NGS and methylation profiling is sufficient DNA quality and quantity which restricts its feasibility, especially in cases with small biopsy samples or low tumor cell content, both frequent challenges in specimen of diffusely growing CNS lesions. Addressing these challenges, we demonstrate a novel application, namely NePSTA (NeuroPathology Spatial Transcriptomic Analysis), which is capable of generating comprehensive morphological and molecular neuropathological diagnostics from single 5 µm tissue sections. Our framework employs 10x Visium spatial transcriptomics with graph neural networks for automated histological and molecular evaluations. Trained and evaluated across 130 patients with CNS malignancies and healthy donors across four medical centers, NePSTA integrates spatial gene expression data and inferred CNAs to predict tissue histology and methylation-based subclasses with high accuracy. Further, we demonstrate the ability to reconstruct immunohistochemistry and genotype profiling on single thin slides of minute tissue biopsies. Our approach has minimal tissue requirements, often inadequate for conventional molecular diagnostics, demonstrating the potential to transform neuropathological diagnostics and enhance tumor subtype identification with implications for fast and precise diagnostic work-up.

Manuscript will be avaiable soon!

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
