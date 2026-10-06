# Resource-Efficient CNN for Post-Wildfire Severity Classification

> Image classification of post-wildfire areas using lightweight CNN architectures under computational constraints.

## Overview

This project develops a computer vision model to classify the severity of post-wildfire areas into five severity levels while considering computational efficiency.

The study compares four lightweight CNN architectures using transfer learning:

* MobileNet V3-Small
* SqueezeNet V1.1
* ShuffleNetV2-x0.5
* EfficientNet-Lite0

The models were evaluated not only based on classification performance, but also on model size and training/validation time to identify an architecture suitable for resource-constrained computing environments.

## Problem

Assessing post-wildfire severity manually can be time-consuming and susceptible to human error. A computer vision-based classification system can support a more consistent and automated assessment process.

However, higher-performing CNN architectures often come with increased computational requirements. Therefore, this project investigates the trade-off between classification performance and computational efficiency.

## Dataset

The dataset consists of **2,242 images**:

* 1,626 images obtained from previous research
* 616 images collected independently

The images were classified into five severity levels:

1. Very Light
2. Light
3. Moderate
4. Severe
5. Very Severe

### Data Labeling

Because the original dataset required severity labels, image features were extracted using **VGG19** and grouped into five clusters using **K-Means clustering**.

Cluster quality was evaluated using the **Silhouette Index**, with the best configuration achieving an average Silhouette Index of approximately **0.407**.

Final cluster labels were assigned based on visual indicators of vegetation and soil conditions using the referenced wildfire severity criteria.

## Methodology

The overall workflow is:

```text
Image Collection
      ↓
Feature Extraction (VGG19)
      ↓
K-Means Clustering
      ↓
Silhouette Index Evaluation
      ↓
Label Assignment
      ↓
Image Preprocessing
      ↓
Train / Validation / Test Split
      ↓
CNN Model Development
      ↓
Baseline Training
      ↓
Hyperparameter Tuning
      ↓
Test Evaluation
      ↓
Performance & Efficiency Comparison
```

## Preprocessing

The images were preprocessed using:

* Resizing
* Normalization
* Data augmentation

  * Horizontal flipping
  * Random rotation
  * Random brightness adjustment

The dataset was split into:

| Split      | Proportion |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        20% |
| Test       |        10% |

Augmented versions of an image were kept within the same partition as the original image to avoid data leakage across train, validation, and test sets.

## Model Development

The project uses pretrained CNN architectures and adapts their final classification layers for the five severity classes.

### Architectures

| Model              | Framework | Approach          |
| ------------------ | --------- | ----------------- |
| MobileNet V3-Small | PyTorch   | Transfer learning |
| SqueezeNet V1.1    | PyTorch   | Transfer learning |
| ShuffleNetV2-x0.5  | PyTorch   | Transfer learning |
| EfficientNet-Lite0 | PyTorch   | Transfer learning |

The pretrained weights were loaded before modifying the classification layers and dropout configuration.

## Hyperparameter Tuning

Hyperparameter optimization was performed using **grid search**.

The explored hyperparameters include:

| Hyperparameter    | Values        |
| ----------------- | ------------- |
| Batch size        | 8, 16, 32     |
| Dropout           | 0.3, 0.5      |
| Optimizer         | Adam, RMSProp |
| Learning rate     | 0.001, 0.0001 |
| L1 regularization | 0.001, 0.01   |
| L2 regularization | 0.001, 0.01   |

Each configuration was trained for **100 epochs**.

The model configuration was selected based on validation **macro F1-score**.

## Evaluation

The final models were evaluated using the test set with:

* Accuracy
* Precision
* Recall
* F1-score
* Macro F1-score
* Micro F1-score
* Confusion matrix

Model efficiency was additionally evaluated using:

* Number of parameters
* Estimated model size
* Training and validation time

## Results

### Best-performing model

**SqueezeNet V1.1** achieved the best overall test performance among the tuned models, with:

* Test Accuracy: **0.56**
* Macro F1-score: **0.56**
* Micro F1-score: **0.56**

The model also provided a favorable balance between classification performance and computational efficiency.

### Model comparison

| Model               | Test Accuracy | Macro F1 | Micro F1 |
| ------------------- | ------------: | -------: | -------: |
| MobileNet V3-Small  |          0.52 |     0.53 |     0.52 |
| EfficientNet-Lite0  |          0.54 |     0.53 |     0.54 |
| **SqueezeNet V1.1** |      **0.56** | **0.56** | **0.56** |
| ShuffleNetV2-x0.5   |          0.54 |     0.55 |     0.56 |

> Results are based on the final experimental evaluation reported in the thesis.

## Model Efficiency

The project also compares model size and training/validation time because the objective is not solely to maximize classification performance.

The estimated model sizes ranged from approximately:

* **1.35 MB** — ShuffleNetV2-x0.5
* **5.86 MB** — MobileNet V3-Small
* **28.02 MB** — SqueezeNet V1.1
* **43.40 MB** — EfficientNet-Lite0

This comparison highlights the trade-off between predictive performance and computational requirements.

## Key Takeaways

* Built an end-to-end image classification pipeline from data labeling to model evaluation.
* Applied **VGG19 feature extraction and K-Means clustering** for initial dataset labeling.
* Implemented image preprocessing and augmentation.
* Built and compared four lightweight CNN architectures using **PyTorch**.
* Applied **grid-search hyperparameter tuning** across batch size, dropout, optimizer, learning rate, L1, and L2 regularization.
* Selected models based on validation macro F1-score.
* Evaluated both predictive performance and computational efficiency.
* Identified **SqueezeNet V1.1** as the most suitable architecture for the study's objective.

## Repository Structure

```text
├── notebooks/
│   ├── 01_data_labeling.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_data_splitting.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_hyperparameter_tuning_mobilenet_squeezenet.ipynb
│   ├── 06_hyperparameter_tuning_shufflenet_efficientnet.ipynb
│   └── 07_model_evaluation.ipynb
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── models/
│   └── README.md
│
├── data/
│   └── README.md
│
└── docs/
    └── thesis_poster.pdf
```

## Reproducibility

The notebooks contain the experimental workflow used in this research.

Because the original image dataset and trained model weights are not included in this repository, the complete experiment cannot be reproduced directly from the repository without access to the original data and model artifacts.

## Technologies

* Python
* PyTorch
* TorchVision
* Scikit-learn
* OpenCV
* Pandas
* NumPy
* Matplotlib
* VGG19
* K-Means Clustering
* Transfer Learning
* Convolutional Neural Networks

## Author

**Nurul Fadillah**
Computer Science — IPB University

[LinkedIn](https://linkedin.com/in/nurul-fadillah-766431215) · [GitHub](https://github.com/annorance)
