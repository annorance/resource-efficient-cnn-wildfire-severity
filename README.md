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

However, higher-performing CNN architectures often come with increased computational requirements. Therefore, this project investigates the trade-off between classification performance and computational efficiency to identify the model that best balances accuracy and computational efficiency.

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
Image Preprocessing (image resizing; normalization; and augmentation, incl. horizontal flipping, random rotation, and random brightness transformation )
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

Augmented images were kept in the same partition as their original images to ensure a balanced distribution of augmented samples across the training, validation, and test sets.

## Model Development

The project uses pretrained CNN architectures and adapts their final classification layers for the five severity classes.

### Architectures

| Model              | Framework | Approach          |
| ------------------ | --------- | ----------------- |
| MobileNet V3-Small | PyTorch   | Transfer learning |
| SqueezeNet V1.1    | PyTorch   | Transfer learning |
| ShuffleNetV2-x0.5  | PyTorch   | Transfer learning |
| EfficientNet-Lite0 | PyTorch   | Transfer learning |

MobileNet V3-Small, SqueezeNet V1.1, and ShuffleNetV2-x0.5 were implemented using PyTorch's torchvision.models module. EfficientNet-Lite0 was implemented using the architecture released by RangiLyu (2020).

- PyTorch / TorchVision: https://pytorch.org/vision/stable/models.html
- EfficientNet-Lite: RangiLyu/EfficientNet-Lite

The pretrained weights were loaded before modifying the classification layers and dropout configuration.

## Baseline Model Training

Before hyperparameter tuning, a baseline model was trained for each CNN architecture using the training set and evaluated on the validation set. All baseline models were trained for 100 epochs using GPU acceleration.

The baseline configuration was kept consistent across architectures:

| Hyperparameter | Baseline Value |
|---|---|
| Batch size | 32 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Dropout | 0.2 |
| Loss function | Categorical Cross-Entropy |
| Evaluation metric | macro F1-score |
| Epochs | 100 |

This stage produced one baseline model for each architecture, resulting in **four baseline models**. The baseline results were then used as a reference for evaluating the impact of hyperparameter tuning.

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
* Estimated model size, calculated assuming FP32 precision (4 bytes per parameter)
* Training and validation time

## Results

### Best-performing model

**SqueezeNet V1.1** achieved the best overall test performance among the tuned models, improving upon the best baseline model, **MobileNet V3-Small**.

| Model | Test Accuracy | Macro F1-score | Micro F1-score |
|---|---:|---:|---:|
| Baseline — MobileNet V3-Small | 0.52 | 0.53 | 0.52 |
| Tuned — SqueezeNet V1.1 | **0.56** | **0.56** | **0.56** |

The best-performing SqueezeNet V1.1 configuration was:

| Hyperparameter | Value |
|---|---:|
| Batch size | 16 |
| Dropout | 0.5 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| L1 regularization | 0.001 |
| L2 regularization | 0.01 |

Compared with the best baseline model, SqueezeNet V1.1 improved test accuracy by **0.04**, macro F1-score by **0.03**, and micro F1-score by **0.04**. The model also provided a favorable balance between classification performance and computational efficiency.

### Model comparison

| Model               | Test Accuracy | Macro F1 | Micro F1 |
| ------------------- | ------------: | -------: | -------: |
| MobileNet V3-Small  |          0.52 |     0.53 |     0.52 |
| EfficientNet-Lite0  |          0.54 |     0.53 |     0.54 |
| **SqueezeNet V1.1** |      **0.56** | **0.56** | **0.56** |
| ShuffleNetV2-x0.5   |          0.54 |     0.55 |     0.56 |

> Results are based on the final experimental evaluation reported in the thesis.

## Model Efficiency

Model efficiency was evaluated based on **model size** and **training and validation time** over 100 epochs. Model size was calculated from the number of model parameters.

| Model | Model Size (MB) | Training & Validation Time (minutes) |
|---|---:|---:|
| ShuffleNetV2-x0.5 | 1.35 | 28.02 |
| SqueezeNet V1.1 | 2.77 | 23.83 |
| MobileNet V3-Small | 5.86 | 43.40 |
| EfficientNet-Lite0 | 13.04 | 53.86 |

**SqueezeNet V1.1** had the shortest training and validation time (**23.83 minutes**), while **ShuffleNetV2-x0.5** had the smallest model size (**1.35 MB**). This makes the model provided a favorable **balance** between **classification performance** and **computational efficiency**. 

On the other hand, **EfficientNet-Lite0** had the largest model size (**13.04 MB**) and longest training and validation time (**53.86 minutes**). These results highlight that there is still a trade-off between classification performance and computational efficiency in resource-constrained environments.

## Key Takeaways

* Built an end-to-end image classification pipeline from data labeling to model evaluation.
* Applied **VGG19 feature extraction and K-Means clustering** for initial dataset labeling.
* Implemented image preprocessing and augmentation.
* Built and compared four lightweight CNN architectures using **PyTorch**.
* Applied **grid-search hyperparameter tuning** across batch size, dropout, optimizer, learning rate, L1, and L2 regularization.
* Selected models based on validation macro F1-score.
* Evaluated both predictive performance and computational efficiency.
* Identified **SqueezeNet V1.1** as the most suitable architecture for the study's objective.

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
