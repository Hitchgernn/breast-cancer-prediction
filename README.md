# Breast Cancer Prediction with From-Scratch Logistic Regression

This repository implements a binary breast cancer classification pipeline using **Logistic Regression built from scratch** with **NumPy**, plus **Pandas** for data handling and **Matplotlib/Seaborn** for visualization.

The project uses the **Mammographic Masses** dataset and focuses on:

- data preprocessing
- exploratory data analysis
- feature engineering
- manual train/validation/test splitting
- manual scaling
- Logistic Regression from scratch
- threshold tuning for medical use cases
- k-fold validation and grid search

## Repository Structure

```text
.
├── dataset/
│   ├── mammographic_masses.data
│   └── mammographic_masses.names
├── prediction.ipynb
├── optimized_version.py
└── README.md
```

## Dataset

- Dataset: `Mammographic Masses`
- Files:
  - [`dataset/mammographic_masses.data`](./dataset/mammographic_masses.data)
  - [`dataset/mammographic_masses.names`](./dataset/mammographic_masses.names)

Source reference:
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass

Main variables used in the project:

- `BI-RADS Assesment`
- `Age`
- `Shape`
- `Margin`
- `Density`
- `Severity` (target)

Target meaning:

- `0` = benign
- `1` = malignant

## Project Goal

The goal of this project is to predict whether a mammographic mass is **benign** or **malignant** using a manually implemented Logistic Regression model without relying on high-level machine learning libraries such as Scikit-Learn.

## Methodology

The pipeline is organized into the following stages:

1. Data loading
2. Exploratory Data Analysis (EDA)
3. Missing-value handling and preprocessing
4. Feature engineering
5. Train/validation/test split
6. Feature scaling
7. Baseline Logistic Regression training
8. Validation with k-fold cross-validation and manual grid search
9. Final tuned model training
10. Threshold selection and final evaluation

## Feature Engineering

The current workflow includes:

- one-hot encoding from scratch for categorical features such as `Shape` and `Margin`
- numerical feature transformation such as:
  - `Age^2`
  - `BI-RADS Assesment * Age`

This is useful because Logistic Regression is a linear model, so manual feature construction can improve its ability to model more complex patterns.

## Logistic Regression Formulas

### 1. Linear Model

For each sample:

```math
z = Xw + b
```

Where:

- `X` = input features
- `w` = weight vector
- `b` = bias

### 2. Sigmoid Function

The sigmoid function converts the linear output into a probability:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

This gives a value between 0 and 1, interpreted as the probability that the sample belongs to class 1.

### 3. Binary Cross-Entropy Loss

The loss function for binary classification is:

```math
J(w, b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]
```

Where:

- `m` = number of samples
- `y` = true label
- `y_hat` = predicted probability

### 4. L2 Regularization

To reduce overfitting, the implementation may include L2 regularization:

```math
J_{reg}(w, b) = J(w, b) + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2
```

Where:

- `lambda` = regularization strength

### 5. Gradient Descent Updates

The gradients are computed as:

```math
dw = \frac{1}{m}X^T(\hat{y} - y)
```

```math
db = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})
```

With L2 regularization, the weight gradient becomes:

```math
dw = \frac{1}{m}X^T(\hat{y} - y) + \frac{\lambda}{m}w
```

Parameter updates:

```math
w := w - \alpha \cdot dw
```

```math
b := b - \alpha \cdot db
```

Where:

- `alpha` = learning rate

## Thresholding

The model outputs probabilities, not final class labels directly. A threshold is used to convert probabilities into class predictions:

```math
\hat{y} =
\begin{cases}
1 & \text{if } p \ge \text{threshold} \\
0 & \text{if } p < \text{threshold}
\end{cases}
```

This project uses both:

- an **optimal threshold** based on F1-score
- a **medical threshold** chosen to prioritize recall

This is important because, in medical classification, missing a malignant case can be more costly than generating extra false positives.

## Evaluation Metrics

The project evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- Confusion Matrix

Metric definitions:

```math
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
```

```math
Precision = \frac{TP}{TP + FP}
```

```math
Recall = \frac{TP}{TP + FN}
```

```math
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
```

```math
Specificity = \frac{TN}{TN + FP}
```

Where:

- `TP` = True Positive
- `TN` = True Negative
- `FP` = False Positive
- `FN` = False Negative

## Validation Strategy

To improve reliability, the repository includes:

- manual **k-fold cross-validation**
- manual **grid search**
- separate **validation** and **test** usage

The intended evaluation flow is:

1. Train a baseline model
2. Measure baseline metrics
3. Run validation with k-fold and grid search
4. Train a tuned model using the best parameters
5. Select a threshold on the validation set
6. Report final metrics on the test set

## Visualizations

The project includes several useful plots:

- feature distribution plots
- correlation heatmap
- grid search comparison plot
- baseline vs tuned comparison plot
- precision-recall curve
- confusion matrix
- ROC curve
- threshold trade-off plot
- prediction probability distribution
- training loss curve

## How to Run

### Option 1: Notebook

Open:

- [`prediction.ipynb`](./prediction.ipynb)

and run the cells in order.

### Option 2: Script Reference

Use:

- [`optimized_version.py`](./optimized_version.py)

as a structured reference for:

- code organization
- section ordering
- formulas and evaluation logic

## Requirements

Install the basic dependencies:

```bash
pip install numpy pandas matplotlib seaborn
```

## Notes

- This project intentionally avoids Scikit-Learn for model implementation.
- Core model logic, thresholding, scaling, and validation are written manually.
- The script is designed to be easy to map into notebook sections such as:
  - EDA
  - Feature Engineering
  - Baseline Results
  - Result Validation
  - Final Evaluation

## Future Improvements

Possible next steps:

- compare one-hot encoding vs ordinal treatment for more features
- evaluate the usefulness of `Density` more systematically
- add weighted loss for class imbalance handling
- compare multiple manually implemented optimizers
- export final metrics and plots for report generation
