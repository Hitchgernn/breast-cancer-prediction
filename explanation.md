# Breast Cancer Prediction Project Explanation

## Introduction

Breast cancer is one of the leading causes of death among women worldwide. Early detection plays a critical role in improving treatment outcomes, and mammography is one of the most widely used screening tools for this purpose. However, interpreting mammographic findings is still subject to human error, especially when the visual characteristics of a mass are ambiguous.

In this project, an automatic binary classification model is developed to assist medical practitioners in distinguishing between **benign** and **malignant** breast masses. The model is implemented using **Logistic Regression from scratch**, with NumPy as the main numerical library and without relying on high-level machine learning frameworks such as Scikit-Learn.

## Problem Statement

Mammography is an effective screening method, but its **Positive Predictive Value (PPV)** is still relatively low. In practice, many breast biopsies are performed even though the mass later turns out to be benign. This creates several problems:

- unnecessary stress and anxiety for patients
- painful and invasive medical procedures
- inefficient use of hospital time and cost

Therefore, a classification model that helps separate benign and malignant cases more consistently can support decision-making and reduce avoidable interventions.

## Dataset Details

The project uses the **Mammographic Masses** dataset. The data was collected by the **Institute of Radiology of the University Erlangen-Nuremberg** between **2003 and 2006**.

Dataset summary:

- Total samples: `961`
- Benign cases (`Severity = 0`): `516`
- Malignant cases (`Severity = 1`): `445`

Original variables in the dataset:

- `BI-RADS Assesment`: initial clinical assessment score
- `Age`: patient age
- `Shape`: shape of the tumor mass
- `Margin`: margin characteristic of the mass
- `Density`: mass density
- `Severity`: target label (`0 = benign`, `1 = malignant`)

Feature interpretation in this project:

- `Age` is treated as a numerical feature
- `Shape` and `Margin` are treated as categorical labels and encoded manually
- `Severity` is the binary target variable

Dataset source:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass

## Objective

The main objective of this project is to build a **from-scratch Logistic Regression classifier** that can predict the severity of a breast mass from mammographic features.

More specifically, the project aims to:

- support clinical decision-making
- improve classification reliability for benign vs malignant masses
- provide a transparent baseline model that is easy to interpret
- evaluate whether threshold tuning can make the model safer for medical use

## Data Loading

The dataset is loaded from the local repository using Pandas. Missing values are marked as `NaN` so they can be handled explicitly during preprocessing.

Main steps:

- define the dataset column names
- read the raw `.data` file
- inspect the basic structure of the data

## Exploratory Data Analysis (EDA)

EDA is used to understand the dataset before training the model. This stage helps identify:

- the distribution of each feature
- missing-value patterns
- class balance in the target
- the relationship between features and the target

The EDA in this project includes:

- histogram for `Age`
- count plots for categorical features
- correlation heatmap
- manual inspection of feature usefulness

This stage is important because the feature selection decision is not made blindly. Instead, it is supported by empirical patterns found in the data.

## Missing-Value Handling and Preprocessing

The dataset contains missing values in several columns, especially `Shape`, `Margin`, and `Density`. A simple and interpretable preprocessing strategy is applied:

- categorical-like features are imputed with the **mode**
- `Age` is imputed with the **median**
- data types are converted to numeric format
- incorrect entries such as `55` in `BI-RADS Assesment` are corrected when necessary

This preprocessing is designed to keep the dataset usable without introducing unnecessary complexity.

## Feature Engineering

Feature engineering is used to improve the representational quality of the input before training Logistic Regression.

The following strategies are applied:

### 1. One-Hot Encoding from Scratch

Because `Shape` and `Margin` are treated as categorical labels rather than ordinal values, they are encoded manually using one-hot encoding.

This avoids forcing the model to interpret category IDs as if they had meaningful numerical order.

### 2. Numerical Feature Transformations

Additional transformed features are added manually:

- `Age^2`
- `BI-RADS Assesment * Age`

These help a linear model capture patterns that are not fully represented by the original raw features alone.

## Train/Validation/Test Split

The data is divided into three subsets:

- **training set**: used to learn the model parameters
- **validation set**: used to tune thresholds and inspect model behavior
- **test set**: used only for final evaluation

This separation is important because the test set should remain unseen during model selection. Otherwise, the final evaluation would be biased.

## Feature Scaling

Before training, features are standardized using a manually implemented scaler.

The scaling formula is:

```math
X_{scaled} = \frac{X - \mu}{\sigma}
```

Where:

- `mu` is the mean of the training feature
- `sigma` is the standard deviation of the training feature

Feature scaling is important because Logistic Regression optimized with gradient descent is sensitive to feature scale. Standardization helps the optimization process converge more reliably.

## Baseline Logistic Regression Training

A baseline model is first trained using manually chosen hyperparameters. This provides a reference point before validation and tuning.

The baseline stage helps answer:

- how well the raw pipeline performs before optimization
- whether later tuning actually improves the model

## Logistic Regression Formulation

### 1. Linear Combination

The model computes a linear score:

```math
z = Xw + b
```

Where:

- `X` is the feature matrix
- `w` is the weight vector
- `b` is the bias term

### 2. Sigmoid Function

The score is converted into a probability using the sigmoid function:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

This produces an output between `0` and `1`, which is interpreted as the probability that a sample belongs to class `1`.

### 3. Binary Cross-Entropy Loss

The training objective is the binary cross-entropy loss:

```math
J(w, b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]
```

Where:

- `m` is the number of samples
- `y` is the true class label
- `y_hat` is the predicted probability

### 4. L2 Regularization

To reduce overfitting, L2 regularization can be added:

```math
J_{reg}(w, b) = J(w, b) + \frac{\lambda}{2m}\sum_{j=1}^{n} w_j^2
```

Where:

- `lambda` controls the strength of regularization

### 5. Gradient Descent

The gradients are computed as:

```math
dw = \frac{1}{m}X^T(\hat{y} - y)
```

```math
db = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})
```

With L2 regularization:

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

- `alpha` is the learning rate

## Validation with K-Fold Cross-Validation and Manual Grid Search

After the baseline model is obtained, validation is performed using:

- **k-fold cross-validation**
- **manual grid search**

This stage is used to search for better combinations of:

- learning rate
- number of iterations
- regularization strength

Cross-validation is useful because it evaluates the model on multiple train-validation splits, making the estimate more robust than relying on a single split.

## Final Tuned Model Training

After the best hyperparameters are found, the final model is retrained on the training data using those parameters. This tuned model is then used for threshold analysis and final evaluation.

The tuned model is compared with the baseline model to check whether tuning improves practical performance.

## Threshold Selection and Final Evaluation

Logistic Regression outputs probabilities, not class labels directly. A threshold is needed to convert probabilities into predictions.

Prediction rule:

```math
\hat{y} =
\begin{cases}
1 & \text{if } p \ge \text{threshold} \\
0 & \text{if } p < \text{threshold}
\end{cases}
```

Two thresholding strategies can be used:

- **optimal threshold**: selected using F1-score
- **medical threshold**: selected to prioritize high recall

This distinction matters in medical classification. A model with a lower threshold may reduce false negatives, even if accuracy becomes slightly lower.

## Results and Metrics

Model performance is evaluated using standard binary classification metrics:

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

- `TP` = true positives
- `TN` = true negatives
- `FP` = false positives
- `FN` = false negatives

In this project, **recall** is especially important because missing a malignant case is more dangerous than producing additional false positives.

## Result Validation

Result validation is supported by:

- k-fold cross-validation
- comparison between baseline and tuned models
- threshold tuning on the validation set instead of the test set

This ensures that the final reported performance is more credible and not based on a single arbitrary choice of parameters.

## Conclusion

This project demonstrates that a breast cancer classification pipeline can be implemented successfully using **Logistic Regression from scratch**. Even without high-level machine learning libraries, it is possible to build a structured workflow that includes preprocessing, feature engineering, optimization, validation, threshold tuning, and evaluation.

The overall methodology is suitable for educational and analytical purposes because it provides:

- clear mathematical interpretability
- transparent implementation details
- practical evaluation for a medical classification task

The final model can support decision analysis for distinguishing benign and malignant mammographic masses, while still leaving room for future improvements such as weighted loss, more advanced feature engineering, or comparison with other from-scratch classifiers.
