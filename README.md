# CodeAlpha_Tasks_Iris_flower
Iris Flower Classification with Machine Learning
# Iris Flower Classification with Machine Learning

![Iris Dataset Visualization](https://archive.ics.uci.edu/ml/assets/MLimages/Large53.jpg)

## Project Overview

This project demonstrates a comprehensive machine learning workflow for classifying Iris flowers into three species (Setosa, Versicolor, or Virginica) based on their sepal and petal measurements. The implementation includes exploratory data analysis, feature engineering, multiple machine learning models, and performance evaluation.

## Key Features

- Complete EDA with statistical analysis and visualizations
- Correlation and PCA analysis
- KMeans clustering with elbow method
- Comparison of 7 different classification algorithms
- Feature importance analysis
- Model persistence for future predictions

## Dataset

The Iris dataset contains 150 samples with:
- **Features**:
  - SepalLengthCm
  - SepalWidthCm 
  - PetalLengthCm
  - PetalWidthCm
- **Target**: Species (Iris-setosa, Iris-versicolor, Iris-virginica)

Analysis Highlights
Exploratory Data Analysis
Statistical summary of features

Correlation matrix visualization

Pair plots showing feature relationships

Outlier detection using IQR method

Dimensionality Reduction
PCA analysis showing 92.5% variance explained by first component

2D visualization of PCA-transformed data

Clustering
- KMeans clustering with elbow method

- Optimal cluster count determined as 3 (matching actual species)

Model Comparison
Tested 7 classification algorithms:

- Logistic Regression

- K-Nearest Neighbors

- Support Vector Machine

- Decision Tree

- Random Forest

- Gradient Boosting

- Naive Bayes

Feature Importance
Random Forest identified most important features:

PetalLengthCm

PetalWidthCm

SepalLengthCm

SepalWidthCm

Results
All non-probabilistic models achieved 100% accuracy on the test set, demonstrating the dataset is well-structured for classification. The best performing model was saved as iris_model.joblib.

Key findings:

- Petal measurements are most discriminative for classification

- Sepal width has the most variability but least importance

- Clear separation between species in PCA space

- Optimal clustering matches the three actual species

