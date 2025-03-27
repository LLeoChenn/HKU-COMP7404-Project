# Local LDA for face recognize

[Reference paper](https://ieeexplore.ieee.org/document/1388259).

## 1. Introduction

This repository contains Python code for dimensionality reduction and classification tasks, specifically focusing on face data. The implemented algorithms include Locality - Preserving Discriminant Analysis with Gaussian Mixture Model (LLDA_GMM), Locality - Preserving Discriminant Analysis with K - Means (LLDA_KMeans), Linear Discriminant Analysis (LDA), and Kernel - based Linear Discriminant Analysis (KernelLDA). These algorithms are used to reduce the dimensionality of face data and then classify it using a K - Nearest Neighbors (KNN) classifier.

## 2. Prerequisites

Python: The code is written in Python. It is recommended to use Python 3.6 or higher.

Libraries:
numpy: Fundamental package for scientific computing with Python.
scikit - learn: Machine learning library in Python, used for various tasks like data splitting, clustering, discriminant analysis, and classification.
scipy: Scientific computing library, used for matrix operations and calculating distances.
matplotlib: Library for creating visualizations, used to plot eigenvalues and accuracy trends.

## 3. Excecution

**For simulated data**: run numeric_simulation.ipynb

**For Olivett data and FERET dataset** face recognition:

run FERET2jpg.m in matlab firstly to transform the photos to .jpg format;

then run LLDA_V3.ipynb for LLDA models.

remark: FERET data should be applied in [official website](https://www.nist.gov/itl/products-and-services/color-feret-database).
