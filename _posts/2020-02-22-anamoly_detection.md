---
title: "Anamoly Detection"
last_modified_at: 2020-02-22T16:20:02-05:00
categories:
  - Blog
tags:
  - machine learning
  - python
  - data science
#header:
#  image: "/assets/images/frauddetection.jpeg"
excerpt: "Machine Learning, Perceptron, Data Science"
---


#### [Credit Card Fraud Detection](https://github.com/praveenpareek11/Anamoly-Detection)


## Introduction
In this kernel we will use various predictive models to see how accurate they are in detecting whether a transaction is a normal payment or a fraud. As described in the dataset, the features are scaled and the names of the features are not shown due to privacy reasons. Nevertheless, we can still analyze some important aspects of the dataset. Let's start!

## Goals:
- Understand the little distribution of the "little" data that was provided to us.
- Create a 50/50 sub-dataframe ratio of "Fraud" and "Non-Fraud" transactions. (NearMiss Algorithm)
- Determine the Classifiers we are going to use and decide which one has a higher accuracy.
- Create a Neural Network and compare the accuracy to our best classifier.
- Understand common mistaked made with imbalanced datasets.

## Outline:
1. Understanding our data
    1. Gather Sense of our data

2. Preprocessing
    1. Scaling and Distributing
    2. Splitting the Data


3. Random UnderSampling and Oversampling
    1. Distributing and Correlating
    2. Anomaly Detection
    3. Dimensionality Reduction and Clustering (t-SNE)
    4. Classifiers
    5. A Deeper Look into Logistic Regression
    6. Oversampling with SMOTE


4. Testing
    1. Testing with Logistic Regression
    2. Neural Networks Testing (Undersampling vs Oversampling)

For more reading follow this [link](https://github.com/praveenpareek11/Anamoly-Detection)
