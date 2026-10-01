# 📰 Fake News Detection using NLP & Machine Learning

An end-to-end **Fake News Detection system** built using **Natural Language Processing (NLP) and Machine Learning** to classify news articles as **Fake or Real**.

This project covers the complete machine learning workflow from **data collection and exploration to text preprocessing, feature engineering, model training, evaluation, and prediction**.

## 🚀 Project Workflow

**Raw News Data → Data Cleaning → EDA → NLP Preprocessing → Feature Extraction → Train/Test Split → Model Training → Evaluation → Prediction**

### 🔹 1. Data Understanding & Exploration

* Loaded and inspected the news dataset using **Pandas**
* Checked dataset shape, columns, data types, and missing values
* Identified duplicate records
* Analyzed the distribution of Fake and Real news
* Performed exploratory data analysis (EDA)
* Examined text length and other relevant patterns

### 🔹 2. Data Preprocessing

Applied NLP techniques to clean and prepare the news text:

* Handling missing values
* Removing duplicate data
* Converting text to lowercase
* Removing punctuation and unnecessary characters
* Removing stopwords
* Tokenization
* Text normalization
* Cleaning unwanted/irrelevant content

### 🔹 3. Feature Engineering with NLP

Converted unstructured text into numerical features that machine learning models can understand.

* **Bag of Words**
* **TF-IDF (Term Frequency–Inverse Document Frequency)**
* Understanding word frequency and importance
* Creating numerical representations of news articles

### 🔹 4. Machine Learning

Experimented with machine learning classification algorithms to identify patterns between Fake and Real news.

Models explored include:

* Logistic Regression
* Naive Bayes
* Decision Tree
* Random Forest
* Other classification approaches where applicable

### 🔹 5. Model Evaluation

Evaluated the trained models using multiple classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

The goal was not only to measure overall accuracy but also to understand **false positives and false negatives**, which are particularly important in fake-news classification.

### 🔹 6. Prediction Pipeline

Created a prediction workflow where new/unseen news text can be passed through the same preprocessing and feature-extraction pipeline and classified as:

> 🟢 **REAL NEWS**
> 🔴 **FAKE NEWS**

## 🧠 Key Concepts Covered

* Natural Language Processing (NLP)
* Text preprocessing
* Tokenization
* Stopword removal
* Feature extraction
* Bag of Words
* TF-IDF
* Supervised Machine Learning
* Binary Classification
* Train/Test Split
* Model Evaluation
* Confusion Matrix
* Precision, Recall & F1-Score
* Generalization to unseen text

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **NLTK**
* **Scikit-learn**
* **Jupyter Notebook**

## 📌 Important Note

This project demonstrates **automated text classification**, not factual verification of news. A model predicting that an article is "Fake" does not independently prove that the underlying claim is false. Real-world misinformation detection can require additional sources, fact-checking, source credibility analysis, and contextual verification.

## 🎯 Learning Objective

The main objective of this project was to understand and implement an **end-to-end NLP + Machine Learning pipeline** rather than simply training a classification model.

This project serves as a practical implementation of concepts learned in **Machine Learning, NLP, and text classification**.
