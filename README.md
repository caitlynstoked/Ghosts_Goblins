
# Ghouls, Goblins, and Ghosts Classification

## Overview

This project tackles Kaggle’s *Ghouls, Goblins, and Ghosts… Boo!* competition, where the goal is to classify fictional creatures as **Ghoul**, **Goblin**, or **Ghost** based on physical and categorical attributes.

## Problem Statement

Given a set of features describing creatures (e.g., bone length, color, hair length), the task is to build a supervised classification model that accurately predicts the creature type. Model performance is evaluated using **classification accuracy**.

## Data

* **Training data:** Includes creature features and the target variable `type`.
* **Test data:** Includes the same features without labels; predictions are submitted to Kaggle.

## Exploratory Data Analysis (EDA)

Initial analysis included:

* Inspecting distributions and summaries (`skim`, `glimpse`)
* Visualizing numeric features by class using boxplots
* Exploring relationships between categorical variables and class labels with mosaic plots

## Feature Engineering

Preprocessing was handled using a `recipes` pipeline:

* Removed identifier column (`id`)
* Handled unknown and missing categorical values
* One-hot encoded categorical predictors
* Removed zero-variance predictors
* Normalized numeric features

## Models

Multiple classification models were trained and compared using **5-fold cross-validation**:

* **Naive Bayes** (klaR)
* **Support Vector Machine (Polynomial Kernel)**
* **Support Vector Machine (Radial Basis Function Kernel)**
* **Support Vector Machine (Linear Kernel)**

Each model’s hyperparameters were tuned via grid search to maximize accuracy.

## Model Selection & Training

For each algorithm:

1. Hyperparameters were tuned using cross-validation.
2. The best configuration (by accuracy) was selected.
3. The finalized workflow was trained on the full training dataset.

## Evaluation Metric

* **Accuracy:** Percentage of correctly classified creatures.

## Submission

Predictions were generated on the test set and saved in Kaggle’s required format:

```
id,type
0,Ghost
1,Goblin
2,Ghoul
```

Separate submission files were produced for each model to compare leaderboard performance.

## Tools & Libraries

* R, tidyverse
* tidymodels, recipes, workflows
* klaR, kernlab
* DataExplorer, skimr

## Citation

Wendy Kan. *Ghouls, Goblins, and Ghosts… Boo!* Kaggle Competition, 2016.


