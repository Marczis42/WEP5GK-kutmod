
# Titanic — Machine Learning from Disaster

[Kaggle: Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview)

## Project Overview

The Titanic dataset is a classic binary classification problem: given passenger information (name, age, gender, ticket class, etc.), predict whether a passenger survived the disaster.

This repository demonstrates a simple end-to-end workflow: data loading and exploration, data cleaning and feature engineering, model training and evaluation, and generating predictions for submission.

## Goals

- Build and evaluate baseline machine learning models to predict passenger survival.
- Apply data-processing and feature-engineering techniques to improve model performance.
- Produce a submission file compatible with the Kaggle competition format.

## Contents

- `data/raw/` — original dataset files (train/test).
- `data/processed/` — processed datasets used for modeling.
- `notebooks/` — Jupyter notebooks for exploration, processing, and modeling.
- `src/` — helper modules for data loading, processing, and feature engineering.

## Methodology

1. Explore the raw data and identify missing values and feature distributions.
2. Clean and impute missing values; construct informative features.
3. Encode categorical variables and split the train set for validation.
4. Train models (e.g., Random Forest, SVC), evaluate on validation data, and analyze feature importance.
5. Generate predictions on the test set and export a submission CSV.

## Usage

Set up the Python environment (see `requirements.txt`), then run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_processing.ipynb`
3. `notebooks/03_modeling.ipynb`

Optionally, import helpers from `src/` to run parts of the pipeline programmatically.

