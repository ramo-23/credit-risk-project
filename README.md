# Credit Risk - Probability of Default (PD) Modelling Project

## Overview of the Project

**Objective**: This projects seeks to build, and validate a Probability of Default (PD) model for unsecured lending of funds. This project covers a wide range of topics such as: data cleaning, feature engineering, model development, validation and monitoring.

## Dataset used
- Raw Data: the raw dataset used can be found at [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Processed datasets and artifacts can be found in this folder once the project has been cloned: [data/processed](data/processed) and [artifacts](artifacts)

# Project workflow
The project itself is divided into six detailed steps, and scripts to go through all six steps.

- **Phase 1 - Data Cleaning**
	- Purpose: clean inputs, handle missingness, and produce EDA summaries.
	- Key outputs: cleaned dataset and EDA plots.
	- Files: [src/data_preprocessing.py](src/data_preprocessing.py)

- **Phase 2 - Feature Engineering & Selection**
	- Purpose: derive predictive features, encode categories, and select stable predictors.
	- Key outputs: `data/processed/accepted_fe.csv`, feature importance tables.
	- Files: [src/feature_engineering.py](src/feature_engineering.py)

- **Phase 3 — Modeling & Calibration**
	- Purpose: train PD models, calibrate probabilities, and produce primary (champion) model.
	- Key outputs: serialized model artifacts, calibration plots, performance metrics.
	- Files: [src/modeling.py](src/modeling.py), [src/calibration.py](src/calibration.py)