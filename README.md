# Snow Prediction for Tomorrow

Predict whether it will snow tomorrow for each weather station using time-aware data splits, temporal features, and tuned binary classifiers.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)

## Project Overview

This project builds a snow prediction model using historical weather data from the GSOD (Global Summary of the Day) dataset. The goal is to predict binary snow occurrence for the next day at each station.

**Key Characteristics:**
- **Target Variable**: `y_tomorrow` - indicates if snow occurs the next calendar day
- **Time Period**: 2000-2005 (6 years of data)
- **Stations**: 725300-725330 range with complete coverage across all years
- **Approach**: Time-aware train/validation/test splits with temporal feature engineering

## Repository Structure

```
snow-prediction/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── data/
│   ├── external/
│   │   └── coding_challenge.csv       # Original data source
│   ├── interim/
│   │   ├── part1_task1.csv           # Extracted weather data from BigQuery
│   │   └── stations_ok.csv           # Filtered stations with full coverage
│   └── processed/
│       ├── train.csv                  # Training split
│       ├── valid.csv                  # Validation split
│       ├── test.csv                   # Test split
│       ├── test_predictions_lightgbm.csv      # LightGBM predictions
│       ├── test_metrics_lightgbm.csv          # LightGBM metrics
│       ├── test_predictions_logreg.csv        # Logistic Regression predictions
│       └── test_metrics_logreg.csv            # Logistic Regression metrics
└── notebook/
    ├── 00_setup_and_checks.ipynb              # Environment setup
    ├── 01_data_extraction_bigquery.ipynb      # Extract data from BigQuery
    ├── 02_station_coverage_analysis.ipynb     # Filter stations by coverage
    ├── 03_data_splitting_preprocessing.ipynb  # Create train/valid/test splits
    ├── 04_model_training_lightgbm.ipynb       # Train LightGBM model
    └── 05_model_training_logistic_regression.ipynb  # Train Logistic Regression model
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Execute notebooks in sequence:

1. **00_setup_and_checks.ipynb** - Verify environment and dependencies
2. **01_data_extraction_bigquery.ipynb** - Query and extract weather data
3. **02_station_coverage_analysis.ipynb** - Identify stations with complete temporal coverage
4. **03_data_splitting_preprocessing.ipynb** - Generate time-aware splits and target variable
5. **04_model_training_lightgbm.ipynb** - Train gradient boosting model
6. **05_model_training_logistic_regression.ipynb** - Train baseline linear model

## Data Pipeline

### 1. Data Extraction
- **Source**: BigQuery `bigquery-public-data.samples.gsod`
- **Station Range**: 725300-725330
- **Temporal Range**: 2000-2005

### 2. Station Filtering
- **Criteria**: Must have data for all 6 years (2000-2005)
- **Output**: `stations_ok.csv` with qualifying stations

### 3. Target Engineering
- **Method**: Per-station sort by date, then shift snow flag by -1 day
- **Result**: `y_tomorrow` indicates next-day snow occurrence

### 4. Time-Aware Splitting
- **Test Date**: 20 years before today minus 1 day
- **Validation Window**: 60 days before test date
- **Training Window**: All data before validation window
- **Prevents**: Temporal leakage and ensures realistic evaluation

## Feature Engineering

### Numeric Features
- `total_precipitation` - Daily precipitation amount
- `mean_temp` - Average temperature
- `max_temperature` - Maximum temperature

### Boolean Weather Flags
- `rain`, `fog`, `hail`, `thunder`, `tornado`

### Temporal Features

**Lag Features** (per-station historical values):
- Lag 1, 3, 7, and 14 days for numeric features

**Rolling Statistics**:
- 3-day rolling mean for numeric features

**Seasonality**:
- Month and day-of-year encoded as sine/cosine pairs
- Captures cyclical patterns without ordinality

### Excluded Features
- `min_temperature` - 100% missing values
- `snow_depth` - ~96% missing values

## Models

### LightGBM (Gradient Boosting)
- **Algorithm**: `LGBMClassifier` 
- **Strengths**: Captures non-linear interactions, handles feature interactions automatically
- **Use Case**: Primary model for complex weather patterns

### Logistic Regression
- **Algorithm**: Linear classifier with L2 regularization
- **Preprocessing**: Median imputation + StandardScaler
- **Use Case**: Fast, interpretable baseline model

### Threshold Tuning
- **Method**: Grid search over probability thresholds on validation set
- **Metric**: Optimize F1 score
- **Rationale**: Default 0.5 threshold is suboptimal for imbalanced classes
- **Workflow**: 
  1. Fit model on training data
  2. Tune threshold on validation set
  3. Refit on train+valid with chosen threshold
  4. Evaluate on test set

## Results

### Model Comparison

| Model | AUC | F1 | Accuracy | Precision | Recall |
|-------|-----|----| ---------|-----------|--------|
| LightGBM | - | - | - | 1.0 | 0.5 |
| Logistic Regression | Higher | - | - | 1.0 | 0.5 |

**Key Findings**:
- ✅ Both models achieved **perfect precision** (no false positives)
- ⚠️ Both models achieved **50% recall** (caught 1 of 2 snow events)
- 📊 Logistic Regression had **higher AUC** (better probability ranking)
- 🎯 After threshold tuning, both produced **identical label predictions**

### Metrics Explained

**AUC (Area Under ROC Curve)**
- Measures how well the model ranks positive cases above negative cases
- Independent of threshold choice
- Higher values indicate better probability estimates

**F1 Score**
- Harmonic mean of precision and recall
- Balances false positives vs false negatives
- Optimal for imbalanced classification tasks

**Threshold**
- Decision boundary for converting probabilities to labels
- Default 0.5 is often suboptimal for imbalanced data
- Tuned on validation set to maximize F1 score

### Outputs

All results are saved to `data/processed/`:

- **Train/Valid/Test Splits**: `train.csv`, `valid.csv`, `test.csv`
- **LightGBM**: `test_predictions_lightgbm.csv`, `test_metrics_lightgbm.csv`
- **Logistic Regression**: `test_predictions_logreg.csv`, `test_metrics_logreg.csv`

---

**Note**: This project demonstrates end-to-end ML workflow with proper temporal validation, feature engineering, and threshold optimization for production-ready snow prediction.
