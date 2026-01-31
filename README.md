# Order Book Alpha Modeling

## Overview

This project investigates whether **limit order book dynamics contain predictive information** about short-horizon mid-price movements. Using high-frequency order book data for **AAPL**, the goal is to build a **reproducible alpha research pipeline** that spans raw data parsing, feature engineering, model training, and rigorous out-of-sample validation.

The emphasis of this repository is **predictive signal discovery and validation**, not the construction of a deployable trading strategy.

The project is structured to reflect real-world quantitative research workflows, combining **C++ for high-throughput data processing** and **Python for modeling and evaluation**.

---

## Research Objective

The primary objective is to predict **200ms ahead mid-price log returns** using features derived from the limit order book (MBP-10).

Success is evaluated using **out-of-sample predictive metrics**, with particular focus on robustness and temporal stability rather than raw profitability.

At ultra-short horizons, price movements are dominated by noise; therefore, even modest predictive performance can be statistically meaningful.

---

## Data

- **Instrument:** AAPL  
- **Venue / Source:** Databento  
- **Data Type:** MBP-10 (top 10 levels of the order book)  
- **Sampling Frequency:** 200ms (calendar time)  
- **Time Span:** July 3, 2025 – October 2, 2025  
- **Mid-price:** (Best Bid + Best Ask) / 2  

Raw order book messages are parsed and reconstructed in C++, then cleaned and sampled into fixed-interval snapshots before feature generation.

---

## Feature Engineering

Features are engineered from reconstructed order book snapshots and include:

- Top-of-book metrics (spread, normalized spread)
- Depth imbalance across multiple levels
- Order book slope and convexity
- Event and trade intensity measures
- Short-horizon realized volatility proxies

All features are computed using **only information available at or before time _t_** to prevent data leakage.

---

## Prediction Target

The primary target is the **log return of the mid-price at a 200ms horizon**, defined as:

y_t = log(m_{t+200ms}) - log(m_t)

A dead zone (epsilon threshold) is used for certain directional evaluations to reduce the influence of microstructure noise.

---

## Models

The following models are implemented and compared:

### Linear Models
- Ordinary Least Squares
- Ridge Regression
- Lasso Regression

### Tree-Based Models
- Random Forest
- Gradient Boosting
- XGBoost

Linear models serve as interpretable baselines, while tree-based models capture nonlinear interactions and feature dependencies.

---

## Evaluation Framework

Model evaluation focuses on **out-of-sample predictive validity** and robustness rather than trading performance.

### Primary Metrics
1. Out-of-sample R² (main success metric)
2. Information Coefficient (IC)
3. Directional accuracy with a dead zone

### Stability & Robustness
4. Per-day / per-session stability analysis
5. Horizon decay analysis
6. Robustness checks:
   - Latency simulation (label shifting)
   - Feature ablation studies

All train / validation / test splits are performed **chronologically** to avoid look-ahead bias.

---

## Indicative Trading Sanity Check

A simplified trading simulation is included only as a **sanity check** to verify that the learned signals are directionally consistent with realized price movements under conservative assumptions.

This simulation:
- uses fixed, non-optimized rules,
- includes basic transaction costs,
- is **not** intended to represent a deployable strategy.

Trading metrics (e.g., Sharpe ratio) are treated as **secondary diagnostics** and are not used for model selection.

---

## Results

> 📌 **Placeholder — results to be added**

This section will summarize:
- Out-of-sample R² across models
- IC distributions and stability
- Performance by trading day
- Horizon decay behavior
- Key robustness findings

---

## Repository Structure

```
.
├── AAPL_data.cpp
├── MSFT_data.cpp
├── cleaned_data/
├── Data/
├── models/
├── plots/
├── results/
├── checks/
├── utils.py
├── create_features_AAPL.py
├── create_labels_AAPL.py
├── phase1_linear.py
├── phase2_randomforest.py
├── phase3_gradientboosting.py
├── phase4_xgboost.py
├── validate_models.py
└── README.md
```

---

## Reproducibility

- All preprocessing and modeling steps are deterministic given fixed random seeds.
- Feature schemas and labels are versioned.
- Evaluation is strictly out-of-sample with chronological splits.
- Raw data is excluded due to licensing constraints.

---

## Disclaimer

This repository is intended for **research and educational purposes only**.
It does not constitute investment advice or a production trading system.
