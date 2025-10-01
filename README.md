A2 – Option 2: Vancouver Daily Crime Forecasting & Extreme-Value Risk Analysis

Author: Seungjin Han Course: Machine Learning (A2) · UTS FEIT Date: 2025-10-01

📄 Executive Summary

This project implements a practical system for forecasting daily crime counts for Vancouver (2003–2017) using a Seq2Seq LSTM with calendar embeddings for multi-horizon forecasts ($H=1..7$). The model consistently outperforms naive baselines across all horizons.
Crucially, to address rare but operationally critical spike days, the system complements point forecasts with an Extreme Value Theory (EVT) layer applied to residual block maxima. This analysis estimates return levels with bootstrap confidence intervals, providing quantifiable risk assessment for resource allocation.
The project is designed to be fully reproducible (seeded, time-ordered split) and includes explicit Deployment & I/O Contracts.

--------------------------------------------------------------------------------
🔗 Project Files and Reproducibility
The project implementation is self-contained, including environment setup and data preprocessing, as required by the A2 specification.
File / Document
Description
Status
A2 Full Project Journal
Comprehensive report detailing Task Definition (A), Implementation (B), and Evaluation/Reflection (C).
[A2 – Full Journal PDF Link]
Colab Notebook
Self-contained implementation code for training, evaluation, and EVT analysis.
[Colab File Link]
Dataset (crime.csv)
Vancouver Police Department incident data (2003–2017 daily aggregation).
[Dataset CSV Download Link]
Required Environment/Libraries: Python 3.12, TensorFlow 2.17+, numpy, pandas, scipy, kerashypetune (for grid search), and fitter (for GEV distribution fitting).
