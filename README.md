# 🚨 Vancouver Daily Crime Forecasting & Extreme-Value Risk Analysis

**Author:** Seungjin Han  
**Course:** Machine Learning (A2), UTS FEIT  
**Date:** 2025-10-01

---

## 📄 Executive Summary

This project implements a robust, reproducible system for forecasting daily crime counts in Vancouver (2003–2017), using a **Seq2Seq LSTM** model with calendar embeddings for multi-horizon forecasts ($H = 1..7$ days ahead).

To address rare but critical spike days, the system augments standard forecasts with an **Extreme Value Theory (EVT)** layer applied to residual block maxima—yielding quantitative risk estimates for extreme crime surges.

**Key Features:**
- End-to-end reproducibility (seeded, strict time-split)
- Explicit deployment & I/O contracts
- Integrated EVT analysis for operational risk

---

## 🔗 Project Files & Resources

| File / Document             | Description                                                                        | Status / Link                                                                                       |
|-----------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **A2 Full Project Journal** | Comprehensive report: Task Definition (A), Implementation (B), Evaluation/Reflection (C) | A2 – Full Journal PDF                                                                              |
| **Colab Notebook**          | Self-contained code: model training, evaluation, EVT analysis                      | Colab File(github repo)                                                                                         |
| **Dataset (`crime.csv`)**   | Vancouver PD incident data (daily, 2003–2017)                                     | [Download CSV (Google Drive)](https://drive.google.com/file/d/1RO1h4JfhLKDZd6OtXeDGo0VhgkR9YZWP/view?usp=share_link) |

> **Environment / Libraries:**  
> Python 3.12 · TensorFlow 2.17+ · numpy · pandas · scipy  
> `kerashypetune` (grid search) · `fitter` (GEV distribution fitting)

---

## 📈 Methodology Overview

- **Forecasting Model:** Seq2Seq LSTM with calendar feature embeddings
- **Multi-horizon Prediction:** Predicts $H=1..7$ days ahead
- **EVT Risk Layer:** Fits Generalized Extreme Value (GEV) distributions to block maxima of forecast residuals for operational risk analysis

---

## 📦 Reproducibility & Deployment

- Fully seeded, time-ordered train/test split
- All code and results reproducible in Colab
- Explicit I/O contracts for deployment

---
