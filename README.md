<h1 align="center">Attention-based-Pandemic-Forecasting</h1>
<h3 align="center">Self-Attention based Architecture for Forecasting the Pandemic Spread in India</h3>

This repository contains the implementation of the paper titled **"Self-Attention based Architecture for Forecasting the Pandemic Spread in India"** published at **INDICON 24**. The work aims to use a self-attention mechanism for accurate prediction of COVID-19 spread in India. 
Paper Link - [Indicon 24'](https://ieeekharagpur.org/event/21st-ieee-india-council-international-conference-indicon-2024/)

## Overview

The goal of this project is to develop an effective forecasting model using self-attention mechanisms to predict the pandemic spread, particularly focusing on COVID-19 in India. We leverage the Self-Attention architecture, a part of transformer models, to capture long-range dependencies and improve the accuracy of pandemic predictions for different districts in India.

## Key Features

- **Self-Attention Mechanism**: We employ self-attention to capture important patterns in the data across districts and time, helping the model to learn interdependencies more efficiently.
- **District-wise Forecasting**: The model is designed to forecast the pandemic spread in over 700 districts across India, using historical case data.
- **Model Evaluation**: The accuracy and performance of the model are evaluated using metrics like MAE, RMSE, and others, with comparative analysis to traditional time-series models.

## Requirements

- Python 3.8+
- PyTorch
- Pandas
- Numpy
- Matplotlib
- Scikit-learn
- TensorFlow (if required for additional features)

You can install the necessary packages via `requirements.txt`:

```bash
pip install -r requirements.txt
