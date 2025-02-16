<h1 align="center">Attention-based-Pandemic-Forecasting</h1>
<h3 align="center">Self-Attention based Architecture for Forecasting the Pandemic Spread in India</h3>

This repository contains the implementation of the paper titled **"Self-Attention based Architecture for Forecasting the Pandemic Spread in India"** published at **INDICON 24**. The work aims to use a self-attention mechanism for accurate prediction of COVID-19 spread in India. 
Paper Link - [Indicon 24'](https://ieeekharagpur.org/event/21st-ieee-india-council-international-conference-indicon-2024/)

## Overview

The goal of this project is to develop an effective forecasting model using self-attention mechanisms to predict the pandemic spread, particularly focusing on COVID-19 in India. We leverage the Self-Attention architecture, a part of transformer models, to capture long-range dependencies and improve the accuracy of pandemic predictions for different districts in India.

![Forecasting on the Map of India](https://drive.google.com/file/d/1Xtam5En9RnkK2O7yuvwKvycGaJImlpDx/view?usp=sharing)

## Key Features

- **Self-Attention Mechanism**: We employ self-attention to capture important patterns in the data across districts and time, helping the model to learn interdependencies more efficiently.
- **District-wise Forecasting**: The model is designed to forecast the pandemic spread in over 700 districts across India, using historical case data.
- **Model Evaluation**: The accuracy and performance of the model are evaluated using metrics like MAE, RMSE, and others, with comparative analysis to traditional time-series models.

## Requirements

You can install the necessary packages via `requirements.txt`:

```bash
pip install -r requirements.txt
```

# Usage
You can directly train the model by running the following command:
```
python main.py
```
for hyperparameter tunning, edit the `hyperparameter.py` file.

# Citation
If you find this work is useful in your research or applications, please consider citing our work by the following BibTeX entry.
```
    @article{
      }
```


## Acknowledgment
This project is supported by the Indian Council of Medical Research ([ICMR](https://covid19dashboard.mohfw.gov.in/)). We extend our gratitude to ICMR for providing the COVID-19 India Dataset, which was invaluable for this work.
