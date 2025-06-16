# RiskQPP Evaluation

This repository contains code for evaluating **Query Performance Prediction (QPP)** methods using both traditional and **risk-sensitive** metrics (GeoRisk, URisk, TRisk). It supports bootstrap sampling for statistical testing.

## Features

- **Traditional metrics:** Kendall's 𝜏, Pearson's r, Spearman's 𝜌, and sMARE
- **Risk-sensitive metrics:** GeoRisk, URisk, TRisk
- Bootstrap-based CSV export for metric variability

## Usage

Set your `BASE_PATH` and `PREDICTORS_ALL` in `config.py`, then run:

```bash
python run.py
```
Results are saved to `output/` and `output/bootstrap/`.

### Notes
- No preprocessed data or results are included
- Datasets and predictors are customizable

### Credits
- The implementation of GeoRisk was adapted from the following repository: [Haiga/risk-loss](https://github.com/Haiga/risk-loss).

<!-- ### Citation
If you use this code in your work, please cite our paper:

> Marçal, R. et al.  
> *A Robustness Assessment of Query Performance Prediction (QPP) Methods based on Risk-Sensitive Analysis*  
> In Proceedings of ICTIR, 2025.   -->