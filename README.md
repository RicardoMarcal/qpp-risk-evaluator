# RiskQPP Evaluator

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

### Data Format

The input prediction files must be CSVs with the following columns:

- `tid`: query/topic identifier
- `scores`: predictor scores
- `value`: ground truth effectiveness metric (e.g., MAP or nDCG)

### Credits
- The implementation of GeoRisk was adapted from the following repository: [Haiga/risk-loss](https://github.com/Haiga/risk-loss).

### Citation

This framework was developed to support the paper: *A Robustness Assessment of Query Performance Prediction (QPP) Methods based on Risk-Sensitive Analysis*.
If you use this code or ideas from it in your work, please cite the corresponding publication.

### License

Apache 2.0 License